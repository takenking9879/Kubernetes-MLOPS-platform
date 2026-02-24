/**
 * Central Zustand store for the pipeline editor.
 *
 * Manages: nodes, edges, schema, validation, dry-run, and console messages.
 * All async actions propagate errors to the console (fail-fast to the UI).
 */

import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import type { Node } from 'reactflow';
import type { NodeData } from '../types/nodes';
import { isStageNode } from '../types/nodes';
import type { FlowEdge, EdgeSelector } from '../types/edges';
import { createDefaultSelector } from '../types/edges';
import type { SparkSchema } from '../types/schema';
import type { ValidationResult } from '../types/validation';
import type { DryRunResult } from '../types/dryrun';
import {
  fetchSchemaFromIceberg,
  uploadSchemaFromCSV,
  executeDryRun as apiDryRun,
} from '../api/client';
import { validatePipelineLocally } from '../lib/validation/lafValidator';
import { generateYAML } from '../lib/yaml/generator';
import { importFromYAML } from '../lib/yaml/importer';
import { computeSchemaHashSync } from '../lib/schemaHash';
import { DATASET_NODE_CONTRACT, FINAL_FEATURES_NODE_CONTRACT } from '../registry/NodeRegistry';
import {
  collectFinalFeatureColumns,
  dedupeBuckets,
  validateFinalFeatureSelections,
} from '../lib/finalFeatures/model';

// ─── Console message ────────────────────────────────────────────────

export interface ConsoleMessage {
  readonly id: string;
  readonly timestamp: number;
  readonly type: 'info' | 'error' | 'warning';
  readonly message: string;
}

let msgCounter = 0;
function createMsg(message: string, type: ConsoleMessage['type']): ConsoleMessage {
  return { id: `msg-${++msgCounter}`, timestamp: Date.now(), type, message };
}

// ─── Dirty tracking helper ──────────────────────────────────────────

/**
 * Find all downstream node IDs reachable from `startIds` via edges.
 * Returns the full set of dirty nodes (including startIds).
 */
function findDownstream(startIds: Iterable<string>, edges: readonly FlowEdge[]): Set<string> {
  const outgoing = new Map<string, string[]>();
  for (const edge of edges) {
    const arr = outgoing.get(edge.source) ?? [];
    arr.push(edge.target);
    outgoing.set(edge.source, arr);
  }

  const dirty = new Set<string>();
  const queue = [...startIds];

  while (queue.length > 0) {
    const current = queue.shift()!;
    if (dirty.has(current)) continue;
    dirty.add(current);
    for (const child of outgoing.get(current) ?? []) {
      if (!dirty.has(child)) queue.push(child);
    }
  }

  return dirty;
}

// ─── Store shape ────────────────────────────────────────────────────

interface PipelineState {
  // State
  nodes: Node<NodeData>[];
  edges: FlowEdge[];
  dirtyNodeIds: Set<string>;
  datasetSchema: SparkSchema | null;
  validationResult: ValidationResult | null;
  isValidating: boolean;
  dryRunResult: DryRunResult | null;
  isDryRunning: boolean;
  selectedNodeId: string | null;
  selectedEdgeId: string | null;
  consoleMessages: ConsoleMessage[];

  // Pipeline metadata
  pipelineName: string;
  pipelineVersion: string;

  // Node actions
  addNode: (node: Node<NodeData>) => void;
  updateNodeData: (id: string, updater: (prev: NodeData) => NodeData) => void;
  deleteNode: (id: string) => void;
  setNodes: (nodes: Node<NodeData>[]) => void;

  // Edge actions
  addEdge: (source: string, target: string, selector?: EdgeSelector) => FlowEdge;
  updateEdgeSelector: (id: string, selector: EdgeSelector) => void;
  deleteEdge: (id: string) => void;
  setEdges: (edges: FlowEdge[]) => void;

  // Dataset
  uploadSchemaFromCSV: (file: File) => Promise<void>;
  loadSchemaFromIceberg: (table: string) => Promise<void>;
  setDatasetSchema: (schema: SparkSchema) => void;

  // Validation
  validate: (opts?: { silent?: boolean }) => ValidationResult;

  // YAML
  exportYAML: () => string;
  importYAML: (yamlString: string) => void;

  // Dry-run
  dryRun: () => Promise<DryRunResult>;

  // Pipeline metadata
  setPipelineName: (name: string) => void;
  setPipelineVersion: (version: string) => void;

  // UI
  selectNode: (id: string | null) => void;
  selectEdge: (id: string | null) => void;
  log: (message: string, type: ConsoleMessage['type']) => void;
  clearConsole: () => void;
}

// ─── Store implementation ───────────────────────────────────────────

export const usePipelineStore = create<PipelineState>()(
  devtools(
    (set, get) => ({
      // Initial state
      nodes: [],
      edges: [],
      dirtyNodeIds: new Set(),
      datasetSchema: null,
      validationResult: null,
      isValidating: false,
      dryRunResult: null,
      isDryRunning: false,
      selectedNodeId: null,
      selectedEdgeId: null,
      consoleMessages: [],
      pipelineName: 'visual_pipeline',
      pipelineVersion: '1.0',

      // ── Node actions ───────────────────────────────────────────

      addNode: (node) => {
        const { nodes } = get();
        const instances = nodes.filter((n) => n.data.type === node.data.type).length;
        if (node.data.type === 'dataset' && instances >= DATASET_NODE_CONTRACT.maxInstances) {
          get().log('Dataset already exists. Only one Dataset node is allowed.', 'warning');
          return;
        }
        if (node.data.type === 'final_features' && instances >= FINAL_FEATURES_NODE_CONTRACT.maxInstances) {
          get().log('Final Features already exists. Only one Final Features node is allowed.', 'warning');
          return;
        }

        set((s) => ({
          nodes: [...s.nodes, node],
          dirtyNodeIds: new Set(s.nodes.map((n) => n.id).concat(node.id)),
        }));
      },

      updateNodeData: (id, updater) => {
        set((s) => ({
          nodes: s.nodes.map((n) =>
            n.id === id ? { ...n, data: updater(n.data) } : n,
          ),
          dirtyNodeIds: new Set([...s.dirtyNodeIds, ...findDownstream([id], s.edges)]),
        }));
      },

      deleteNode: (id) => {
        set((s) => ({
          nodes: s.nodes.filter((n) => n.id !== id),
          edges: s.edges.filter((e) => e.source !== id && e.target !== id),
          selectedNodeId: s.selectedNodeId === id ? null : s.selectedNodeId,
          dirtyNodeIds: new Set(s.nodes.map((n) => n.id)),
        }));
      },

      setNodes: (nodes) => set({ nodes }),

      // ── Edge actions ───────────────────────────────────────────

      addEdge: (source, target, selector) => {
        const edge: FlowEdge = {
          id: `e-${source}-${target}-${Date.now()}`,
          source,
          target,
          selector: selector ?? createDefaultSelector(),
          resolved: false,
          resolvedColumns: [],
        };
        set((s) => ({
          edges: [...s.edges, edge],
          dirtyNodeIds: new Set([...s.dirtyNodeIds, ...findDownstream([target], [...s.edges, edge])]),
        }));
        return edge;
      },

      updateEdgeSelector: (id, selector) => {
        set((s) => {
          const edge = s.edges.find((e) => e.id === id);
          const targetId = edge?.target;
          const downstream = targetId ? findDownstream([targetId], s.edges) : new Set<string>();
          return {
            edges: s.edges.map((e) =>
              e.id === id ? { ...e, selector, resolved: false, resolvedColumns: [] } : e,
            ),
            dirtyNodeIds: new Set([...s.dirtyNodeIds, ...downstream]),
          };
        });
      },

      deleteEdge: (id) => {
        set((s) => {
          const edge = s.edges.find((e) => e.id === id);
          const newEdges = s.edges.filter((e) => e.id !== id);
          const targetId = edge?.target;
          const downstream = targetId ? findDownstream([targetId], newEdges) : new Set<string>();
          return {
            edges: newEdges,
            selectedEdgeId: s.selectedEdgeId === id ? null : s.selectedEdgeId,
            dirtyNodeIds: new Set([...s.dirtyNodeIds, ...downstream]),
          };
        });
      },

      setEdges: (edges) => set({ edges }),

      // ── Dataset loading ────────────────────────────────────────


      uploadSchemaFromCSV: async (file) => {
        try {
          const schema = await uploadSchemaFromCSV(file);
          set((state) => ({
            datasetSchema: schema,
            nodes: state.nodes.map((node) => {
              if (node.data.type !== 'dataset') return node;
              if (!schema.uploadedPath) return node;
              return {
                ...node,
                data: {
                  ...node.data,
                  path: schema.uploadedPath,
                },
              };
            }),
          }));

          const pathInfo = schema.uploadedPath ? ` (${schema.uploadedPath})` : '';
          get().log(
            `Schema loaded: ${schema.columns.length} columns from uploaded file: ${file.name}${pathInfo}`,
            'info',
          );
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          get().log(`Failed to upload CSV: ${msg}`, 'error');
          throw err;
        }
      },

      loadSchemaFromIceberg: async (table) => {
        try {
          const schema = await fetchSchemaFromIceberg(table);
          set((state) => ({
            datasetSchema: schema,
            nodes: state.nodes.map((node) => {
              if (node.data.type !== 'dataset') return node;
              return {
                ...node,
                data: {
                  ...node.data,
                  path: schema.uploadedPath || table,
                },
              };
            }),
          }));
          get().log(`Schema loaded: ${schema.columns.length} columns from ${table}`, 'info');
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          get().log(`Failed to load Iceberg schema: ${msg}`, 'error');
          throw err;
        }
      },

      setDatasetSchema: (schema) => set({ datasetSchema: schema }),

      // ── Validation ─────────────────────────────────────────────

      validate: (opts) => {
        set({ isValidating: true });

        const { nodes, edges, datasetSchema, dirtyNodeIds, validationResult: prevResult } = get();
        const result = validatePipelineLocally(nodes, edges, datasetSchema, {
          previousResult: prevResult ?? undefined,
          dirtyNodeIds: dirtyNodeIds.size > 0 ? dirtyNodeIds : undefined,
        });

        set((state) => ({
          validationResult: result,
          isValidating: false,
          dirtyNodeIds: new Set(),
          nodes: state.nodes.map((node) => {
            const data = node.data;
            if (!isStageNode(data)) return node;

            const compiled = result.compiledNodes.get(node.id);
            if (!compiled) {
              return {
                ...node,
                data: {
                  ...data,
                  compilation: { status: 'draft' },
                  resolvedInputs: [],
                  resolvedOutputs: [],
                  outputSchema: null,
                },
              };
            }

            return {
              ...node,
              data: {
                ...data,
                compilation: compiled.compilation,
                resolvedInputs: [...compiled.resolvedInputs],
                resolvedOutputs: [...compiled.resolvedOutputs],
                outputSchema: compiled.outputSchema,
              },
            };
          }),
          edges: state.edges.map((edge) => {
            const compiled = result.compiledEdges.get(edge.id);
            if (!compiled) {
              return { ...edge, resolved: false, resolvedColumns: [] };
            }
            return {
              ...edge,
              resolved: true,
              resolvedColumns: [...compiled.resolvedColumns],
            };
          }),
        }));

        const silent = opts?.silent === true;
        if (!silent) {
          if (result.valid) {
            get().log(`Pipeline valid (${result.topologicalOrder.length} stages)`, 'info');
          } else {
            get().log(`Validation failed: ${result.errors.length} error(s)`, 'error');
            for (const err of result.errors) {
              get().log(`  ${err.message}`, 'error');
            }
          }

          for (const w of result.warnings) {
            get().log(`  Warning: ${w.message}`, 'warning');
          }
        }

        return result;
      },

      // ── YAML export ────────────────────────────────────────────

      exportYAML: () => {
        const result = get().validate();
        if (!result.valid) {
          get().log('Cannot export YAML: pipeline has validation errors', 'error');
          return '';
        }

        const hasUncompiledStages = get().nodes.some(
          (node) => isStageNode(node.data) && node.data.compilation.status !== 'compiled',
        );

        if (hasUncompiledStages) {
          get().log('Cannot export YAML: some stages are still draft or have compilation errors', 'error');
          return '';
        }

        const { nodes, edges, pipelineName, pipelineVersion, datasetSchema, validationResult } = get();

        const datasetNodes = nodes.filter((n) => n.data.type === 'dataset');
        if (datasetNodes.length !== 1) {
          get().log('Cannot export YAML: exactly one Dataset node is required (E_DATASET_SINGLETON)', 'error');
          return '';
        }

        const finalNodes = nodes.filter((n) => n.data.type === 'final_features');
        if (finalNodes.length !== 1) {
          get().log('Cannot export YAML: exactly one Final Features node is required (E_FF_MISSING)', 'error');
          return '';
        }

        const finalNode = finalNodes[0];
        if (!finalNode || finalNode.data.type !== 'final_features') {
          get().log('Cannot export YAML: invalid Final Features node state', 'error');
          return '';
        }

        const incomingToFinal = edges.filter((e) => e.target === finalNode.id);
        const outgoingFromFinal = edges.filter((e) => e.source === finalNode.id);
        if (incomingToFinal.length < 1) {
          get().log('Cannot export YAML: Final Features must have at least one upstream producer (E_FF_DISCONNECTED)', 'error');
          return '';
        }
        if (outgoingFromFinal.length > 0) {
          get().log('Cannot export YAML: Final Features is terminal and cannot have outgoing edges (E_FF_OUTGOING_EDGE)', 'error');
          return '';
        }

        const selected = dedupeBuckets({
          feature: finalNode.data.features,
          target: finalNode.data.target,
          metadata: finalNode.data.metadata,
          passthrough: finalNode.data.passthrough,
        });

        const availableColumns = collectFinalFeatureColumns(
          nodes,
          edges,
          finalNode.id,
          validationResult,
          datasetSchema,
        );

        const ffIssues = validateFinalFeatureSelections(
          availableColumns,
          selected,
          finalNode.data.confirmedColumns,
        );

        if (ffIssues.length > 0) {
          for (const issue of ffIssues) {
            get().log(`Cannot export YAML: ${issue.message} (${issue.code})`, 'error');
          }
          return '';
        }

        get().updateNodeData(finalNode.id, (prev) => ({
          ...prev,
          ...selected,
        }));

        const schemaHash = datasetSchema ? computeSchemaHashSync(datasetSchema) : undefined;
        const yaml = generateYAML(nodes, result.topologicalOrder, {
          pipelineName,
          pipelineVersion,
          schemaHash,
        });
        get().log('YAML generated successfully', 'info');
        return yaml;
      },

      importYAML: (yamlString) => {
        try {
          const { datasetSchema, nodes: oldNodes } = get();
          const result = importFromYAML(yamlString);

          // Preserve dataset path and schema if they were already loaded in the session
          const oldDataset = oldNodes.find((n) => n.data.type === 'dataset');
          const nodes = result.nodes.map((node) => {
            if (node.data.type === 'dataset') {
              const prevPath = oldDataset?.data.type === 'dataset' ? oldDataset.data.path : '';
              const prevSource = oldDataset?.data.type === 'dataset' ? oldDataset.data.source : 'csv';
              return {
                ...node,
                data: {
                  ...node.data,
                  path: node.data.path || prevPath,
                  source: node.data.path ? node.data.source : prevSource,
                  schema: node.data.schema || datasetSchema,
                },
              };
            }
            return node;
          });

          set({
            nodes,
            edges: result.edges,
            pipelineName: result.pipelineName,
            pipelineVersion: result.pipelineVersion,
            validationResult: null,
            dryRunResult: null,
            selectedNodeId: null,
            selectedEdgeId: null,
          });

          get().log(
            `Imported pipeline "${result.pipelineName}" with ${result.nodes.length - 1} stages`,
            'info',
          );

          for (const w of result.warnings) {
            get().log(`Import warning: ${w}`, 'warning');
          }

          if (result.schemaHash) {
            get().log(`Schema hash from YAML: ${result.schemaHash}`, 'info');
          }

          // Run silent validation after import
          get().validate({ silent: true });
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          get().log(`YAML import failed: ${msg}`, 'error');
        }
      },

      // ── Dry-run ────────────────────────────────────────────────

      dryRun: async () => {
        const yaml = get().exportYAML();
        if (!yaml) {
          const fail: DryRunResult = {
            success: false,
            error: { message: 'Cannot dry-run: YAML generation failed' },
          };
          set({ dryRunResult: fail });
          return fail;
        }

        set({ isDryRunning: true });

        try {
          const datasetNode = get().nodes.find((n) => n.data.type === 'dataset');
          if (!datasetNode || datasetNode.data.type !== 'dataset') {
            throw new Error('No dataset node in pipeline');
          }

          const result = await apiDryRun({
            yaml,
            datasetPath: datasetNode.data.path,
            sampleLimit: 1000,
          });

          set({ dryRunResult: result, isDryRunning: false });

          if (result.success) {
            get().log(`Dry-run succeeded (${result.metrics.executionTime.toFixed(2)}s)`, 'info');
          } else {
            get().log(`Dry-run failed: ${result.error.message}`, 'error');
            if (result.error.sparkTrace) {
              get().log(result.error.sparkTrace, 'error');
            }
          }

          return result;
        } catch (err) {
          set({ isDryRunning: false });
          const msg = err instanceof Error ? err.message : String(err);
          const fail: DryRunResult = { success: false, error: { message: msg } };
          set({ dryRunResult: fail });
          get().log(`Dry-run error: ${msg}`, 'error');
          return fail;
        }
      },

      // ── Pipeline metadata ───────────────────────────────────────

      setPipelineName: (name) => set({ pipelineName: name }),
      setPipelineVersion: (version) => set({ pipelineVersion: version }),

      // ── UI actions ─────────────────────────────────────────────

      selectNode: (id) => set({ selectedNodeId: id, selectedEdgeId: null }),
      selectEdge: (id) => set({ selectedEdgeId: id, selectedNodeId: null }),

      log: (message, type) => {
        set((s) => ({
          consoleMessages: [...s.consoleMessages, createMsg(message, type)],
        }));
      },

      clearConsole: () => set({ consoleMessages: [] }),
    }),
    { name: 'PipelineStore' },
  ),
);
