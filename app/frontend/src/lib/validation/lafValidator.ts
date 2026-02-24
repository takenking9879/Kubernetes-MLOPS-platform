/**
 * LAF Compiler — Local Ahead-of-File validator.
 *
 * Validates the pipeline DAG entirely on the frontend:
 *   1. Dataset + schema presence
 *   2. Topological sort with cycle detection
 *   3. Incremental edge selector resolution per source schema
 *   4. Incremental schema simulation (symbolic, no Spark)
 *   5. Per-node parameter validation
 *   6. Disconnected-node warnings
 *
 * FAIL-FAST: errors are accumulated and returned; the pipeline is
 * considered invalid if *any* error (not warning) is present.
 */

import type { Node } from 'reactflow';
import type { NodeData } from '../../types/nodes';
import { isStageNode } from '../../types/nodes';
import type { FlowEdge } from '../../types/edges';
import type { SparkSchema, SchemaEvolution } from '../../types/schema';
import type { ValidationResult, ValidationError, CompiledNodeState } from '../../types/validation';
import { validationError, validationWarning } from '../../types/validation';
import { topologicalSort } from './toposort';
import { resolveSelector } from '../selectors/resolver';
import { mergeSchemas, simulateStage } from './schemaSimulator';
import { validateNodeParams, getNodeDefinition } from '../../registry/NodeRegistry';
import { isNamingStrategyComplete } from '../../types/naming';

export interface IncrementalOptions {
  readonly previousResult?: ValidationResult;
  readonly dirtyNodeIds?: ReadonlySet<string>;
}

export function validatePipelineLocally(
  nodes: readonly Node<NodeData>[],
  edges: readonly FlowEdge[],
  datasetSchema: SparkSchema | null,
  incremental?: IncrementalOptions,
): ValidationResult {
  const errors: ValidationError[] = [];
  const warnings: ValidationError[] = [];
  const schemaEvolutions = new Map<string, SchemaEvolution>();
  const compiledNodes = new Map<string, CompiledNodeState>();
  const compiledEdges = new Map<string, { readonly resolvedColumns: readonly string[] }>();

  // ── 1. Dataset existence ────────────────────────────────────────
  const datasetNodes = nodes.filter((n) => n.data.type === 'dataset');
  if (datasetNodes.length > 1) {
    errors.push(validationError('multiple_dataset', 'Only one Dataset node is allowed'));
  }

  const datasetNode = datasetNodes[0];
  if (!datasetNode) {
    errors.push(validationError('missing_dataset', 'Pipeline must have a Dataset node'));
    return {
      valid: false,
      errors,
      warnings,
      schemaEvolutions,
      topologicalOrder: [],
      compiledNodes,
      compiledEdges,
    };
  }

  if (!datasetSchema || datasetSchema.columns.length === 0) {
    errors.push(validationError(
      'missing_schema',
      'Dataset schema is not loaded. Load a CSV or Iceberg source first.',
      { nodeId: datasetNode.id },
    ));
    return {
      valid: false,
      errors,
      warnings,
      schemaEvolutions,
      topologicalOrder: [],
      compiledNodes,
      compiledEdges,
    };
  }

  const datasetIncomingEdges = edges.filter((e) => e.target === datasetNode.id);
  if (datasetIncomingEdges.length > 0) {
    errors.push(validationError(
      'cardinality_mismatch',
      'Dataset node cannot have incoming edges',
      { nodeId: datasetNode.id },
    ));
  }

  const finalNodes = nodes.filter((n) => n.data.type === 'final_features');
  if (finalNodes.length === 0) {
    errors.push(validationError('missing_final_features', 'Pipeline must include exactly one Final Features node'));
  }
  if (finalNodes.length > 1) {
    errors.push(validationError('multiple_final_features', 'Only one Final Features node is allowed'));
  }

  const finalNode = finalNodes[0];
  if (finalNode) {
    const incoming = edges.filter((e) => e.target === finalNode.id);
    const outgoing = edges.filter((e) => e.source === finalNode.id);
    if (incoming.length === 0) {
      errors.push(validationError(
        'final_features_disconnected',
        'Final Features must have at least one upstream producer',
        { nodeId: finalNode.id },
      ));
    }
    if (outgoing.length > 0) {
      errors.push(validationError(
        'final_features_outgoing_edge',
        'Final Features is terminal and cannot have outgoing edges',
        { nodeId: finalNode.id },
      ));
    }
  }

  // ── 2. Topological sort ─────────────────────────────────────────
  const nodeIds = nodes.map((n) => n.id);
  const topoResult = topologicalSort(nodeIds, edges);

  if (topoResult.hasCycle) {
    errors.push(validationError(
      'cycle_detected',
      `Cycle detected in pipeline: ${topoResult.cyclePath.join(' -> ')}`,
    ));
    return {
      valid: false,
      errors,
      warnings,
      schemaEvolutions,
      topologicalOrder: [],
      compiledNodes,
      compiledEdges,
    };
  }

  const nodeMap = new Map(nodes.map((n) => [n.id, n]));
  const edgesByTarget = new Map<string, FlowEdge[]>();
  for (const edge of edges) {
    const arr = edgesByTarget.get(edge.target) ?? [];
    arr.push(edge);
    edgesByTarget.set(edge.target, arr);
  }

  // Dataset node compile state
  compiledNodes.set(datasetNode.id, {
    compilation: { status: 'compiled', inputSchema: datasetSchema, outputSchema: datasetSchema },
    resolvedInputs: [],
    resolvedOutputs: datasetSchema.columns.map((c) => c.name),
    outputSchema: datasetSchema,
  });

  // Incremental compilation: reuse previous results for clean nodes
  const prevCompiledNodes = incremental?.previousResult?.compiledNodes;
  const prevCompiledEdges = incremental?.previousResult?.compiledEdges;
  const dirtyNodes = incremental?.dirtyNodeIds;

  // ── 3-5. Incremental selector resolution + simulation + params ──
  for (const nodeId of topoResult.order) {
    const node = nodeMap.get(nodeId);
    if (!node) continue;
    if (node.data.type === 'dataset') continue;
    if (!isStageNode(node.data)) continue;

    // Incremental: if node is clean and has previous compiled state, reuse it
    if (dirtyNodes && !dirtyNodes.has(nodeId) && prevCompiledNodes?.has(nodeId)) {
      const prevState = prevCompiledNodes.get(nodeId)!;
      compiledNodes.set(nodeId, prevState);

      // Also reuse edge resolutions for this node's incoming edges
      const incomingEdges = edgesByTarget.get(nodeId) ?? [];
      for (const edge of incomingEdges) {
        const prevEdge = prevCompiledEdges?.get(edge.id);
        if (prevEdge) {
          compiledEdges.set(edge.id, prevEdge);
        }
      }
      continue;
    }

    const data = node.data;
    const incomingEdges = edgesByTarget.get(nodeId) ?? [];
    const sourceSchemas: SparkSchema[] = [];
    const resolvedInputs: string[] = [];
    const seenInputs = new Set<string>();

    let nodeHasErrors = false;
    const nodeCompilationErrors: string[] = [];

    if (incomingEdges.length === 0) {
      compiledNodes.set(nodeId, {
        compilation: { status: 'draft' },
        resolvedInputs: [],
        resolvedOutputs: [],
        outputSchema: null,
      });
      continue;
    }

    if (!isNamingStrategyComplete(data.namingStrategy)) {
      compiledNodes.set(nodeId, {
        compilation: { status: 'draft' },
        resolvedInputs: [],
        resolvedOutputs: [],
        outputSchema: null,
      });
      continue;
    }

    for (const edge of incomingEdges) {
      const sourceSchema = getCompiledOutputSchema(edge.source, datasetNode.id, datasetSchema, compiledNodes);

      if (!sourceSchema) {
        compiledNodes.set(nodeId, {
          compilation: { status: 'draft' },
          resolvedInputs: [],
          resolvedOutputs: [],
          outputSchema: null,
        });
        nodeHasErrors = false;
        nodeCompilationErrors.length = 0;
        break;
      }

      sourceSchemas.push(sourceSchema);

      if (!isSelectorConfigured(edge)) {
        compiledNodes.set(nodeId, {
          compilation: { status: 'draft' },
          resolvedInputs: [],
          resolvedOutputs: [],
          outputSchema: null,
        });
        nodeHasErrors = false;
        nodeCompilationErrors.length = 0;
        break;
      }

      try {
        const resolved = resolveSelector(edge.selector, sourceSchema);
        if (resolved.length === 0) {
          throw new Error('Selector resolved to 0 columns');
        }

        compiledEdges.set(edge.id, { resolvedColumns: resolved });
        for (const col of resolved) {
          if (!seenInputs.has(col)) {
            seenInputs.add(col);
            resolvedInputs.push(col);
          }
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        const type = msg.includes('0 columns') || msg.includes('empty column list')
          ? 'selector_empty'
          : 'selector_resolution_failed';
        errors.push(validationError(
          type,
          `Edge ${edge.source} -> ${edge.target}: ${msg}`,
          { edgeId: edge.id, nodeId },
        ));
        nodeHasErrors = true;
        nodeCompilationErrors.push(msg);
      }
    }

    if (compiledNodes.get(nodeId)?.compilation.status === 'draft') {
      continue;
    }

    const def = getNodeDefinition(data.type);
    const inputCount = resolvedInputs.length;

    if (def.inputCardinality.kind === 'exactly' && inputCount !== def.inputCardinality.n) {
      const msg = `Stage '${data.label}' (${data.type}) requires exactly ${def.inputCardinality.n} input(s) but got ${inputCount}`;
      errors.push(validationError(
        'cardinality_mismatch',
        msg,
        { nodeId: node.id },
      ));
      nodeHasErrors = true;
      nodeCompilationErrors.push(msg);
    }

    if (def.inputCardinality.kind === 'atLeast' && inputCount < def.inputCardinality.n) {
      const msg = `Stage '${data.label}' (${data.type}) requires at least ${def.inputCardinality.n} input(s) but got ${inputCount}`;
      errors.push(validationError(
        'cardinality_mismatch',
        msg,
        { nodeId: node.id },
      ));
      nodeHasErrors = true;
      nodeCompilationErrors.push(msg);
    }

    const paramErrors = validateNodeParams(data.type, data.params);
    for (const msg of paramErrors) {
      nodeHasErrors = true;
      nodeCompilationErrors.push(msg);
    }

    if (nodeHasErrors) {
      for (const errMsg of nodeCompilationErrors) {
        errors.push(validationError('invalid_params', `${data.label}: ${errMsg}`, { nodeId: node.id }));
      }
      compiledNodes.set(nodeId, {
        compilation: { status: 'error', errors: [...nodeCompilationErrors] },
        resolvedInputs: [...resolvedInputs],
        resolvedOutputs: [],
        outputSchema: null,
      });
      continue;
    }

    let inputSchema: SparkSchema;
    try {
      inputSchema = sourceSchemas.length > 0 ? mergeSchemas(sourceSchemas) : { columns: [] };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      errors.push(validationError('missing_input_column', msg, { nodeId: node.id }));
      compiledNodes.set(nodeId, {
        compilation: { status: 'error', errors: [msg] },
        resolvedInputs: [...resolvedInputs],
        resolvedOutputs: [],
        outputSchema: null,
      });
      continue;
    }

    try {
      const simulation = simulateStage(data, inputSchema, resolvedInputs);
      compiledNodes.set(nodeId, {
        compilation: {
          status: 'compiled',
          inputSchema,
          outputSchema: simulation.outputSchema,
        },
        resolvedInputs: [...resolvedInputs],
        resolvedOutputs: [...simulation.resolvedOutputs],
        outputSchema: simulation.outputSchema,
      });

      schemaEvolutions.set(nodeId, {
        stageId: nodeId,
        inputSchema,
        outputSchema: simulation.outputSchema,
        columnsAdded: simulation.columnsAdded,
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      const type = err instanceof TypeError
        ? 'type_mismatch'
        : msg.includes('duplicate output column')
          ? 'duplicate_output_column'
          : 'missing_input_column';
      errors.push(validationError(type, msg, { nodeId: node.id }));
      compiledNodes.set(nodeId, {
        compilation: { status: 'error', errors: [msg] },
        resolvedInputs: [...resolvedInputs],
        resolvedOutputs: [],
        outputSchema: null,
      });
    }
  }

  // ── 6. Disconnected node warnings ───────────────────────────────
  const reachable = findReachableFrom(datasetNode.id, edges);
  for (const node of nodes) {
    if (!reachable.has(node.id) && node.data.type !== 'dataset') {
      warnings.push(validationWarning(
        'disconnected_node',
        `Node '${node.data.label}' is not connected to the pipeline`,
        { nodeId: node.id },
      ));
    }
  }

  // Column count warning
  const totalInputs = Array.from(compiledEdges.values()).reduce(
    (sum, edgeState) => sum + edgeState.resolvedColumns.length,
    0,
  );
  if (totalInputs > 60) {
    warnings.push(validationWarning(
      'cardinality_mismatch',
      `Pipeline handles ${totalInputs} column-flows. Consider using group selectors for maintainability.`,
    ));
  }

  return {
    valid: errors.length === 0,
    errors,
    warnings,
    schemaEvolutions,
    topologicalOrder: topoResult.order,
    compiledNodes,
    compiledEdges,
  };
}

function getCompiledOutputSchema(
  sourceNodeId: string,
  datasetNodeId: string,
  datasetSchema: SparkSchema,
  compiledNodes: ReadonlyMap<string, CompiledNodeState>,
): SparkSchema | null {
  if (sourceNodeId === datasetNodeId) {
    return datasetSchema;
  }

  const sourceState = compiledNodes.get(sourceNodeId);
  if (!sourceState || sourceState.compilation.status !== 'compiled' || !sourceState.outputSchema) return null;
  return sourceState.outputSchema;
}

function isSelectorConfigured(edge: FlowEdge): boolean {
  if (edge.selector.type === 'manual') {
    return edge.selector.columns.length > 0;
  }
  return true;
}

function findReachableFrom(
  startId: string,
  edges: readonly FlowEdge[],
): Set<string> {
  const outgoing = new Map<string, string[]>();
  for (const edge of edges) {
    const arr = outgoing.get(edge.source) ?? [];
    arr.push(edge.target);
    outgoing.set(edge.source, arr);
  }

  const visited = new Set<string>();
  const queue: string[] = [startId];

  while (queue.length > 0) {
    const current = queue.shift();
    if (!current || visited.has(current)) continue;
    visited.add(current);

    const neighbors = outgoing.get(current) ?? [];
    for (const neighbor of neighbors) {
      if (!visited.has(neighbor)) {
        queue.push(neighbor);
      }
    }
  }

  return visited;
}
