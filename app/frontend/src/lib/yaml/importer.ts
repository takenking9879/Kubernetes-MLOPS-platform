/**
 * YAML Importer.
 *
 * Parses a PySpark DSL YAML config and reconstructs the visual DAG:
 *   - Creates ReactFlow nodes from each stage
 *   - Infers OutputNamingStrategy from input/output column pairs
 *   - Builds edges by tracking column producers
 *   - Handles both v1 (no meta) and v2 (with meta) formats
 */

import { parse } from 'yaml';
import type { Node } from 'reactflow';
import type {
  NodeData,
  StageNodeData,
  StageType,
  FinalFeatureSelectorData,
  DatasetNodeData,
} from '../../types/nodes';
import type { FlowEdge, ManualSelector } from '../../types/edges';
import type { OutputNamingStrategy } from '../../types/naming';
import { NODE_REGISTRY } from '../../registry/NodeRegistry';
import { getNodeDefinition } from '../../registry/NodeRegistry';
import { autoLayoutNodes } from '../layout/autoLayout';

// ─── YAML shape (what we parse) ──────────────────────────────────────

interface YamlStageRaw {
  type: string;
  name: string;
  inputs: string | string[];
  outputs: string | string[];
  params?: Record<string, unknown>;
}

interface YamlPipelineRaw {
  dslVersion?: number;
  generatedAt?: string;
  schemaHash?: string;
  final_features?: {
    features?: string[];
    target?: string[];
    metadata?: string[];
    passthrough?: string[];
  };
  pipeline: {
    name?: string;
    version?: string;
    stages: YamlStageRaw[];
  };
  meta?: {
    dslVersion?: number;
    schemaHash?: string;
    generatedAt?: string;
  };
}

// ─── Import result ───────────────────────────────────────────────────

export interface ImportResult {
  nodes: Node<NodeData>[];
  edges: FlowEdge[];
  pipelineName: string;
  pipelineVersion: string;
  schemaHash?: string;
  warnings: string[];
}

// ─── Main entry point ────────────────────────────────────────────────

export function importFromYAML(yamlString: string): ImportResult {
  const warnings: string[] = [];
  const raw = parse(yamlString) as YamlPipelineRaw;

  if (!raw?.pipeline) {
    throw new Error('Invalid YAML: missing top-level "pipeline" key');
  }

  const { pipeline, meta } = raw;

  if (!Array.isArray(pipeline.stages) || pipeline.stages.length === 0) {
    throw new Error('Invalid YAML: "pipeline.stages" must be a non-empty array');
  }

  // Check for v1 (legacy) format
  if (!meta?.dslVersion) {
    warnings.push('No meta.dslVersion found — treating as legacy v1 format');
  }

  const pipelineName = pipeline.name ?? 'imported_pipeline';
  const pipelineVersion = pipeline.version ?? '1.0';
  const schemaHash = raw.schemaHash ?? meta?.schemaHash;

  // ── Build stage nodes ────────────────────────────────────────────

  const nodes: Node<NodeData>[] = [];
  const edges: FlowEdge[] = [];

  // Track which stage produced each column: columnName -> nodeId
  const columnProducer = new Map<string, string>();

  // Create dataset node (placeholder — user will attach real schema)
  const datasetId = 'dataset_root';
  const datasetNode: Node<NodeData> = {
    id: datasetId,
    type: 'dataset',
    position: { x: 0, y: 0 },
    data: {
      id: datasetId,
      label: 'Dataset',
      type: 'dataset',
      source: 'csv',
      path: '',
      schema: null,
    } as DatasetNodeData,
  };
  nodes.push(datasetNode);

  // Process stages in order (YAML order = topological order)
  for (const rawStage of pipeline.stages) {
    const stageType = rawStage.type;

    // Validate stage type exists in registry
    if (!(stageType in NODE_REGISTRY)) {
      warnings.push(`Unknown stage type "${stageType}" (stage: ${rawStage.name}) — skipping`);
      continue;
    }

    const def = getNodeDefinition(stageType);

    // Normalize inputs/outputs to arrays
    const inputs = normalizeToArray(rawStage.inputs);
    const outputs = normalizeToArray(rawStage.outputs);

    if (inputs.length === 0) {
      warnings.push(`Stage "${rawStage.name}" has no inputs — skipping`);
      continue;
    }

    if (outputs.length === 0) {
      warnings.push(`Stage "${rawStage.name}" has no outputs — skipping`);
      continue;
    }

    // Infer naming strategy from input/output pairs
    const namingStrategy = inferNamingStrategy(inputs, outputs, stageType);

    const nodeId = rawStage.name;
    const params = rawStage.params ?? { ...def.defaultParams };

    const data: StageNodeData = {
      id: nodeId,
      label: def.label,
      type: stageType as StageType,
      stageType: def.category,
      inputs: [],
      outputs: [],
      params,
      namingStrategy,
      resolvedInputs: [...inputs],
      resolvedOutputs: [...outputs],
      outputSchema: null,
      compilation: { status: 'draft' },
    } as StageNodeData;

    const stageNode: Node<NodeData> = {
      id: nodeId,
      type: 'stage',
      position: { x: 0, y: 0 }, // auto-layout later
      data,
    };
    nodes.push(stageNode);

    // ── Create edges based on column provenance ──────────────────

    // Group input columns by their producer node
    const producerGroups = new Map<string, string[]>();

    for (const inputCol of inputs) {
      const producerId = columnProducer.get(inputCol);
      if (producerId) {
        const group = producerGroups.get(producerId);
        if (group) {
          group.push(inputCol);
        } else {
          producerGroups.set(producerId, [inputCol]);
        }
      } else {
        // Column not produced by any stage — assume it comes from the dataset
        const group = producerGroups.get(datasetId);
        if (group) {
          group.push(inputCol);
        } else {
          producerGroups.set(datasetId, [inputCol]);
        }
      }
    }

    // Create one edge per producer
    for (const [sourceId, columns] of producerGroups) {
      const edgeId = `e-${sourceId}-${nodeId}-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
      const selector: ManualSelector = { type: 'manual', columns };

      edges.push({
        id: edgeId,
        source: sourceId,
        target: nodeId,
        selector,
        resolved: false,
        resolvedColumns: [],
      });
    }

    // Register this stage's outputs as produced by this node
    for (const outputCol of outputs) {
      columnProducer.set(outputCol, nodeId);
    }
  }

  // ── FinalFeatures node ───────────────────────────────────────────

  const topLevelFinalFeatures = raw.final_features;

  if (topLevelFinalFeatures) {
    const ffId = 'final_features';

    const ffData: FinalFeatureSelectorData = {
      id: ffId,
      label: 'Final Features',
      type: 'final_features',
      features: topLevelFinalFeatures.features ?? [],
      target: topLevelFinalFeatures.target ?? [],
      metadata: topLevelFinalFeatures.metadata ?? [],
      passthrough: topLevelFinalFeatures.passthrough ?? [],
      manualOverrides: {},
      confirmedColumns: [],
    };

    const ffNode: Node<NodeData> = {
      id: ffId,
      type: 'final_features',
      position: { x: 0, y: 0 },
      data: ffData,
    };
    nodes.push(ffNode);

    // Create edges from the last stages that produced the final feature columns
    const allFinalCols = [
      ...ffData.features,
      ...ffData.target,
      ...ffData.metadata,
      ...ffData.passthrough,
    ];

    const ffProducerGroups = new Map<string, string[]>();
    for (const col of allFinalCols) {
      const producerId = columnProducer.get(col) ?? datasetId;
      const group = ffProducerGroups.get(producerId);
      if (group) {
        group.push(col);
      } else {
        ffProducerGroups.set(producerId, [col]);
      }
    }

    for (const [sourceId, columns] of ffProducerGroups) {
      edges.push({
        id: `e-${sourceId}-${ffId}-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
        source: sourceId,
        target: ffId,
        selector: { type: 'manual', columns },
        resolved: false,
        resolvedColumns: [],
      });
    }
  }

  // ── Auto-layout ──────────────────────────────────────────────────

  const layoutedNodes = autoLayoutNodes(nodes, edges);

  return {
    nodes: layoutedNodes,
    edges,
    pipelineName,
    pipelineVersion,
    schemaHash,
    warnings,
  };
}

// ─── Helpers ─────────────────────────────────────────────────────────

function normalizeToArray(value: string | string[] | undefined): string[] {
  if (!value) return [];
  if (typeof value === 'string') return [value];
  return [...value];
}

/**
 * Infer the OutputNamingStrategy by comparing input and output column names.
 *
 * Checks for common suffix/prefix patterns across all I/O pairs.
 * Falls back to 'custom' with explicit output names.
 */
function inferNamingStrategy(
  inputs: string[],
  outputs: string[],
  _stageType: string,
): OutputNamingStrategy {
  // Passthrough: inputs and outputs are identical
  if (
    inputs.length === outputs.length &&
    inputs.every((inp, i) => inp === outputs[i])
  ) {
    return { mode: 'passthrough' };
  }

  // For many_to_one (concat, ratio) or mismatched counts, use custom
  if (inputs.length !== outputs.length) {
    return { mode: 'custom', columns: outputs };
  }

  // Try to detect a common suffix: output = input + suffix
  const suffixes = inputs.map((inp, i) => {
    const out = outputs[i]!;
    if (out.startsWith(inp)) {
      return out.slice(inp.length);
    }
    return null;
  });

  if (suffixes.every((s) => s !== null) && new Set(suffixes).size === 1 && suffixes[0] !== '') {
    return { mode: 'suffix', value: suffixes[0]! };
  }

  // Try to detect a common prefix: output = prefix + input
  const prefixes = inputs.map((inp, i) => {
    const out = outputs[i]!;
    if (out.endsWith(inp)) {
      return out.slice(0, out.length - inp.length);
    }
    return null;
  });

  if (prefixes.every((p) => p !== null) && new Set(prefixes).size === 1 && prefixes[0] !== '') {
    return { mode: 'prefix', value: prefixes[0]! };
  }

  // Fall back to custom
  return { mode: 'custom', columns: outputs };
}
