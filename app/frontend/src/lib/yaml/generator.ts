/**
 * YAML Generator.
 *
 * Walks the validated DAG in topological order and produces a YAML
 * config string that is compatible with the PySpark DSL backend
 * (src/dsl/pipeline.py -> Pipeline.from_yaml).
 *
 * INVARIANT: The generated YAML contains ONLY explicit column lists
 * in inputs/outputs -- never edge selectors, rules, or group references.
 */

import { stringify } from 'yaml';
import type { Node } from 'reactflow';
import type { NodeData, StageNodeData, FinalFeatureSelectorData } from '../../types/nodes';
import { isStageNode, isFinalFeatures } from '../../types/nodes';
import { dedupeBuckets } from '../finalFeatures/model';

interface YamlStage {
  type: string;
  name: string;
  inputs: string[] | string;
  outputs: string[] | string;
  params: Record<string, unknown>;
}

interface YamlMeta {
  dslVersion: string;
  schemaHash?: string;
  generatedAt: string;
}

interface YamlPipeline {
  meta: YamlMeta;
  pipeline: {
    name: string;
    version: string;
    stages: YamlStage[];
  };
  final_features?: {
    features: string[];
    target: string[];
    metadata: string[];
    passthrough: string[];
  };
}

export interface GenerateYAMLOptions {
  readonly pipelineName?: string;
  readonly pipelineVersion?: string;
  readonly schemaHash?: string;
}

export function generateYAML(
  nodes: readonly Node<NodeData>[],
  topologicalOrder: readonly string[],
  options: GenerateYAMLOptions = {},
): string {
  const {
    pipelineName = 'visual_pipeline',
    pipelineVersion = '1.0',
    schemaHash,
  } = options;

  const nodeMap = new Map(nodes.map((n) => [n.id, n]));

  const stages: YamlStage[] = [];

  for (const nodeId of topologicalOrder) {
    const node = nodeMap.get(nodeId);
    if (!node) continue;

    const data = node.data;

    // Skip non-stage nodes
    if (!isStageNode(data)) continue;

    const stage = buildStageConfig(data, nodeId);
    stages.push(stage);
  }

  // Build final_features from FinalFeatureSelectorNode if present
  const finalNode = nodes.find((n) => isFinalFeatures(n.data));
  let finalFeatures: YamlPipeline['final_features'] | undefined;

  if (finalNode && isFinalFeatures(finalNode.data)) {
    const fd = finalNode.data as FinalFeatureSelectorData;
    const deduped = dedupeBuckets({
      feature: fd.features,
      target: fd.target,
      metadata: fd.metadata,
      passthrough: fd.passthrough,
    });

    finalFeatures = {
      features: deduped.feature,
      target: deduped.target,
      metadata: deduped.metadata,
      passthrough: deduped.passthrough,
    };
  }

  const meta: YamlMeta = {
    dslVersion: '2.0',
    generatedAt: new Date().toISOString(),
    ...(schemaHash ? { schemaHash } : {}),
  };

  const config: YamlPipeline = {
    meta,
    pipeline: {
      name: pipelineName,
      version: pipelineVersion,
      stages,
    },
    ...(finalFeatures ? { final_features: finalFeatures } : {}),
  };

  const yaml = stringify(config, { indent: 2, lineWidth: 120 });
  return formatYAMLSpacing(yaml);
}

function formatYAMLSpacing(yaml: string): string {
  const lines = yaml.split('\n');
  const out: string[] = [];

  let inStages = false;
  let seenStageItem = false;

  for (const line of lines) {
    if (/^ {2}stages:$/.test(line)) {
      if (out.length > 0 && out[out.length - 1] !== '') {
        out.push('');
      }
      out.push(line);
      inStages = true;
      seenStageItem = false;
      continue;
    }

    if (inStages) {
      if (/^ {4}- /.test(line)) {
        if (seenStageItem && out[out.length - 1] !== '') {
          out.push('');
        }
        seenStageItem = true;
        out.push(line);
        continue;
      }

      if (line.trim() !== '' && !/^ {6}/.test(line)) {
        inStages = false;
      }
    }

    if (/^(pipeline:|final_features:)$/.test(line)) {
      if (out.length > 0 && out[out.length - 1] !== '') {
        out.push('');
      }
      out.push(line);
      continue;
    }

    out.push(line);
  }

  return out.join('\n').replace(/\n{3,}/g, '\n\n');
}

// --- Build a single stage config ------------------------------------

function buildStageConfig(
  data: StageNodeData,
  nodeId: string,
): YamlStage {
  const inputs = [...data.resolvedInputs];
  if (inputs.length === 0) {
    throw new Error(`Stage '${nodeId}' has no resolvedInputs. Run validation before export.`);
  }

  const outputs = [...data.resolvedOutputs];
  if (outputs.length === 0) {
    throw new Error(`Stage '${nodeId}' has no resolvedOutputs. Run validation before export.`);
  }

  // Clean params: remove undefined/null values
  const cleanParams: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(data.params)) {
    if (value !== undefined && value !== null && value !== '') {
      cleanParams[key] = value;
    }
  }

  return {
    type: data.type,
    name: data.id,
    inputs: inputs.length === 1 ? inputs[0]! : inputs,
    outputs: outputs.length === 1 ? outputs[0]! : outputs,
    params: cleanParams,
  };
}
