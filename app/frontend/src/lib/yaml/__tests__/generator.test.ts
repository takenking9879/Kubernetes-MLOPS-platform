import { describe, it, expect } from 'vitest';
import { generateYAML } from '../generator';
import { parse } from 'yaml';
import type { Node } from 'reactflow';
import type { NodeData, CastTransformerData, FinalFeatureSelectorData } from '../../../types/nodes';

function makeNode(id: string, data: NodeData, type: string = 'stage'): Node<NodeData> {
  return { id, type, position: { x: 0, y: 0 }, data };
}

const datasetNode = makeNode('ds', {
  id: 'ds',
  label: 'Dataset',
  type: 'dataset',
  source: 'csv',
  path: '/data/test.csv',
  schema: null,
} as NodeData, 'dataset');

const castNode = makeNode('cast-1', {
  id: 'cast-1',
  label: 'Cast Type',
  type: 'cast_transformer',
  stageType: 'transformer',
  inputs: [],
  outputs: [],
  params: { target_type: 'timestamp' },
  namingStrategy: { mode: 'suffix', value: '_ts' },
  resolvedInputs: ['raw_ts'],
  resolvedOutputs: ['raw_ts_ts'],
  outputSchema: null,
  compilation: { status: 'compiled', inputSchema: { columns: [] }, outputSchema: { columns: [] } },
} as unknown as CastTransformerData, 'stage');

describe('generateYAML', () => {
  it('generates valid YAML with stages', () => {
    const yaml = generateYAML([datasetNode, castNode], ['ds', 'cast-1']);
    const parsed = parse(yaml);

    expect(parsed.pipeline).toBeDefined();
    expect(parsed.pipeline.stages).toHaveLength(1);
    expect(parsed.pipeline.stages[0].type).toBe('cast_transformer');
    expect(parsed.pipeline.stages[0].name).toBe('cast-1');
  });

  it('uses pipeline name and version from options', () => {
    const yaml = generateYAML([datasetNode, castNode], ['ds', 'cast-1'], {
      pipelineName: 'my_pipeline',
      pipelineVersion: '2.0',
    });
    const parsed = parse(yaml);

    expect(parsed.pipeline.name).toBe('my_pipeline');
    expect(parsed.pipeline.version).toBe('2.0');
  });

  it('includes meta block', () => {
    const yaml = generateYAML([datasetNode, castNode], ['ds', 'cast-1'], {
      schemaHash: 'abc123',
    });
    const parsed = parse(yaml);

    expect(parsed.meta).toBeDefined();
    expect(parsed.meta.dslVersion).toBe('2.0');
    expect(parsed.meta.schemaHash).toBe('abc123');
    expect(parsed.meta.generatedAt).toBeDefined();
  });

  it('adds blank lines between sections and stages', () => {
    const ffNode = makeNode('ff', {
      id: 'ff',
      label: 'Final Features',
      type: 'final_features',
      features: ['cat_idx'],
      target: ['label'],
      metadata: ['id'],
      passthrough: [],
      manualOverrides: {},
      confirmedColumns: [],
    } as FinalFeatureSelectorData, 'final_features');

    const extraStage = makeNode('cast-2', {
      id: 'cast-2',
      label: 'Cast Type 2',
      type: 'cast_transformer',
      stageType: 'transformer',
      inputs: [],
      outputs: [],
      params: { target_type: 'double' },
      namingStrategy: { mode: 'suffix', value: '_d' },
      resolvedInputs: ['raw_amount'],
      resolvedOutputs: ['raw_amount_d'],
      outputSchema: null,
      compilation: { status: 'compiled', inputSchema: { columns: [] }, outputSchema: { columns: [] } },
    } as unknown as CastTransformerData, 'stage');

    const yaml = generateYAML([datasetNode, castNode, extraStage, ffNode], ['ds', 'cast-1', 'cast-2']);

    expect(yaml).toContain('meta:\n  dslVersion: "2.0"');
    expect(yaml).toContain('\n\npipeline:');
    expect(yaml).toContain('\n\n  stages:');
    expect(yaml).toContain('\n\n    - type:');
    expect(yaml).toContain('\n\nfinal_features:');
  });

  it('unwraps single-element inputs/outputs to scalar', () => {
    const yaml = generateYAML([datasetNode, castNode], ['ds', 'cast-1']);
    const parsed = parse(yaml);
    const stage = parsed.pipeline.stages[0];

    // Single-element arrays should be serialized as scalars
    expect(typeof stage.inputs).toBe('string');
    expect(typeof stage.outputs).toBe('string');
  });

  it('includes final_features block', () => {
    const ffNode = makeNode('ff', {
      id: 'ff',
      label: 'Final Features',
      type: 'final_features',
      features: ['cat_idx', 'amount_norm'],
      target: ['label'],
      metadata: ['id'],
      passthrough: [],
      manualOverrides: {},
      confirmedColumns: [],
    } as FinalFeatureSelectorData, 'final_features');

    const yaml = generateYAML([datasetNode, castNode, ffNode], ['ds', 'cast-1']);
    const parsed = parse(yaml);

    expect(parsed.final_features).toBeDefined();
    expect(parsed.final_features.features).toEqual(['cat_idx', 'amount_norm']);
    expect(parsed.final_features.target).toEqual(['label']);
    expect(parsed.final_features.metadata).toEqual(['id']);
  });

  it('strips empty/null params', () => {
    const nodeWithNulls = makeNode('s1', {
      id: 's1',
      label: 'Cast',
      type: 'cast_transformer',
      stageType: 'transformer',
      inputs: [],
      outputs: [],
      params: { target_type: 'double', extra: null, empty: '' },
      namingStrategy: { mode: 'suffix', value: '_cast' },
      resolvedInputs: ['col1'],
      resolvedOutputs: ['col1_cast'],
      outputSchema: null,
      compilation: { status: 'compiled', inputSchema: { columns: [] }, outputSchema: { columns: [] } },
    } as unknown as NodeData, 'stage');

    const yaml = generateYAML([datasetNode, nodeWithNulls], ['ds', 's1']);
    const parsed = parse(yaml);

    expect(parsed.pipeline.stages[0].params).toEqual({ target_type: 'double' });
  });
});
