import { describe, it, expect } from 'vitest';
import { importFromYAML } from '../importer';
import type { CastTransformerData, StageNodeData } from '../../../types/nodes';

const simplePipelineYaml = `
pipeline:
  name: "test_pipeline"
  version: "1.0"
  stages:
    - type: "cast_transformer"
      name: "timestamp_parser"
      inputs: ["timestamp"]
      outputs: "timestamp_ts"
      params:
        target_type: "timestamp"
    - type: "temporal_extractor"
      name: "hour_extractor"
      inputs: ["timestamp_ts"]
      outputs: "hour"
      params:
        component: "hour"
`;

const pipelineWithFinalFeatures = `
pipeline:
  name: "feature_pipeline"
  version: "2.0"
  stages:
    - type: "cast_transformer"
      name: "cast_1"
      inputs: "amount"
      outputs: "amount_ts"
      params:
        target_type: "timestamp"
final_features:
  features:
    - "cat_a"
    - "amount_ts"
  target:
    - "label"
  metadata:
    - "id"
schemaHash: "abc123"
dslVersion: 2
generatedAt: "2025-01-01T00:00:00Z"
`;

describe('importFromYAML', () => {
  it('parses a simple pipeline', () => {
    const result = importFromYAML(simplePipelineYaml);

    expect(result.pipelineName).toBe('test_pipeline');
    expect(result.pipelineVersion).toBe('1.0');
    // 1 dataset + 2 stages
    expect(result.nodes).toHaveLength(3);
    // 2 edges: dataset->timestamp_parser, timestamp_parser->hour_extractor
    expect(result.edges).toHaveLength(2);
  });

  it('creates dataset node', () => {
    const result = importFromYAML(simplePipelineYaml);
    const datasetNode = result.nodes.find((n) => n.type === 'dataset');
    expect(datasetNode).toBeDefined();
    expect(datasetNode!.data.type).toBe('dataset');
  });

  it('creates stage nodes with correct types', () => {
    const result = importFromYAML(simplePipelineYaml);
    const stageNodes = result.nodes.filter((n) => n.type === 'stage');
    expect(stageNodes).toHaveLength(2);

    const castStage = stageNodes.find((n) => n.id === 'timestamp_parser');
    expect(castStage).toBeDefined();
    expect((castStage!.data as CastTransformerData).type).toBe('cast_transformer');
    expect((castStage!.data as CastTransformerData).params.target_type).toBe('timestamp');
  });

  it('infers edges from column provenance', () => {
    const result = importFromYAML(simplePipelineYaml);

    // The hour_extractor takes 'timestamp_ts' which was produced by timestamp_parser
    const hourEdge = result.edges.find((e) => e.target === 'hour_extractor');
    expect(hourEdge).toBeDefined();
    expect(hourEdge!.source).toBe('timestamp_parser');
    expect(hourEdge!.selector.type).toBe('manual');
    if (hourEdge!.selector.type === 'manual') {
      expect(hourEdge!.selector.columns).toEqual(['timestamp_ts']);
    }
  });

  it('infers suffix naming strategy', () => {
    const result = importFromYAML(simplePipelineYaml);
    const castNode = result.nodes.find((n) => n.id === 'timestamp_parser');
    const data = castNode!.data as StageNodeData;

    expect(data.namingStrategy.mode).toBe('suffix');
    if (data.namingStrategy.mode === 'suffix') {
      expect(data.namingStrategy.value).toBe('_ts');
    }
  });

  it('infers custom naming for non-pattern outputs', () => {
    const yamlWithCustom = `
pipeline:
  name: "test"
  version: "1.0"
  stages:
    - type: "cast_transformer"
      name: "cast_1"
      inputs: "amount"
      outputs: "total_revenue"
      params:
        target_type: "double"
`;
    const result = importFromYAML(yamlWithCustom);
    const castNode = result.nodes.find((n) => n.id === 'cast_1');
    const data = castNode!.data as StageNodeData;

    expect(data.namingStrategy.mode).toBe('custom');
    if (data.namingStrategy.mode === 'custom') {
      expect(data.namingStrategy.columns).toEqual(['total_revenue']);
    }
  });

  it('handles pipeline with final_features', () => {
    const result = importFromYAML(pipelineWithFinalFeatures);
    const ffNode = result.nodes.find((n) => n.type === 'final_features');

    expect(ffNode).toBeDefined();
    if (ffNode && ffNode.data.type === 'final_features') {
      expect(ffNode.data.features).toEqual(['cat_a', 'amount_ts']);
      expect(ffNode.data.target).toEqual(['label']);
    }
  });

  it('extracts schema hash from meta block', () => {
    const result = importFromYAML(pipelineWithFinalFeatures);
    expect(result.schemaHash).toBe('abc123');
  });

  it('warns on legacy format (no meta)', () => {
    const result = importFromYAML(simplePipelineYaml);
    expect(result.warnings.some((w) => w.includes('legacy'))).toBe(true);
  });

  it('throws on invalid YAML structure', () => {
    expect(() => importFromYAML('not_a_pipeline: true')).toThrow('missing');
  });

  it('throws on empty stages', () => {
    expect(() => importFromYAML('pipeline:\n  stages: []')).toThrow('non-empty');
  });

  it('warns on unknown stage type', () => {
    const yaml = `
pipeline:
  name: "test"
  version: "1.0"
  stages:
    - type: "unknown_type"
      name: "unknown_1"
      inputs: "col"
      outputs: "col_out"
`;
    const result = importFromYAML(yaml);
    expect(result.warnings.some((w) => w.includes('Unknown stage type'))).toBe(true);
  });

  it('auto-layouts nodes with non-zero positions', () => {
    const result = importFromYAML(simplePipelineYaml);
    // At least some nodes should have non-zero Y positions (layers)
    const yPositions = result.nodes.map((n) => n.position.y);
    const uniqueY = new Set(yPositions);
    // Dataset at layer 0, stages at layer 1+
    expect(uniqueY.size).toBeGreaterThan(1);
  });
});
