import { describe, it, expect } from 'vitest';
import { simulateStage, mergeSchemas } from '../schemaSimulator';
import type { SparkSchema } from '../../../types/schema';
import type { StageNodeData, CastTransformerData, LogTransformerData, ConcatTransformerData } from '../../../types/nodes';

const baseSchema: SparkSchema = {
  columns: [
    { name: 'amount', sparkType: 'double', nullable: true },
    { name: 'count', sparkType: 'integer', nullable: false },
    { name: 'name', sparkType: 'string', nullable: false },
    { name: 'ts', sparkType: 'timestamp', nullable: false },
  ],
};

function makeStageData(overrides: Partial<StageNodeData>): StageNodeData {
  return {
    id: 'test-stage',
    label: 'Test Stage',
    type: 'cast_transformer',
    stageType: 'transformer',
    inputs: [],
    outputs: [],
    params: { target_type: 'double' },
    namingStrategy: { mode: 'suffix', value: '_out' },
    resolvedInputs: [],
    resolvedOutputs: [],
    outputSchema: null,
    compilation: { status: 'draft' },
    ...overrides,
  } as StageNodeData;
}

describe('simulateStage', () => {
  it('produces correct output with suffix naming', () => {
    const data = makeStageData({
      type: 'cast_transformer',
      params: { target_type: 'timestamp' },
      namingStrategy: { mode: 'suffix', value: '_ts' },
    } as Partial<CastTransformerData>);

    const result = simulateStage(data, baseSchema, ['amount']);

    expect(result.resolvedOutputs).toEqual(['amount_ts']);
    expect(result.columnsAdded).toEqual(['amount_ts']);
    expect(result.outputSchema.columns.some((c) => c.name === 'amount_ts')).toBe(true);
  });

  it('produces correct output with prefix naming', () => {
    const data = makeStageData({
      type: 'log_transformer',
      params: { log_type: 'log1p' },
      namingStrategy: { mode: 'prefix', value: 'log_' },
    } as Partial<LogTransformerData>);

    const result = simulateStage(data, baseSchema, ['amount']);

    expect(result.resolvedOutputs).toEqual(['log_amount']);
  });

  it('produces correct output with passthrough naming', () => {
    const data = makeStageData({
      type: 'cast_transformer',
      params: { target_type: 'double' },
      namingStrategy: { mode: 'passthrough' },
    } as Partial<CastTransformerData>);

    // Passthrough would create a duplicate 'amount' - should throw
    expect(() => simulateStage(data, baseSchema, ['amount'])).toThrow('duplicate');
  });

  it('produces correct output with custom naming', () => {
    const data = makeStageData({
      type: 'cast_transformer',
      params: { target_type: 'double' },
      namingStrategy: { mode: 'custom', columns: ['my_custom_col'] },
    } as Partial<CastTransformerData>);

    const result = simulateStage(data, baseSchema, ['amount']);

    expect(result.resolvedOutputs).toEqual(['my_custom_col']);
  });

  it('throws on missing input column', () => {
    const data = makeStageData({});

    expect(() => simulateStage(data, baseSchema, ['nonexistent'])).toThrow('does not exist in schema');
  });

  it('type constraint: temporal_extractor rejects double column', () => {
    const data = makeStageData({
      type: 'temporal_extractor',
      stageType: 'transformer',
      params: { component: 'hour' },
      namingStrategy: { mode: 'suffix', value: '_time' },
    });

    expect(() => simulateStage(data, baseSchema, ['amount'])).toThrow('timestamp');
  });

  it('type constraint: temporal_extractor accepts timestamp column', () => {
    const data = makeStageData({
      type: 'temporal_extractor',
      stageType: 'transformer',
      params: { component: 'hour' },
      namingStrategy: { mode: 'suffix', value: '_time' },
    });

    const result = simulateStage(data, baseSchema, ['ts']);
    expect(result.resolvedOutputs).toEqual(['ts_time']);
  });

  it('type constraint: string_indexer rejects numeric column', () => {
    const data = makeStageData({
      type: 'string_indexer',
      stageType: 'estimator',
      params: { handle_invalid: 'keep' },
      namingStrategy: { mode: 'suffix', value: '_idx' },
    });

    expect(() => simulateStage(data, baseSchema, ['count'])).toThrow('string');
  });

  it('type constraint: log_transformer rejects string column', () => {
    const data = makeStageData({
      type: 'log_transformer',
      stageType: 'transformer',
      params: { log_type: 'log1p' },
      namingStrategy: { mode: 'suffix', value: '_log' },
    });

    expect(() => simulateStage(data, baseSchema, ['name'])).toThrow('numeric');
  });

  it('many_to_one (concat) produces single output', () => {
    const data = makeStageData({
      type: 'concat_transformer',
      stageType: 'transformer',
      params: { separator: '_' },
      namingStrategy: { mode: 'suffix', value: '_concat' },
    } as Partial<ConcatTransformerData>);

    const stringSchema: SparkSchema = {
      columns: [
        { name: 'city', sparkType: 'string', nullable: false },
        { name: 'country', sparkType: 'string', nullable: false },
      ],
    };

    const result = simulateStage(data, stringSchema, ['city', 'country']);
    expect(result.resolvedOutputs).toHaveLength(1);
    expect(result.resolvedOutputs[0]).toBe('city_concat');
  });
});

describe('mergeSchemas', () => {
  it('merges non-overlapping schemas', () => {
    const s1: SparkSchema = { columns: [{ name: 'a', sparkType: 'double', nullable: false }] };
    const s2: SparkSchema = { columns: [{ name: 'b', sparkType: 'string', nullable: false }] };

    const merged = mergeSchemas([s1, s2]);
    expect(merged.columns).toHaveLength(2);
  });

  it('allows same column with same type', () => {
    const s1: SparkSchema = { columns: [{ name: 'a', sparkType: 'double', nullable: false }] };
    const s2: SparkSchema = { columns: [{ name: 'a', sparkType: 'double', nullable: true }] };

    const merged = mergeSchemas([s1, s2]);
    expect(merged.columns).toHaveLength(1);
  });

  it('throws on conflicting types for same column', () => {
    const s1: SparkSchema = { columns: [{ name: 'a', sparkType: 'double', nullable: false }] };
    const s2: SparkSchema = { columns: [{ name: 'a', sparkType: 'string', nullable: false }] };

    expect(() => mergeSchemas([s1, s2])).toThrow('Conflicting Spark types');
  });
});
