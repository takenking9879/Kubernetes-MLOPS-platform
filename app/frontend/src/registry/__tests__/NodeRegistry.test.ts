import { describe, it, expect } from 'vitest';
import {
  NODE_REGISTRY,
  getNodeDefinition,
  getTransformers,
  getEstimators,
  validateNodeParams,
} from '../NodeRegistry';

describe('NodeRegistry', () => {
  it('has all 14 stage types registered', () => {
    expect(Object.keys(NODE_REGISTRY)).toHaveLength(14);
  });

  it('has 9 transformers', () => {
    const transformers = getTransformers();
    expect(transformers).toHaveLength(9);
    expect(transformers.every((t) => t.category === 'transformer')).toBe(true);
  });

  it('has 5 estimators', () => {
    const estimators = getEstimators();
    expect(estimators).toHaveLength(5);
    expect(estimators.every((e) => e.category === 'estimator')).toBe(true);
  });

  it('getNodeDefinition returns merged definition with behavior', () => {
    const def = getNodeDefinition('cast_transformer');

    expect(def.type).toBe('cast_transformer');
    expect(def.label).toBe('Cast Type');
    expect(def.transformSignature).toBe('one_to_one');
    expect(def.defaultNamingStrategy).toBeDefined();
    expect(def.outputExample).toBeDefined();
  });

  it('annotates array element types for structured editors', () => {
    const binning = getNodeDefinition('binning_transformer');
    const conditional = getNodeDefinition('conditional_transformer');

    expect(binning.paramSchema.find((p) => p.name === 'bins')?.elementType).toBe('number');
    expect(binning.paramSchema.find((p) => p.name === 'labels')?.elementType).toBe('string');
    expect(conditional.paramSchema.find((p) => p.name === 'values')?.elementType).toBe('mixed');
  });

  it('getNodeDefinition throws for unknown type', () => {
    expect(() => getNodeDefinition('nonexistent')).toThrow('Unknown node type');
  });

  describe('inputTypeConstraint', () => {
    it('temporal_extractor requires timestamp', () => {
      const def = getNodeDefinition('temporal_extractor');
      expect(def.inputTypeConstraint).toBeDefined();
      expect(def.inputTypeConstraint!.allowedTypes).toContain('timestamp');
    });

    it('log_transformer requires numeric', () => {
      const def = getNodeDefinition('log_transformer');
      expect(def.inputTypeConstraint).toBeDefined();
      expect(def.inputTypeConstraint!.allowedTypes).toContain('double');
      expect(def.inputTypeConstraint!.allowedTypes).toContain('integer');
      expect(def.inputTypeConstraint!.allowedTypes).toContain('long');
    });

    it('string_indexer requires string', () => {
      const def = getNodeDefinition('string_indexer');
      expect(def.inputTypeConstraint).toBeDefined();
      expect(def.inputTypeConstraint!.allowedTypes).toContain('string');
    });

    it('cast_transformer has no type constraint', () => {
      const def = getNodeDefinition('cast_transformer');
      expect(def.inputTypeConstraint).toBeUndefined();
    });

    it('standard_scaler has no type constraint', () => {
      const def = getNodeDefinition('standard_scaler');
      expect(def.inputTypeConstraint).toBeUndefined();
    });
  });
});

describe('validateNodeParams', () => {
  it('returns empty array for valid cast_transformer params', () => {
    const errors = validateNodeParams('cast_transformer', { target_type: 'double' });
    expect(errors).toEqual([]);
  });

  it('returns error for missing required param', () => {
    const errors = validateNodeParams('cast_transformer', {});
    expect(errors.length).toBeGreaterThan(0);
    expect(errors[0]).toContain('required');
  });

  it('returns error for invalid select value', () => {
    const errors = validateNodeParams('cast_transformer', { target_type: 'invalid_type' });
    expect(errors.length).toBeGreaterThan(0);
    expect(errors[0]).toContain('must be one of');
  });

  it('validates number min constraint', () => {
    const errors = validateNodeParams('cyclic_transformer', { function: 'sin', period: -1 });
    expect(errors.some((e) => e.includes('>= 0.001'))).toBe(true);
  });

  it('validates numeric array items for bins', () => {
    const errors = validateNodeParams('binning_transformer', {
      bins: [0, '10', 50, 100],
      labels: ['low', 'mid', 'high'],
    });

    expect(errors.some((e) => e.includes('must contain only numbers'))).toBe(true);
  });

  it('accepts string labels and mixed conditional values', () => {
    const errors = validateNodeParams('conditional_transformer', {
      condition: 'isin',
      values: [1, 'two', true],
      true_value: 1,
      false_value: 0,
    });

    expect(errors).toEqual([]);
  });

  it('returns empty array for valid standard_scaler params', () => {
    const errors = validateNodeParams('standard_scaler', { with_mean: true, with_std: true });
    expect(errors).toEqual([]);
  });
});
