import { describe, it, expect } from 'vitest';
import { resolveSelector } from '../resolver';
import type { SparkSchema } from '../../../types/schema';
import type { ManualSelector, GroupSelector, RuleSelector } from '../../../types/edges';

const testSchema: SparkSchema = {
  columns: [
    { name: 'id', sparkType: 'long', nullable: false },
    { name: 'name', sparkType: 'string', nullable: false },
    { name: 'amount', sparkType: 'double', nullable: true, cardinality: 100 },
    { name: 'count', sparkType: 'integer', nullable: false, cardinality: 10 },
    { name: 'ts', sparkType: 'timestamp', nullable: false },
    { name: 'flag', sparkType: 'boolean', nullable: false },
    { name: 'amount_norm', sparkType: 'double', nullable: true, cardinality: 500 },
  ],
};

describe('resolveSelector - manual', () => {
  it('resolves valid columns', () => {
    const sel: ManualSelector = { type: 'manual', columns: ['id', 'amount'] };
    expect(resolveSelector(sel, testSchema)).toEqual(['id', 'amount']);
  });

  it('throws on missing columns', () => {
    const sel: ManualSelector = { type: 'manual', columns: ['nonexistent'] };
    expect(() => resolveSelector(sel, testSchema)).toThrow('not in schema');
  });

  it('throws on empty column list', () => {
    const sel: ManualSelector = { type: 'manual', columns: [] };
    expect(() => resolveSelector(sel, testSchema)).toThrow('empty column list');
  });
});

describe('resolveSelector - group', () => {
  it('resolves all_numeric', () => {
    const sel: GroupSelector = { type: 'group', kind: 'all_numeric' };
    const result = resolveSelector(sel, testSchema);
    expect(result).toContain('id');
    expect(result).toContain('amount');
    expect(result).toContain('count');
    expect(result).toContain('amount_norm');
    expect(result).not.toContain('name');
    expect(result).not.toContain('ts');
  });

  it('resolves all_categorical', () => {
    const sel: GroupSelector = { type: 'group', kind: 'all_categorical' };
    const result = resolveSelector(sel, testSchema);
    expect(result).toEqual(['name']);
  });

  it('throws when no columns match', () => {
    const emptySchema: SparkSchema = {
      columns: [{ name: 'ts', sparkType: 'timestamp', nullable: false }],
    };
    const sel: GroupSelector = { type: 'group', kind: 'all_categorical' };
    expect(() => resolveSelector(sel, emptySchema)).toThrow('0 columns');
  });
});

describe('resolveSelector - rule', () => {
  it('filters by sparkTypeIn', () => {
    const sel: RuleSelector = { type: 'rule', sparkTypeIn: ['double'] };
    const result = resolveSelector(sel, testSchema);
    expect(result).toEqual(['amount', 'amount_norm']);
  });

  it('filters by nameContains', () => {
    const sel: RuleSelector = { type: 'rule', nameContains: '_norm' };
    const result = resolveSelector(sel, testSchema);
    expect(result).toEqual(['amount_norm']);
  });

  it('filters by cardinalityGte', () => {
    const sel: RuleSelector = { type: 'rule', cardinalityGte: 100 };
    const result = resolveSelector(sel, testSchema);
    expect(result).toContain('amount');
    expect(result).toContain('amount_norm');
  });

  it('filters by cardinalityLte', () => {
    const sel: RuleSelector = { type: 'rule', cardinalityLte: 10 };
    const result = resolveSelector(sel, testSchema);
    expect(result).toEqual(['count']);
  });

  it('combines multiple rule filters', () => {
    const sel: RuleSelector = {
      type: 'rule',
      sparkTypeIn: ['double'],
      nameContains: 'norm',
    };
    const result = resolveSelector(sel, testSchema);
    expect(result).toEqual(['amount_norm']);
  });

  it('throws when no columns match a rule', () => {
    const sel: RuleSelector = { type: 'rule', nameContains: 'zzz_nonexistent' };
    expect(() => resolveSelector(sel, testSchema)).toThrow('0 columns');
  });
});
