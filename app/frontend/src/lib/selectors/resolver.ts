/**
 * Edge Selector Resolver.
 *
 * Resolves EdgeSelector (manual / group / rule) against a SparkSchema
 * to produce a concrete list of column names.
 *
 * FAIL-FAST: throws if any selector resolves to 0 columns.
 */

import type { SparkSchema, ColumnMeta } from '../../types/schema';
import { NUMERIC_SPARK_TYPES, CATEGORICAL_SPARK_TYPES } from '../../types/schema';
import type { EdgeSelector, ManualSelector, GroupSelector, RuleSelector } from '../../types/edges';

export function resolveSelector(
  selector: EdgeSelector,
  schema: SparkSchema,
): string[] {
  switch (selector.type) {
    case 'manual':
      return resolveManual(selector, schema);
    case 'group':
      return resolveGroup(selector, schema);
    case 'rule':
      return resolveRule(selector, schema);
  }
}

// ─── Manual ─────────────────────────────────────────────────────────

function resolveManual(sel: ManualSelector, schema: SparkSchema): string[] {
  const available = new Set(schema.columns.map((c) => c.name));
  const missing = sel.columns.filter((col) => !available.has(col));

  if (missing.length > 0) {
    throw new Error(
      `Manual selector references columns not in schema: ${missing.join(', ')}`,
    );
  }

  if (sel.columns.length === 0) {
    throw new Error('Manual selector has an empty column list');
  }

  return [...sel.columns];
}

// ─── Group ──────────────────────────────────────────────────────────

function resolveGroup(sel: GroupSelector, schema: SparkSchema): string[] {
  let predicate: (col: ColumnMeta) => boolean;

  switch (sel.kind) {
    case 'all_numeric':
      predicate = (col) => NUMERIC_SPARK_TYPES.has(col.sparkType);
      break;
    case 'all_categorical':
      predicate = (col) => CATEGORICAL_SPARK_TYPES.has(col.sparkType);
      break;
  }

  const result = schema.columns.filter(predicate).map((c) => c.name);

  if (result.length === 0) {
    throw new Error(`Group selector '${sel.kind}' matched 0 columns in schema`);
  }

  return result;
}

// ─── Rule ───────────────────────────────────────────────────────────

function resolveRule(sel: RuleSelector, schema: SparkSchema): string[] {
  let cols: readonly ColumnMeta[] = schema.columns;

  if (sel.sparkTypeIn && sel.sparkTypeIn.length > 0) {
    const types = new Set(sel.sparkTypeIn);
    cols = cols.filter((c) => types.has(c.sparkType));
  }

  if (sel.nameContains) {
    const pattern = sel.nameContains;
    cols = cols.filter((c) => c.name.includes(pattern));
  }

  if (sel.cardinalityGte !== undefined) {
    const gte = sel.cardinalityGte;
    cols = cols.filter((c) => (c.cardinality ?? 0) >= gte);
  }

  if (sel.cardinalityLte !== undefined) {
    const lte = sel.cardinalityLte;
    cols = cols.filter((c) => (c.cardinality ?? Infinity) <= lte);
  }

  const result = cols.map((c) => c.name);

  if (result.length === 0) {
    throw new Error('Rule selector matched 0 columns in schema');
  }

  return result;
}
