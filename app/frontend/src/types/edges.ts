import type { SparkDataType } from './schema';

// ─── Edge Selector discriminated union ──────────────────────────────

/** Manual selector: user picks explicit column names */
export interface ManualSelector {
  readonly type: 'manual';
  columns: string[];
}

/** Group selector: select all numeric or all categorical columns */
export interface GroupSelector {
  readonly type: 'group';
  kind: 'all_numeric' | 'all_categorical';
}

/** Rule selector: deterministic filter by column metadata */
export interface RuleSelector {
  readonly type: 'rule';
  cardinalityGte?: number;
  cardinalityLte?: number;
  nameContains?: string;
  sparkTypeIn?: SparkDataType[];
}

export type EdgeSelector = ManualSelector | GroupSelector | RuleSelector;

// ─── Flow Edge ──────────────────────────────────────────────────────

/**
 * Pipeline edge carrying an EdgeSelector.
 * `resolvedColumns` is populated only after selector resolution.
 */
export interface FlowEdge {
  readonly id: string;
  readonly source: string;
  readonly target: string;
  sourceHandle?: string;
  targetHandle?: string;
  selector: EdgeSelector;
  resolved: boolean;
  resolvedColumns: string[];
}

// ─── Helpers ────────────────────────────────────────────────────────

/** Create a default manual selector with an empty column list. */
export function createDefaultSelector(): ManualSelector {
  return { type: 'manual', columns: [] };
}

/** Summarize an edge selector for display on the edge label. */
export function summarizeSelector(edge: FlowEdge): string {
  const sel = edge.selector;
  switch (sel.type) {
    case 'manual': {
      if (edge.resolved) return `manual (${edge.resolvedColumns.length})`;
      if (sel.columns.length === 0) return 'manual: (none)';
      if (sel.columns.length <= 3) return `manual: ${sel.columns.join(', ')}`;
      return `manual (${sel.columns.length})`;
    }
    case 'group':
      return edge.resolved
        ? `${sel.kind} (${edge.resolvedColumns.length})`
        : sel.kind;
    case 'rule': {
      const parts: string[] = [];
      if (sel.nameContains) parts.push(`name~${sel.nameContains}`);
      if (sel.sparkTypeIn) parts.push(`type=${sel.sparkTypeIn.join(',')}`);
      if (sel.cardinalityGte !== undefined) parts.push(`card>=${sel.cardinalityGte}`);
      if (sel.cardinalityLte !== undefined) parts.push(`card<=${sel.cardinalityLte}`);
      const label = parts.length > 0 ? parts.join(' & ') : 'rule';
      return edge.resolved ? `${label} (${edge.resolvedColumns.length})` : label;
    }
  }
}
