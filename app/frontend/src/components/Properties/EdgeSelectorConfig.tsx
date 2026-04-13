import { useCallback } from 'react';
import type { FlowEdge, EdgeSelector } from '../../types/edges';
import { usePipelineStore } from '../../store/pipelineStore';
import type { SparkDataType } from '../../types/schema';
import { isStageNode } from '../../types/nodes';
import { ChipInput } from '../forms/ChipInput';
import { NumericInput } from '../forms/NumericInput';
import { parseFiniteNumber, parseStringChip, stringifyNumberValue } from '../../lib/formValues';

interface Props {
  readonly edge: FlowEdge;
}

export function EdgeSelectorConfig({ edge }: Props) {
  const updateEdgeSelector = usePipelineStore((s) => s.updateEdgeSelector);
  const validate = usePipelineStore((s) => s.validate);
  const deleteEdge = usePipelineStore((s) => s.deleteEdge);
  const datasetSchema = usePipelineStore((s) => s.datasetSchema);
  const nodes = usePipelineStore((s) => s.nodes);
  const validationResult = usePipelineStore((s) => s.validationResult);

  const sel = edge.selector;

  const setSelector = useCallback(
    (selector: EdgeSelector) => {
      updateEdgeSelector(edge.id, selector);
      validate({ silent: true });
    },
    [edge.id, updateEdgeSelector, validate],
  );

  const switchType = useCallback(
    (type: string) => {
      switch (type) {
        case 'manual':
          setSelector({ type: 'manual', columns: [] });
          break;
        case 'group':
          setSelector({ type: 'group', kind: 'all_numeric' });
          break;
        case 'rule':
          setSelector({ type: 'rule' });
          break;
      }
    },
    [setSelector],
  );

  const sourceNode = nodes.find((n) => n.id === edge.source);
  const compiledSource = validationResult?.compiledNodes.get(edge.source);

  const availableColumns = (() => {
    if (!sourceNode) {
      return datasetSchema?.columns.map((c) => c.name) ?? [];
    }

    if (sourceNode.data.type === 'dataset') {
      return datasetSchema?.columns.map((c) => c.name) ?? [];
    }

    if (compiledSource?.outputSchema) {
      return compiledSource.outputSchema.columns.map((c) => c.name);
    }

    if (isStageNode(sourceNode.data) && sourceNode.data.outputSchema) {
      return sourceNode.data.outputSchema.columns.map((c) => c.name);
    }

    return datasetSchema?.columns.map((c) => c.name) ?? [];
  })();

  return (
    <div className="space-y-3">
      <h3 className="text-xs font-semibold uppercase tracking-wider text-blue-400">
        Edge Selector
      </h3>

      <p className="text-[10px] text-slate-500">
        {edge.source} &rarr; {edge.target}
      </p>

      {/* Selector type */}
      <div>
        <label className="mb-1 block text-[10px] font-medium text-slate-400">Mode</label>
        <select
          value={sel.type}
          onChange={(e) => switchType(e.target.value)}
          className="input-field"
        >
          <option value="manual">Manual (pick columns)</option>
          <option value="group">Group (by type)</option>
          <option value="rule">Rule (by metadata)</option>
        </select>
      </div>

      {/* Manual selector */}
      {sel.type === 'manual' && (
        <div>
          <label className="mb-1 block text-[10px] font-medium text-slate-400">Columns</label>
          <div className="max-h-40 overflow-y-auto rounded border border-slate-700 bg-slate-800 p-2">
            {availableColumns.length === 0 ? (
              <p className="text-[10px] text-slate-500">Load a dataset schema first</p>
            ) : (
              availableColumns.map((col) => (
                <label key={col} className="flex items-center gap-2 py-0.5">
                  <input
                    type="checkbox"
                    checked={sel.columns.includes(col)}
                    onChange={(e) => {
                      const newCols = e.target.checked
                        ? [...sel.columns, col]
                        : sel.columns.filter((c) => c !== col);
                      setSelector({ type: 'manual', columns: newCols });
                    }}
                    className="rounded border-slate-600 bg-slate-700 text-blue-500"
                  />
                  <span className="text-xs text-slate-300">{col}</span>
                </label>
              ))
            )}
          </div>
          <p className="mt-1 text-[10px] text-slate-500">
            Selected: {sel.columns.length}
          </p>
        </div>
      )}

      {/* Group selector */}
      {sel.type === 'group' && (
        <div>
          <label className="mb-1 block text-[10px] font-medium text-slate-400">Kind</label>
          <select
            value={sel.kind}
            onChange={(e) =>
              setSelector({
                type: 'group',
                kind: e.target.value as 'all_numeric' | 'all_categorical',
              })
            }
            className="input-field"
          >
            <option value="all_numeric">All Numeric</option>
            <option value="all_categorical">All Categorical</option>
          </select>
        </div>
      )}

      {/* Rule selector */}
      {sel.type === 'rule' && (
        <div className="space-y-2">
          <div>
            <label className="mb-1 block text-[10px] font-medium text-slate-400">
              Name Contains
            </label>
            <input
              type="text"
              value={sel.nameContains ?? ''}
              onChange={(e) =>
                setSelector({
                  ...sel,
                  nameContains: e.target.value || undefined,
                })
              }
              placeholder="e.g. _norm"
              className="input-field"
            />
          </div>

          <div>
            <label className="mb-1 block text-[10px] font-medium text-slate-400">
              Spark Types (comma-separated)
            </label>
            <ChipInput
              value={(sel.sparkTypeIn ?? []) as SparkDataType[]}
              onChange={(types) =>
                setSelector({
                  ...sel,
                  sparkTypeIn: types.length > 0 ? types : undefined,
                })
              }
              parseItem={(raw) => parseStringChip(raw) as SparkDataType | null}
              formatItem={(item) => item}
              placeholder="integer, double"
              className="space-y-1"
            />
          </div>

          <div className="flex gap-2">
            <div className="flex-1">
              <label className="mb-1 block text-[10px] font-medium text-slate-400">
                Cardinality &ge;
              </label>
              <NumericInput
                value={stringifyNumberValue(sel.cardinalityGte)}
                onChange={(next) => {
                  if (next === '') {
                    setSelector({
                      ...sel,
                      cardinalityGte: undefined,
                    });
                    return;
                  }
                  const parsed = parseFiniteNumber(next);
                  setSelector({
                    ...sel,
                    cardinalityGte: parsed ?? sel.cardinalityGte,
                  });
                }}
                className="input-field"
              />
            </div>
            <div className="flex-1">
              <label className="mb-1 block text-[10px] font-medium text-slate-400">
                Cardinality &le;
              </label>
              <NumericInput
                value={stringifyNumberValue(sel.cardinalityLte)}
                onChange={(next) => {
                  if (next === '') {
                    setSelector({
                      ...sel,
                      cardinalityLte: undefined,
                    });
                    return;
                  }
                  const parsed = parseFiniteNumber(next);
                  setSelector({
                    ...sel,
                    cardinalityLte: parsed ?? sel.cardinalityLte,
                  });
                }}
                className="input-field"
              />
            </div>
          </div>
        </div>
      )}

      {/* Resolved preview */}
      {edge.resolved && (
        <div className="rounded border border-slate-700 bg-slate-800 p-2">
          <p className="text-[10px] font-medium text-slate-400">
            Resolved: {edge.resolvedColumns.length} column(s)
          </p>
          <p className="mt-0.5 truncate text-[10px] text-slate-500">
            {edge.resolvedColumns.slice(0, 5).join(', ')}
            {edge.resolvedColumns.length > 5 ? '...' : ''}
          </p>
        </div>
      )}

      {/* Delete edge */}
      <button
        onClick={() => deleteEdge(edge.id)}
        className="w-full rounded-md bg-red-900/30 px-3 py-1.5 text-xs text-red-400
                   hover:bg-red-900/50"
      >
        Delete Edge
      </button>
    </div>
  );
}
