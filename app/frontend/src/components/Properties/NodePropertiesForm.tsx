import { useCallback, useEffect, useMemo, useState } from 'react';
import type { Node } from 'reactflow';
import type { NodeData, StageNodeData, DatasetNodeData, FinalFeatureSelectorData } from '../../types/nodes';
import { isStageNode, isDatasetNode, isFinalFeatures } from '../../types/nodes';
import { usePipelineStore } from '../../store/pipelineStore';
import { getNodeDefinition, type ParamSchema } from '../../registry/NodeRegistry';
import type { OutputNamingStrategy } from '../../types/naming';
import {
  collectFinalFeatureColumns,
  dedupeBuckets,
  type FeatureBucket,
  type FinalFeatureColumn,
  isNumericSparkType,
} from '../../lib/finalFeatures/model';

interface Props {
  readonly node: Node<NodeData>;
}

export function NodePropertiesForm({ node }: Props) {
  const updateNodeData = usePipelineStore((s) => s.updateNodeData);
  const datasetSchema = usePipelineStore((s) => s.datasetSchema);
  const uploadSchemaFromCSV = usePipelineStore((s) => s.uploadSchemaFromCSV);

  const data = node.data;

  // ── Dataset node ────────────────────────────────────────────────
  if (isDatasetNode(data)) {
    return (
      <div className="space-y-3">
        <SectionTitle title="Dataset" color="emerald" />

        <Field label="Source">
          <select
            value={data.source}
            onChange={(e) =>
              updateNodeData(node.id, (prev) => ({
                ...prev,
                source: e.target.value as 'csv' | 'iceberg',
              } as DatasetNodeData))
            }
            className="input-field"
          >
            <option value="csv">Local file (CSV / Parquet)</option>
            <option value="iceberg">Iceberg Table</option>
          </select>
        </Field>

        <Field label="Path / Table">
          <input
            type="text"
            value={data.path}
            onChange={(e) =>
              updateNodeData(node.id, (prev) => ({
                ...prev,
                path: e.target.value,
              } as DatasetNodeData))
            }
            placeholder={data.source === 'csv' ? '/data/file.csv or /data/file.parquet' : 'catalog.db.table'}
            className="input-field"
          />
        </Field>

        {data.source === 'csv' && (
          <div className="space-y-1.5">
            <label className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
              Upload Local File (CSV / Parquet)
            </label>
            <input
              type="file"
              accept=".csv,.parquet"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) {
                  uploadSchemaFromCSV(file).catch(() => {});
                }
              }}
              className="w-full text-xs text-slate-400 file:mr-2 file:rounded-md file:border-0 
                         file:bg-slate-800 file:px-2 file:py-1 file:text-xs file:font-medium 
                         file:text-emerald-400 hover:file:bg-slate-700"
            />
          </div>
        )}

        
        {datasetSchema && (
          <p className="text-xs text-emerald-400">
            {datasetSchema.columns.length} columns loaded
          </p>
        )}
      </div>
    );
  }

  // ── Final features node ─────────────────────────────────────────
  if (isFinalFeatures(data)) {
    return <FinalFeaturesPropertiesForm node={node} data={data} />;
  }

  // ── Stage node ──────────────────────────────────────────────────
  if (isStageNode(data)) {
    return <StagePropertiesForm node={node} data={data} />;
  }

  return null;
}

function FinalFeaturesPropertiesForm({
  node,
  data,
}: {
  readonly node: Node<NodeData>;
  readonly data: FinalFeatureSelectorData;
}) {
  const updateNodeData = usePipelineStore((s) => s.updateNodeData);
  const nodes = usePipelineStore((s) => s.nodes);
  const edges = usePipelineStore((s) => s.edges);
  const validationResult = usePipelineStore((s) => s.validationResult);
  const datasetSchema = usePipelineStore((s) => s.datasetSchema);

  const [search, setSearch] = useState('');
  const [onlyUncertain, setOnlyUncertain] = useState(false);

  const availableColumns = useMemo(
    () => collectFinalFeatureColumns(nodes, edges, node.id, validationResult, datasetSchema),
    [datasetSchema, edges, node.id, nodes, validationResult],
  );

  const normalized = dedupeBuckets({
    feature: data.features,
    target: data.target,
    metadata: data.metadata,
    passthrough: data.passthrough,
  });

  const selectedBucketByColumn = new Map<string, FeatureBucket>();
  (['feature', 'target', 'metadata', 'passthrough'] as const).forEach((bucket) => {
    for (const col of normalized[bucket]) {
      selectedBucketByColumn.set(col, bucket);
    }
  });

  const filteredColumns = availableColumns.filter((col) => {
    const q = search.trim().toLowerCase();
    if (!q && !onlyUncertain) return true;
    const matchSearch = q.length === 0
      || col.name.toLowerCase().includes(q)
      || col.sparkType.toLowerCase().includes(q)
      || col.provenance?.stageType.toLowerCase().includes(q);
    const matchUncertain = !onlyUncertain || col.autoBucket === 'uncertain' || Boolean(col.conflictReason);
    return matchSearch && matchUncertain;
  });

  const setSelections = (next: Record<FeatureBucket, string[]>) => {
    const deduped = dedupeBuckets(next);
    updateNodeData(node.id, (prev) => ({
      ...(prev as FinalFeatureSelectorData),
      features: deduped.feature,
      target: deduped.target,
      metadata: deduped.metadata,
      passthrough: deduped.passthrough,
    }));
  };

  const setColumn = (bucket: FeatureBucket, columnName: string, checked: boolean) => {
    const next = {
      feature: [...normalized.feature],
      target: [...normalized.target],
      metadata: [...normalized.metadata],
      passthrough: [...normalized.passthrough],
    };

    const buckets: FeatureBucket[] = ['feature', 'target', 'metadata', 'passthrough'];
    for (const b of buckets) {
      next[b] = next[b].filter((col) => col !== columnName);
    }

    if (checked) {
      next[bucket].push(columnName);
    }

    setSelections(next);
  };

  const setManualOverride = (columnName: string, bucket: FeatureBucket) => {
    setColumn(bucket, columnName, true);
    updateNodeData(node.id, (prev) => {
      const old = prev as FinalFeatureSelectorData;
      return {
        ...old,
        manualOverrides: {
          ...old.manualOverrides,
          [columnName]: bucket,
        },
      };
    });
  };

  const setConfirmed = (columnName: string, confirmed: boolean) => {
    updateNodeData(node.id, (prev) => {
      const old = prev as FinalFeatureSelectorData;
      const set = new Set(old.confirmedColumns);
      if (confirmed) set.add(columnName);
      else set.delete(columnName);
      return {
        ...old,
        confirmedColumns: Array.from(set),
      };
    });
  };

  const selectAllAuto = () => {
    const next = {
      feature: [] as string[],
      target: [] as string[],
      metadata: [] as string[],
      passthrough: [] as string[],
    };

    for (const col of availableColumns) {
      if (isNumericSparkType(col.sparkType)) next.feature.push(col.name);
    }

    setSelections(next);
  };

  const clearAll = () => {
    setSelections({
      feature: [],
      target: [],
      metadata: [],
      passthrough: [],
    });
  };

  useEffect(() => {
    const hasAnySelection =
      data.features.length
      + data.target.length
      + data.metadata.length
      + data.passthrough.length > 0;

    if (hasAnySelection || availableColumns.length === 0) {
      return;
    }

    const initial = {
      feature: [] as string[],
      target: [] as string[],
      metadata: [] as string[],
      passthrough: [] as string[],
    };

    for (const col of availableColumns) {
      if (isNumericSparkType(col.sparkType)) initial.feature.push(col.name);
    }

    setSelections(initial);
  }, [availableColumns, data.features.length, data.metadata.length, data.target.length, data.passthrough.length]);

  const renderPane = (bucket: FeatureBucket, title: string) => {
    // Show in pane if: 
    // 1. It is already selected for this bucket
    // 2. OR it "belongs" here (numeric -> feature) and is not selected elsewhere
    const fromSchema = filteredColumns.filter((col) => {
      const currentBucket = selectedBucketByColumn.get(col.name);
      if (currentBucket === bucket) return true;
      if (currentBucket) return false; // Already taken by another bucket

      if (bucket === 'feature') return isNumericSparkType(col.sparkType);
      return true; // Target and Metadata show all remaining
    });

    // Add columns that are selected but NOT in availableColumns (manually added or missing)
    const availableNames = new Set(availableColumns.map((c) => c.name));
    const manualOnly = normalized[bucket]
      .filter((name) => !availableNames.has(name))
      .map((name) => ({
        name,
        sparkType: 'unknown',
        isManual: true as const,
      }));

    type PaneItem = (FinalFeatureColumn & { isManual: false }) | { name: string; sparkType: string; isManual: true };

    const allItems: PaneItem[] = [
      ...fromSchema.map((c) => ({ ...c, isManual: false as const })),
      ...manualOnly,
    ];

    return (
      <details key={bucket} open className="rounded border border-slate-800 bg-slate-900/40">
        <summary className="cursor-pointer px-2 py-2 text-xs font-semibold text-slate-300">
          {title} ({normalized[bucket].length})
        </summary>
        <div className="space-y-2 px-2 pb-2">
          {bucket === 'passthrough' && (
            <div className="mb-2 flex gap-2">
              <input
                type="text"
                placeholder="Add manual column... (Enter)"
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    const val = e.currentTarget.value.trim();
                    if (val) {
                      setColumn('passthrough', val, true);
                      e.currentTarget.value = '';
                    }
                  }
                }}
                className="w-full rounded border border-slate-700 bg-slate-950 px-2 py-1 text-[10px] text-slate-200 focus:border-emerald-500 focus:outline-none"
              />
            </div>
          )}

          {allItems.length === 0 && <p className="text-xs text-slate-500">No columns available</p>}
          {allItems.map((item) => {
            const isManual = item.isManual;
            const col = isManual ? null : item;
            const name = item.name;
            const selectedBucket = selectedBucketByColumn.get(name);
            const checked = selectedBucket === bucket;
            const isNonNumeric = col ? !isNumericSparkType(col.sparkType) : false;
            const isInvalidFeature = bucket === 'feature' && isNonNumeric && checked;
            const showUncertain = col ? (col.autoBucket === 'uncertain' || Boolean(col.conflictReason)) : false;

            return (
              <div 
                key={`${bucket}-${name}`} 
                className={`rounded border p-2 text-xs transition-colors ${
                  isInvalidFeature ? 'border-red-500/50 bg-red-950/20' : 'border-slate-800'
                } ${isManual ? 'border-amber-500/30 bg-amber-950/5' : ''}`}
              >
                <div className="flex items-start justify-between gap-2">
                  <label className="flex items-center gap-2 text-slate-200">
                    <input
                      type="checkbox"
                      checked={checked}
                      onChange={(e) => setColumn(bucket, name, e.target.checked)}
                    />
                    <span className={isInvalidFeature ? 'font-bold text-red-200' : ''}>
                      {name}
                    </span>
                  </label>
                  <span className={`rounded px-1.5 py-0.5 text-[10px] ${!isNonNumeric ? 'bg-emerald-900/40 text-emerald-300' : 'bg-amber-900/40 text-amber-300'}`}>
                    {item.sparkType}
                  </span>
                </div>

                <div className="mt-1 flex flex-wrap items-center gap-1 text-[10px] text-slate-400">
                  {col ? (
                    <>
                      <span>{col.provenance?.stageName ? `${col.provenance.stageName} (${col.provenance.stageType})` : 'dataset'}</span>
                      {isInvalidFeature && (
                        <span className="rounded bg-red-900 px-1.5 py-0.5 font-bold text-red-200 ring-1 ring-red-400">
                          Error: Numeric Required
                        </span>
                      )}
                      {col.conflictReason && <span title={col.conflictReason} className="rounded bg-amber-900/40 px-1.5 py-0.5 text-amber-300">Conflict</span>}
                      {col.autoBucket === 'uncertain' && !isInvalidFeature && <span className="rounded bg-slate-700 px-1.5 py-0.5 text-slate-200">Uncertain</span>}
                    </>
                  ) : (
                    <span className="italic text-amber-400/60 font-medium">Manual passthrough (inferred at runtime)</span>
                  )}
                </div>

                <div className="mt-2 flex flex-wrap gap-2">
                  <select
                    value={selectedBucket ?? ''}
                    onChange={(e) => {
                      const nextBucket = e.target.value as FeatureBucket;
                      if (!nextBucket) return;
                      setManualOverride(name, nextBucket);
                    }}
                    className="rounded border border-slate-700 bg-slate-950 px-2 py-1 text-[10px] text-slate-200"
                  >
                    <option value="">Move to...</option>
                    <option value="feature">Feature</option>
                    <option value="target">Target</option>
                    <option value="metadata">Metadata</option>
                    <option value="passthrough">Passthrough</option>
                  </select>

                  {showUncertain && checked && (
                    <label className="flex items-center gap-1 text-[10px] text-amber-300">
                      <input
                        type="checkbox"
                        checked={data.confirmedColumns.includes(name)}
                        onChange={(e) => setConfirmed(name, e.target.checked)}
                      />
                      confirm uncertain/conflict
                    </label>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </details>
    );
  };

  return (
    <div className="space-y-3">
      <SectionTitle title="Final Features" color="red" />

      <div className="grid grid-cols-2 gap-2">
        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search columns"
          className="input-field col-span-2"
        />
        <button className="rounded border border-slate-700 px-2 py-1 text-xs text-slate-200" onClick={selectAllAuto}>
          Select all auto-detected
        </button>
        <button className="rounded border border-slate-700 px-2 py-1 text-xs text-slate-200" onClick={clearAll}>
          Clear
        </button>
      </div>

      <label className="flex items-center gap-2 text-xs text-slate-300">
        <input
          type="checkbox"
          checked={onlyUncertain}
          onChange={(e) => setOnlyUncertain(e.target.checked)}
        />
        Only uncertain/conflicting
      </label>

      {renderPane('feature', 'Features (numeric only)')}
      {renderPane('target', 'Target')}
      {renderPane('metadata', 'Metadata')}
      {renderPane('passthrough', 'Inference/Optional (Passthrough)')}
    </div>
  );
}

// ─── Stage-specific form ──────────────────────────────────────────

function StagePropertiesForm({
  node,
  data,
}: {
  readonly node: Node<NodeData>;
  readonly data: StageNodeData;
}) {
  const updateNodeData = usePipelineStore((s) => s.updateNodeData);
  const edges = usePipelineStore((s) => s.edges);
  const validate = usePipelineStore((s) => s.validate);
  const def = getNodeDefinition(data.type);
  const isEstimator = data.stageType === 'estimator';
  const stageParams = data.params as Record<string, unknown>;
  const incomingEdges = edges.filter((edge) => edge.target === node.id);

  const updateParam = useCallback(
    (name: string, value: unknown) => {
      updateNodeData(node.id, (prev) => {
        const stage = prev as StageNodeData;
        return {
          ...stage,
          params: { ...stage.params, [name]: value },
        } as NodeData;
      });
      validate({ silent: true });
    },
    [node.id, updateNodeData, validate],
  );

  const updateNamingStrategy = useCallback(
    (namingStrategy: OutputNamingStrategy) => {
      updateNodeData(node.id, (prev) => ({
        ...(prev as StageNodeData),
        namingStrategy,
      } as NodeData));
      validate({ silent: true });
    },
    [node.id, updateNodeData, validate],
  );

  const updateLabel = useCallback(
    (label: string) => {
      updateNodeData(node.id, (prev) => ({
        ...prev,
        label,
      } as NodeData));
      validate({ silent: true });
    },
    [node.id, updateNodeData, validate],
  );

  return (
    <div className="space-y-3">
      <SectionTitle
        title={isEstimator ? 'Estimator' : 'Transformer'}
        color={isEstimator ? 'purple' : 'blue'}
      />

      <Field label="Name">
        <input
          type="text"
          value={data.label}
          onChange={(e) => updateLabel(e.target.value)}
          className="input-field"
        />
      </Field>

      <p className="text-[10px] text-slate-500">{def.description}</p>

      {/* Dynamic params */}
      {def.paramSchema.map((schema) => (
        <ParamField
          key={schema.name}
          schema={schema}
          value={stageParams[schema.name]}
          onChange={(v) => updateParam(schema.name, v)}
        />
      ))}

      <OutputNamingSection
        strategy={data.namingStrategy}
        onChange={updateNamingStrategy}
      />

      <InputModeSection incomingEdges={incomingEdges} />

      <PreviewOutputSection
        resolvedInputs={data.resolvedInputs}
        resolvedOutputs={data.resolvedOutputs}
      />

      {def.generatesNewColumns && (
        <ExampleHelp
          input={def.outputExample.input}
          output={def.outputExample.output}
        />
      )}
    </div>
  );
}

function InputModeSection({
  incomingEdges,
}: {
  readonly incomingEdges: readonly { readonly id: string; readonly selector: { readonly type: string } }[];
}) {
  if (incomingEdges.length === 0) {
    return (
      <div className="rounded border border-slate-800 bg-slate-900/40 p-2 text-[11px] text-slate-500">
        Waiting for upstream connection.
      </div>
    );
  }

  return (
    <div className="space-y-1">
      <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">
        Input Mode
      </p>
      {incomingEdges.map((edge) => (
        <p key={edge.id} className="text-xs text-slate-300">
          {`Input Mode: ${edge.selector.type}`}
        </p>
      ))}
    </div>
  );
}

function OutputNamingSection({
  strategy,
  onChange,
}: {
  readonly strategy: OutputNamingStrategy;
  readonly onChange: (strategy: OutputNamingStrategy) => void;
}) {
  return (
    <div className="space-y-2 rounded border border-slate-800 bg-slate-900/40 p-3">
      <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">
        Output Naming
      </p>
      <select
        value={strategy.mode}
        onChange={(e) => {
          const mode = e.target.value as OutputNamingStrategy['mode'];
          switch (mode) {
            case 'auto':
            case 'passthrough':
              onChange({ mode });
              break;
            case 'prefix':
            case 'suffix':
              onChange({ mode, value: '' });
              break;
            case 'custom':
              onChange({ mode, columns: [] });
              break;
          }
        }}
        className="input-field"
      >
        <option value="auto">Auto</option>
        <option value="prefix">Prefix</option>
        <option value="suffix">Suffix</option>
        <option value="passthrough">Passthrough</option>
        <option value="custom">Custom</option>
      </select>

      {(strategy.mode === 'prefix' || strategy.mode === 'suffix') && (
        <input
          type="text"
          value={strategy.value}
          onChange={(e) => onChange({ mode: strategy.mode, value: e.target.value })}
          placeholder={strategy.mode === 'prefix' ? 'e.g. fe_' : 'e.g. _feat'}
          className="input-field"
        />
      )}

      {strategy.mode === 'custom' && (
        <OnePerLineField
          label="Custom Output Columns"
          value={strategy.columns}
          onChange={(columns) => onChange({ mode: 'custom', columns })}
        />
      )}
    </div>
  );
}

function PreviewOutputSection({
  resolvedInputs,
  resolvedOutputs,
}: {
  readonly resolvedInputs: readonly string[];
  readonly resolvedOutputs: readonly string[];
}) {
  return (
    <div className="space-y-1 rounded border border-slate-800 bg-slate-900/40 p-3">
      <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">
        Preview Output
      </p>
      {resolvedInputs.length === 0 || resolvedOutputs.length === 0 ? (
        <p className="text-xs text-slate-500">Preview available after compilation.</p>
      ) : (
        resolvedInputs.map((input, index) => {
          const output = resolvedOutputs[index] ?? resolvedOutputs[0];
          return (
            <p key={`${input}-${output}-${index}`} className="text-xs text-slate-300">
              {`${input} → ${output}`}
            </p>
          );
        })
      )}
    </div>
  );
}

function ExampleHelp({
  input,
  output,
}: {
  readonly input: readonly string[];
  readonly output: readonly string[];
}) {
  return (
    <details className="rounded border border-slate-800 bg-slate-900/40 p-2">
      <summary className="cursor-pointer text-xs text-blue-400">Show example</summary>
      <div className="mt-2 space-y-1 text-xs text-slate-300">
        {input.map((src, index) => (
          <p key={`${src}-${index}`}>
            {`${src} → ${output[index] ?? output[0] ?? ''}`}
          </p>
        ))}
      </div>
    </details>
  );
}

// ─── Param field renderer ─────────────────────────────────────────

function ParamField({
  schema,
  value,
  onChange,
}: {
  readonly schema: ParamSchema;
  readonly value: unknown;
  readonly onChange: (v: unknown) => void;
}) {
  switch (schema.type) {
    case 'select':
      return (
        <Field label={schema.label}>
          <select
            value={String(value ?? schema.default ?? '')}
            onChange={(e) => onChange(e.target.value)}
            className="input-field"
          >
            {schema.options?.map((opt) => (
              <option key={String(opt.value)} value={String(opt.value)}>
                {opt.label}
              </option>
            ))}
          </select>
        </Field>
      );

    case 'number':
      return (
        <Field label={schema.label}>
          <input
            type="number"
            value={value !== undefined && value !== null ? Number(value) : ''}
            onChange={(e) => {
              const n = parseFloat(e.target.value);
              onChange(isNaN(n) ? undefined : n);
            }}
            min={schema.min}
            max={schema.max}
            className="input-field"
          />
        </Field>
      );

    case 'boolean':
      return (
        <Field label={schema.label}>
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={Boolean(value ?? schema.default)}
              onChange={(e) => onChange(e.target.checked)}
              className="rounded border-slate-600 bg-slate-800 text-blue-500"
            />
            <span className="text-xs text-slate-300">
              {Boolean(value ?? schema.default) ? 'Yes' : 'No'}
            </span>
          </label>
        </Field>
      );

    case 'string':
      return (
        <Field label={schema.label}>
          <input
            type="text"
            value={String(value ?? schema.default ?? '')}
            onChange={(e) => onChange(e.target.value)}
            className="input-field"
          />
        </Field>
      );

    case 'array':
      return (
        <Field label={schema.label}>
          <input
            type="text"
            value={Array.isArray(value) ? value.join(', ') : ''}
            onChange={(e) => {
              const parts = e.target.value
                .split(',')
                .map((s) => s.trim())
                .filter(Boolean);
              // Try to parse numbers
              const parsed = parts.map((p) => {
                const n = Number(p);
                return isNaN(n) ? p : n;
              });
              onChange(parsed);
            }}
            placeholder="Comma-separated values"
            className="input-field"
          />
        </Field>
      );

    default:
      return null;
  }
}

// ─── Shared UI primitives ─────────────────────────────────────────

function SectionTitle({ title, color }: { title: string; color: string }) {
  const colorMap: Record<string, string> = {
    blue: 'text-blue-400',
    purple: 'text-purple-400',
    emerald: 'text-emerald-400',
    red: 'text-red-400',
  };

  return (
    <h3 className={`text-xs font-semibold uppercase tracking-wider ${colorMap[color] ?? 'text-slate-400'}`}>
      {title}
    </h3>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="mb-1 block text-[10px] font-medium text-slate-400">{label}</label>
      {children}
    </div>
  );
}

function OnePerLineField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: string[];
  onChange: (v: string[]) => void;
}) {
  return (
    <Field label={label}>
      <textarea
        value={value.join('\n')}
        onChange={(e) => {
          const parts = e.target.value
            .split('\n')
            .map((s) => s.trim())
            .filter(Boolean);
          onChange(parts);
        }}
        placeholder="One column name per line"
        className="input-field"
        rows={4}
      />
    </Field>
  );
}
