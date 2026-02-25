/**
 * RunPage — ML platform execution control surface.
 *
 * Generates params.yaml + dataset schema YAMLs with a live preview panel.
 *
 * Sections:
 *   A. Basic Configuration  (execution, dataset/DSL, splits, model config)
 *   B. Hyperparameters      (dynamic: tune=false → *_PARAMS, tune=true → three sub-panels)
 *   C. Dataset Schema Builder (collapsed, generates raw/full/preprocessed YAMLs)
 *   D. Advanced Configuration (collapsed, kuberay/spark/iceberg)
 *
 * Right column: live YAML preview with tabs for params.yaml / raw / full / preprocessed.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import { ChevronDown, ChevronUp, Copy, RefreshCw } from 'lucide-react';
import {
  checkArtifact,
  getIcebergSample,
  getRunStatus,
  listDsls,
  submitRun,
  uploadSchemas,
  type DslVersion,
  type RunResult,
  type SchemaUploadResult,
} from '../api/platformClient';
import { useDatasetStore } from '../store/datasetStore';
import {
  generateParamsYaml,
  DEFAULT_ADVANCED_CONFIG,
  type AdvancedConfig,
  type ParamsYamlInput,
} from '../lib/paramsYaml';
import {
  generateRawYaml,
  generateFullYaml,
  generatePreprocessedYaml,
  validateSchemaBuilder,
  type SchemaBuilderState,
  type SchemaColumn,
} from '../lib/schemaYaml';
import {
  XGBOOST_DEFAULTS,
  XGBOOST_TUNE_SETTINGS_DEFAULTS,
  getDefaults,
  getTuneSettingsDefaults,
  getAllowedKeys,
  getSearchSpace,
  getParamMeta,
  getTuneSettingsMeta,
  getNonTunableKeys,
  type ParamMeta,
  type SearchSpaceEntry,
} from '../types/hyperparams';

// ─── Constants ────────────────────────────────────────────────────────────────

const DEFAULT_SPLITS = {
  train: { start: '2026-01-01 00:00:00', end: '2026-01-01 12:00:00' },
  val:   { start: '2026-01-01 12:00:00', end: '2026-01-02 00:00:00' },
  test:  { start: '2026-01-02 00:00:00', end: '2026-01-03 00:00:00' },
};

const DEFAULT_MODEL_CONFIG = {
  experiment_name: 'kuberay-attack-detection',
  registry_model_name: 'attack-detection',
  target: 'attack',
  num_classes: 6,
  seed: 42,
};

const DATE_RE = /^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$/;

type Framework = 'xgboost' | 'pytorch';
type TuneMode = 'predefined' | 'override';
type PreviewTab = 'params.yaml' | 'raw.yaml' | 'full.yaml' | 'preprocessed.yaml';

function nowId(): string {
  const d = new Date();
  return [
    d.getUTCFullYear(),
    String(d.getUTCMonth() + 1).padStart(2, '0'),
    String(d.getUTCDate()).padStart(2, '0'),
    '_',
    String(d.getUTCHours()).padStart(2, '0'),
    String(d.getUTCMinutes()).padStart(2, '0'),
    String(d.getUTCSeconds()).padStart(2, '0'),
  ].join('');
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function AccordionSection({
  title,
  defaultOpen = false,
  badge,
  children,
}: {
  title: string;
  defaultOpen?: boolean;
  badge?: React.ReactNode;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="rounded-lg border border-slate-700 bg-slate-900">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between px-4 py-3 text-sm font-medium text-slate-300 hover:text-slate-100"
      >
        <span className="flex items-center gap-2">
          {title}
          {badge}
        </span>
        {open ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      </button>
      {open && (
        <div className="border-t border-slate-700 px-4 pb-4 pt-3 space-y-3">
          {children}
        </div>
      )}
    </div>
  );
}

function Label({ children, title }: { children: React.ReactNode; title?: string }) {
  return (
    <label className="text-xs text-slate-400" title={title}>
      {children}
    </label>
  );
}

function Field({ label, tooltip, children }: { label: string; tooltip?: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1">
      <Label title={tooltip}>{label}{tooltip && <span className="ml-1 text-slate-600 cursor-help">?</span>}</Label>
      {children}
    </div>
  );
}

const INPUT_CLS = 'rounded bg-slate-800 px-2 py-1 text-xs text-slate-100 outline-none focus:ring-1 focus:ring-blue-500 w-full';
const SELECT_CLS = 'rounded bg-slate-800 px-2 py-1 text-xs text-slate-100 outline-none focus:ring-1 focus:ring-blue-500 w-full';
const BTN_NEUTRAL = 'rounded bg-slate-700 px-2 py-1 text-xs text-slate-300 hover:bg-slate-600 disabled:opacity-40';
const BTN_PRIMARY = 'rounded bg-blue-600 px-4 py-1.5 text-xs font-medium text-white hover:bg-blue-500 disabled:opacity-40';
const SUB_HEADING = 'text-xs font-semibold text-slate-400 uppercase tracking-wider';

function ToggleButton({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={`rounded-full px-3 py-1 text-xs font-medium transition-colors ${
        active
          ? 'bg-blue-600 text-white'
          : 'bg-slate-700 text-slate-400 hover:bg-slate-600'
      }`}
    >
      {children}
    </button>
  );
}

function ReadOnlyBadge({ children }: { children: React.ReactNode }) {
  return (
    <span className="rounded bg-slate-700 px-1.5 py-0.5 text-[10px] text-slate-400 font-mono">
      {children}
    </span>
  );
}

// ─── Hyperparameter input row ─────────────────────────────────────────────────

function ParamInput({
  paramKey,
  meta,
  value,
  onChange,
  readOnly = false,
}: {
  paramKey: string;
  meta: { type: string; defaultValue: unknown; options?: string[]; min?: number; max?: number; step?: number };
  value: number | string | string[];
  onChange?: (key: string, val: number | string | string[]) => void;
  readOnly?: boolean;
}) {
  if (readOnly || meta.type === 'array') {
    const display = Array.isArray(value) ? value.join(', ') : String(value);
    return <ReadOnlyBadge>{display}</ReadOnlyBadge>;
  }
  if (meta.type === 'string' && meta.options) {
    return (
      <select
        className={SELECT_CLS}
        value={value as string}
        onChange={(e) => onChange?.(paramKey, e.target.value)}
      >
        {meta.options.map((o) => (
          <option key={o} value={o}>{o}</option>
        ))}
      </select>
    );
  }
  return (
    <input
      type="number"
      className={INPUT_CLS}
      value={value as number}
      min={meta.min}
      max={meta.max}
      step={meta.step ?? 'any'}
      onChange={(e) => onChange?.(paramKey, parseFloat(e.target.value) || 0)}
    />
  );
}

// ─── Search space table row ───────────────────────────────────────────────────

function SearchSpaceRow({
  paramKey,
  entry,
  override,
  onOverrideChange,
  overrideMode,
}: {
  paramKey: string;
  entry: SearchSpaceEntry;
  override?: { min: number; max: number };
  onOverrideChange?: (key: string, field: 'min' | 'max', val: number) => void;
  overrideMode: boolean;
}) {
  const renderRange = () => {
    if (entry.type === 'fixed') {
      const v = Array.isArray(entry.value) ? entry.value.join(', ') : String(entry.value);
      return <ReadOnlyBadge>fixed: {v}</ReadOnlyBadge>;
    }
    if (entry.type === 'choice') {
      return <ReadOnlyBadge>choice: [{entry.options.join(', ')}]</ReadOnlyBadge>;
    }
    // entry.type is now narrowed to 'randint' | 'uniform' | 'loguniform'
    if (overrideMode) {
      return (
        <div className="flex items-center gap-1">
          <input
            type="number"
            className="w-20 rounded bg-slate-800 px-1.5 py-0.5 text-xs text-slate-100 outline-none focus:ring-1 focus:ring-blue-500"
            value={override?.min ?? entry.min}
            step="any"
            onChange={(e) => onOverrideChange?.(paramKey, 'min', parseFloat(e.target.value))}
          />
          <span className="text-slate-500 text-xs">–</span>
          <input
            type="number"
            className="w-20 rounded bg-slate-800 px-1.5 py-0.5 text-xs text-slate-100 outline-none focus:ring-1 focus:ring-blue-500"
            value={override?.max ?? entry.max}
            step="any"
            onChange={(e) => onOverrideChange?.(paramKey, 'max', parseFloat(e.target.value))}
          />
        </div>
      );
    }
    // Predefined read-only
    if (entry.type === 'randint') return <ReadOnlyBadge>randint({entry.min}, {entry.max})</ReadOnlyBadge>;
    if (entry.type === 'uniform') return <ReadOnlyBadge>uniform({entry.min}, {entry.max})</ReadOnlyBadge>;
    return <ReadOnlyBadge>loguniform({entry.min.toExponential(0)}, {entry.max})</ReadOnlyBadge>;
  };

  return (
    <div className="grid grid-cols-[140px_1fr] items-center gap-2 py-1">
      <span className="font-mono text-xs text-slate-300">{paramKey}</span>
      {renderRange()}
    </div>
  );
}

// ─── YAML Preview Panel ───────────────────────────────────────────────────────

function YamlPreviewPanel({
  tabs,
  activeTab,
  onTabChange,
}: {
  tabs: Array<{ key: PreviewTab; label: string; content: string }>;
  activeTab: PreviewTab;
  onTabChange: (tab: PreviewTab) => void;
}) {
  const [copied, setCopied] = useState(false);
  const content = tabs.find((t) => t.key === activeTab)?.content ?? '';

  const handleCopy = () => {
    void navigator.clipboard.writeText(content).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };

  return (
    <div className="flex h-full flex-col rounded-lg border border-slate-700 bg-slate-900 overflow-hidden">
      {/* Tab bar */}
      <div className="flex items-center justify-between border-b border-slate-700 px-3 py-2 shrink-0">
        <div className="flex gap-1">
          {tabs.map((t) => (
            <button
              key={t.key}
              onClick={() => onTabChange(t.key)}
              className={`rounded px-2 py-0.5 text-[10px] font-medium transition-colors ${
                activeTab === t.key
                  ? 'bg-slate-700 text-slate-100'
                  : 'text-slate-500 hover:text-slate-300'
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1 rounded px-2 py-0.5 text-[10px] text-slate-400 hover:text-slate-200 hover:bg-slate-800"
          title="Copy to clipboard"
        >
          <Copy size={10} />
          {copied ? 'Copied!' : 'Copy'}
        </button>
      </div>
      {/* Content */}
      <pre className="flex-1 overflow-auto p-3 text-[11px] leading-relaxed font-mono text-green-300 bg-slate-950 whitespace-pre">
        {content}
      </pre>
    </div>
  );
}

// ─── Main RunPage Component ───────────────────────────────────────────────────

export function RunPage() {
  const { activeDataset } = useDatasetStore();

  // ── Execution ────────────────────────────────────────────────────────────
  const [dataset, setDataset] = useState(activeDataset ?? '');
  const [dsls, setDsls] = useState<DslVersion[]>([]);
  const [dslVersion, setDslVersion] = useState<number | ''>('');
  const [executionId, setExecutionId] = useState(nowId);

  // ── Tuning ───────────────────────────────────────────────────────────────
  const [tuningEnabled, setTuningEnabled] = useState(true);
  const [numberOfTrials, setNumberOfTrials] = useState(3);
  const [sampleFraction, setSampleFraction] = useState(0.2);

  // ── Model ────────────────────────────────────────────────────────────────
  const [framework, setFramework] = useState<Framework>('xgboost');
  const [modelConfig, setModelConfig] = useState({ ...DEFAULT_MODEL_CONFIG });

  // ── Splits ───────────────────────────────────────────────────────────────
  const [splits, setSplits] = useState(DEFAULT_SPLITS);

  // ── Hyperparams (*_PARAMS) — reset on framework change ──────────────────
  const [hyperparams, setHyperparams] = useState<Record<string, number | string | string[]>>(
    () => ({ ...XGBOOST_DEFAULTS }),
  );

  // ── Tune settings (*_TUNE_SETTINGS) ─────────────────────────────────────
  const [tuneSettings, setTuneSettings] = useState<Record<string, number>>(
    () => ({ ...XGBOOST_TUNE_SETTINGS_DEFAULTS }),
  );

  // ── Search space (UI-only override mode) ─────────────────────────────────
  const [tuneMode, setTuneMode] = useState<TuneMode>('predefined');
  const [searchSpaceOverrides, setSearchSpaceOverrides] = useState<
    Record<string, { min: number; max: number }>
  >({});

  // ── Schema builder ───────────────────────────────────────────────────────
  const [schemaBuilder, setSchemaBuilder] = useState<SchemaBuilderState>({
    allColumns: [],
    rawTopLevel: [],
    propertiesFields: [],
    fullColumns: [],
    preprocessedColumns: [],
    targetColumn: '',
    idColumn: '',
    predictionColumn: '',
  });
  const [schemaLoading, setSchemaLoading] = useState(false);
  const [schemaSaving, setSchemaSaving] = useState(false);
  const [schemaError, setSchemaError] = useState('');
  const [schemaSaveResult, setSchemaSaveResult] = useState<SchemaUploadResult | null>(null);

  // ── Advanced ─────────────────────────────────────────────────────────────
  const [advanced, setAdvanced] = useState<AdvancedConfig>({ ...DEFAULT_ADVANCED_CONFIG });

  // ── Preview ──────────────────────────────────────────────────────────────
  const [activePreviewTab, setActivePreviewTab] = useState<PreviewTab>('params.yaml');

  // ── Submission ───────────────────────────────────────────────────────────
  const [validationErrors, setValidationErrors] = useState<string[]>([]);
  const [artifactCheck, setArtifactCheck] = useState<{
    loading: boolean; exists?: boolean; processedTable?: string | null; error?: string;
  }>({ loading: false });
  const [submitResult, setSubmitResult] = useState<RunResult | null>(null);
  const [submitError, setSubmitError] = useState('');
  const [runStatus, setRunStatus] = useState('');
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Effects ──────────────────────────────────────────────────────────────

  useEffect(() => {
    if (activeDataset) setDataset(activeDataset);
  }, [activeDataset]);

  useEffect(() => {
    if (!dataset) { setDsls([]); return; }
    listDsls(dataset)
      .then((r) => setDsls(r.dsls))
      .catch(() => setDsls([]));
  }, [dataset]);

  useEffect(() => {
    setHyperparams({ ...getDefaults(framework) });
    setTuneSettings({ ...getTuneSettingsDefaults(framework) });
    setSearchSpaceOverrides({});
  }, [framework]);

  useEffect(() => {
    if (!submitResult?.dag_run_id) return;
    const TERMINAL = ['success', 'failed', 'upstream_failed'];
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const s = await getRunStatus(submitResult.dag_run_id);
        setRunStatus(s.state);
        if (TERMINAL.includes(s.state.toLowerCase())) {
          if (pollRef.current) clearInterval(pollRef.current);
        }
      } catch { /* keep polling */ }
    }, 15_000);
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [submitResult]);

  // ── useMemo — YAML previews ───────────────────────────────────────────────

  const paramsInput = useMemo<ParamsYamlInput>(() => ({
    execution_id: executionId,
    dataset,
    dslS3Path: dslVersion !== ''
      ? `s3://<S3_BUCKET>/dsl/dsl_${dataset}/v${dslVersion}__<slug>.yaml`
      : '<select a DSL version>',
    tuning: { enabled: tuningEnabled, number_of_trials: numberOfTrials },
    splits,
    framework,
    model: modelConfig,
    sample_fraction_for_tuning: sampleFraction,
    hyperparams,
    tuneSettings: tuningEnabled ? tuneSettings : {},
    advanced,
  }), [
    executionId, dataset, dslVersion, tuningEnabled, numberOfTrials,
    splits, framework, modelConfig, sampleFraction, hyperparams, tuneSettings, advanced,
  ]);

  const paramsYamlPreview = useMemo(() => generateParamsYaml(paramsInput), [paramsInput]);
  const rawYamlPreview    = useMemo(() => schemaBuilder.allColumns.length ? generateRawYaml(schemaBuilder, dataset) : '# Load columns to preview schema', [schemaBuilder, dataset]);
  const fullYamlPreview   = useMemo(() => schemaBuilder.allColumns.length ? generateFullYaml(schemaBuilder, dataset) : '# Load columns to preview schema', [schemaBuilder, dataset]);
  const preprocessedYamlPreview = useMemo(
    () => schemaBuilder.allColumns.length ? generatePreprocessedYaml(schemaBuilder, dataset) : '# Load columns to preview schema',
    [schemaBuilder, dataset],
  );

  // ── Validation ────────────────────────────────────────────────────────────

  function validateForm(): string[] {
    const errors: string[] = [];
    const allowed = getAllowedKeys(framework);

    if (!tuningEnabled) {
      for (const k of Object.keys(hyperparams)) {
        if (!allowed.has(k)) errors.push(`Unknown hyperparameter key: "${k}"`);
      }
    }
    if (modelConfig.num_classes < 2) errors.push('num_classes must be at least 2');
    if (tuningEnabled && numberOfTrials < 1) errors.push('number_of_trials must be at least 1');
    if (tuningEnabled && (sampleFraction < 0.01 || sampleFraction > 1.0)) {
      errors.push('sample_fraction_for_tuning must be between 0.01 and 1.0');
    }

    for (const [s, f] of [
      ['train', 'start'], ['train', 'end'],
      ['val', 'start'],   ['val', 'end'],
      ['test', 'start'],  ['test', 'end'],
    ] as const) {
      const val = splits[s][f];
      if (!DATE_RE.test(val)) errors.push(`${s}.${f}: must be YYYY-MM-DD HH:MM:SS`);
    }

    const schemaErrors = validateSchemaBuilder(schemaBuilder);
    errors.push(...schemaErrors);

    return errors;
  }

  // ── Handlers ──────────────────────────────────────────────────────────────

  const handleCheckArtifact = async () => {
    if (!dataset || !executionId) return;
    setArtifactCheck({ loading: true });
    try {
      const r = await checkArtifact(executionId, dataset);
      setArtifactCheck({ loading: false, exists: r.exists, processedTable: r.processed_table });
    } catch (e) {
      setArtifactCheck({ loading: false, error: e instanceof Error ? e.message : String(e) });
    }
  };

  const handleSubmit = async () => {
    setSubmitError('');
    setSubmitResult(null);
    setRunStatus('');
    const errors = validateForm();
    setValidationErrors(errors);
    if (errors.length > 0) return;
    if (!dataset || dslVersion === '') {
      setValidationErrors(['Select a dataset and DSL version first.']);
      return;
    }

    try {
      const result = await submitRun({
        dataset,
        dsl_version: dslVersion as number,
        execution_id: executionId.trim(),
        framework,
        splits,
        tuning: { enabled: tuningEnabled, number_of_trials: numberOfTrials },
        model: modelConfig,
        sample_fraction_for_tuning: sampleFraction,
        hyperparams: tuningEnabled ? {} : hyperparams,
        tune_settings: tuningEnabled ? tuneSettings : {},
      });
      setSubmitResult(result);
      setRunStatus('queued');
    } catch (e) {
      // Surface 422 validation details from backend
      const msg = e instanceof Error ? e.message : String(e);
      setSubmitError(msg);
    }
  };

  const handleLoadSchema = async () => {
    if (!dataset) { setSchemaError('Select a dataset first'); return; }
    setSchemaLoading(true);
    setSchemaError('');
    try {
      const r = await getIcebergSample(dataset, 10);
      const cols = r.columns as SchemaColumn[];
      setSchemaBuilder({
        allColumns: cols,
        rawTopLevel: cols.filter((c) => !c.name.includes('.')).slice(0, 3).map((c) => c.name),
        propertiesFields: [],
        fullColumns: cols.map((c) => c.name),
        preprocessedColumns: [],
        targetColumn: '',
        idColumn: cols[0]?.name ?? '',
        predictionColumn: 'label',
      });
    } catch (e) {
      setSchemaError(e instanceof Error ? e.message : String(e));
    } finally {
      setSchemaLoading(false);
    }
  };

  const handleSaveSchemas = async () => {
    if (!dataset) return;
    setSchemaSaving(true);
    setSchemaSaveResult(null);
    setSchemaError('');
    try {
      const r = await uploadSchemas(dataset, {
        raw: rawYamlPreview,
        full: fullYamlPreview,
        preprocessed: preprocessedYamlPreview,
      });
      setSchemaSaveResult(r);
    } catch (e) {
      setSchemaError(e instanceof Error ? e.message : String(e));
    } finally {
      setSchemaSaving(false);
    }
  };

  const downloadYaml = (content: string, filename: string) => {
    const blob = new Blob([content], { type: 'text/yaml' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  // ── Helpers ───────────────────────────────────────────────────────────────

  const updateHyperparam = (key: string, val: number | string | string[]) => {
    setHyperparams((prev) => ({ ...prev, [key]: val }));
  };

  const updateTuneSetting = (key: string, val: number) => {
    setTuneSettings((prev) => ({ ...prev, [key]: val }));
  };

  const updateSearchSpaceOverride = (key: string, field: 'min' | 'max', val: number) => {
    setSearchSpaceOverrides((prev) => ({
      ...prev,
      [key]: { ...(prev[key] ?? { min: 0, max: 1 }), [field]: val },
    }));
  };

  const updateSplit = (
    split: keyof typeof splits,
    field: 'start' | 'end',
    val: string,
  ) => {
    setSplits((prev) => ({ ...prev, [split]: { ...prev[split], [field]: val } }));
  };

  const updateSchemaToggle = (colName: string, field: 'rawTopLevel' | 'propertiesFields' | 'fullColumns' | 'preprocessedColumns', checked: boolean) => {
    setSchemaBuilder((prev) => {
      const arr = prev[field];
      return {
        ...prev,
        [field]: checked ? [...arr, colName] : arr.filter((n) => n !== colName),
      };
    });
  };

  const stateColor = (state: string) =>
    state === 'success' ? 'text-green-400'
      : state === 'running' ? 'text-yellow-400'
        : state.includes('fail') ? 'text-red-400'
          : 'text-slate-400';

  // ── Derived ───────────────────────────────────────────────────────────────

  const searchSpace = getSearchSpace(framework);
  const paramMeta = getParamMeta(framework);
  const tuneSettingsMeta = getTuneSettingsMeta(framework);
  const nonTunableKeys = getNonTunableKeys(framework);

  const previewTabs = [
    { key: 'params.yaml' as PreviewTab, label: 'params.yaml', content: paramsYamlPreview },
    { key: 'raw.yaml' as PreviewTab, label: 'raw.yaml', content: rawYamlPreview },
    { key: 'full.yaml' as PreviewTab, label: 'full.yaml', content: fullYamlPreview },
    { key: 'preprocessed.yaml' as PreviewTab, label: 'preprocessed.yaml', content: preprocessedYamlPreview },
  ];

  // ─────────────────────────────────────────────────────────────────────────────
  // RENDER
  // ─────────────────────────────────────────────────────────────────────────────

  return (
    <div className="flex h-full min-h-0 overflow-hidden">
      {/* ── LEFT: scrollable form ── */}
      <div className="flex-1 min-w-0 overflow-y-auto p-4 space-y-3">

        {/* ── A. BASIC CONFIGURATION ── */}
        <AccordionSection title="Basic Configuration" defaultOpen>

          {/* Execution identity */}
          <div className="space-y-2">
            <p className={SUB_HEADING}>Execution</p>
            <div className="grid grid-cols-2 gap-3">
              <Field label="Execution ID" tooltip="Unique identifier for this run. Auto-generated, but editable.">
                <div className="flex gap-1">
                  <input
                    value={executionId}
                    onChange={(e) => setExecutionId(e.target.value)}
                    className={INPUT_CLS}
                  />
                  <button
                    onClick={() => setExecutionId(nowId())}
                    className={BTN_NEUTRAL}
                    title="Regenerate"
                  >
                    <RefreshCw size={12} />
                  </button>
                </div>
              </Field>
              <Field label="Dataset">
                <input
                  value={dataset}
                  onChange={(e) => setDataset(e.target.value)}
                  placeholder="network_traffic_raw"
                  className={INPUT_CLS}
                />
              </Field>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <Field label="DSL Version" tooltip="Select a saved DSL pipeline version for this dataset.">
                <select
                  value={dslVersion}
                  onChange={(e) => setDslVersion(e.target.value === '' ? '' : Number(e.target.value))}
                  className={SELECT_CLS}
                >
                  <option value="">— select —</option>
                  {dsls.map((d) => (
                    <option key={d.version} value={d.version}>
                      v{d.version} — {d.slug}
                    </option>
                  ))}
                </select>
              </Field>
              <Field label="">
                <div className="flex items-center gap-2 pt-4">
                  <button
                    onClick={() => void handleCheckArtifact()}
                    disabled={artifactCheck.loading}
                    className={BTN_NEUTRAL}
                  >
                    {artifactCheck.loading ? 'Checking…' : 'Check artifact'}
                  </button>
                  {artifactCheck.exists !== undefined && (
                    <span className={`text-xs font-medium ${artifactCheck.exists ? 'text-yellow-400' : 'text-green-400'}`}>
                      {artifactCheck.exists
                        ? `⚠ Exists — preprocessing will be skipped`
                        : '✓ New run — full pipeline'}
                    </span>
                  )}
                  {artifactCheck.error && <span className="text-xs text-red-400">{artifactCheck.error}</span>}
                </div>
              </Field>
            </div>
          </div>

          {/* Date splits */}
          <div className="space-y-2 pt-1">
            <p className={SUB_HEADING}>Date Splits</p>
            <div className="grid grid-cols-2 gap-2">
              {(['train', 'val', 'test'] as const).map((split) =>
                (['start', 'end'] as const).map((field) => (
                  <Field
                    key={`${split}-${field}`}
                    label={`${split.charAt(0).toUpperCase() + split.slice(1)} ${field}`}
                    tooltip="Format: YYYY-MM-DD HH:MM:SS"
                  >
                    <input
                      type="text"
                      value={splits[split][field]}
                      onChange={(e) => updateSplit(split, field, e.target.value)}
                      placeholder="YYYY-MM-DD HH:MM:SS"
                      className={INPUT_CLS}
                    />
                  </Field>
                ))
              )}
            </div>
          </div>

          {/* Model config */}
          <div className="space-y-2 pt-1">
            <p className={SUB_HEADING}>Model</p>
            <div className="grid grid-cols-2 gap-3">
              <Field label="Framework">
                <select
                  value={framework}
                  onChange={(e) => setFramework(e.target.value as Framework)}
                  className={SELECT_CLS}
                >
                  <option value="xgboost">xgboost</option>
                  <option value="pytorch">pytorch</option>
                </select>
              </Field>
              <Field label="Experiment Name" tooltip="MLflow experiment name.">
                <input
                  value={modelConfig.experiment_name}
                  onChange={(e) => setModelConfig((m) => ({ ...m, experiment_name: e.target.value }))}
                  className={INPUT_CLS}
                />
              </Field>
              <Field label="Registry Model Name" tooltip="MLflow Model Registry name.">
                <input
                  value={modelConfig.registry_model_name}
                  onChange={(e) => setModelConfig((m) => ({ ...m, registry_model_name: e.target.value }))}
                  className={INPUT_CLS}
                />
              </Field>
              <Field label="Target Column" tooltip="Name of the label column in the dataset.">
                <input
                  value={modelConfig.target}
                  onChange={(e) => setModelConfig((m) => ({ ...m, target: e.target.value }))}
                  className={INPUT_CLS}
                />
              </Field>
              <Field label="Num Classes" tooltip="Number of output classes (≥ 2).">
                <input
                  type="number"
                  min={2}
                  value={modelConfig.num_classes}
                  onChange={(e) => setModelConfig((m) => ({ ...m, num_classes: parseInt(e.target.value) || 2 }))}
                  className={INPUT_CLS}
                />
              </Field>
              <Field label="Seed" tooltip="Random seed for reproducibility.">
                <input
                  type="number"
                  value={modelConfig.seed}
                  onChange={(e) => setModelConfig((m) => ({ ...m, seed: parseInt(e.target.value) || 0 }))}
                  className={INPUT_CLS}
                />
              </Field>
            </div>

            {/* Tuning toggle */}
            <div className="flex items-center gap-3 pt-1">
              <span className="text-xs text-slate-400">Tuning</span>
              <div className="flex gap-1">
                <ToggleButton active={tuningEnabled} onClick={() => setTuningEnabled(true)}>
                  Enabled
                </ToggleButton>
                <ToggleButton active={!tuningEnabled} onClick={() => setTuningEnabled(false)}>
                  Disabled
                </ToggleButton>
              </div>
            </div>

            {tuningEnabled && (
              <div className="grid grid-cols-2 gap-3 pl-2 border-l border-slate-700">
                <Field
                  label="Sample Fraction"
                  tooltip="Fraction of data used for hyperparameter search (0.01–1.0)."
                >
                  <div className="flex items-center gap-2">
                    <input
                      type="range"
                      min={0.01}
                      max={1}
                      step={0.01}
                      value={sampleFraction}
                      onChange={(e) => setSampleFraction(parseFloat(e.target.value))}
                      className="flex-1 accent-blue-500"
                    />
                    <span className="w-10 text-xs text-slate-300 text-right">{sampleFraction.toFixed(2)}</span>
                  </div>
                </Field>
                <Field
                  label="Number of Trials"
                  tooltip="Total HPT trials for Ray Tune. Stored in execution.tuning.number_of_trials."
                >
                  <input
                    type="number"
                    min={1}
                    value={numberOfTrials}
                    onChange={(e) => setNumberOfTrials(parseInt(e.target.value) || 1)}
                    className={INPUT_CLS}
                  />
                </Field>
              </div>
            )}
          </div>
        </AccordionSection>

        {/* ── B. HYPERPARAMETERS ── */}
        <AccordionSection title="Hyperparameters" defaultOpen>
          {!tuningEnabled ? (
            /* tune=false: all *_PARAMS editable */
            <div className="space-y-2">
              <p className="text-xs text-slate-500">
                All <code className="bg-slate-800 px-1 rounded">{framework === 'xgboost' ? 'XGBOOST_PARAMS' : 'PYTORCH_PARAMS'}</code> keys.
                Only these keys are accepted by the backend.
              </p>
              <div className="grid grid-cols-2 gap-2">
                {Object.keys(getDefaults(framework)).map((key) => {
                  const meta: ParamMeta | undefined = paramMeta[key];
                  if (!meta) return null;
                  const m = meta;
                  return (
                    <Field key={key} label={key}>
                      <ParamInput
                        paramKey={key}
                        meta={m}
                        value={hyperparams[key] ?? m.defaultValue}
                        onChange={updateHyperparam}
                      />
                    </Field>
                  );
                })}
              </div>
            </div>
          ) : (
            /* tune=true: three sub-panels */
            <div className="space-y-4">

              {/* 1. Trial Budget (*_TUNE_SETTINGS) */}
              <div className="space-y-2">
                <p className={SUB_HEADING} title="How long each trial runs. Overrides *_PARAMS training-length for the tuning phase.">
                  Trial Execution Budget
                </p>
                <p className="text-xs text-slate-500">
                  Controls training steps per trial (e.g. epochs / num_boost_round). Not sampled by Ray Tune.
                </p>
                <div className="grid grid-cols-2 gap-2">
                  {Object.keys(tuneSettingsMeta).map((key) => {
                    const meta: ParamMeta | undefined = tuneSettingsMeta[key];
                    if (!meta) return null;
                    const m = meta;
                    return (
                      <Field key={key} label={key}>
                        <input
                          type="number"
                          min={m.min}
                          step={m.step ?? 1}
                          value={tuneSettings[key] ?? m.defaultValue}
                          onChange={(e) => updateTuneSetting(key, parseInt(e.target.value) || 1)}
                          className={INPUT_CLS}
                        />
                      </Field>
                    );
                  })}
                </div>
              </div>

              {/* 2. Search Space (SEARCH_SPACE_*) */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <p className={SUB_HEADING} title="Parameters Ray Tune will optimize across trials.">
                    Optimized Parameters (Ray Tune)
                  </p>
                  <div className="flex gap-1">
                    <ToggleButton active={tuneMode === 'predefined'} onClick={() => setTuneMode('predefined')}>
                      Predefined
                    </ToggleButton>
                    <ToggleButton active={tuneMode === 'override'} onClick={() => setTuneMode('override')}>
                      Override Ranges
                    </ToggleButton>
                  </div>
                </div>
                {tuneMode === 'override' && (
                  <p className="text-xs text-slate-500 italic">
                    UI preview only — backend reads search space from <code className="bg-slate-800 px-1 rounded">src/schemas/model/</code>
                  </p>
                )}
                <div className="rounded border border-slate-800 bg-slate-950 px-3 py-2 space-y-1">
                  {Object.entries(searchSpace).map(([key, entry]) => (
                    <SearchSpaceRow
                      key={key}
                      paramKey={key}
                      entry={entry}
                      override={searchSpaceOverrides[key]}
                      onOverrideChange={updateSearchSpaceOverride}
                      overrideMode={tuneMode === 'override'}
                    />
                  ))}
                </div>
              </div>

              {/* 3. Non-Tunable Constants (*_PARAMS keys not in SEARCH_SPACE_*) */}
              {nonTunableKeys.length > 0 && (
                <div className="space-y-2">
                  <p
                    className={SUB_HEADING}
                    title="These *_PARAMS values apply as constants to every trial (not sampled by Ray Tune)."
                  >
                    Non-Tunable Parameters (constant across all trials)
                  </p>
                  <div className="grid grid-cols-2 gap-2">
                    {nonTunableKeys.map((key) => {
                      const meta: ParamMeta | undefined = paramMeta[key];
                      if (!meta) return null;
                      const m = meta;
                      return (
                        <Field key={key} label={key} tooltip="Applied as a constant fallback in *_PARAMS for all trials.">
                          <ParamInput
                            paramKey={key}
                            meta={m}
                            value={hyperparams[key] ?? m.defaultValue}
                            onChange={updateHyperparam}
                          />
                        </Field>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          )}
        </AccordionSection>

        {/* ── C. DATASET SCHEMA BUILDER ── */}
        <AccordionSection
          title="Dataset Schema Builder"
          badge={
            schemaSaveResult ? (
              <span className="rounded-full bg-green-900/50 px-2 py-0.5 text-[10px] text-green-400">
                v{schemaSaveResult.version} saved
              </span>
            ) : undefined
          }
        >
          <div className="space-y-3">
            <p className="text-xs text-slate-500">
              Generate versioned schema YAMLs (raw / full / preprocessed) from the Iceberg table.
              Saved to <code className="bg-slate-800 px-1 rounded">s3://.../schemas/datasets/{'{name}'}/v{'{N}'}/</code>
            </p>

            <div className="flex items-center gap-2">
              <button
                onClick={() => void handleLoadSchema()}
                disabled={schemaLoading || !dataset}
                className={BTN_NEUTRAL}
              >
                {schemaLoading ? 'Loading…' : 'Load columns from Iceberg'}
              </button>
              {schemaBuilder.allColumns.length > 0 && (
                <span className="text-xs text-green-400">
                  ✓ {schemaBuilder.allColumns.length} columns loaded
                </span>
              )}
              {schemaError && <span className="text-xs text-red-400">{schemaError}</span>}
            </div>

            {schemaBuilder.allColumns.length > 0 && (
              <>
                {/* Column assignment */}
                <div className="space-y-1">
                  <p className={SUB_HEADING}>Column Assignment</p>
                  <div className="rounded border border-slate-800 divide-y divide-slate-800 overflow-hidden">
                    {/* Header row */}
                    <div className="grid grid-cols-[1fr_80px_80px_80px_80px] px-3 py-1.5 bg-slate-800/50 text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                      <span>Column</span>
                      <span className="text-center">Top-level</span>
                      <span className="text-center">Properties</span>
                      <span className="text-center">Full</span>
                      <span className="text-center">Target</span>
                    </div>
                    {schemaBuilder.allColumns.map((col) => (
                      <div
                        key={col.name}
                        className="grid grid-cols-[1fr_80px_80px_80px_80px] items-center px-3 py-1.5 hover:bg-slate-800/30"
                      >
                        <div>
                          <span className="font-mono text-xs text-slate-200">{col.name}</span>
                          <span className="ml-2 text-[10px] text-slate-500">{col.sparkType}</span>
                        </div>
                        <div className="flex justify-center">
                          <input
                            type="checkbox"
                            checked={schemaBuilder.rawTopLevel.includes(col.name)}
                            onChange={(e) => updateSchemaToggle(col.name, 'rawTopLevel', e.target.checked)}
                            className="accent-blue-500"
                          />
                        </div>
                        <div className="flex justify-center">
                          <input
                            type="checkbox"
                            checked={schemaBuilder.propertiesFields.includes(col.name)}
                            onChange={(e) => updateSchemaToggle(col.name, 'propertiesFields', e.target.checked)}
                            className="accent-blue-500"
                          />
                        </div>
                        <div className="flex justify-center">
                          <input
                            type="checkbox"
                            checked={schemaBuilder.fullColumns.includes(col.name)}
                            onChange={(e) => updateSchemaToggle(col.name, 'fullColumns', e.target.checked)}
                            className="accent-blue-500"
                          />
                        </div>
                        <div className="flex justify-center">
                          <input
                            type="radio"
                            name="targetColumn"
                            checked={schemaBuilder.targetColumn === col.name}
                            onChange={() => setSchemaBuilder((s) => ({ ...s, targetColumn: col.name }))}
                            className="accent-blue-500"
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Preprocessed features */}
                <div className="space-y-1">
                  <p className={SUB_HEADING}>Preprocessed Features (DSL output)</p>
                  <p className="text-xs text-slate-500">
                    Select the DSL output feature columns for preprocessed.yaml.
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {schemaBuilder.allColumns.map((col) => (
                      <label key={col.name} className="flex items-center gap-1 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={schemaBuilder.preprocessedColumns.includes(col.name)}
                          onChange={(e) => updateSchemaToggle(col.name, 'preprocessedColumns', e.target.checked)}
                          className="accent-blue-500"
                        />
                        <span className="font-mono text-xs text-slate-300">{col.name}</span>
                      </label>
                    ))}
                  </div>
                </div>

                {/* id + prediction columns */}
                <div className="grid grid-cols-2 gap-3">
                  <Field label="ID Column" tooltip="Column used to identify each record (e.g. event_id).">
                    <select
                      value={schemaBuilder.idColumn}
                      onChange={(e) => setSchemaBuilder((s) => ({ ...s, idColumn: e.target.value }))}
                      className={SELECT_CLS}
                    >
                      <option value="">— none —</option>
                      {schemaBuilder.allColumns.map((c) => (
                        <option key={c.name} value={c.name}>{c.name}</option>
                      ))}
                    </select>
                  </Field>
                  <Field label="Prediction Column" tooltip="Name of the output prediction column (e.g. label).">
                    <input
                      value={schemaBuilder.predictionColumn}
                      onChange={(e) => setSchemaBuilder((s) => ({ ...s, predictionColumn: e.target.value }))}
                      className={INPUT_CLS}
                    />
                  </Field>
                </div>

                {/* Schema validation errors */}
                {(() => {
                  const errs = validateSchemaBuilder(schemaBuilder);
                  return errs.length > 0 ? (
                    <div className="rounded border border-red-800/40 bg-red-900/20 px-3 py-2 space-y-0.5">
                      {errs.map((e, i) => (
                        <p key={i} className="text-xs text-red-400">{e}</p>
                      ))}
                    </div>
                  ) : null;
                })()}

                {/* Actions */}
                <div className="flex flex-wrap gap-2 pt-1">
                  <button onClick={() => downloadYaml(rawYamlPreview, 'raw.yaml')} className={BTN_NEUTRAL}>
                    ↓ raw.yaml
                  </button>
                  <button onClick={() => downloadYaml(fullYamlPreview, 'full.yaml')} className={BTN_NEUTRAL}>
                    ↓ full.yaml
                  </button>
                  <button onClick={() => downloadYaml(preprocessedYamlPreview, 'preprocessed.yaml')} className={BTN_NEUTRAL}>
                    ↓ preprocessed.yaml
                  </button>
                  <button
                    onClick={() => void handleSaveSchemas()}
                    disabled={schemaSaving || !dataset}
                    className={BTN_PRIMARY}
                  >
                    {schemaSaving ? 'Saving…' : 'Save schemas to S3'}
                  </button>
                </div>

                {schemaSaveResult && (
                  <div className="rounded border border-green-800/40 bg-green-900/10 px-3 py-2 text-xs text-green-400 space-y-1">
                    <p className="font-medium">Schemas saved — version {schemaSaveResult.version}</p>
                    {Object.entries(schemaSaveResult.uploaded).map(([k, v]) => (
                      <p key={k} className="text-slate-400"><span className="text-green-500">{k}:</span> {v}</p>
                    ))}
                  </div>
                )}
              </>
            )}
          </div>
        </AccordionSection>

        {/* ── D. ADVANCED CONFIGURATION ── */}
        <AccordionSection title="Advanced Configuration">
          <div className="space-y-4">

            {/* KubeRay / MLflow */}
            <div className="space-y-2">
              <p className={SUB_HEADING}>KubeRay / MLflow</p>
              <div className="grid grid-cols-2 gap-3">
                <Field label="MLflow Tracking URI">
                  <input value={advanced.mlflow_tracking_uri} onChange={(e) => setAdvanced((a) => ({ ...a, mlflow_tracking_uri: e.target.value }))} className={INPUT_CLS} />
                </Field>
                <Field label="MLflow Artifact Location">
                  <input value={advanced.mlflow_artifact_location} onChange={(e) => setAdvanced((a) => ({ ...a, mlflow_artifact_location: e.target.value }))} className={INPUT_CLS} />
                </Field>
                <Field label="Serving Alias" tooltip="MLflow Model Registry alias for the production model.">
                  <input value={advanced.serving_alias} onChange={(e) => setAdvanced((a) => ({ ...a, serving_alias: e.target.value }))} className={INPUT_CLS} />
                </Field>
                <Field label="Canary">
                  <div className="flex gap-1 pt-0.5">
                    <ToggleButton active={advanced.canary} onClick={() => setAdvanced((a) => ({ ...a, canary: true }))}>On</ToggleButton>
                    <ToggleButton active={!advanced.canary} onClick={() => setAdvanced((a) => ({ ...a, canary: false }))}>Off</ToggleButton>
                  </div>
                </Field>
                {advanced.canary && (
                  <>
                    <Field label="Canary Alias">
                      <input value={advanced.canary_alias} onChange={(e) => setAdvanced((a) => ({ ...a, canary_alias: e.target.value }))} className={INPUT_CLS} />
                    </Field>
                    <Field label="Canary Probability" tooltip="Fraction of traffic routed to canary (0–1).">
                      <input type="number" min={0} max={1} step={0.01} value={advanced.canary_probability} onChange={(e) => setAdvanced((a) => ({ ...a, canary_probability: parseFloat(e.target.value) }))} className={INPUT_CLS} />
                    </Field>
                  </>
                )}
                <Field label="Webhook Base URL">
                  <input value={advanced.webhook_public_base_url} onChange={(e) => setAdvanced((a) => ({ ...a, webhook_public_base_url: e.target.value }))} className={INPUT_CLS} />
                </Field>
                <Field label="Webhook Path">
                  <input value={advanced.webhook_path} onChange={(e) => setAdvanced((a) => ({ ...a, webhook_path: e.target.value }))} className={INPUT_CLS} />
                </Field>
                <Field label="Webhook Name">
                  <input value={advanced.webhook_name} onChange={(e) => setAdvanced((a) => ({ ...a, webhook_name: e.target.value }))} className={INPUT_CLS} />
                </Field>
                <Field label="Webhook Max Timestamp Age (s)">
                  <input type="number" min={0} value={advanced.webhook_max_timestamp_age_seconds} onChange={(e) => setAdvanced((a) => ({ ...a, webhook_max_timestamp_age_seconds: parseInt(e.target.value) || 0 }))} className={INPUT_CLS} />
                </Field>
              </div>
            </div>

            {/* Spark */}
            <div className="space-y-2">
              <p className={SUB_HEADING}>Spark</p>
              <div className="grid grid-cols-2 gap-3">
                <Field label="Read Batch Size" tooltip="Number of rows per Spark read batch.">
                  <input type="number" min={1} value={advanced.spark_read_batch_size} onChange={(e) => setAdvanced((a) => ({ ...a, spark_read_batch_size: parseInt(e.target.value) || 512 }))} className={INPUT_CLS} />
                </Field>
                <Field label="Write Batch Size">
                  <input type="number" min={1} value={advanced.spark_write_batch_size} onChange={(e) => setAdvanced((a) => ({ ...a, spark_write_batch_size: parseInt(e.target.value) || 100000 }))} className={INPUT_CLS} />
                </Field>
              </div>
            </div>

            {/* Iceberg */}
            <div className="space-y-2">
              <p className={SUB_HEADING}>Iceberg</p>
              <Field label="Warehouse Path" tooltip="S3 path to the Iceberg warehouse.">
                <input value={advanced.iceberg_warehouse} onChange={(e) => setAdvanced((a) => ({ ...a, iceberg_warehouse: e.target.value }))} className={INPUT_CLS} />
              </Field>
            </div>
          </div>
        </AccordionSection>

        {/* ── Submit ── */}
        <div className="space-y-3">
          {validationErrors.length > 0 && (
            <div className="rounded border border-red-800/40 bg-red-900/20 px-3 py-2 space-y-0.5">
              {validationErrors.map((e, i) => (
                <p key={i} className="text-xs text-red-400">{e}</p>
              ))}
            </div>
          )}
          <button
            onClick={() => void handleSubmit()}
            disabled={!!submitResult}
            className="w-full rounded bg-blue-600 py-2 text-sm font-semibold text-white hover:bg-blue-500 disabled:opacity-40 transition-colors"
          >
            Run Pipeline
          </button>
          {submitError && (
            <div className="rounded border border-red-800/40 bg-red-900/20 px-3 py-2">
              <p className="text-xs text-red-400">{submitError}</p>
            </div>
          )}

          {submitResult && (
            <div className="rounded-lg border border-slate-700 bg-slate-900 p-4 space-y-2">
              <p className="text-sm font-medium text-slate-300">Run Status</p>
              <div className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-xs">
                <span className="text-slate-400">DAG run ID</span>
                <span className="break-all text-slate-200">{submitResult.dag_run_id}</span>
                <span className="text-slate-400">Execution ID</span>
                <span className="text-slate-200">{submitResult.execution_id}</span>
                <span className="text-slate-400">params.yaml</span>
                <span className="break-all text-slate-200">{submitResult.params_s3_path}</span>
                <span className="text-slate-400">DSL</span>
                <span className="break-all text-slate-200">{submitResult.dsl_s3_path}</span>
                <span className="text-slate-400">State</span>
                <span className={`font-semibold ${stateColor(runStatus.toLowerCase())}`}>
                  {runStatus || 'queued'}
                </span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ── RIGHT: sticky YAML preview ── */}
      <div className="w-[440px] shrink-0 min-h-0 p-4">
        <YamlPreviewPanel
          tabs={previewTabs}
          activeTab={activePreviewTab}
          onTabChange={setActivePreviewTab}
        />
      </div>
    </div>
  );
}
