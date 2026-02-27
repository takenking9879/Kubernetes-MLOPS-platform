/**
 * RunPage — ML platform execution control surface.
 *
 * 5-step workflow stepper:
 *   1. Dataset   — dataset, DSL version, execution ID
 *   2. Splits    — temporal boundaries with locked constraints
 *   3. Model     — framework, config, hyperparams (tune=false)
 *   4. Tuning    — tuning toggle + trial budget / search space
 *   5. Review    — summary + validation + launch
 *
 * Right column: live YAML preview (params.yaml / raw / full / preprocessed).
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import { Check, ChevronDown, ChevronUp, Copy, RefreshCw } from 'lucide-react';
import {
  getIcebergSample,
  getPreprocessParams,
  getRunStatus,
  listDatasets,
  listProcessingRuns,
  submitRun,
  uploadSchemas,
  type DatasetInfo,
  type ProcessedTableEntry,
  type RunResult,
  type SchemaUploadResult,
} from '../api/platformClient';

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
  train: { start: '2026-01-01 00:00:00', end: '2026-01-05 00:00:00' },
  val:   { start: '2026-01-07 00:00:00', end: '2026-01-09 00:00:00' },
  test:  { start: '2026-01-11 00:00:00', end: '2026-01-13 00:00:00' },
};

const DEFAULT_MODEL_CONFIG = {
  experiment_name: 'kuberay-attack-detection',
  registry_model_name: 'attack-detection',
  target: 'attack',
  num_classes: 6,
  seed: 42,
};

const DATE_RE = /^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$/;
const STEP_LABELS = ['Dataset', 'Splits', 'Model', 'Tuning', 'Review'];

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

function formatDuration(start: string, end: string): string {
  if (!DATE_RE.test(start) || !DATE_RE.test(end)) return '—';
  const ms =
    Date.parse(end.replace(' ', 'T') + 'Z') -
    Date.parse(start.replace(' ', 'T') + 'Z');
  if (ms <= 0) return '⚠';
  const h = Math.floor(ms / 3_600_000);
  if (h >= 48) return `${Math.floor(h / 24)}d ${h % 24}h`;
  if (h > 0) {
    const m = Math.floor((ms % 3_600_000) / 60_000);
    return m > 0 ? `${h}h ${m}m` : `${h}h`;
  }
  return `${Math.floor((ms % 3_600_000) / 60_000)}m`;
}

// ─── Style tokens ─────────────────────────────────────────────────────────────

const INPUT_CLS =
  'rounded bg-slate-800 px-2 py-1 text-xs text-slate-100 outline-none focus:ring-1 focus:ring-blue-500 w-full';
const SELECT_CLS =
  'rounded bg-slate-800 px-2 py-1 text-xs text-slate-100 outline-none focus:ring-1 focus:ring-blue-500 w-full';
const BTN_NEUTRAL =
  'rounded bg-slate-700 px-2 py-1 text-xs text-slate-300 hover:bg-slate-600 disabled:opacity-40';
const BTN_PRIMARY =
  'rounded bg-blue-600 px-4 py-1.5 text-xs font-medium text-white hover:bg-blue-500 disabled:opacity-40';
const SUB_HEADING = 'text-xs font-semibold text-slate-400 uppercase tracking-wider';

// ─── Primitive UI components ───────────────────────────────────────────────────

function Label({ children, title }: { children: React.ReactNode; title?: string }) {
  return (
    <label className="text-xs text-slate-400" title={title}>
      {children}
    </label>
  );
}

function Field({
  label,
  tooltip,
  children,
}: {
  label: string;
  tooltip?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-1">
      <Label title={tooltip}>
        {label}
        {tooltip && <span className="ml-1 cursor-help text-slate-600">ⓘ</span>}
      </Label>
      {children}
    </div>
  );
}

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
    <span className="rounded bg-slate-700 px-1.5 py-0.5 font-mono text-[10px] text-slate-400">
      {children}
    </span>
  );
}

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
        <div className="space-y-3 border-t border-slate-700 px-4 pb-4 pt-3">
          {children}
        </div>
      )}
    </div>
  );
}

// ─── StepperHeader ────────────────────────────────────────────────────────────

function StepperHeader({
  currentStep,
  stepStatus,
  onStepClick,
}: {
  currentStep: number;
  stepStatus: (n: number) => 'active' | 'completed' | 'error' | 'pending';
  onStepClick: (n: number) => void;
}) {
  return (
    <div className="flex w-full select-none items-start gap-0">
      {STEP_LABELS.map((label, i) => {
        const n = i + 1;
        const st = stepStatus(n);
        const clickable = n <= currentStep;

        const nodeStyle =
          st === 'active'
            ? 'bg-blue-600 text-white ring-2 ring-blue-400/30'
            : st === 'completed'
              ? 'bg-green-700 text-white'
              : st === 'error'
                ? 'bg-red-700 text-white'
                : 'bg-slate-700 text-slate-500';

        const labelStyle =
          st === 'active'
            ? 'text-slate-100 font-semibold'
            : st === 'completed'
              ? 'text-slate-400'
              : 'text-slate-600';

        return (
          <div key={n} className="flex flex-1 items-start">
            <div
              className={`flex flex-col items-center ${clickable ? 'cursor-pointer' : 'cursor-default'}`}
              onClick={() => clickable && onStepClick(n)}
            >
              <div
                className={`flex h-7 w-7 items-center justify-center rounded-full text-xs font-bold transition-colors ${nodeStyle}`}
              >
                {st === 'completed' ? (
                  <Check size={13} strokeWidth={3} />
                ) : st === 'error' ? (
                  '!'
                ) : (
                  n
                )}
              </div>
              <span className={`mt-1 text-[10px] transition-colors ${labelStyle}`}>
                {label}
              </span>
            </div>
            {i < STEP_LABELS.length - 1 && (
              <div
                className={`mx-2 mt-3.5 h-px flex-1 transition-colors ${
                  st === 'completed' ? 'bg-green-700' : 'bg-slate-700'
                }`}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}

// ─── SplitRow ─────────────────────────────────────────────────────────────────

function SplitRow({
  label,
  start,
  end,
  onStartChange,
  onEndChange,
  duration,
}: {
  label: string;
  start: string;
  end: string;
  onStartChange: (v: string) => void;
  onEndChange: (v: string) => void;
  duration: string;
}) {
  return (
    <div className="grid grid-cols-[52px_1fr_18px_1fr_48px] items-center gap-2">
      <span className="text-xs font-medium text-slate-400">{label}</span>
      <input
        value={start}
        onChange={(e) => onStartChange(e.target.value)}
        placeholder="YYYY-MM-DD HH:MM:SS"
        className={INPUT_CLS}
      />
      <span className="text-center text-xs text-slate-600">→</span>
      <input
        value={end}
        onChange={(e) => onEndChange(e.target.value)}
        placeholder="YYYY-MM-DD HH:MM:SS"
        className={INPUT_CLS}
      />
      <span
        className={`text-right font-mono text-[10px] ${
          duration.startsWith('⚠') ? 'text-red-400' : 'text-slate-500'
        }`}
      >
        {duration}
      </span>
    </div>
  );
}

// ─── ParamInput ───────────────────────────────────────────────────────────────

function ParamInput({
  paramKey,
  meta,
  value,
  onChange,
  readOnly = false,
}: {
  paramKey: string;
  meta: {
    type: string;
    defaultValue: unknown;
    options?: string[];
    min?: number;
    max?: number;
    step?: number;
  };
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
          <option key={o} value={o}>
            {o}
          </option>
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

// ─── SearchSpaceRow ───────────────────────────────────────────────────────────

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
      const v = Array.isArray(entry.value)
        ? entry.value.join(', ')
        : String(entry.value);
      return <ReadOnlyBadge>fixed: {v}</ReadOnlyBadge>;
    }
    if (entry.type === 'choice') {
      return <ReadOnlyBadge>choice: [{entry.options.join(', ')}]</ReadOnlyBadge>;
    }
    // narrowed to 'randint' | 'uniform' | 'loguniform'
    if (overrideMode) {
      return (
        <div className="flex items-center gap-1">
          <input
            type="number"
            className="w-20 rounded bg-slate-800 px-1.5 py-0.5 text-xs text-slate-100 outline-none focus:ring-1 focus:ring-blue-500"
            value={override?.min ?? entry.min}
            step="any"
            onChange={(e) =>
              onOverrideChange?.(paramKey, 'min', parseFloat(e.target.value))
            }
          />
          <span className="text-xs text-slate-500">–</span>
          <input
            type="number"
            className="w-20 rounded bg-slate-800 px-1.5 py-0.5 text-xs text-slate-100 outline-none focus:ring-1 focus:ring-blue-500"
            value={override?.max ?? entry.max}
            step="any"
            onChange={(e) =>
              onOverrideChange?.(paramKey, 'max', parseFloat(e.target.value))
            }
          />
        </div>
      );
    }
    if (entry.type === 'randint')
      return (
        <ReadOnlyBadge>
          randint({entry.min}, {entry.max})
        </ReadOnlyBadge>
      );
    if (entry.type === 'uniform')
      return (
        <ReadOnlyBadge>
          uniform({entry.min}, {entry.max})
        </ReadOnlyBadge>
      );
    return (
      <ReadOnlyBadge>
        loguniform({entry.min.toExponential(0)}, {entry.max})
      </ReadOnlyBadge>
    );
  };

  return (
    <div className="grid grid-cols-[140px_1fr] items-center gap-2 py-1">
      <span className="font-mono text-xs text-slate-300">{paramKey}</span>
      {renderRange()}
    </div>
  );
}

// ─── SummaryCard ──────────────────────────────────────────────────────────────

function SummaryCard({
  title,
  items,
  onEdit,
}: {
  title: string;
  items: [string, string][];
  onEdit?: () => void;
}) {
  return (
    <div className="rounded-lg border border-slate-700 bg-slate-800/40 px-4 py-3">
      <div className="mb-2 flex items-center justify-between">
        <span className={SUB_HEADING}>{title}</span>
        {onEdit && (
          <button
            onClick={onEdit}
            className="text-[10px] text-blue-400 hover:text-blue-300"
          >
            Edit
          </button>
        )}
      </div>
      <div className="grid grid-cols-[120px_1fr] gap-x-3 gap-y-1">
        {items.map(([k, v]) => (
          <div key={k} className="contents">
            <span className="text-[11px] text-slate-500">{k}</span>
            <span className="break-all font-mono text-[11px] text-slate-200">
              {v || '—'}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── YamlPreviewPanel ─────────────────────────────────────────────────────────

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
    <div className="flex h-full flex-col overflow-hidden rounded-lg border border-slate-700 bg-slate-900">
      <div className="flex shrink-0 items-center justify-between border-b border-slate-700 px-3 py-2">
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
          className="flex items-center gap-1 rounded px-2 py-0.5 text-[10px] text-slate-400 hover:bg-slate-800 hover:text-slate-200"
          title="Copy to clipboard"
        >
          <Copy size={10} />
          {copied ? 'Copied!' : 'Copy'}
        </button>
      </div>
      <pre className="flex-1 overflow-auto whitespace-pre bg-slate-950 p-3 font-mono text-[11px] leading-relaxed text-green-300">
        {content}
      </pre>
    </div>
  );
}

// ─── RunPage ──────────────────────────────────────────────────────────────────

export function RunPage() {
  // ── Step navigation ───────────────────────────────────────────────────────
  const [currentStep, setCurrentStep] = useState(1);
  const [navErrors, setNavErrors] = useState<string[]>([]);

  // ── Execution ────────────────────────────────────────────────────────────
  const [dataset, setDataset] = useState('');
  // rawDatasetFilter: selector de dataset fuente — NO se guarda en params.yaml
  const [rawDatasetFilter, setRawDatasetFilter] = useState('');
  const [dslVersion, setDslVersion] = useState<number | ''>('');
  const [availableDatasets, setAvailableDatasets] = useState<DatasetInfo[]>([]);
  const [processingRuns, setProcessingRuns] = useState<ProcessedTableEntry[]>([]);
  const [selectedProcessedTable, setSelectedProcessedTable] = useState('');
  const [executionId, setExecutionId] = useState(nowId);

  // ── Params inspection modal ───────────────────────────────────────────────
  const [paramsModalOpen, setParamsModalOpen] = useState(false);
  const [paramsModalContent, setParamsModalContent] = useState('');
  const [paramsModalLoading, setParamsModalLoading] = useState(false);
  const [paramsModalError, setParamsModalError] = useState('');

  // ── Tuning ───────────────────────────────────────────────────────────────
  const [tuningEnabled, setTuningEnabled] = useState(true);
  const [numberOfTrials, setNumberOfTrials] = useState(3);
  const [sampleFraction, setSampleFraction] = useState(0.2);

  // ── Model ────────────────────────────────────────────────────────────────
  const [framework, setFramework] = useState<Framework>('xgboost');
  const [modelConfig, setModelConfig] = useState({ ...DEFAULT_MODEL_CONFIG });

  // ── Splits ───────────────────────────────────────────────────────────────
  const [splits, setSplits] = useState(DEFAULT_SPLITS);

  // ── Hyperparams (*_PARAMS) ────────────────────────────────────────────────
  const [hyperparams, setHyperparams] = useState<
    Record<string, number | string | string[]>
  >(() => ({ ...XGBOOST_DEFAULTS }));

  // ── Tune settings (*_TUNE_SETTINGS) ──────────────────────────────────────
  const [tuneSettings, setTuneSettings] = useState<Record<string, number>>(
    () => ({ ...XGBOOST_TUNE_SETTINGS_DEFAULTS }),
  );

  // ── Search space (UI override mode) ──────────────────────────────────────
  const [tuneMode, setTuneMode] = useState<TuneMode>('predefined');
  const [searchSpaceOverrides, setSearchSpaceOverrides] = useState<
    Record<string, { min: number; max: number }>
  >({});

  // ── Schema builder ────────────────────────────────────────────────────────
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
  const [schemaSaveResult, setSchemaSaveResult] =
    useState<SchemaUploadResult | null>(null);

  // ── Advanced ─────────────────────────────────────────────────────────────
  const [advanced, setAdvanced] = useState<AdvancedConfig>({
    ...DEFAULT_ADVANCED_CONFIG,
  });

  // ── Preview ───────────────────────────────────────────────────────────────
  const [activePreviewTab, setActivePreviewTab] =
    useState<PreviewTab>('params.yaml');

  // ── Submission ────────────────────────────────────────────────────────────
  const [submitResult, setSubmitResult] = useState<RunResult | null>(null);
  const [submitError, setSubmitError] = useState('');
  const [runStatus, setRunStatus] = useState('');
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Effects ───────────────────────────────────────────────────────────────

  // Load available datasets on mount (para el selector de Raw Dataset)
  useEffect(() => {
    listDatasets()
      .then((ds) => setAvailableDatasets(ds))
      .catch(() => setAvailableDatasets([]));
  }, []);

  // Load processing runs filtered by rawDatasetFilter
  useEffect(() => {
    if (!rawDatasetFilter) {
      setProcessingRuns([]);
      setSelectedProcessedTable('');
      setDslVersion('');
      return;
    }
    listProcessingRuns(rawDatasetFilter)
      .then((r) => setProcessingRuns(r.runs))
      .catch(() => setProcessingRuns([]));
  }, [rawDatasetFilter]);

  // Sync dslVersion when selectedProcessedTable changes
  useEffect(() => {
    if (selectedProcessedTable && processingRuns.length > 0) {
      const match = processingRuns.find(
        (r) => r.processed_table_name === selectedProcessedTable
      );
      if (match) {
        setDslVersion(match.dsl_version);
      }
    } else {
      setDslVersion('');
    }
  }, [selectedProcessedTable, processingRuns]);

  // dataset = rawDatasetFilter (ya no se deriva del entry)
  useEffect(() => {
    setDataset(rawDatasetFilter);
  }, [rawDatasetFilter]);

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
      } catch {
        /* keep polling */
      }
    }, 15_000);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [submitResult]);

  // ── useMemo — YAML previews ────────────────────────────────────────────────

  const paramsInput = useMemo<ParamsYamlInput>(
    () => ({
      execution_id: executionId,
      dataset,
      dslS3Path: '',
      tuning: { enabled: tuningEnabled, number_of_trials: numberOfTrials },
      splits,
      framework,
      model: modelConfig,
      sample_fraction_for_tuning: sampleFraction,
      hyperparams,
      tuneSettings: tuningEnabled ? tuneSettings : {},
      advanced,
    }),
    [
      executionId,
      dataset,
      tuningEnabled,
      numberOfTrials,
      splits,
      framework,
      modelConfig,
      sampleFraction,
      hyperparams,
      tuneSettings,
      advanced,
    ],
  );

  const paramsYamlPreview = useMemo(
    () => generateParamsYaml(paramsInput),
    [paramsInput],
  );
  const rawYamlPreview = useMemo(
    () =>
      schemaBuilder.allColumns.length
        ? generateRawYaml(schemaBuilder, dataset)
        : '# Load columns to preview schema',
    [schemaBuilder, dataset],
  );
  const fullYamlPreview = useMemo(
    () =>
      schemaBuilder.allColumns.length
        ? generateFullYaml(schemaBuilder, dataset)
        : '# Load columns to preview schema',
    [schemaBuilder, dataset],
  );
  const preprocessedYamlPreview = useMemo(
    () =>
      schemaBuilder.allColumns.length
        ? generatePreprocessedYaml(schemaBuilder, dataset)
        : '# Load columns to preview schema',
    [schemaBuilder, dataset],
  );

  // ── Step validation ────────────────────────────────────────────────────────

  function getStepErrors(step: number): string[] {
    switch (step) {
      case 1: {
        const e: string[] = [];
        if (!selectedProcessedTable) e.push('Select a processed table');
        return e;
      }
      case 2: {
        const e: string[] = [];
        const allFields: Array<[keyof typeof splits, 'start' | 'end']> = [
          ['train', 'start'], ['train', 'end'],
          ['val',   'start'], ['val',   'end'],
          ['test',  'start'], ['test',  'end'],
        ];
        for (const [s, f] of allFields) {
          if (!DATE_RE.test(splits[s][f]))
            e.push(`${s}.${f}: must be YYYY-MM-DD HH:MM:SS`);
        }
        if (e.length === 0) {
          const ts = (str: string) => Date.parse(str.replace(' ', 'T') + 'Z');
          const parsed = {
            train: [ts(splits.train.start), ts(splits.train.end)] as [number, number],
            val:   [ts(splits.val.start),   ts(splits.val.end)]   as [number, number],
            test:  [ts(splits.test.start),  ts(splits.test.end)]  as [number, number],
          };
          // Positive duration per split
          const labels: Record<string, string> = { train: 'Train', val: 'Validation', test: 'Test' };
          for (const [k, [s, en]] of Object.entries(parsed)) {
            if (s >= en) e.push(`${labels[k]} start must be before ${labels[k]} end`);
          }
          // No pairwise overlap
          const splitsOverlap = (
            [as_, ae]: [number, number],
            [bs, be]: [number, number],
          ) => as_ < be && bs < ae;
          const pairs: Array<[keyof typeof parsed, keyof typeof parsed, string]> = [
            ['train', 'val',  '"Train" and "Validation"'],
            ['train', 'test', '"Train" and "Test"'],
            ['val',   'test', '"Validation" and "Test"'],
          ];
          for (const [a, b, label] of pairs) {
            if (splitsOverlap(parsed[a], parsed[b]))
              e.push(`${label} ranges overlap`);
          }
        }
        return e;
      }
      case 3: {
        const e: string[] = [];
        if (!modelConfig.target.trim()) e.push('Target column is required');
        if (modelConfig.num_classes < 2)
          e.push('num_classes must be at least 2');
        return e;
      }
      case 4: {
        const e: string[] = [];
        if (tuningEnabled) {
          if (numberOfTrials < 1)
            e.push('number_of_trials must be at least 1');
          if (sampleFraction < 0.01 || sampleFraction > 1.0)
            e.push('sample_fraction must be 0.01–1.0');
        }
        return e;
      }
      default:
        return [];
    }
  }

  function validateForm(): string[] {
    const errors: string[] = [];
    for (let s = 1; s <= 4; s++) errors.push(...getStepErrors(s));
    if (!tuningEnabled) {
      const allowed = getAllowedKeys(framework);
      for (const k of Object.keys(hyperparams)) {
        if (!allowed.has(k)) errors.push(`Unknown hyperparameter key: "${k}"`);
      }
    }
    errors.push(...validateSchemaBuilder(schemaBuilder));
    return errors;
  }

  // ── Step navigation ────────────────────────────────────────────────────────

  const stepStatus = (
    n: number,
  ): 'active' | 'completed' | 'error' | 'pending' => {
    if (n === currentStep) return 'active';
    if (n > currentStep) return 'pending';
    return getStepErrors(n).length === 0 ? 'completed' : 'error';
  };

  const goForward = () => {
    const errs = getStepErrors(currentStep);
    if (errs.length > 0) {
      setNavErrors(errs);
      return;
    }
    setNavErrors([]);
    setCurrentStep((s) => Math.min(s + 1, 5));
  };

  const goBack = () => {
    setNavErrors([]);
    setCurrentStep((s) => Math.max(s - 1, 1));
  };

  const goToStep = (n: number) => {
    if (n <= currentStep) {
      setNavErrors([]);
      setCurrentStep(n);
    }
  };

  // ── Handlers ──────────────────────────────────────────────────────────────


  const handleInspectParams = async () => {
    const entry = processingRuns.find((r) => r.processed_table_name === selectedProcessedTable);
    if (!entry) return;
    setParamsModalContent('');
    setParamsModalError('');
    setParamsModalLoading(true);
    setParamsModalOpen(true);
    try {
      const result = await getPreprocessParams(entry.execution_id);
      setParamsModalContent(result.yaml_content);
    } catch (e) {
      setParamsModalError(e instanceof Error ? e.message : String(e));
    } finally {
      setParamsModalLoading(false);
    }
  };

  const handleSubmit = async () => {
    setSubmitError('');
    setSubmitResult(null);
    setRunStatus('');
    const errors = validateForm();
    if (errors.length > 0) return;
    try {
      const requestBody = {
        processed_table: selectedProcessedTable,
        execution_id: executionId.trim(),
        framework,
        splits,
        tuning: { enabled: tuningEnabled, number_of_trials: numberOfTrials },
        model: modelConfig,
        sample_fraction_for_tuning: sampleFraction,
        hyperparams: tuningEnabled ? {} : hyperparams,
        tune_settings: tuningEnabled ? tuneSettings : {},
      };
      const result = await submitRun(requestBody);
      setSubmitResult(result);
      setRunStatus('queued');
    } catch (e) {
      setSubmitError(e instanceof Error ? e.message : String(e));
    }
  };

  const handleLoadSchema = async () => {
    if (!dataset) {
      setSchemaError('Select a dataset first');
      return;
    }
    setSchemaLoading(true);
    setSchemaError('');
    try {
      const r = await getIcebergSample(dataset, 10);
      const cols = r.columns as SchemaColumn[];
      setSchemaBuilder({
        allColumns: cols,
        rawTopLevel: cols
          .filter((c) => !c.name.includes('.'))
          .slice(0, 3)
          .map((c) => c.name),
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

  const updateSearchSpaceOverride = (
    key: string,
    field: 'min' | 'max',
    val: number,
  ) => {
    setSearchSpaceOverrides((prev) => ({
      ...prev,
      [key]: { ...(prev[key] ?? { min: 0, max: 1 }), [field]: val },
    }));
  };

  const updateSplit = (
    split: 'train' | 'val' | 'test',
    field: 'start' | 'end',
    val: string,
  ) => {
    setSplits((prev) => ({
      ...prev,
      [split]: { ...prev[split], [field]: val },
    }));
  };

  const updateSchemaToggle = (
    colName: string,
    field:
      | 'rawTopLevel'
      | 'propertiesFields'
      | 'fullColumns'
      | 'preprocessedColumns',
    checked: boolean,
  ) => {
    setSchemaBuilder((prev) => {
      const arr = prev[field];
      return {
        ...prev,
        [field]: checked
          ? [...arr, colName]
          : arr.filter((n) => n !== colName),
      };
    });
  };

  const stateColor = (state: string) =>
    state === 'success'
      ? 'text-green-400'
      : state === 'running'
        ? 'text-yellow-400'
        : state.includes('fail')
          ? 'text-red-400'
          : 'text-slate-400';

  // ── Derived values ────────────────────────────────────────────────────────

  const searchSpace = getSearchSpace(framework);
  const paramMeta = getParamMeta(framework);
  const tuneSettingsMeta = getTuneSettingsMeta(framework);
  const nonTunableKeys = getNonTunableKeys(framework);

  const maxTKey =
    framework === 'xgboost' ? 'num_boost_round' : 'max_epochs';
  const roundsPerTrial =
    (tuneSettings[maxTKey] as number | undefined) ?? 10;
  const totalSteps = numberOfTrials * roundsPerTrial;

  const fullValidationErrors = validateForm();

  const previewTabs = [
    {
      key: 'params.yaml' as PreviewTab,
      label: 'params.yaml',
      content: paramsYamlPreview,
    },
    {
      key: 'raw.yaml' as PreviewTab,
      label: 'raw.yaml',
      content: rawYamlPreview,
    },
    {
      key: 'full.yaml' as PreviewTab,
      label: 'full.yaml',
      content: fullYamlPreview,
    },
    {
      key: 'preprocessed.yaml' as PreviewTab,
      label: 'preprocessed.yaml',
      content: preprocessedYamlPreview,
    },
  ];

  // ─────────────────────────────────────────────────────────────────────────────
  // RENDER
  // ─────────────────────────────────────────────────────────────────────────────

  return (
    <div className="flex h-full min-h-0 overflow-hidden">
      {/* ── LEFT: scrollable form ── */}
      <div className="min-w-0 flex-1 space-y-3 overflow-y-auto p-4">

        {/* Stepper header */}
        <StepperHeader
          currentStep={currentStep}
          stepStatus={stepStatus}
          onStepClick={goToStep}
        />

        {/* Step content card */}
        <div className="rounded-lg border border-slate-700 bg-slate-900 px-4 py-4">

          {/* ── STEP 1: Processed Table & Execution ── */}
          {currentStep === 1 && (
            <div className="space-y-4">
              <p className={SUB_HEADING}>Processed Table & Execution</p>

              <div className="space-y-3">
                <p className="text-xs text-slate-500">
                  Elige el dataset fuente para filtrar las tablas procesadas,
                  luego selecciona la tabla sobre la que entrenar.
                </p>

                {/* Raw Dataset filter — solo filtra, NO se guarda en params_training.yaml */}
                <Field label="Raw Dataset" tooltip="Filtra las tablas procesadas por dataset fuente. No se incluye en params_training.yaml.">
                  {availableDatasets.length > 0 ? (
                    <select
                      value={rawDatasetFilter}
                      onChange={(e) => {
                        setRawDatasetFilter(e.target.value);
                        setSelectedProcessedTable('');
                      }}
                      className={SELECT_CLS}
                    >
                      <option value="">— select dataset —</option>
                      {availableDatasets.map((d) => (
                        <option key={d.name} value={d.name}>{d.name}</option>
                      ))}
                    </select>
                  ) : (
                    <input
                      value={rawDatasetFilter}
                      onChange={(e) => {
                        setRawDatasetFilter(e.target.value);
                        setSelectedProcessedTable('');
                      }}
                      placeholder="network_traffic"
                      className={INPUT_CLS}
                    />
                  )}
                </Field>

                {/* Processed Table — solo nombre, con botón de inspección */}
                <Field label="Processed Table" tooltip="Tabla Iceberg generada por un run de preprocesamiento.">
                  <div className="flex gap-1">
                    <select
                      value={selectedProcessedTable}
                      onChange={(e) => setSelectedProcessedTable(e.target.value)}
                      className={SELECT_CLS}
                      disabled={!rawDatasetFilter || processingRuns.length === 0}
                    >
                      <option value="">
                        {!rawDatasetFilter
                          ? '— select a dataset first —'
                          : processingRuns.length === 0
                            ? '— no processed tables found —'
                            : '— select processed table —'}
                      </option>
                      {processingRuns.map((r) => (
                        <option key={r.execution_id} value={r.processed_table_name}>
                          {r.processed_table_name}
                        </option>
                      ))}
                    </select>
                    <button
                      onClick={() => void handleInspectParams()}
                      disabled={!selectedProcessedTable}
                      className={BTN_NEUTRAL}
                      title="Inspeccionar params_preprocess.yaml de esta tabla"
                    >
                      🔍
                    </button>
                  </div>
                </Field>

                <Field
                  label="Execution ID"
                  tooltip="Unique run identifier. Auto-generated, editable."
                >
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
              </div>
            </div>
          )}

          {/* ── STEP 2: Splits ── */}
          {currentStep === 2 && (
            <div className="space-y-4">
              <div>
                <p className={SUB_HEADING}>Temporal Split Boundaries</p>
                <p className="mt-1 text-xs text-slate-500">
                  Each split is independent — ranges can have gaps or be in any
                  order.{' '}
                  <span className="font-mono text-slate-400">
                    YYYY-MM-DD HH:MM:SS
                  </span>
                  . Overlaps between splits are not allowed.
                </p>
              </div>

              <div className="space-y-2.5">
                <SplitRow
                  label="Train"
                  start={splits.train.start}
                  end={splits.train.end}
                  onStartChange={(v) => updateSplit('train', 'start', v)}
                  onEndChange={(v) => updateSplit('train', 'end', v)}
                  duration={formatDuration(
                    splits.train.start,
                    splits.train.end,
                  )}
                />
                <SplitRow
                  label="Val"
                  start={splits.val.start}
                  end={splits.val.end}
                  onStartChange={(v) => updateSplit('val', 'start', v)}
                  onEndChange={(v) => updateSplit('val', 'end', v)}
                  duration={formatDuration(splits.val.start, splits.val.end)}
                />
                <SplitRow
                  label="Test"
                  start={splits.test.start}
                  end={splits.test.end}
                  onStartChange={(v) => updateSplit('test', 'start', v)}
                  onEndChange={(v) => updateSplit('test', 'end', v)}
                  duration={formatDuration(
                    splits.test.start,
                    splits.test.end,
                  )}
                />
              </div>

              {/* Duration summary */}
              {[splits.train, splits.val, splits.test].every(
                (s) => DATE_RE.test(s.start) && DATE_RE.test(s.end),
              ) && (
                <div className="flex items-center gap-2 pt-1 text-[10px] text-slate-500">
                  <span>Duration:</span>
                  <span>
                    train{' '}
                    <span className="text-slate-300">
                      {formatDuration(splits.train.start, splits.train.end)}
                    </span>
                  </span>
                  <span>·</span>
                  <span>
                    val{' '}
                    <span className="text-slate-300">
                      {formatDuration(splits.val.start, splits.val.end)}
                    </span>
                  </span>
                  <span>·</span>
                  <span>
                    test{' '}
                    <span className="text-slate-300">
                      {formatDuration(splits.test.start, splits.test.end)}
                    </span>
                  </span>
                </div>
              )}
            </div>
          )}

          {/* ── STEP 3: Model ── */}
          {currentStep === 3 && (
            <div className="space-y-4">
              {/* Framework */}
              <div className="space-y-2">
                <p className={SUB_HEADING}>Framework</p>
                <div className="flex gap-2">
                  <ToggleButton
                    active={framework === 'xgboost'}
                    onClick={() => setFramework('xgboost')}
                  >
                    XGBoost
                  </ToggleButton>
                  <ToggleButton
                    active={framework === 'pytorch'}
                    onClick={() => setFramework('pytorch')}
                  >
                    PyTorch
                  </ToggleButton>
                </div>
              </div>

              {/* Experiment */}
              <div className="space-y-2">
                <p className={SUB_HEADING}>Experiment</p>
                <div className="grid grid-cols-2 gap-3">
                  <Field label="Experiment Name">
                    <input
                      value={modelConfig.experiment_name}
                      onChange={(e) =>
                        setModelConfig((m) => ({
                          ...m,
                          experiment_name: e.target.value,
                        }))
                      }
                      className={INPUT_CLS}
                    />
                  </Field>
                  <Field label="Registry Model Name">
                    <input
                      value={modelConfig.registry_model_name}
                      onChange={(e) =>
                        setModelConfig((m) => ({
                          ...m,
                          registry_model_name: e.target.value,
                        }))
                      }
                      className={INPUT_CLS}
                    />
                  </Field>
                </div>
              </div>

              {/* Target */}
              <div className="space-y-2">
                <p className={SUB_HEADING}>Target</p>
                <div className="grid grid-cols-3 gap-3">
                  <Field label="Target Column">
                    <input
                      value={modelConfig.target}
                      onChange={(e) =>
                        setModelConfig((m) => ({
                          ...m,
                          target: e.target.value,
                        }))
                      }
                      className={INPUT_CLS}
                    />
                  </Field>
                  <Field
                    label="Num Classes"
                    tooltip="Number of output classes (≥ 2)."
                  >
                    <input
                      type="number"
                      min={2}
                      value={modelConfig.num_classes}
                      onChange={(e) =>
                        setModelConfig((m) => ({
                          ...m,
                          num_classes: parseInt(e.target.value) || 2,
                        }))
                      }
                      className={INPUT_CLS}
                    />
                  </Field>
                  <Field label="Seed">
                    <input
                      type="number"
                      value={modelConfig.seed}
                      onChange={(e) =>
                        setModelConfig((m) => ({
                          ...m,
                          seed: parseInt(e.target.value) || 0,
                        }))
                      }
                      className={INPUT_CLS}
                    />
                  </Field>
                </div>
              </div>

              {/* Hyperparams — only when tune=false */}
              {!tuningEnabled ? (
                <div className="space-y-2">
                  <p className={SUB_HEADING}>
                    {framework === 'xgboost' ? 'XGBoost' : 'PyTorch'}{' '}
                    Parameters
                  </p>
                  <p className="text-xs text-slate-500">
                    Fixed parameters for training (tuning is disabled).
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
                <div className="rounded border border-slate-700 bg-slate-800/30 px-3 py-2.5">
                  <p className="text-xs text-slate-400">
                    Tuning is enabled — hyperparameter ranges are configured in
                    the{' '}
                    <span className="text-blue-400">Tuning</span> step.
                    Non-tunable parameters will use defaults.
                  </p>
                </div>
              )}
            </div>
          )}

          {/* ── STEP 4: Tuning ── */}
          {currentStep === 4 && (
            <div className="space-y-4">
              {/* Toggle */}
              <div className="flex items-center justify-between">
                <p className={SUB_HEADING}>Hyperparameter Tuning</p>
                <div className="flex gap-1">
                  <ToggleButton
                    active={tuningEnabled}
                    onClick={() => setTuningEnabled(true)}
                  >
                    Enabled
                  </ToggleButton>
                  <ToggleButton
                    active={!tuningEnabled}
                    onClick={() => setTuningEnabled(false)}
                  >
                    Disabled
                  </ToggleButton>
                </div>
              </div>

              {!tuningEnabled ? (
                /* Disabled — muted parameter summary */
                <div className="space-y-2 rounded border border-slate-700 bg-slate-800/30 px-4 py-3">
                  <p className="text-xs text-slate-400">
                    Training will run with fixed parameters for the full number
                    of rounds/epochs.
                  </p>
                  <p className="text-xs text-slate-500">
                    Active parameters (from Model step):
                  </p>
                  <div className="flex flex-wrap gap-1.5">
                    {Object.entries(hyperparams)
                      .slice(0, 9)
                      .map(([k, v]) => (
                        <span
                          key={k}
                          className="rounded bg-slate-700 px-1.5 py-0.5 font-mono text-[10px] text-slate-400"
                        >
                          {k}={Array.isArray(v) ? v.join(',') : String(v)}
                        </span>
                      ))}
                  </div>
                </div>
              ) : (
                /* Enabled — three sub-panels */
                <div className="space-y-4">
                  {/* Budget controls */}
                  <div className="grid grid-cols-2 gap-3">
                    <Field
                      label="Number of Trials"
                      tooltip="Total Ray Tune trials. Stored in execution.tuning.number_of_trials."
                    >
                      <input
                        type="number"
                        min={1}
                        value={numberOfTrials}
                        onChange={(e) =>
                          setNumberOfTrials(parseInt(e.target.value) || 1)
                        }
                        className={INPUT_CLS}
                      />
                    </Field>
                    <Field
                      label="Sample Fraction"
                      tooltip="Fraction of training data used per trial (0.01–1.0)."
                    >
                      <div className="flex items-center gap-2">
                        <input
                          type="range"
                          min={0.01}
                          max={1}
                          step={0.01}
                          value={sampleFraction}
                          onChange={(e) =>
                            setSampleFraction(parseFloat(e.target.value))
                          }
                          className="flex-1 accent-blue-500"
                        />
                        <span className="w-10 text-right text-xs text-slate-300">
                          {sampleFraction.toFixed(2)}
                        </span>
                      </div>
                    </Field>
                  </div>

                  {/* Trial Execution Budget */}
                  <div className="space-y-2">
                    <p
                      className={SUB_HEADING}
                      title="How long each trial runs. Overrides *_PARAMS training-length for the tuning phase."
                    >
                      Trial Execution Budget
                    </p>
                    <p className="text-xs text-slate-500">
                      Training steps per trial — not sampled by Ray Tune.
                    </p>
                    <div className="grid grid-cols-2 gap-2">
                      {Object.keys(tuneSettingsMeta).map((key) => {
                        const meta: ParamMeta | undefined =
                          tuneSettingsMeta[key];
                        if (!meta) return null;
                        const m = meta;
                        return (
                          <Field key={key} label={key}>
                            <input
                              type="number"
                              min={m.min}
                              step={m.step ?? 1}
                              value={tuneSettings[key] ?? m.defaultValue}
                              onChange={(e) =>
                                updateTuneSetting(
                                  key,
                                  parseInt(e.target.value) || 1,
                                )
                              }
                              className={INPUT_CLS}
                            />
                          </Field>
                        );
                      })}
                    </div>
                  </div>

                  {/* Search Space */}
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <p
                        className={SUB_HEADING}
                        title="Parameters Ray Tune optimizes across trials."
                      >
                        Optimized Parameters (Ray Tune)
                      </p>
                      <div className="flex gap-1">
                        <ToggleButton
                          active={tuneMode === 'predefined'}
                          onClick={() => setTuneMode('predefined')}
                        >
                          Predefined
                        </ToggleButton>
                        <ToggleButton
                          active={tuneMode === 'override'}
                          onClick={() => setTuneMode('override')}
                        >
                          Override
                        </ToggleButton>
                      </div>
                    </div>
                    {tuneMode === 'override' && (
                      <p className="text-xs italic text-slate-500">
                        UI preview only — backend reads from{' '}
                        <code className="rounded bg-slate-800 px-1">
                          src/schemas/model/
                        </code>
                      </p>
                    )}
                    <div className="space-y-0.5 rounded border border-slate-800 bg-slate-950 px-3 py-2">
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

                  {/* Non-Tunable Constants */}
                  {nonTunableKeys.length > 0 && (
                    <div className="space-y-2">
                      <p
                        className={SUB_HEADING}
                        title="These *_PARAMS values apply as constants to every trial."
                      >
                        Non-Tunable Constants
                      </p>
                      <div className="grid grid-cols-2 gap-2">
                        {nonTunableKeys.map((key) => {
                          const meta: ParamMeta | undefined = paramMeta[key];
                          if (!meta) return null;
                          const m = meta;
                          return (
                            <Field
                              key={key}
                              label={key}
                              tooltip="Constant fallback across all trials."
                            >
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

                  {/* Estimate */}
                  <div className="rounded border border-slate-700 bg-slate-800/30 px-3 py-2.5 text-xs text-slate-400">
                    <span className="text-slate-200">{numberOfTrials}</span>{' '}
                    trials ×{' '}
                    <span className="text-slate-200">{roundsPerTrial}</span>{' '}
                    rounds ={' '}
                    <span className="text-slate-200">~{totalSteps}</span> total
                    training steps
                    <span className="ml-1 text-slate-600">
                      (ASHA may terminate trials early)
                    </span>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ── STEP 5: Review & Run ── */}
          {currentStep === 5 && (
            <div className="space-y-3">
              <p className={SUB_HEADING}>Review Configuration</p>

              <SummaryCard
                title="Dataset"
                onEdit={() => goToStep(1)}
                items={[
                  ['Dataset', dataset],
                  [
                    'DSL Version',
                    dslVersion !== '' ? `v${dslVersion}` : '—',
                  ],
                  ['Execution ID', executionId],
                ]}
              />

              <SummaryCard
                title="Splits"
                onEdit={() => goToStep(2)}
                items={[
                  [
                    'Train',
                    `${splits.train.start}  →  ${splits.train.end}  (${formatDuration(splits.train.start, splits.train.end)})`,
                  ],
                  [
                    'Val',
                    `${splits.val.start}  →  ${splits.val.end}  (${formatDuration(splits.val.start, splits.val.end)})`,
                  ],
                  [
                    'Test',
                    `${splits.test.start}  →  ${splits.test.end}  (${formatDuration(splits.test.start, splits.test.end)})`,
                  ],
                ]}
              />

              <SummaryCard
                title="Model"
                onEdit={() => goToStep(3)}
                items={[
                  ['Framework', framework],
                  [
                    'Target',
                    `${modelConfig.target} (${modelConfig.num_classes} classes)`,
                  ],
                  ['Experiment', modelConfig.experiment_name],
                  ['Registry', modelConfig.registry_model_name],
                ]}
              />

              <SummaryCard
                title="Tuning"
                onEdit={() => goToStep(4)}
                items={
                  tuningEnabled
                    ? [
                        ['Enabled', 'Yes'],
                        ['Trials', String(numberOfTrials)],
                        ['Rounds/trial', String(roundsPerTrial)],
                        ['Sample fraction', String(sampleFraction)],
                      ]
                    : [['Enabled', 'No — fixed parameters']]
                }
              />

              {/* Validation result */}
              {fullValidationErrors.length > 0 ? (
                <div className="space-y-1 rounded border border-red-800/40 bg-red-900/20 px-3 py-2">
                  <p className="text-xs font-semibold text-red-400">
                    Issues to resolve before launching:
                  </p>
                  {fullValidationErrors.map((e, i) => (
                    <p key={i} className="text-xs text-red-400">
                      • {e}
                    </p>
                  ))}
                </div>
              ) : (
                <div className="rounded border border-green-800/30 bg-green-900/10 px-3 py-2">
                  <p className="text-xs text-green-400">
                    ✓ All configuration valid
                  </p>
                </div>
              )}

              {/* Launch / Result */}
              {submitResult ? (
                <div className="space-y-2 rounded-lg border border-slate-700 bg-slate-800/40 p-4">
                  <p className="text-sm font-medium text-green-400">
                    ✓ Pipeline launched
                  </p>
                  <div className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-xs">
                    <span className="text-slate-400">DAG run ID</span>
                    <span className="break-all text-slate-200">
                      {submitResult.dag_run_id}
                    </span>
                    <span className="text-slate-400">Execution ID</span>
                    <span className="text-slate-200">
                      {submitResult.execution_id}
                    </span>
                    <span className="text-slate-400">params.yaml</span>
                    <span className="break-all text-slate-200">
                      {submitResult.params_s3_path}
                    </span>
                    <span className="text-slate-400">DSL</span>
                    <span className="break-all text-slate-200">
                      {submitResult.dsl_s3_path}
                    </span>
                    <span className="text-slate-400">State</span>
                    <span
                      className={`font-semibold ${stateColor(runStatus.toLowerCase())}`}
                    >
                      {runStatus || 'queued'}
                    </span>
                  </div>
                  <button
                    onClick={() => {
                      setSubmitResult(null);
                      setCurrentStep(1);
                      setNavErrors([]);
                    }}
                    className={BTN_NEUTRAL}
                  >
                    New Run
                  </button>
                </div>
              ) : (
                <>
                  {submitError && (
                    <div className="rounded border border-red-800/40 bg-red-900/20 px-3 py-2">
                      <p className="text-xs text-red-400">{submitError}</p>
                    </div>
                  )}
                  <button
                    onClick={() => void handleSubmit()}
                    disabled={fullValidationErrors.length > 0}
                    className="w-full rounded bg-blue-600 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-blue-500 disabled:opacity-40"
                  >
                    Launch Pipeline
                  </button>
                </>
              )}
            </div>
          )}
        </div>

        {/* Navigation errors */}
        {navErrors.length > 0 && (
          <div className="space-y-0.5 rounded border border-red-800/40 bg-red-900/20 px-3 py-2">
            {navErrors.map((e, i) => (
              <p key={i} className="text-xs text-red-400">
                • {e}
              </p>
            ))}
          </div>
        )}

        {/* Navigation buttons */}
        {currentStep < 5 ? (
          <div className="flex items-center justify-between">
            {currentStep > 1 ? (
              <button onClick={goBack} className={BTN_NEUTRAL}>
                ← Back
              </button>
            ) : (
              <div />
            )}
            <button onClick={goForward} className={BTN_PRIMARY}>
              Continue →
            </button>
          </div>
        ) : (
          <div>
            <button onClick={goBack} className={BTN_NEUTRAL}>
              ← Edit Configuration
            </button>
          </div>
        )}

        {/* ── Dataset Schema Builder ── */}
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
              Generate versioned schema YAMLs from the Iceberg table. Saved to{' '}
              <code className="rounded bg-slate-800 px-1">
                s3://.../schemas/datasets/{'{name}'}/v{'{N}'}/
              </code>
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
              {schemaError && (
                <span className="text-xs text-red-400">{schemaError}</span>
              )}
            </div>

            {schemaBuilder.allColumns.length > 0 && (
              <>
                {/* Column assignment */}
                <div className="space-y-1">
                  <p className={SUB_HEADING}>Column Assignment</p>
                  <div className="divide-y divide-slate-800 overflow-hidden rounded border border-slate-800">
                    <div className="grid grid-cols-[1fr_80px_80px_80px_80px] bg-slate-800/50 px-3 py-1.5 text-[10px] font-bold uppercase tracking-wider text-slate-400">
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
                          <span className="font-mono text-xs text-slate-200">
                            {col.name}
                          </span>
                          <span className="ml-2 text-[10px] text-slate-500">
                            {col.sparkType}
                          </span>
                        </div>
                        {(
                          [
                            'rawTopLevel',
                            'propertiesFields',
                            'fullColumns',
                          ] as const
                        ).map((field) => (
                          <div key={field} className="flex justify-center">
                            <input
                              type="checkbox"
                              checked={schemaBuilder[field].includes(col.name)}
                              onChange={(e) =>
                                updateSchemaToggle(
                                  col.name,
                                  field,
                                  e.target.checked,
                                )
                              }
                              className="accent-blue-500"
                            />
                          </div>
                        ))}
                        <div className="flex justify-center">
                          <input
                            type="radio"
                            name="targetColumn"
                            checked={schemaBuilder.targetColumn === col.name}
                            onChange={() =>
                              setSchemaBuilder((s) => ({
                                ...s,
                                targetColumn: col.name,
                              }))
                            }
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
                  <div className="flex flex-wrap gap-2">
                    {schemaBuilder.allColumns.map((col) => (
                      <label
                        key={col.name}
                        className="flex cursor-pointer items-center gap-1"
                      >
                        <input
                          type="checkbox"
                          checked={schemaBuilder.preprocessedColumns.includes(
                            col.name,
                          )}
                          onChange={(e) =>
                            updateSchemaToggle(
                              col.name,
                              'preprocessedColumns',
                              e.target.checked,
                            )
                          }
                          className="accent-blue-500"
                        />
                        <span className="font-mono text-xs text-slate-300">
                          {col.name}
                        </span>
                      </label>
                    ))}
                  </div>
                </div>

                {/* ID + prediction columns */}
                <div className="grid grid-cols-2 gap-3">
                  <Field label="ID Column">
                    <select
                      value={schemaBuilder.idColumn}
                      onChange={(e) =>
                        setSchemaBuilder((s) => ({
                          ...s,
                          idColumn: e.target.value,
                        }))
                      }
                      className={SELECT_CLS}
                    >
                      <option value="">— none —</option>
                      {schemaBuilder.allColumns.map((c) => (
                        <option key={c.name} value={c.name}>
                          {c.name}
                        </option>
                      ))}
                    </select>
                  </Field>
                  <Field label="Prediction Column">
                    <input
                      value={schemaBuilder.predictionColumn}
                      onChange={(e) =>
                        setSchemaBuilder((s) => ({
                          ...s,
                          predictionColumn: e.target.value,
                        }))
                      }
                      className={INPUT_CLS}
                    />
                  </Field>
                </div>

                {/* Schema errors */}
                {(() => {
                  const errs = validateSchemaBuilder(schemaBuilder);
                  return errs.length > 0 ? (
                    <div className="space-y-0.5 rounded border border-red-800/40 bg-red-900/20 px-3 py-2">
                      {errs.map((e, i) => (
                        <p key={i} className="text-xs text-red-400">
                          {e}
                        </p>
                      ))}
                    </div>
                  ) : null;
                })()}

                {/* Actions */}
                <div className="flex flex-wrap gap-2 pt-1">
                  <button
                    onClick={() => downloadYaml(rawYamlPreview, 'raw.yaml')}
                    className={BTN_NEUTRAL}
                  >
                    ↓ raw.yaml
                  </button>
                  <button
                    onClick={() => downloadYaml(fullYamlPreview, 'full.yaml')}
                    className={BTN_NEUTRAL}
                  >
                    ↓ full.yaml
                  </button>
                  <button
                    onClick={() =>
                      downloadYaml(preprocessedYamlPreview, 'preprocessed.yaml')
                    }
                    className={BTN_NEUTRAL}
                  >
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
                  <div className="space-y-1 rounded border border-green-800/40 bg-green-900/10 px-3 py-2 text-xs text-green-400">
                    <p className="font-medium">
                      Schemas saved — version {schemaSaveResult.version}
                    </p>
                    {Object.entries(schemaSaveResult.uploaded).map(
                      ([k, v]) => (
                        <p key={k} className="text-slate-400">
                          <span className="text-green-500">{k}:</span> {v}
                        </p>
                      ),
                    )}
                  </div>
                )}
              </>
            )}
          </div>
        </AccordionSection>

        {/* ── Advanced Configuration ── */}
        <AccordionSection title="Advanced Configuration">
          <div className="space-y-4">
            <div className="space-y-2">
              <p className={SUB_HEADING}>KubeRay / MLflow</p>
              <div className="grid grid-cols-2 gap-3">
                <Field label="MLflow Tracking URI">
                  <input
                    value={advanced.mlflow_tracking_uri}
                    onChange={(e) =>
                      setAdvanced((a) => ({
                        ...a,
                        mlflow_tracking_uri: e.target.value,
                      }))
                    }
                    className={INPUT_CLS}
                  />
                </Field>
                <Field label="MLflow Artifact Location">
                  <input
                    value={advanced.mlflow_artifact_location}
                    onChange={(e) =>
                      setAdvanced((a) => ({
                        ...a,
                        mlflow_artifact_location: e.target.value,
                      }))
                    }
                    className={INPUT_CLS}
                  />
                </Field>
                <Field label="Serving Alias">
                  <input
                    value={advanced.serving_alias}
                    onChange={(e) =>
                      setAdvanced((a) => ({
                        ...a,
                        serving_alias: e.target.value,
                      }))
                    }
                    className={INPUT_CLS}
                  />
                </Field>
                <Field label="Canary">
                  <div className="flex gap-1 pt-0.5">
                    <ToggleButton
                      active={advanced.canary}
                      onClick={() =>
                        setAdvanced((a) => ({ ...a, canary: true }))
                      }
                    >
                      On
                    </ToggleButton>
                    <ToggleButton
                      active={!advanced.canary}
                      onClick={() =>
                        setAdvanced((a) => ({ ...a, canary: false }))
                      }
                    >
                      Off
                    </ToggleButton>
                  </div>
                </Field>
                {advanced.canary && (
                  <>
                    <Field label="Canary Alias">
                      <input
                        value={advanced.canary_alias}
                        onChange={(e) =>
                          setAdvanced((a) => ({
                            ...a,
                            canary_alias: e.target.value,
                          }))
                        }
                        className={INPUT_CLS}
                      />
                    </Field>
                    <Field label="Canary Probability">
                      <input
                        type="number"
                        min={0}
                        max={1}
                        step={0.01}
                        value={advanced.canary_probability}
                        onChange={(e) =>
                          setAdvanced((a) => ({
                            ...a,
                            canary_probability: parseFloat(e.target.value),
                          }))
                        }
                        className={INPUT_CLS}
                      />
                    </Field>
                  </>
                )}
                <Field label="Webhook Base URL">
                  <input
                    value={advanced.webhook_public_base_url}
                    onChange={(e) =>
                      setAdvanced((a) => ({
                        ...a,
                        webhook_public_base_url: e.target.value,
                      }))
                    }
                    className={INPUT_CLS}
                  />
                </Field>
                <Field label="Webhook Path">
                  <input
                    value={advanced.webhook_path}
                    onChange={(e) =>
                      setAdvanced((a) => ({
                        ...a,
                        webhook_path: e.target.value,
                      }))
                    }
                    className={INPUT_CLS}
                  />
                </Field>
                <Field label="Webhook Name">
                  <input
                    value={advanced.webhook_name}
                    onChange={(e) =>
                      setAdvanced((a) => ({
                        ...a,
                        webhook_name: e.target.value,
                      }))
                    }
                    className={INPUT_CLS}
                  />
                </Field>
                <Field label="Webhook Max Age (s)">
                  <input
                    type="number"
                    min={0}
                    value={advanced.webhook_max_timestamp_age_seconds}
                    onChange={(e) =>
                      setAdvanced((a) => ({
                        ...a,
                        webhook_max_timestamp_age_seconds:
                          parseInt(e.target.value) || 0,
                      }))
                    }
                    className={INPUT_CLS}
                  />
                </Field>
              </div>
            </div>

            <div className="space-y-2">
              <p className={SUB_HEADING}>Spark</p>
              <div className="grid grid-cols-2 gap-3">
                <Field label="Read Batch Size">
                  <input
                    type="number"
                    min={1}
                    value={advanced.spark_read_batch_size}
                    onChange={(e) =>
                      setAdvanced((a) => ({
                        ...a,
                        spark_read_batch_size: parseInt(e.target.value) || 512,
                      }))
                    }
                    className={INPUT_CLS}
                  />
                </Field>
                <Field label="Write Batch Size">
                  <input
                    type="number"
                    min={1}
                    value={advanced.spark_write_batch_size}
                    onChange={(e) =>
                      setAdvanced((a) => ({
                        ...a,
                        spark_write_batch_size:
                          parseInt(e.target.value) || 100000,
                      }))
                    }
                    className={INPUT_CLS}
                  />
                </Field>
              </div>
            </div>

            <div className="space-y-2">
              <p className={SUB_HEADING}>Iceberg</p>
              <Field label="Warehouse Path">
                <input
                  value={advanced.iceberg_warehouse}
                  onChange={(e) =>
                    setAdvanced((a) => ({
                      ...a,
                      iceberg_warehouse: e.target.value,
                    }))
                  }
                  className={INPUT_CLS}
                />
              </Field>
            </div>
          </div>
        </AccordionSection>
      </div>

      {/* ── RIGHT: sticky YAML preview ── */}
      <div className="w-[440px] shrink-0 min-h-0 p-4">
        <YamlPreviewPanel
          tabs={previewTabs}
          activeTab={activePreviewTab}
          onTabChange={setActivePreviewTab}
        />
      </div>

      {/* ── Modal: inspección de params_preprocess.yaml ── */}
      {paramsModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
          <div className="flex w-[600px] max-h-[80vh] flex-col rounded-lg border border-slate-700 bg-slate-900 shadow-xl">
            <div className="flex items-center justify-between border-b border-slate-700 px-4 py-3">
              <p className="text-xs font-semibold text-slate-200">
                params_preprocess.yaml — {selectedProcessedTable}
              </p>
              <button
                onClick={() => setParamsModalOpen(false)}
                className="text-xs text-slate-500 hover:text-slate-300"
              >
                Close ✕
              </button>
            </div>
            <div className="flex-1 overflow-auto p-3">
              {paramsModalLoading && (
                <p className="text-xs text-slate-400">Loading…</p>
              )}
              {paramsModalError && (
                <p className="text-xs text-red-400">{paramsModalError}</p>
              )}
              {paramsModalContent && (
                <pre className="whitespace-pre font-mono text-[11px] leading-relaxed text-green-300">
                  {paramsModalContent}
                </pre>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
