/**
 * RunPage — ML training pipeline.
 *
 * 4-step workflow stepper:
 *   1. Preprocessing Run — select preprocess_run_id, auto-load lineage info
 *   2. Model     — framework, config, hyperparams (tune=false)
 *   3. Tuning    — tuning toggle + trial budget / search space
 *   4. Review    — summary + launch
 *
 * Right column: live params_training.yaml preview.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Check, Copy, RefreshCw } from 'lucide-react';
import {
  getPreprocessParams,
  getRunStatus,
  getTrainingConfig,
  listDatasets,
  listPreprocessRunIds,
  submitRun,
  type RunRequest,
  type DatasetInfo,
  type ModelTypeConfig,
  type PreprocessRunId,
  type ResourceConstraints,
  type TaskTypeConfig,
  type TrainingConfig,
  type TrainingRunResult,
} from '../api/platformClient';
import { GPUResourceSelector } from '../components/GPUResourceSelector';
import { ChipInput } from '../components/forms/ChipInput';
import { NumericInput } from '../components/forms/NumericInput';
import { parseFiniteNumber, parseStringChip, stringifyNumberValue } from '../lib/formValues';
import {
  TASK_INFO,
  XGBOOST_DEFAULTS,
  XGBOOST_TUNE_SETTINGS_DEFAULTS,
  getDefaults,
  getDefaultsForTask,
  getParamMetaForTask,
  getTuneSettingsDefaults,
  getAllowedKeys,
  getSearchSpace,
  getTuneSettingsMeta,
  getNonTunableKeys,
  type ParamMeta,
  type SearchSpaceEntry,
  type TaskType,
} from '../types/hyperparams';

// ─── Constants ────────────────────────────────────────────────────────────────

const DEFAULT_MODEL_CONFIG = {
  experiment_name: 'kuberay-attack-detection',
  registry_model_name: 'attack-detection',
  mlflow_tracking_uri: '',
  mlflow_artifact_location: 's3://k8s-mlops-platform-bucket/mlflow-artifacts/',
  target: 'attack',
  num_classes: 6,
  seed: 42,
};

const STEP_LABELS = ['Preprocessing Run', 'Model', 'Tuning', 'Review'];

type Framework = 'xgboost' | 'pytorch';
type TuneMode = 'predefined' | 'override';
type SkyLaunchMode = 'launch' | 'jobs_launch';

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
    onOverrideChange?: (key: string, field: 'min' | 'max' | 'options', val: number | number[] | undefined) => void;
    options?: string[];
    min?: number;
    max?: number;
    step?: number;
  };
  value: number | string | Array<number | string>;
  onChange?: (key: string, val: number | string | Array<number | string>) => void;
  readOnly?: boolean;
}) {
  if (readOnly || meta.type === 'array') {
    if (readOnly) {
      const display = Array.isArray(value) ? value.join(', ') : String(value);
      return <ReadOnlyBadge>{display}</ReadOnlyBadge>;
    }

    return (
      <ChipInput
        value={Array.isArray(value) ? (value as Array<string | number>) : []}
        onChange={(next) => onChange?.(paramKey, next)}
        parseItem={(raw) => parseStringChip(raw)}
        formatItem={(item) => String(item)}
        placeholder="Type a value and press Enter"
      />
    );
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
    <NumericInput
      className={INPUT_CLS}
      value={stringifyNumberValue(value as number | string | null | undefined)}
      min={meta.min}
      max={meta.max}
      step={meta.step ?? 'any'}
      onChange={(next) => onChange?.(paramKey, next)}
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
  override?: { min?: number; max?: number; options?: number[] };
  onOverrideChange?: (key: string, field: 'min' | 'max' | 'options', val: number | number[] | undefined) => void;
  overrideMode: boolean;
}) {
  const renderRange = () => {
    if (entry.type === 'fixed') {
      const v = Array.isArray(entry.value) ? entry.value.join(', ') : String(entry.value);
      return <ReadOnlyBadge>fixed: {v}</ReadOnlyBadge>;
    }
    if (entry.type === 'choice') {
      if (overrideMode) {
        const currentOptions = override?.options ?? (entry.options as number[]);
        return (
          <ChipInput
            value={currentOptions}
            onChange={(opts) => onOverrideChange?.(paramKey, 'options', opts)}
            parseItem={(raw) => {
              const parsed = parseFiniteNumber(raw);
              return parsed === null ? null : parsed;
            }}
            formatItem={(item) => String(item)}
            placeholder="64, 128, 256"
            className="w-44"
          />
        );
      }
      return <ReadOnlyBadge>choice: [{entry.options.join(', ')}]</ReadOnlyBadge>;
    }
    if (overrideMode) {
      return (
        <div className="flex items-center gap-1">
          <NumericInput
            className="w-20 rounded bg-slate-800 px-1.5 py-0.5 text-xs text-slate-100 outline-none focus:ring-1 focus:ring-blue-500"
            value={stringifyNumberValue(override?.min ?? entry.min)}
            step="any"
            onChange={(next) => {
              const parsed = parseFiniteNumber(next);
              onOverrideChange?.(paramKey, 'min', next === '' ? undefined : parsed ?? undefined);
            }}
          />
          <span className="text-xs text-slate-500">–</span>
          <NumericInput
            className="w-20 rounded bg-slate-800 px-1.5 py-0.5 text-xs text-slate-100 outline-none focus:ring-1 focus:ring-blue-500"
            value={stringifyNumberValue(override?.max ?? entry.max)}
            step="any"
            onChange={(next) => {
              const parsed = parseFiniteNumber(next);
              onOverrideChange?.(paramKey, 'max', next === '' ? undefined : parsed ?? undefined);
            }}
          />
        </div>
      );
    }
    if (entry.type === 'randint')
      return <ReadOnlyBadge>randint({entry.min}, {entry.max})</ReadOnlyBadge>;
    if (entry.type === 'uniform')
      return <ReadOnlyBadge>uniform({entry.min}, {entry.max})</ReadOnlyBadge>;
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
          <button onClick={onEdit} className="text-[10px] text-blue-400 hover:text-blue-300">
            Edit
          </button>
        )}
      </div>
      <div className="grid grid-cols-[140px_1fr] gap-x-3 gap-y-1">
        {items.map(([k, v]) => (
          <div key={k} className="contents">
            <span className="text-[11px] text-slate-500">{k}</span>
            <span className="break-all font-mono text-[11px] text-slate-200">{v || '—'}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── YamlPreviewPanel ─────────────────────────────────────────────────────────

function YamlPreviewPanel({ content }: { content: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    void navigator.clipboard.writeText(content).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };

  return (
    <div className="flex h-full flex-col overflow-hidden rounded-lg border border-slate-700 bg-slate-900">
      <div className="flex shrink-0 items-center justify-between border-b border-slate-700 px-3 py-2">
        <span className="text-[10px] font-medium text-slate-400">params_training.yaml preview</span>
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

// ─── buildPreviewYaml ─────────────────────────────────────────────────────────

function buildPreviewYaml(opts: {
  preprocess_run_id: string;
  framework: Framework;
  taskType: TaskType;
  modelType: string;
  useGpu: boolean;
  tuningEnabled: boolean;
  numberOfTrials: number;
  sampleFraction: number;
  modelConfig: typeof DEFAULT_MODEL_CONFIG;
  hyperparams: Record<string, number | string | Array<number | string>>;
  tuneSettings: Record<string, number | ''>;
}): string {
  const lines: string[] = [
    '# params_training.yaml preview',
    '',
    'execution:',
    `  preprocess_run_id: ${opts.preprocess_run_id || '(not selected)'}`,
    `  framework: ${opts.framework}`,
    '',
    'model:',
    `  target: ${opts.modelConfig.target}`,
    `  num_classes: ${opts.modelConfig.num_classes}`,
    `  task_type: ${opts.taskType}`,
    ...(opts.framework === 'pytorch' ? [`  model_type: ${opts.modelType}`] : []),
    `  use_gpu: ${opts.useGpu}`,
    `  experiment_name: ${opts.modelConfig.experiment_name}`,
    `  registry_model_name: ${opts.modelConfig.registry_model_name}`,
    `  mlflow_tracking_uri: ${opts.modelConfig.mlflow_tracking_uri || '(default from Airflow env)'}`,
    `  mlflow_artifact_location: ${opts.modelConfig.mlflow_artifact_location}`,
    `  seed: ${opts.modelConfig.seed}`,
    '',
    'tuning:',
    `  enabled: ${opts.tuningEnabled}`,
  ];
  if (opts.tuningEnabled) {
    lines.push(`  number_of_trials: ${opts.numberOfTrials}`);
    lines.push(`  sample_fraction_for_tuning: ${opts.sampleFraction.toFixed(2)}`);
  }
  if (!opts.tuningEnabled && Object.keys(opts.hyperparams).length > 0) {
    lines.push('');
    lines.push('hyperparams:');
    for (const [k, v] of Object.entries(opts.hyperparams)) {
      lines.push(`  ${k}: ${String(v)}`);
    }
  }
  if (opts.tuningEnabled && Object.keys(opts.tuneSettings).length > 0) {
    lines.push('');
    lines.push('tune_settings:');
    for (const [k, v] of Object.entries(opts.tuneSettings)) {
      lines.push(`  ${k}: ${String(v)}`);
    }
  }
  return lines.join('\n') + '\n';
}

// ─── RunPage ──────────────────────────────────────────────────────────────────

export function RunPage() {
  // ── Step navigation ───────────────────────────────────────────────────────
  const [currentStep, setCurrentStep] = useState(1);
  const [navErrors, setNavErrors] = useState<string[]>([]);

  // ── Step 1: Preprocessing Run ─────────────────────────────────────────────
  const [availableDatasets, setAvailableDatasets] = useState<DatasetInfo[]>([]);
  const [selectedDatasetFilter, setSelectedDatasetFilter] = useState('');
  const [preprocessRunIds, setPreprocessRunIds] = useState<PreprocessRunId[]>([]);
  const [preprocessIdsLoading, setPreprocessIdsLoading] = useState(false);
  const [selectedPreprocessRunId, setSelectedPreprocessRunId] = useState('');
  const [preprocessInfo, setPreprocessInfo] = useState<{
    dataset: string;
    dsl_version: number | null;
    dsl_s3_path: string;
    schema_version: number | null;
  } | null>(null);
  const [preprocessInfoLoading, setPreprocessInfoLoading] = useState(false);
  const [preprocessInfoError, setPreprocessInfoError] = useState('');

  // ── Params inspection modal ───────────────────────────────────────────────
  const [paramsModalOpen, setParamsModalOpen] = useState(false);
  const [paramsModalContent, setParamsModalContent] = useState('');
  const [paramsModalLoading, setParamsModalLoading] = useState(false);
  const [paramsModalError, setParamsModalError] = useState('');

  // ── Model ─────────────────────────────────────────────────────────────────
  const [framework, setFramework] = useState<Framework>('xgboost');
  const [modelConfig, setModelConfig] = useState({ ...DEFAULT_MODEL_CONFIG });

  // ── Hyperparams ───────────────────────────────────────────────────────────
  const [hyperparams, setHyperparams] = useState<Record<string, number | string | Array<number | string>>>(
    () => ({ ...XGBOOST_DEFAULTS }),
  );

  // ── Tune settings ─────────────────────────────────────────────────────────
  const [tuneSettings, setTuneSettings] = useState<Record<string, number>>(
    () => ({ ...XGBOOST_TUNE_SETTINGS_DEFAULTS }),
  );

  // ── Search space (UI override mode) ──────────────────────────────────────
  const [tuneMode, setTuneMode] = useState<TuneMode>('predefined');
  const [searchSpaceOverrides, setSearchSpaceOverrides] = useState<
    Record<string, { min?: number; max?: number; options?: number[] }>
  >({});

  // ── Tuning ────────────────────────────────────────────────────────────────
  const [tuningEnabled, setTuningEnabled] = useState(true);
  const [numberOfTrials, setNumberOfTrials] = useState(3);
  const [sampleFraction, setSampleFraction] = useState(0.2);

  // ── Task / model / GPU ────────────────────────────────────────────────────
  const [taskType, setTaskType] = useState<TaskType>('classification');
  const [modelType, setModelType] = useState<string>('mlp');
  const [useGpu, setUseGpu] = useState<boolean>(false);
  const [skyLaunchMode, setSkyLaunchMode] = useState<SkyLaunchMode>('launch');
  const [resourceConstraints, setResourceConstraints] = useState<ResourceConstraints>({
    providers: ['runpod'],
    prefer_spot: true,
    min_vram_gb: 0,
    max_price_per_hour: 9999,
  });
  const [useResourceSelector, setUseResourceSelector] = useState<boolean>(false);
  const [trainingConfig, setTrainingConfig] = useState<TrainingConfig | null>(null);

  // ── Submission ────────────────────────────────────────────────────────────
  const [submitResult, setSubmitResult] = useState<TrainingRunResult | null>(null);
  const [submitError, setSubmitError] = useState('');
  const [runStatus, setRunStatus] = useState('');
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Effects ───────────────────────────────────────────────────────────────

  const loadPreprocessIds = useCallback(() => {
    setPreprocessIdsLoading(true);
    listPreprocessRunIds(selectedDatasetFilter || undefined)
      .then((r) => setPreprocessRunIds(r.runs))
      .catch(() => setPreprocessRunIds([]))
      .finally(() => setPreprocessIdsLoading(false));
  }, [selectedDatasetFilter]);

  useEffect(() => {
    listDatasets()
      .then((datasets) => setAvailableDatasets(datasets))
      .catch(() => setAvailableDatasets([]));
    // Load training config once; silently ignore failures (UI falls back to local constants).
    getTrainingConfig()
      .then(setTrainingConfig)
      .catch(() => {});
  }, []);

  useEffect(() => {
    setSelectedPreprocessRunId('');
    setPreprocessInfo(null);
    setPreprocessInfoError('');
    loadPreprocessIds();
  }, [loadPreprocessIds]);

  useEffect(() => {
    if (!selectedPreprocessRunId) {
      setPreprocessInfo(null);
      setPreprocessInfoError('');
      return;
    }
    setPreprocessInfoLoading(true);
    setPreprocessInfoError('');
    setPreprocessInfo(null);
    getPreprocessParams(selectedPreprocessRunId)
      .then(async (r) => {
        const { parse } = await import('yaml');
        const parsed = parse(r.yaml_content) as {
          execution?: { dsl_s3_path?: string; dsl_version?: number; raw_dataset_name?: string };
          schema_ref?: { version?: number };
          run_metadata?: { dataset?: string; dsl_version?: number };
        };
        setPreprocessInfo({
          dataset: parsed.run_metadata?.dataset ?? parsed.execution?.raw_dataset_name ?? '',
          dsl_version: parsed.run_metadata?.dsl_version ?? parsed.execution?.dsl_version ?? null,
          dsl_s3_path: parsed.execution?.dsl_s3_path ?? '',
          schema_version: parsed.schema_ref?.version ?? null,
        });
      })
      .catch((e) =>
        setPreprocessInfoError(e instanceof Error ? e.message : String(e)),
      )
      .finally(() => setPreprocessInfoLoading(false));
  }, [selectedPreprocessRunId]);

  useEffect(() => {
    setHyperparams({ ...getDefaultsForTask(framework, taskType) });
    setTuneSettings({ ...getTuneSettingsDefaults(framework) });
    setSearchSpaceOverrides({});
  }, [framework, taskType]);

  useEffect(() => {
    if (!submitResult?.dag_run_id) return;
    const TERMINAL = ['success', 'failed', 'upstream_failed'];
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const s = await getRunStatus(
          submitResult.dag_run_id,
          Boolean(submitResult.skypilot),
        );
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

  // ── YAML preview ──────────────────────────────────────────────────────────

  const previewYaml = useMemo(
    () =>
      buildPreviewYaml({
        preprocess_run_id: selectedPreprocessRunId,
        framework,
        taskType,
        modelType,
        useGpu,
        tuningEnabled,
        numberOfTrials,
        sampleFraction,
        modelConfig,
        hyperparams,
        tuneSettings,
      }),
    [
      selectedPreprocessRunId,
      framework,
      taskType,
      modelType,
      useGpu,
      tuningEnabled,
      numberOfTrials,
      sampleFraction,
      modelConfig,
      hyperparams,
      tuneSettings,
    ],
  );

  // ── Step validation ────────────────────────────────────────────────────────

  function getStepErrors(step: number): string[] {
    switch (step) {
      case 1: {
        const e: string[] = [];
        if (!selectedPreprocessRunId) e.push('Select a preprocessing run');
        return e;
      }
      case 2: {
        const e: string[] = [];
        if (!modelConfig.target.trim()) e.push('Target column is required');
        if (modelConfig.num_classes < 2) e.push('num_classes must be at least 2');
        return e;
      }
      case 3: {
        const e: string[] = [];
        if (tuningEnabled) {
          if (numberOfTrials < 1) e.push('number_of_trials must be at least 1');
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
    for (let s = 1; s <= 3; s++) errors.push(...getStepErrors(s));
    if (!tuningEnabled) {
      const allowed = getAllowedKeys(framework);
      for (const k of Object.keys(hyperparams)) {
        if (!allowed.has(k)) errors.push(`Unknown hyperparameter key: "${k}"`);
      }
    }
    return errors;
  }

  // ── Step navigation ────────────────────────────────────────────────────────

  const stepStatus = (n: number): 'active' | 'completed' | 'error' | 'pending' => {
    if (n === currentStep) return 'active';
    if (n > currentStep) return 'pending';
    return getStepErrors(n).length === 0 ? 'completed' : 'error';
  };

  const goForward = () => {
    const errs = getStepErrors(currentStep);
    if (errs.length > 0) { setNavErrors(errs); return; }
    setNavErrors([]);
    setCurrentStep((s) => Math.min(s + 1, 4));
  };

  const goBack = () => {
    setNavErrors([]);
    setCurrentStep((s) => Math.max(s - 1, 1));
  };

  const goToStep = (n: number) => {
    if (n <= currentStep) { setNavErrors([]); setCurrentStep(n); }
  };

  // ── Handlers ──────────────────────────────────────────────────────────────

  const handleInspectParams = async () => {
    if (!selectedPreprocessRunId) return;
    setParamsModalContent('');
    setParamsModalError('');
    setParamsModalLoading(true);
    setParamsModalOpen(true);
    try {
      const r = await getPreprocessParams(selectedPreprocessRunId);
      setParamsModalContent(r.yaml_content);
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
      // Build search_space payload when override mode is active.
      // Only include params that the user actually changed (non-empty overrides).
      const searchSpacePayload: NonNullable<RunRequest['search_space']> = {};
      if (tuningEnabled && tuneMode === 'override') {
        const baseSpace = getSearchSpace(framework);
        for (const [key, entry] of Object.entries(baseSpace)) {
          const ov = searchSpaceOverrides[key];
          if (!ov) continue;
          if (entry.type === 'choice' && ov.options && ov.options.length > 0) {
            searchSpacePayload[key] = { type: 'choice', options: ov.options };
          } else if (entry.type !== 'choice' && (ov.min !== undefined || ov.max !== undefined)) {
            searchSpacePayload[key] = {
              type: entry.type,
              min: ov.min ?? (entry as { min: number }).min,
              max: ov.max ?? (entry as { max: number }).max,
            };
          }
        }
      }

      const result = await submitRun({
        preprocess_run_id: selectedPreprocessRunId,
        framework,
        use_gpu: useGpu,
        use_managed_jobs: skyLaunchMode === 'jobs_launch',
        resource_constraints: useResourceSelector ? resourceConstraints : null,
        tuning: { enabled: tuningEnabled, number_of_trials: numberOfTrials },
        model: {
          ...modelConfig,
          task_type: taskType,
          model_type: framework === 'pytorch' ? modelType : undefined,
        },
        sample_fraction_for_tuning: sampleFraction,
        hyperparams: tuningEnabled ? {} : hyperparams,
        tune_settings: tuningEnabled ? tuneSettings : {},
        search_space: Object.keys(searchSpacePayload).length > 0 ? searchSpacePayload : undefined,
      });
      setSubmitResult(result);
      setRunStatus('queued');
    } catch (e) {
      setSubmitError(e instanceof Error ? e.message : String(e));
    }
  };

  // ── Derived values ────────────────────────────────────────────────────────

  const searchSpace = getSearchSpace(framework);
  const paramMeta = getParamMetaForTask(framework, taskType);
  const tuneSettingsMeta = getTuneSettingsMeta(framework);
  const nonTunableKeys = getNonTunableKeys(framework);

  const maxTKey = framework === 'xgboost' ? 'num_boost_round' : 'max_epochs';
  const roundsPerTrial = (tuneSettings[maxTKey] as number | undefined) ?? 10;
  const totalSteps = numberOfTrials * roundsPerTrial;

  const fullValidationErrors = validateForm();

  const updateHyperparam = (key: string, val: number | string | Array<number | string>) =>
    setHyperparams((prev) => ({ ...prev, [key]: val }));

  const updateTuneSetting = (key: string, val: number) =>
    setTuneSettings((prev) => ({ ...prev, [key]: val }));

  const updateSearchSpaceOverride = (
    key: string,
    field: 'min' | 'max' | 'options',
    val: number | number[] | undefined,
  ) =>
    setSearchSpaceOverrides((prev) => ({
      ...prev,
      [key]: { ...prev[key], [field]: val },
    }));

  const stateColor = (state: string) =>
    state === 'success'
      ? 'text-green-400'
      : state === 'running'
        ? 'text-yellow-400'
        : state.includes('fail')
          ? 'text-red-400'
          : 'text-slate-400';

  // ─────────────────────────────────────────────────────────────────────────────
  // RENDER
  // ─────────────────────────────────────────────────────────────────────────────

  return (
    <div className="flex h-full min-h-0 overflow-hidden">
      {/* ── LEFT: scrollable form ── */}
      <div className="min-w-0 flex-1 space-y-3 overflow-y-auto p-4">

        <StepperHeader
          currentStep={currentStep}
          stepStatus={stepStatus}
          onStepClick={goToStep}
        />

        <div className="rounded-lg border border-slate-700 bg-slate-900 px-4 py-4">

          {/* ── STEP 1: Preprocessing Run ── */}
          {currentStep === 1 && (
            <div className="space-y-4">
              <p className={SUB_HEADING}>Select Preprocessing Run</p>

              <p className="text-xs text-slate-500">
                Select the preprocessing run whose processed table this training job will use.
                The dataset, DSL version, and schema version are auto-loaded from the run params.
              </p>

              <Field
                label="Raw Dataset"
                tooltip="Filter preprocessing runs by source raw dataset."
              >
                <select
                  value={selectedDatasetFilter}
                  onChange={(e) => setSelectedDatasetFilter(e.target.value)}
                  className={SELECT_CLS}
                >
                  <option value="">— all datasets —</option>
                  {availableDatasets.map((d) => (
                    <option key={d.name} value={d.name}>
                      {d.name}
                    </option>
                  ))}
                </select>
              </Field>

              <Field
                label="Preprocessing Run"
                tooltip="Identifies the Spark preprocessing run and its processed Iceberg table."
              >
                <div className="flex gap-1">
                  <select
                    value={selectedPreprocessRunId}
                    onChange={(e) => setSelectedPreprocessRunId(e.target.value)}
                    className={SELECT_CLS}
                    disabled={preprocessIdsLoading}
                  >
                    <option value="">
                      {preprocessIdsLoading ? '— loading… —' : '— select run —'}
                    </option>
                    {preprocessRunIds.map((r) => (
                      <option key={r.preprocess_run_id} value={r.preprocess_run_id}>
                        {r.preprocess_run_id}
                        {r.dataset ? ` · ${r.dataset}` : ''}
                      </option>
                    ))}
                  </select>
                  <button
                    onClick={loadPreprocessIds}
                    disabled={preprocessIdsLoading}
                    className={BTN_NEUTRAL}
                    title="Refresh list"
                  >
                    <RefreshCw size={12} />
                  </button>
                  <button
                    onClick={() => void handleInspectParams()}
                    disabled={!selectedPreprocessRunId}
                    className={BTN_NEUTRAL}
                    title="Inspect params_preprocess.yaml"
                  >
                    🔍
                  </button>
                </div>
              </Field>

              {/* Auto-loaded lineage info */}
              {preprocessInfoLoading && (
                <p className="text-xs text-slate-500">Loading run info…</p>
              )}
              {preprocessInfoError && (
                <p className="text-xs text-red-400">{preprocessInfoError}</p>
              )}
              {preprocessInfo && (
                <div className="rounded border border-slate-700 bg-slate-800/40 px-3 py-2.5">
                  <p className={`${SUB_HEADING} mb-2`}>Run Info</p>
                  <div className="grid grid-cols-[120px_1fr] gap-x-3 gap-y-1">
                    <span className="text-[11px] text-slate-500">Dataset</span>
                    <span className="font-mono text-[11px] text-slate-200">
                      {preprocessInfo.dataset || '—'}
                    </span>
                    <span className="text-[11px] text-slate-500">DSL version</span>
                    <span className="font-mono text-[11px] text-slate-200">
                      {preprocessInfo.dsl_version != null ? `v${preprocessInfo.dsl_version}` : '—'}
                    </span>
                    <span className="text-[11px] text-slate-500">Schema version</span>
                    <span className="font-mono text-[11px] text-slate-200">
                      {preprocessInfo.schema_version != null
                        ? `v${preprocessInfo.schema_version}`
                        : '—'}
                    </span>
                    {preprocessInfo.dsl_s3_path && (
                      <>
                        <span className="text-[11px] text-slate-500">DSL path</span>
                        <span className="break-all font-mono text-[11px] text-slate-400">
                          {preprocessInfo.dsl_s3_path}
                        </span>
                      </>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ── STEP 2: Model ── */}
          {currentStep === 2 && (
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

              {/* Task Type / Model Type / GPU */}
              <div className="space-y-2">
                <p className={SUB_HEADING}>Task Configuration</p>
                <div className="grid grid-cols-2 gap-3">
                  <Field label="Task Type">
                    <select
                      value={taskType}
                      onChange={(e) => setTaskType(e.target.value as TaskType)}
                      className={SELECT_CLS}
                    >
                      {(trainingConfig?.task_types ?? [
                        { id: 'classification', label: 'Classification', description: 'Predict a discrete class label.' } as TaskTypeConfig,
                        { id: 'regression',     label: 'Regression',     description: 'Predict a continuous numeric value.' } as TaskTypeConfig,
                      ]).map((t) => (
                        <option key={t.id} value={t.id}>{t.label}</option>
                      ))}
                    </select>
                    <p className="mt-0.5 text-[10px] text-slate-500">
                      {(trainingConfig?.task_types ?? []).find((t) => t.id === taskType)?.description}
                    </p>
                  </Field>

                  {framework === 'pytorch' ? (
                    <Field label="Model Architecture">
                      <select
                        value={modelType}
                        onChange={(e) => setModelType(e.target.value)}
                        className={SELECT_CLS}
                      >
                        {(trainingConfig?.model_types ?? [
                          { id: 'mlp', label: 'MLP', framework: 'pytorch', description: 'Multi-layer perceptron.' } as ModelTypeConfig,
                        ])
                          .filter((m) => m.framework === 'pytorch')
                          .map((m) => (
                            <option key={m.id} value={m.id}>{m.label} — {m.description}</option>
                          ))}
                      </select>
                    </Field>
                  ) : (
                    <Field label="GPU Acceleration">
                      <label className="flex cursor-pointer items-center gap-2 pt-1">
                        <input
                          type="checkbox"
                          checked={useGpu}
                          onChange={(e) => setUseGpu(e.target.checked)}
                          className="accent-blue-500"
                        />
                        <span className="text-xs text-slate-300">
                          {useGpu ? 'Enabled — workers request GPU' : 'Disabled — CPU only'}
                        </span>
                      </label>
                    </Field>
                  )}
                </div>

                {/* GPU toggle on its own row for PyTorch (model type already fills the grid) */}
                {framework === 'pytorch' && (
                  <Field label="GPU Acceleration">
                    <label className="flex cursor-pointer items-center gap-2 pt-1">
                      <input
                        type="checkbox"
                        checked={useGpu}
                        onChange={(e) => setUseGpu(e.target.checked)}
                        className="accent-blue-500"
                      />
                      <span className="text-xs text-slate-300">
                        {useGpu ? 'Enabled — workers request GPU' : 'Disabled — CPU only'}
                      </span>
                    </label>
                  </Field>
                )}
              </div>

              {/* Dynamic GPU resource selector (pytorch only) */}
              {framework === 'pytorch' && (
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <p className={SUB_HEADING}>Resource Selection</p>
                    <label className="flex cursor-pointer items-center gap-1.5">
                      <input
                        type="checkbox"
                        checked={useResourceSelector}
                        onChange={(e) => {
                          setUseResourceSelector(e.target.checked);
                          if (e.target.checked) setUseGpu(true);
                        }}
                        className="accent-blue-500"
                      />
                      <span className="text-xs text-slate-400">Dynamic provider selection</span>
                    </label>
                  </div>
                  {useResourceSelector && (
                    <GPUResourceSelector
                      value={resourceConstraints}
                      onChange={setResourceConstraints}
                    />
                  )}
                </div>
              )}

              <div className="space-y-2">
                <p className={SUB_HEADING}>SkyPilot Launch Mode</p>
                <div className="flex flex-wrap gap-2">
                  <ToggleButton
                    active={skyLaunchMode === 'launch'}
                    onClick={() => setSkyLaunchMode('launch')}
                  >
                    sky launch --retry-until-up
                  </ToggleButton>
                  <ToggleButton
                    active={skyLaunchMode === 'jobs_launch'}
                    onClick={() => setSkyLaunchMode('jobs_launch')}
                  >
                    sky jobs launch (managed)
                  </ToggleButton>
                </div>
                <p className="text-[11px] text-slate-500">
                  This applies when the run is routed to SkyPilot. Direct launch keeps
                  <span className="mx-1 font-mono text-slate-400">--retry-until-up</span>
                  enabled; managed jobs do not use that flag.
                </p>
              </div>

              {/* Experiment */}
              <div className="space-y-2">
                <p className={SUB_HEADING}>Experiment</p>
                <div className="grid grid-cols-2 gap-3">
                  <Field label="Experiment Name">
                    <input
                      value={modelConfig.experiment_name}
                      onChange={(e) =>
                        setModelConfig((m) => ({ ...m, experiment_name: e.target.value }))
                      }
                      className={INPUT_CLS}
                    />
                  </Field>
                  <Field label="Registry Model Name">
                    <input
                      value={modelConfig.registry_model_name}
                      onChange={(e) =>
                        setModelConfig((m) => ({ ...m, registry_model_name: e.target.value }))
                      }
                      className={INPUT_CLS}
                    />
                  </Field>
                  <Field label="MLflow Tracking URI">
                    <input
                      value={modelConfig.mlflow_tracking_uri}
                      onChange={(e) =>
                        setModelConfig((m) => ({ ...m, mlflow_tracking_uri: e.target.value }))
                      }
                      placeholder="Default: from Airflow MLFLOW_TRACKING_URI"
                      className={INPUT_CLS}
                    />
                  </Field>
                  <Field label="MLflow Artifact Location">
                    <input
                      value={modelConfig.mlflow_artifact_location}
                      onChange={(e) =>
                        setModelConfig((m) => ({ ...m, mlflow_artifact_location: e.target.value }))
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
                        setModelConfig((m) => ({ ...m, target: e.target.value }))
                      }
                      className={INPUT_CLS}
                    />
                  </Field>
                  <Field label="Num Classes" tooltip="Number of output classes (≥ 2).">
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
                        setModelConfig((m) => ({ ...m, seed: parseInt(e.target.value) || 0 }))
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
                    {framework === 'xgboost' ? 'XGBoost' : 'PyTorch'} Parameters
                  </p>
                  <p className="text-xs text-slate-500">
                    Fixed parameters for training (tuning is disabled).
                  </p>
                  <div className="grid grid-cols-2 gap-2">
                    {Object.keys(getDefaults(framework)).map((key) => {
                      const meta: ParamMeta | undefined = paramMeta[key];
                      if (!meta) return null;
                      return (
                        <Field key={key} label={key}>
                          <ParamInput
                            paramKey={key}
                            meta={meta}
                            value={hyperparams[key] ?? meta.defaultValue}
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
                    Tuning is enabled — hyperparameter ranges are configured in the{' '}
                    <span className="text-blue-400">Tuning</span> step.
                    Non-tunable parameters will use defaults.
                  </p>
                </div>
              )}
            </div>
          )}

          {/* ── STEP 3: Tuning ── */}
          {currentStep === 3 && (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <p className={SUB_HEADING}>Hyperparameter Tuning</p>
                <div className="flex gap-1">
                  <ToggleButton active={tuningEnabled} onClick={() => setTuningEnabled(true)}>
                    Enabled
                  </ToggleButton>
                  <ToggleButton active={!tuningEnabled} onClick={() => setTuningEnabled(false)}>
                    Disabled
                  </ToggleButton>
                </div>
              </div>

              {!tuningEnabled ? (
                <div className="space-y-2 rounded border border-slate-700 bg-slate-800/30 px-4 py-3">
                  <p className="text-xs text-slate-400">
                    Training will run with fixed parameters for the full number of rounds/epochs.
                  </p>
                  <p className="text-xs text-slate-500">Active parameters (from Model step):</p>
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
                <div className="space-y-4">
                  {/* Budget controls */}
                  <div className="grid grid-cols-2 gap-3">
                    <Field
                      label="Number of Trials"
                      tooltip="Total Ray Tune trials."
                    >
                      <input
                        type="number"
                        min={1}
                        value={numberOfTrials}
                        onChange={(e) => setNumberOfTrials(parseInt(e.target.value) || 1)}
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
                          onChange={(e) => setSampleFraction(parseFloat(e.target.value))}
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
                      title="How long each trial runs."
                    >
                      Trial Execution Budget
                    </p>
                    <p className="text-xs text-slate-500">
                      Training steps per trial — not sampled by Ray Tune.
                    </p>
                    <div className="grid grid-cols-2 gap-2">
                      {Object.keys(tuneSettingsMeta).map((key) => {
                        const meta: ParamMeta | undefined = tuneSettingsMeta[key];
                        if (!meta) return null;
                        return (
                          <Field key={key} label={key}>
                            <input
                              type="number"
                              min={meta.min}
                              step={meta.step ?? 1}
                              value={tuneSettings[key] ?? meta.defaultValue}
                              onChange={(e) =>
                                updateTuneSetting(key, parseInt(e.target.value) || 1)
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
                      <p className={SUB_HEADING} title="Parameters Ray Tune optimizes.">
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
                        Override mode — changes are sent to the backend and applied per run.
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
                      <p className={SUB_HEADING} title="Applied as constants to every trial.">
                        Non-Tunable Constants
                      </p>
                      <div className="grid grid-cols-2 gap-2">
                        {nonTunableKeys.map((key) => {
                          const meta: ParamMeta | undefined = paramMeta[key];
                          if (!meta) return null;
                          return (
                            <Field key={key} label={key} tooltip="Constant fallback across all trials.">
                              <ParamInput
                                paramKey={key}
                                meta={meta}
                                value={hyperparams[key] ?? meta.defaultValue}
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
                    <span className="text-slate-200">{numberOfTrials}</span> trials ×{' '}
                    <span className="text-slate-200">{roundsPerTrial}</span> rounds ={' '}
                    <span className="text-slate-200">~{totalSteps}</span> total training steps
                    <span className="ml-1 text-slate-600">(ASHA may terminate trials early)</span>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ── STEP 4: Review & Run ── */}
          {currentStep === 4 && (
            <div className="space-y-3">
              <p className={SUB_HEADING}>Review Configuration</p>

              <SummaryCard
                title="Preprocessing Run"
                onEdit={() => goToStep(1)}
                items={[
                  ['Preprocess Run ID', selectedPreprocessRunId],
                  ['Dataset', preprocessInfo?.dataset ?? '—'],
                  ['DSL version', preprocessInfo?.dsl_version != null ? `v${preprocessInfo.dsl_version}` : '—'],
                  ['Schema version', preprocessInfo?.schema_version != null ? `v${preprocessInfo.schema_version}` : '—'],
                ]}
              />

              <SummaryCard
                title="Model"
                onEdit={() => goToStep(2)}
                items={[
                  ['Framework', framework],
                  ['Task type', taskType],
                  ...(framework === 'pytorch' ? [['Model type', modelType] as [string, string]] : []),
                  ['GPU', useGpu ? 'enabled' : 'disabled (CPU only)'],
                  [
                    'SkyPilot launch mode',
                    skyLaunchMode === 'jobs_launch'
                      ? 'sky jobs launch (managed)'
                      : 'sky launch --retry-until-up',
                  ],
                  ['Target', `${modelConfig.target} (${modelConfig.num_classes} classes)`],
                  ['Experiment', modelConfig.experiment_name],
                  ['Registry', modelConfig.registry_model_name],
                  ['MLflow tracking URI', modelConfig.mlflow_tracking_uri || '(default from Airflow env)'],
                  ['MLflow artifact location', modelConfig.mlflow_artifact_location],
                ]}
              />

              {/* Metric Reference — task-aware, collapsible */}
              <details className="rounded border border-slate-700 bg-slate-900 px-3 py-2">
                <summary className="cursor-pointer select-none text-xs font-semibold text-slate-300">
                  Metric Reference — {taskType === 'classification' ? 'Classification' : 'Regression'}
                </summary>
                <div className="mt-3 space-y-3">
                  <p className="text-xs text-slate-400">
                    <span className="font-medium text-slate-200">Loss function: </span>
                    <span className="font-mono text-amber-400">{TASK_INFO[taskType].loss[framework].name}</span>
                    {' — '}{TASK_INFO[taskType].loss[framework].description}
                  </p>
                  <div className="overflow-x-auto">
                    <table className="w-full min-w-[480px] text-xs">
                      <thead>
                        <tr className="border-b border-slate-700 text-left text-slate-500">
                          <th className="py-1 pr-4 font-medium">Metric</th>
                          <th className="pr-4 font-medium">Range</th>
                          <th className="pr-4 font-medium">Perfect</th>
                          <th className="font-medium">Interpretation</th>
                        </tr>
                      </thead>
                      <tbody>
                        {TASK_INFO[taskType].metrics.map((m) => (
                          <tr key={m.key} className="border-b border-slate-800 text-slate-300">
                            <td className="py-1 pr-4 font-mono text-blue-400">{m.label}</td>
                            <td className="pr-4 text-slate-400">{m.range}</td>
                            <td className="pr-4 font-semibold text-green-400">{String(m.perfect)}</td>
                            <td className="text-slate-400">{m.note}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </details>

              <SummaryCard
                title="Tuning"
                onEdit={() => goToStep(3)}
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

              {/* Validation */}
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
                  <p className="text-xs text-green-400">✓ All configuration valid</p>
                </div>
              )}

              {/* Launch / Result */}
              {submitResult ? (
                <div className="space-y-2 rounded-lg border border-slate-700 bg-slate-800/40 p-4">
                  <p className="text-sm font-medium text-green-400">✓ Training pipeline launched</p>
                  <div className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-xs">
                    <span className="text-slate-400">DAG run ID</span>
                    <span className="break-all text-slate-200">{submitResult.dag_run_id}</span>
                    <span className="text-slate-400">Train run ID</span>
                    <span className="break-all text-slate-200">{submitResult.train_run_id}</span>
                    <span className="text-slate-400">Processed table</span>
                    <span className="break-all text-slate-200">{submitResult.processed_table}</span>
                    <span className="text-slate-400">params S3 path</span>
                    <span className="break-all text-slate-200">
                      {submitResult.train_params_s3_path}
                    </span>
                    <span className="text-slate-400">State</span>
                    <span className={`font-semibold ${stateColor(runStatus.toLowerCase())}`}>
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
                    Launch Training Pipeline
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
        {currentStep < 4 ? (
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
      </div>

      {/* ── RIGHT: sticky YAML preview ── */}
      <div className="w-[420px] shrink-0 min-h-0 p-4">
        <YamlPreviewPanel content={previewYaml} />
      </div>

      {/* ── Modal: params_preprocess.yaml inspection ── */}
      {paramsModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
          <div className="flex w-[680px] max-h-[80vh] flex-col rounded-lg border border-slate-700 bg-slate-900 shadow-xl">
            <div className="flex items-center justify-between border-b border-slate-700 px-4 py-3">
              <span className="text-xs font-medium text-slate-300">
                params_preprocess.yaml — {selectedPreprocessRunId}
              </span>
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
              {!paramsModalLoading && !paramsModalError && (
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
