/**
 * ServingPage — Model serving configuration and deployment.
 *
 * Supports two modes:
 *   ray_only — 3 steps: Training Run → Serving Config → Review & Deploy
 *   kafka    — 4 steps: Training Run → Kafka Schema → Serving Config → Review & Deploy
 *
 * After saving the config (params_serving.yaml), a "Deploy Now" button triggers
 * the serving_pipeline Airflow DAG (promotion + RayService patch + optional Spark connector).
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { Check, Copy, RefreshCw } from 'lucide-react';
import {
  listTrainingRunIds,
  uploadRawSchema,
  submitServingConfig,
  triggerServingDeploy,
  getServingDeployStatus,
  getLLMCatalog,
  triggerVllmDeploy,
  getVllmEndpoint,
  type TrainingRunId,
  type SchemaUploadSingleResult,
  type ServingConfigResult,
  type ServingDeployResult,
  type LLMModelInfo,
  type VllmDeployResult,
  type VllmEndpointResult,
} from '../api/platformClient';
import {
  generateRawYamlV2,
  type RawFieldEntry,
} from '../lib/schemaYaml';
import { RawYamlEditor } from '../components/schema/RawYamlEditor';
import { NumericInput } from '../components/forms/NumericInput';
import { parseFiniteNumber, parseInteger } from '../lib/formValues';

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

// ─── Stepper types ────────────────────────────────────────────────────────────

interface StepDef {
  label: string;
  logicalStep: number;
}

const STEPS_RAY_ONLY: StepDef[] = [
  { label: 'Training Run', logicalStep: 1 },
  { label: 'Serving Config', logicalStep: 3 },
  { label: 'Review', logicalStep: 4 },
];

const STEPS_KAFKA: StepDef[] = [
  { label: 'Training Run', logicalStep: 1 },
  { label: 'Kafka Schema', logicalStep: 2 },
  { label: 'Serving Config', logicalStep: 3 },
  { label: 'Review', logicalStep: 4 },
];

const TERMINAL_STATES = new Set(['success', 'failed', 'upstream_failed']);

// ─── Primitive UI components ──────────────────────────────────────────────────

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

// ─── StepperHeader ────────────────────────────────────────────────────────────

function StepperHeader({
  steps,
  currentStep,
  stepStatus,
  onStepClick,
}: {
  steps: StepDef[];
  currentStep: number;
  stepStatus: (logicalStep: number) => 'active' | 'completed' | 'error' | 'pending';
  onStepClick: (logicalStep: number) => void;
}) {
  return (
    <div className="flex w-full select-none items-start gap-0">
      {steps.map(({ label, logicalStep }, i) => {
        const st = stepStatus(logicalStep);
        const clickable = logicalStep <= currentStep;

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
          <div key={logicalStep} className="flex flex-1 items-start">
            <div
              className={`flex flex-col items-center ${clickable ? 'cursor-pointer' : 'cursor-default'}`}
              onClick={() => clickable && onStepClick(logicalStep)}
            >
              <div
                className={`flex h-7 w-7 items-center justify-center rounded-full text-xs font-bold transition-colors ${nodeStyle}`}
              >
                {st === 'completed' ? (
                  <Check size={13} strokeWidth={3} />
                ) : st === 'error' ? (
                  '!'
                ) : (
                  i + 1
                )}
              </div>
              <span className={`mt-1 text-[10px] transition-colors ${labelStyle}`}>
                {label}
              </span>
            </div>
            {i < steps.length - 1 && (
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

function YamlPreviewPanel({ content, label }: { content: string; label: string }) {
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
        <span className="text-[10px] font-medium text-slate-400">{label}</span>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1 rounded px-2 py-0.5 text-[10px] text-slate-400 hover:bg-slate-800 hover:text-slate-200"
        >
          <Copy size={10} />
          {copied ? 'Copied!' : 'Copy'}
        </button>
      </div>
      <pre className="flex-1 overflow-auto whitespace-pre bg-slate-950 p-3 font-mono text-[11px] leading-relaxed text-green-300">
        {content || '# (empty)'}
      </pre>
    </div>
  );
}

// ─── ServingPage ──────────────────────────────────────────────────────────────

export function ServingPage() {
  // ── Step navigation ───────────────────────────────────────────────────────
  const [currentStep, setCurrentStep] = useState(1);
  const [navErrors, setNavErrors] = useState<string[]>([]);

  // ── Step 1: Training Run + Mode ───────────────────────────────────────────
  const [trainingRunIds, setTrainingRunIds] = useState<TrainingRunId[]>([]);
  const [trainingIdsLoading, setTrainingIdsLoading] = useState(false);
  const [selectedTrainRunId, setSelectedTrainRunId] = useState('');
  const [resolvedDataset, setResolvedDataset] = useState('');
  const [servingMode, setServingMode] = useState<'ray_only' | 'kafka'>('ray_only');

  // ── Step 2: Kafka Schema (raw.yaml) — only used when serving_mode=kafka ──
  const [rawFields, setRawFields] = useState<RawFieldEntry[]>([]);
  const [rawGroups, setRawGroups] = useState<Record<string, string>>({
    properties: 'properties',
  });
  const [idField, setIdField] = useState('');
  const [isSavingSchema, setIsSavingSchema] = useState(false);
  const [schemaSaveResult, setSchemaSaveResult] = useState<SchemaUploadSingleResult | null>(null);
  const [schemaSaveError, setSchemaSaveError] = useState('');
  const [rawSchemaS3Path, setRawSchemaS3Path] = useState('');

  // ── Step 3: Serving Config ────────────────────────────────────────────────
  const [alias, setAlias] = useState('champion');
  const [canary, setCanary] = useState(false);
  const [canaryAlias, setCanaryAlias] = useState('challenger');
  const [canaryProbability, setCanaryProbability] = useState('0.1');
  const [initialReplicas, setInitialReplicas] = useState('0');
  const [webhookPublicBaseUrl, setWebhookPublicBaseUrl] = useState('');
  const [webhookPath, setWebhookPath] = useState('/infer/webhook');
  const [webhookMaxTimestampAgeSeconds, setWebhookMaxTimestampAgeSeconds] = useState('300');

  // ── Step 4 / Submission ───────────────────────────────────────────────────
  const [submitResult, setSubmitResult] = useState<ServingConfigResult | null>(null);
  const [submitError, setSubmitError] = useState('');

  // ── Deploy state ──────────────────────────────────────────────────────────
  const [deployResult, setDeployResult] = useState<ServingDeployResult | null>(null);
  const [deployState, setDeployState] = useState('');
  const [deployError, setDeployError] = useState('');
  const deployPollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Serving type: tabular (Ray Serve) vs LLM (vLLM) ──────────────────────
  const [servingType, setServingType] = useState<'tabular' | 'llm'>('tabular');

  // ── LLM serving state ─────────────────────────────────────────────────────
  const [llmCatalog, setLlmCatalog] = useState<LLMModelInfo[]>([]);
  const [llmCatalogLoading, setLlmCatalogLoading] = useState(false);
  const [selectedLlmModel, setSelectedLlmModel] = useState('');
  const [hfToken, setHfToken] = useState('');
  const [llmAdapterS3, setLlmAdapterS3] = useState('');
  const [vllmPort, setVllmPort] = useState('8000');
  const [maxModelLen, setMaxModelLen] = useState('4096');
  const [llmDeployResult, setLlmDeployResult] = useState<VllmDeployResult | null>(null);
  const [llmDeployError, setLlmDeployError] = useState('');
  const [llmDeploying, setLlmDeploying] = useState(false);
  const [llmEndpoint, setLlmEndpoint] = useState<VllmEndpointResult | null>(null);
  const llmPollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Dynamic step list based on serving mode ───────────────────────────────
  const steps = servingMode === 'kafka' ? STEPS_KAFKA : STEPS_RAY_ONLY;

  // ── Effects ───────────────────────────────────────────────────────────────

  const loadTrainingIds = useCallback(() => {
    setTrainingIdsLoading(true);
    listTrainingRunIds()
      .then((r) => setTrainingRunIds(r.runs))
      .catch(() => setTrainingRunIds([]))
      .finally(() => setTrainingIdsLoading(false));
  }, []);

  useEffect(() => {
    loadTrainingIds();
  }, [loadTrainingIds]);

  // Resolve dataset whenever selected train_run_id changes
  useEffect(() => {
    if (!selectedTrainRunId) {
      setResolvedDataset('');
      return;
    }
    const entry = trainingRunIds.find((r) => r.train_run_id === selectedTrainRunId);
    if (entry?.dataset) {
      setResolvedDataset(entry.dataset);
      return;
    }
    const m = selectedTrainRunId.match(/^train-(.+)-[0-9a-f]{6}-\d{8}T\d{6}Z-[0-9a-f]{6}$/);
    setResolvedDataset(m?.[1] ?? '');
  }, [selectedTrainRunId, trainingRunIds]);

  // Reset schema save state when dataset or raw fields change
  useEffect(() => {
    setSchemaSaveResult(null);
    setSchemaSaveError('');
    setRawSchemaS3Path('');
  }, [resolvedDataset, rawFields, rawGroups, idField]);

  // When mode switches, reset submission state and jump back to step 1
  useEffect(() => {
    setSubmitResult(null);
    setSubmitError('');
    setDeployResult(null);
    setDeployState('');
    setDeployError('');
    if (deployPollRef.current) clearInterval(deployPollRef.current);
    // If currently on step 2 and switching to ray_only, jump to step 1
    if (currentStep === 2 && servingMode === 'ray_only') {
      setCurrentStep(1);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [servingMode]);

  // Cleanup poll on unmount
  useEffect(() => {
    return () => {
      if (deployPollRef.current) clearInterval(deployPollRef.current);
      if (llmPollRef.current) clearInterval(llmPollRef.current);
    };
  }, []);

  // Load LLM catalog when switching to LLM mode
  useEffect(() => {
    if (servingType !== 'llm' || llmCatalog.length > 0) return;
    setLlmCatalogLoading(true);
    getLLMCatalog()
      .then((c) => { setLlmCatalog(c); if (c.length > 0) setSelectedLlmModel(c[0]?.model_id ?? ''); })
      .catch(() => setLlmCatalog([]))
      .finally(() => setLlmCatalogLoading(false));
  }, [servingType, llmCatalog.length]);

  // ── Derived: raw.yaml content ─────────────────────────────────────────────

  const rawYamlContent =
    rawFields.length > 0
      ? generateRawYamlV2({
          dataset: resolvedDataset,
          dslName: '',
          dslVersion: '',
          processedTableName: '',
          generatedFrom: '',
          allColumns: [],
          rawFields,
          rawGroups,
          idField,
          fullFields: [],
          preprocessedFields: [],
          rawTopLevel: [],
          propertiesFields: [],
          fullColumns: [],
          preprocessedColumns: [],
          targetColumn: '',
          idColumn: '',
          typeOverrides: {},
        })
      : '# Add fields to generate raw.yaml';

  // ── Step validation ────────────────────────────────────────────────────────

  function getStepErrors(step: number): string[] {
    switch (step) {
      case 1: {
        const e: string[] = [];
        if (!selectedTrainRunId) e.push('Select a training run');
        return e;
      }
      case 2: {
        // Schema step is only validated in kafka mode
        if (servingMode !== 'kafka') return [];
        const e: string[] = [];
        if (!rawSchemaS3Path) e.push('Save raw.yaml to S3 before continuing');
        if (rawFields.length === 0) e.push('Add at least one field to the schema');
        if (!idField) e.push('Select an id_field');
        return e;
      }
      case 3: {
        const e: string[] = [];
        if (!alias.trim()) e.push('Alias is required');
        const canaryProb = parseFiniteNumber(canaryProbability);
        if (canary && (canaryProb === null || canaryProb < 0 || canaryProb > 1))
          e.push('Canary probability must be between 0 and 1');
        return e;
      }
      default:
        return [];
    }
  }

  function validateForm(): string[] {
    const errors: string[] = [];
    for (let s = 1; s <= 3; s++) errors.push(...getStepErrors(s));
    return errors;
  }

  // ── Step navigation ────────────────────────────────────────────────────────

  const stepStatus = (logicalStep: number): 'active' | 'completed' | 'error' | 'pending' => {
    if (logicalStep === currentStep) return 'active';
    if (logicalStep > currentStep) return 'pending';
    return getStepErrors(logicalStep).length === 0 ? 'completed' : 'error';
  };

  const goForward = () => {
    const errs = getStepErrors(currentStep);
    if (errs.length > 0) { setNavErrors(errs); return; }
    setNavErrors([]);
    let next = currentStep + 1;
    // Skip step 2 (Kafka Schema) when in ray_only mode
    if (servingMode === 'ray_only' && next === 2) next = 3;
    setCurrentStep(Math.min(next, 4));
  };

  const goBack = () => {
    setNavErrors([]);
    let prev = currentStep - 1;
    // Skip step 2 (Kafka Schema) when in ray_only mode
    if (servingMode === 'ray_only' && prev === 2) prev = 1;
    setCurrentStep(Math.max(prev, 1));
  };

  const goToStep = (logicalStep: number) => {
    if (logicalStep <= currentStep) { setNavErrors([]); setCurrentStep(logicalStep); }
  };

  // ── Handlers ──────────────────────────────────────────────────────────────

  const handleSaveSchema = async () => {
    if (!resolvedDataset) return;
    setIsSavingSchema(true);
    setSchemaSaveError('');
    setSchemaSaveResult(null);
    setRawSchemaS3Path('');
    try {
      const result = await uploadRawSchema(resolvedDataset, rawYamlContent);
      setSchemaSaveResult(result);
      setRawSchemaS3Path(result.s3_path);
    } catch (e) {
      setSchemaSaveError(e instanceof Error ? e.message : String(e));
    } finally {
      setIsSavingSchema(false);
    }
  };

  const handleRawEditorChange = (
    fields: RawFieldEntry[],
    groups: Record<string, string>,
    newIdField: string,
  ) => {
    setRawFields(fields);
    setRawGroups(groups);
    setIdField(newIdField);
  };

  const handleSubmit = async () => {
    setSubmitError('');
    setSubmitResult(null);
    const errors = validateForm();
    if (errors.length > 0) return;
    try {
      const result = await submitServingConfig({
        train_run_id: selectedTrainRunId,
        serving_mode: servingMode,
        // raw_schema_s3_path is only sent in kafka mode
        ...(servingMode === 'kafka' ? { raw_schema_s3_path: rawSchemaS3Path } : {}),
        alias,
        canary,
        canary_alias: canaryAlias,
        canary_probability: parseFiniteNumber(canaryProbability) ?? 0.1,
        initial_replicas: parseInteger(initialReplicas) ?? 0,
        webhook_public_base_url: webhookPublicBaseUrl,
        webhook_path: webhookPath,
        webhook_max_timestamp_age_seconds: parseInteger(webhookMaxTimestampAgeSeconds) ?? 300,
      });
      setSubmitResult(result);
    } catch (e) {
      setSubmitError(e instanceof Error ? e.message : String(e));
    }
  };

  const handleDeploy = async () => {
    if (!submitResult) return;
    setDeployError('');
    setDeployResult(null);
    setDeployState('queued');
    if (deployPollRef.current) clearInterval(deployPollRef.current);
    try {
      const result = await triggerServingDeploy(submitResult.serve_run_id);
      setDeployResult(result);
      // Poll every 15s until terminal state
      deployPollRef.current = setInterval(() => {
        void getServingDeployStatus(submitResult.serve_run_id, result.dag_run_id)
          .then((s) => {
            setDeployState(s.state);
            if (TERMINAL_STATES.has(s.state.toLowerCase())) {
              if (deployPollRef.current) clearInterval(deployPollRef.current);
            }
          })
          .catch(() => { /* keep polling on transient error */ });
      }, 15_000);
    } catch (e) {
      setDeployError(e instanceof Error ? e.message : String(e));
      setDeployState('');
    }
  };

  const handleLlmDeploy = async () => {
    if (!selectedLlmModel) return;
    setLlmDeployError('');
    setLlmDeployResult(null);
    setLlmEndpoint(null);
    setLlmDeploying(true);
    if (llmPollRef.current) clearInterval(llmPollRef.current);
    try {
      const result = await triggerVllmDeploy({
        llm_model_id: selectedLlmModel,
        hf_token: hfToken || undefined,
        llm_adapter_s3: llmAdapterS3 || undefined,
        vllm_port: parseInteger(vllmPort) ?? 8000,
        max_model_len: parseInteger(maxModelLen) ?? 4096,
      });
      setLlmDeployResult(result);
      // Poll endpoint every 20s until healthy
      llmPollRef.current = setInterval(() => {
        void getVllmEndpoint(result.serve_run_id)
          .then((ep) => {
            setLlmEndpoint(ep);
            if (ep.status === 'healthy') {
              if (llmPollRef.current) clearInterval(llmPollRef.current);
            }
          })
          .catch(() => { /* keep polling on transient error */ });
      }, 20_000);
    } catch (e) {
      setLlmDeployError(e instanceof Error ? e.message : String(e));
    } finally {
      setLlmDeploying(false);
    }
  };

  const fullValidationErrors = validateForm();

  // ─────────────────────────────────────────────────────────────────────────────
  // RENDER
  // ─────────────────────────────────────────────────────────────────────────────

  return (
    <div className="flex h-full min-h-0 overflow-hidden">
      {/* ── LEFT: scrollable form ── */}
      <div className="min-w-0 flex-1 space-y-3 overflow-y-auto p-4">

        {/* ── Serving Type toggle ── */}
        <div className="flex items-center gap-3">
          <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
            Serving Type
          </span>
          <ToggleButton
            active={servingType === 'tabular'}
            onClick={() => setServingType('tabular')}
          >
            Tabular (Ray Serve)
          </ToggleButton>
          <ToggleButton
            active={servingType === 'llm'}
            onClick={() => setServingType('llm')}
          >
            LLM (vLLM)
          </ToggleButton>
        </div>

        {/* ── LLM SERVING FORM ── */}
        {servingType === 'llm' && (
          <div className="rounded-lg border border-slate-700 bg-slate-900 px-4 py-4 space-y-4">
            <p className={SUB_HEADING}>vLLM Serving — Deploy a persistent LLM endpoint</p>
            <p className="text-xs text-slate-500">
              Provisions an on-demand GPU VM (RunPod / Vast.ai) running vLLM.
              The cluster stays up after deployment — tear down with{' '}
              <code className="rounded bg-slate-800 px-1">sky down</code>.
            </p>

            {/* Model selection */}
            <div className="grid grid-cols-2 gap-3">
              <div className="flex flex-col gap-1">
                <label className="text-xs text-slate-400">LLM Model</label>
                <select
                  value={selectedLlmModel}
                  onChange={(e) => setSelectedLlmModel(e.target.value)}
                  className={SELECT_CLS}
                  disabled={llmCatalogLoading}
                >
                  <option value="">
                    {llmCatalogLoading ? '— loading… —' : '— select model —'}
                  </option>
                  {llmCatalog.map((m) => (
                    <option key={m.model_id} value={m.model_id}>
                      {m.model_id} ({m.vram_gb} GB VRAM, ≥{m.min_gpus} GPU)
                    </option>
                  ))}
                </select>
              </div>

              {selectedLlmModel && (() => {
                const info = llmCatalog.find((m) => m.model_id === selectedLlmModel);
                return info ? (
                  <div className="flex flex-col justify-end gap-1">
                    <span className="text-[11px] text-slate-500">Recommended GPU</span>
                    <span className="font-mono text-xs text-blue-300">{info.recommended_gpu}</span>
                    <span className="text-[11px] text-slate-500">Min VRAM: {info.vram_gb} GB</span>
                  </div>
                ) : null;
              })()}
            </div>

            {/* HF Token */}
            <div className="flex flex-col gap-1">
              <label className="text-xs text-slate-400">
                HuggingFace Token{' '}
                <span className="text-slate-600">(required for gated models like LLaMA)</span>
              </label>
              <input
                type="password"
                value={hfToken}
                onChange={(e) => setHfToken(e.target.value)}
                placeholder="hf_..."
                className={INPUT_CLS}
              />
            </div>

            {/* Adapter S3 path */}
            <div className="flex flex-col gap-1">
              <label className="text-xs text-slate-400">
                LoRA Adapter S3 Path{' '}
                <span className="text-slate-600">(optional — leave empty for base model)</span>
              </label>
              <input
                value={llmAdapterS3}
                onChange={(e) => setLlmAdapterS3(e.target.value)}
                placeholder="s3://my-bucket/llm-adapters/my-adapter/"
                className={INPUT_CLS}
              />
            </div>

            {/* vLLM config */}
            <div className="grid grid-cols-2 gap-3">
              <div className="flex flex-col gap-1">
                <label className="text-xs text-slate-400">Serving Port</label>
                  <NumericInput
                    min={1}
                    max={65535}
                    value={vllmPort}
                    onChange={setVllmPort}
                    className={INPUT_CLS}
                  />
              </div>
              <div className="flex flex-col gap-1">
                <label className="text-xs text-slate-400">Max Model Length</label>
                  <NumericInput
                    min={256}
                    value={maxModelLen}
                    onChange={setMaxModelLen}
                    className={INPUT_CLS}
                  />
              </div>
            </div>

            <div className="rounded border border-amber-800/30 bg-amber-900/10 px-3 py-2">
              <p className="text-xs text-amber-400">
                On-demand only (no spot preemptions for serving).
                Est. ~$2–4/hr for A100. Tear down manually when done.
              </p>
            </div>

            {/* Deploy button + result */}
            {llmDeployResult ? (
              <div className="space-y-2 rounded-lg border border-slate-700 bg-slate-800/40 p-4">
                <p className="text-sm font-medium text-green-400">✓ vLLM deployment triggered</p>
                <div className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-xs">
                  <span className="text-slate-400">Serve run ID</span>
                  <span className="break-all font-mono text-slate-200">
                    {llmDeployResult.serve_run_id}
                  </span>
                  <span className="text-slate-400">Cluster</span>
                  <span className="font-mono text-slate-200">
                    {llmDeployResult.sky_cluster_name}
                  </span>
                  <span className="text-slate-400">Endpoint</span>
                  <span className={`font-mono ${
                    llmEndpoint?.status === 'healthy'
                      ? 'text-green-400'
                      : 'text-amber-400'
                  }`}>
                    {llmEndpoint?.status === 'healthy'
                      ? llmEndpoint.endpoint_url
                      : llmEndpoint?.status === 'not_found'
                        ? 'provisioning…'
                        : 'polling…'}
                  </span>
                </div>
                {llmEndpoint?.status === 'healthy' && (
                  <div className="rounded border border-green-800/30 bg-green-900/10 px-3 py-2">
                    <p className="text-xs text-green-400">
                      ✓ vLLM healthy at{' '}
                      <span className="font-mono">{llmEndpoint.endpoint_url}/v1</span>
                    </p>
                  </div>
                )}
                <button
                  onClick={() => {
                    setLlmDeployResult(null);
                    setLlmEndpoint(null);
                    if (llmPollRef.current) clearInterval(llmPollRef.current);
                  }}
                  className={BTN_NEUTRAL}
                >
                  New Deployment
                </button>
              </div>
            ) : (
              <div className="space-y-2">
                {llmDeployError && (
                  <p className="text-xs text-red-400">{llmDeployError}</p>
                )}
                <button
                  onClick={() => void handleLlmDeploy()}
                  disabled={llmDeploying || !selectedLlmModel}
                  className="w-full rounded bg-purple-700 py-2 text-sm font-semibold text-white hover:bg-purple-600 disabled:opacity-40"
                >
                  {llmDeploying ? 'Deploying…' : 'Deploy vLLM'}
                </button>
              </div>
            )}
          </div>
        )}

        {/* ── TABULAR SERVING FORM (existing) ── */}
        {servingType === 'tabular' && (
          <>
        <StepperHeader
          steps={steps}
          currentStep={currentStep}
          stepStatus={stepStatus}
          onStepClick={goToStep}
        />

        <div className="rounded-lg border border-slate-700 bg-slate-900 px-4 py-4">

          {/* ── STEP 1: Training Run + Mode ── */}
          {currentStep === 1 && (
            <div className="space-y-4">
              <p className={SUB_HEADING}>Select Training Run</p>
              <p className="text-xs text-slate-500">
                Select the training run to serve. The dataset is inferred from the run ID.
              </p>

              <Field
                label="Training Run"
                tooltip="The train_run_id whose MLflow model will be promoted to champion."
              >
                <div className="flex gap-1">
                  <select
                    value={selectedTrainRunId}
                    onChange={(e) => setSelectedTrainRunId(e.target.value)}
                    className={SELECT_CLS}
                    disabled={trainingIdsLoading}
                  >
                    <option value="">
                      {trainingIdsLoading ? '— loading… —' : '— select run —'}
                    </option>
                    {trainingRunIds.map((r) => (
                      <option key={r.train_run_id} value={r.train_run_id}>
                        {r.train_run_id}
                        {r.dataset ? ` · ${r.dataset}` : ''}
                      </option>
                    ))}
                  </select>
                  <button
                    onClick={loadTrainingIds}
                    disabled={trainingIdsLoading}
                    className={BTN_NEUTRAL}
                    title="Refresh list"
                  >
                    <RefreshCw size={12} />
                  </button>
                </div>
              </Field>

              <Field
                label="Serving Mode"
                tooltip="ray_only: deploys Ray Serve only. kafka: also deploys a Spark Kafka streaming connector."
              >
                <div className="flex gap-2">
                  <ToggleButton
                    active={servingMode === 'ray_only'}
                    onClick={() => setServingMode('ray_only')}
                  >
                    Ray Only
                  </ToggleButton>
                  <ToggleButton
                    active={servingMode === 'kafka'}
                    onClick={() => setServingMode('kafka')}
                  >
                    Kafka
                  </ToggleButton>
                </div>
              </Field>

              {servingMode === 'ray_only' && (
                <p className="text-xs text-slate-500">
                  Promotes the model to <code className="rounded bg-slate-800 px-1">@champion</code>,
                  patches Ray Serve config. No Kafka connector deployed. raw.yaml not required.
                </p>
              )}
              {servingMode === 'kafka' && (
                <p className="text-xs text-slate-500">
                  Promotes the model, patches Ray Serve, and deploys a Spark Kafka streaming connector.
                  You will define a <code className="rounded bg-slate-800 px-1">raw.yaml</code> schema in the next step.
                </p>
              )}

              {selectedTrainRunId && (
                <div className="rounded border border-slate-700 bg-slate-800/40 px-3 py-2.5">
                  <div className="grid grid-cols-[100px_1fr] gap-x-3 gap-y-1">
                    <span className="text-[11px] text-slate-500">Train run ID</span>
                    <span className="break-all font-mono text-[11px] text-slate-200">
                      {selectedTrainRunId}
                    </span>
                    <span className="text-[11px] text-slate-500">Dataset</span>
                    <span className="font-mono text-[11px] text-slate-200">
                      {resolvedDataset || '—'}
                    </span>
                    <span className="text-[11px] text-slate-500">Mode</span>
                    <span className="font-mono text-[11px] text-slate-200">{servingMode}</span>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ── STEP 2: Kafka Schema (only in kafka mode) ── */}
          {currentStep === 2 && (
            <div className="space-y-4">
              <p className={SUB_HEADING}>Kafka Input Schema (raw.yaml)</p>
              <p className="text-xs text-slate-500">
                Define the Kafka message schema for{' '}
                <code className="rounded bg-slate-800 px-1 text-slate-300">
                  {resolvedDataset || '(dataset)'}
                </code>
                . The <code className="rounded bg-slate-800 px-1">id_field</code> is read by{' '}
                <code className="rounded bg-slate-800 px-1">kafka_main.py</code> as the message key.
                Load columns from Iceberg or add them manually.
              </p>

              <RawYamlEditor
                fields={rawFields}
                groups={rawGroups}
                idField={idField}
                onChange={handleRawEditorChange}
                dataset={resolvedDataset}
              />

              {/* Save to S3 */}
              <div className="flex items-center gap-3 pt-1">
                <button
                  onClick={() => void handleSaveSchema()}
                  disabled={
                    isSavingSchema ||
                    rawFields.length === 0 ||
                    !idField ||
                    !resolvedDataset
                  }
                  className={BTN_PRIMARY}
                >
                  {isSavingSchema ? 'Saving…' : 'Save raw.yaml to S3'}
                </button>

                {schemaSaveResult && (
                  <span className="rounded-full bg-green-900/50 px-2 py-0.5 text-[10px] text-green-400">
                    ✓ Saved as v{schemaSaveResult.version}
                  </span>
                )}
                {rawSchemaS3Path && (
                  <span
                    className="max-w-[300px] truncate font-mono text-[10px] text-slate-500"
                    title={rawSchemaS3Path}
                  >
                    {rawSchemaS3Path}
                  </span>
                )}
              </div>

              {schemaSaveError && (
                <p className="text-xs text-red-400">{schemaSaveError}</p>
              )}
              {!rawSchemaS3Path && rawFields.length > 0 && idField && (
                <p className="text-xs text-amber-400">
                  Save raw.yaml to S3 before proceeding to the next step.
                </p>
              )}
            </div>
          )}

          {/* ── STEP 3: Serving Config ── */}
          {currentStep === 3 && (
            <div className="space-y-4">
              {/* Alias */}
              <div className="space-y-2">
                <p className={SUB_HEADING}>Model Alias</p>
                <div className="grid grid-cols-2 gap-3">
                  <Field label="Alias" tooltip="MLflow model alias to promote. Typically 'champion'.">
                    <input
                      value={alias}
                      onChange={(e) => setAlias(e.target.value)}
                      placeholder="champion"
                      className={INPUT_CLS}
                    />
                  </Field>
                </div>
              </div>

              {/* Canary */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <p className={SUB_HEADING}>Canary Deployment</p>
                  <div className="flex gap-1">
                    <ToggleButton active={canary} onClick={() => setCanary(true)}>
                      Enabled
                    </ToggleButton>
                    <ToggleButton active={!canary} onClick={() => setCanary(false)}>
                      Disabled
                    </ToggleButton>
                  </div>
                </div>

                {canary && (
                  <div className="grid grid-cols-3 gap-3">
                    <Field label="Canary Alias">
                      <input
                        value={canaryAlias}
                        onChange={(e) => setCanaryAlias(e.target.value)}
                        placeholder="challenger"
                        className={INPUT_CLS}
                      />
                    </Field>
                    <Field
                      label="Canary Probability"
                      tooltip="Fraction of traffic routed to the canary model (0–1)."
                    >
                      <NumericInput
                        min={0}
                        max={1}
                        step={0.01}
                        value={canaryProbability}
                        onChange={setCanaryProbability}
                        className={INPUT_CLS}
                      />
                    </Field>
                    <Field
                      label="Initial Replicas"
                      tooltip="Number of replicas for the canary service. 0 = not deployed yet."
                    >
                      <NumericInput
                        min={0}
                        value={initialReplicas}
                        onChange={setInitialReplicas}
                        className={INPUT_CLS}
                      />
                    </Field>
                  </div>
                )}
              </div>

              {/* Webhook */}
              <div className="space-y-2">
                <p className={SUB_HEADING}>Webhook</p>
                <p className="text-xs text-slate-500">
                  MLflow alias-change webhook endpoint for hot model reloads.
                </p>
                <div className="grid grid-cols-2 gap-3">
                  <Field label="Public Base URL" tooltip="External base URL of the inference service.">
                    <input
                      value={webhookPublicBaseUrl}
                      onChange={(e) => setWebhookPublicBaseUrl(e.target.value)}
                      placeholder="http://model-serving-serve-svc.ray.svc.cluster.local:8000"
                      className={INPUT_CLS}
                    />
                  </Field>
                  <Field label="Webhook Path">
                    <input
                      value={webhookPath}
                      onChange={(e) => setWebhookPath(e.target.value)}
                      placeholder="/infer/webhook"
                      className={INPUT_CLS}
                    />
                  </Field>
                  <Field
                    label="Max Timestamp Age (s)"
                    tooltip="Reject webhook requests older than this many seconds."
                  >
                    <NumericInput
                      min={1}
                      value={webhookMaxTimestampAgeSeconds}
                      onChange={setWebhookMaxTimestampAgeSeconds}
                      className={INPUT_CLS}
                    />
                  </Field>
                </div>
              </div>
            </div>
          )}

          {/* ── STEP 4: Review & Submit & Deploy ── */}
          {currentStep === 4 && (
            <div className="space-y-3">
              <p className={SUB_HEADING}>Review Configuration</p>

              <SummaryCard
                title="Training Run"
                onEdit={() => goToStep(1)}
                items={[
                  ['Train run ID', selectedTrainRunId],
                  ['Dataset', resolvedDataset],
                  ['Mode', servingMode],
                ]}
              />

              {servingMode === 'kafka' && (
                <SummaryCard
                  title="Kafka Schema"
                  onEdit={() => goToStep(2)}
                  items={[
                    ['Fields', String(rawFields.length)],
                    ['id_field', idField],
                    ['Schema version', schemaSaveResult ? `v${schemaSaveResult.version}` : '—'],
                    ['S3 path', rawSchemaS3Path],
                  ]}
                />
              )}

              <SummaryCard
                title="Serving Config"
                onEdit={() => goToStep(3)}
                items={[
                  ['Alias', alias],
                  ['Canary', canary ? `Yes (${canaryAlias}, p=${canaryProbability})` : 'No'],
                  ['Webhook path', webhookPath || '—'],
                  ['Max timestamp age', `${webhookMaxTimestampAgeSeconds}s`],
                ]}
              />

              {/* Validation */}
              {fullValidationErrors.length > 0 ? (
                <div className="space-y-1 rounded border border-red-800/40 bg-red-900/20 px-3 py-2">
                  <p className="text-xs font-semibold text-red-400">
                    Issues to resolve before submitting:
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

              {/* Submit / Result */}
              {submitResult ? (
                <div className="space-y-3 rounded-lg border border-slate-700 bg-slate-800/40 p-4">
                  <p className="text-sm font-medium text-green-400">✓ Serving config saved</p>
                  <div className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-xs">
                    <span className="text-slate-400">Serve run ID</span>
                    <span className="break-all text-slate-200">{submitResult.serve_run_id}</span>
                    <span className="text-slate-400">Mode</span>
                    <span className="text-slate-200">{submitResult.serving_mode}</span>
                    <span className="text-slate-400">Dataset</span>
                    <span className="text-slate-200">{submitResult.dataset}</span>
                    <span className="text-slate-400">Registry model</span>
                    <span className="break-all text-slate-200">{submitResult.registry_model_name || '—'}</span>
                    <span className="text-slate-400">Params S3 path</span>
                    <span className="break-all text-slate-200">{submitResult.params_s3_path}</span>
                  </div>

                  {/* Deploy section */}
                  {!deployResult ? (
                    <div className="space-y-2 border-t border-slate-700 pt-3">
                      {deployError && (
                        <p className="text-xs text-red-400">{deployError}</p>
                      )}
                      <button
                        onClick={() => void handleDeploy()}
                        className="w-full rounded bg-green-700 py-2 text-sm font-semibold text-white hover:bg-green-600"
                      >
                        Deploy Now
                      </button>
                      <p className="text-center text-[10px] text-slate-500">
                        Triggers: MLflow promotion → Ray Serve patch
                        {submitResult.serving_mode === 'kafka' ? ' → Spark Kafka connector' : ''}
                      </p>
                    </div>
                  ) : (
                    <div className="space-y-1 rounded border border-slate-700 bg-slate-800/40 px-3 py-2">
                      <p className="text-xs font-medium text-slate-300">Deployment triggered</p>
                      <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
                        <span className="text-slate-500">DAG run ID</span>
                        <span className="break-all font-mono text-slate-200">
                          {deployResult.dag_run_id}
                        </span>
                        <span className="text-slate-500">State</span>
                        <span
                          className={`font-mono ${
                            deployState === 'success'
                              ? 'text-green-400'
                              : deployState === 'failed' || deployState === 'upstream_failed'
                                ? 'text-red-400'
                                : 'text-amber-400'
                          }`}
                        >
                          {deployState || 'queued'}
                        </span>
                      </div>
                    </div>
                  )}

                  <button
                    onClick={() => {
                      setSubmitResult(null);
                      setDeployResult(null);
                      setDeployState('');
                      setDeployError('');
                      if (deployPollRef.current) clearInterval(deployPollRef.current);
                      setCurrentStep(1);
                      setNavErrors([]);
                    }}
                    className={BTN_NEUTRAL}
                  >
                    New Serving Config
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
                    Submit Serving Config
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
          </>
        )}
      </div>

      {/* ── RIGHT: raw.yaml preview (only meaningful in tabular kafka mode) ── */}
      <div className="w-[420px] shrink-0 min-h-0 p-4">
        <YamlPreviewPanel
          content={
            servingMode === 'kafka'
              ? rawYamlContent
              : '# raw.yaml not used in this mode'
          }
          label="raw.yaml preview"
        />
      </div>
    </div>
  );
}
