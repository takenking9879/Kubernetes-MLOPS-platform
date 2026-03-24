/**
 * LaunchWizardPage — 4-step unified job launch wizard (Phase 8).
 *
 * Step 1 — Model Selection: job type, model category, hyperparams / LLM picker
 * Step 2 — Resource Configuration: GPUResourceSelector + num_nodes
 * Step 3 — Smart Recommendations: orchestration + cost estimate + YAML preview
 * Step 4 — Launch: summary + launch button + live status
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import {
  ChevronRight,
  Rocket,
  Cpu,
  Layers,
  UploadCloud,
  RefreshCw,
  CheckCircle2,
  XCircle,
  Loader2,
} from 'lucide-react';

import {
  launchJob,
  getLLMCatalog,
  listArchitectures,
  uploadArchitecture,
  selectGPUResources,
  type LaunchRequest,
  type LaunchResponse,
  type OrchestratorRecommendation,
  type LLMModelInfo,
  type ArchitectureInfo,
  type ResourceConstraints,
} from '../api/platformClient';
import { GPUResourceSelector } from '../components/GPUResourceSelector';
import { OrchestrationRecommendation } from '../components/OrchestrationRecommendation';

// ── Helpers ───────────────────────────────────────────────────────────────────

const LABEL_CLS = 'text-[10px] font-semibold uppercase tracking-wider text-slate-500';
const INPUT_CLS =
  'rounded bg-slate-800 px-2 py-1 text-xs text-slate-100 outline-none focus:ring-1 focus:ring-blue-500 w-full';
const SELECT_CLS =
  'rounded bg-slate-800 px-2 py-1 text-xs text-slate-100 outline-none focus:ring-1 focus:ring-blue-500 w-full';
const BTN_PRIMARY =
  'rounded bg-blue-600 px-4 py-1.5 text-xs font-semibold text-white hover:bg-blue-500 transition-colors disabled:opacity-40';
const BTN_SECONDARY =
  'rounded bg-slate-700 px-4 py-1.5 text-xs font-semibold text-slate-300 hover:bg-slate-600 transition-colors';

const STEPS = ['Model', 'Resources', 'Review', 'Launch'];

const JOB_TYPES = [
  { id: 'training', label: 'Training' },
  { id: 'serving', label: 'Serving' },
  { id: 'both', label: 'Training + Serving' },
] as const;

const MODEL_CATEGORIES = [
  { id: 'tabular', label: 'Tabular (ANN / SSM / BAE / XGBoost)' },
  { id: 'llm', label: 'LLM (fine-tune)' },
  { id: 'upload', label: 'Upload custom model (.py)' },
] as const;

const TABULAR_TYPES = ['ann', 'ssm', 'bae', 'xgboost'] as const;

const RUNTIME_PRESETS = [
  { label: '0.5 h', hours: 0.5 },
  { label: '2 h', hours: 2 },
  { label: '8 h', hours: 8 },
  { label: '24 h', hours: 24 },
];

// ── Default resource constraints ──────────────────────────────────────────────

const DEFAULT_CONSTRAINTS: ResourceConstraints = {
  providers: ['runpod'],
  gpu_types: null,
  min_vram_gb: 0,
  max_price_per_hour: 9999,
  prefer_spot: true,
  require_infiniband: false,
  preferred_regions: [],
  num_nodes: 1,
  num_gpus_per_node: 1,
  job_type: 'tabular',
};

// ── Component ─────────────────────────────────────────────────────────────────

export function LaunchWizardPage() {
  const [step, setStep] = useState(0);

  // ── Step 1 state ─────────────────────────────────────────────────────────
  const [jobType, setJobType] = useState<'training' | 'serving' | 'both'>('training');
  const [modelCategory, setModelCategory] = useState<'tabular' | 'llm' | 'upload'>('tabular');
  const [tabularType, setTabularType] = useState<string>('ann');
  const [llmModel, setLlmModel] = useState<string>('');
  const [llmCatalog, setLlmCatalog] = useState<LLMModelInfo[]>([]);
  const [architectures, setArchitectures] = useState<ArchitectureInfo[]>([]);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadName, setUploadName] = useState('');
  const [uploadDesc, setUploadDesc] = useState('');
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'uploading' | 'ok' | 'error'>('idle');
  const [uploadedArchId, setUploadedArchId] = useState('');
  const [uploadedArchS3, setUploadedArchS3] = useState('');
  const [uploadError, setUploadError] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Tabular hyperparams
  const [hyperparams, setHyperparams] = useState<Record<string, string>>({});
  const [hfToken, setHfToken] = useState('');
  const [datasetS3, setDatasetS3] = useState('');
  const [maxSteps, setMaxSteps] = useState('500');
  const [preprocessRunId, setPreprocessRunId] = useState('');
  const [dataset, setDataset] = useState('');

  // ── Step 2 state ─────────────────────────────────────────────────────────
  const [constraints, setConstraints] = useState<ResourceConstraints>(DEFAULT_CONSTRAINTS);
  const [numNodes, setNumNodes] = useState(1);
  const [useInfiniband, setUseInfiniband] = useState(false);

  // ── Step 3 state ─────────────────────────────────────────────────────────
  const [recommendation, setRecommendation] = useState<OrchestratorRecommendation | null>(null);
  const [yamlPreview, setYamlPreview] = useState('');
  const [runtimeHours, setRuntimeHours] = useState(2);
  const [reviewLoading, setReviewLoading] = useState(false);
  const [reviewError, setReviewError] = useState('');

  // ── Step 4 state ─────────────────────────────────────────────────────────
  const [launching, setLaunching] = useState(false);
  const [launchResult, setLaunchResult] = useState<LaunchResponse | null>(null);
  const [launchError, setLaunchError] = useState('');

  // ── Load catalogs on mount ────────────────────────────────────────────────
  useEffect(() => {
    getLLMCatalog().then(setLlmCatalog).catch(() => {});
    listArchitectures().then(setArchitectures).catch(() => {});
  }, []);

  // Auto-select first LLM when catalog loads
  useEffect(() => {
    if (llmCatalog.length > 0 && !llmModel) {
      setLlmModel(llmCatalog[0].model_id);
    }
  }, [llmCatalog]);

  // ── Derived state ────────────────────────────────────────────────────────
  const resolvedModelType =
    modelCategory === 'tabular' ? tabularType
    : modelCategory === 'llm' ? 'llm'
    : 'user_uploaded';

  const selectedLLM = llmCatalog.find((m) => m.model_id === llmModel);

  // ── Step 3: fetch recommendation on enter ────────────────────────────────
  const fetchRecommendation = useCallback(async () => {
    setReviewLoading(true);
    setReviewError('');
    try {
      const req: LaunchRequest = {
        job_type: jobType,
        model: {
          model_type: resolvedModelType,
          model_id: modelCategory === 'llm' ? llmModel : '',
          architecture_s3: uploadedArchS3,
          vram_gb: selectedLLM?.vram_gb ?? 0,
        },
        resource_constraints: {
          ...constraints,
          num_nodes: numNodes,
          require_infiniband: useInfiniband,
          job_type: modelCategory === 'llm' ? 'llm' : 'tabular',
        },
        training: {
          preprocess_run_id: preprocessRunId,
          dataset,
          dataset_s3_path: datasetS3,
          hf_token: hfToken,
          max_steps: parseInt(maxSteps, 10) || 500,
          num_nodes: numNodes,
        },
        serving: {
          hf_token: hfToken,
          num_nodes: numNodes,
          tensor_parallel_size: constraints.num_gpus_per_node ?? 1,
          pipeline_parallel_size: numNodes,
        },
        dry_run: true,
      };
      const res = await launchJob(req);
      setRecommendation(res.recommendation);
      setYamlPreview(res.sky_yaml_preview);
    } catch (err) {
      setReviewError(String(err));
    } finally {
      setReviewLoading(false);
    }
  }, [
    jobType, resolvedModelType, modelCategory, llmModel, uploadedArchS3, selectedLLM,
    constraints, numNodes, useInfiniband, preprocessRunId, dataset, datasetS3,
    hfToken, maxSteps,
  ]);

  useEffect(() => {
    if (step === 2) fetchRecommendation();
  }, [step]);

  // ── Step 4: launch ───────────────────────────────────────────────────────
  const handleLaunch = async () => {
    setLaunching(true);
    setLaunchError('');
    try {
      const req: LaunchRequest = {
        job_type: jobType,
        model: {
          model_type: resolvedModelType,
          model_id: modelCategory === 'llm' ? llmModel : '',
          architecture_s3: uploadedArchS3,
          vram_gb: selectedLLM?.vram_gb ?? 0,
        },
        resource_constraints: {
          ...constraints,
          num_nodes: numNodes,
          require_infiniband: useInfiniband,
          job_type: modelCategory === 'llm' ? 'llm' : 'tabular',
        },
        training: {
          preprocess_run_id: preprocessRunId,
          dataset,
          dataset_s3_path: datasetS3,
          hf_token: hfToken,
          max_steps: parseInt(maxSteps, 10) || 500,
          num_nodes: numNodes,
        },
        serving: {
          hf_token: hfToken,
          num_nodes: numNodes,
          tensor_parallel_size: constraints.num_gpus_per_node ?? 1,
          pipeline_parallel_size: numNodes,
        },
        dry_run: false,
      };
      const result = await launchJob(req);
      setLaunchResult(result);
    } catch (err) {
      setLaunchError(String(err));
    } finally {
      setLaunching(false);
    }
  };

  // ── Upload handler ───────────────────────────────────────────────────────
  const handleUpload = async () => {
    if (!uploadFile || !uploadName) return;
    setUploadStatus('uploading');
    setUploadError('');
    try {
      const result = await uploadArchitecture(uploadFile, uploadName, uploadDesc);
      setUploadedArchId(result.id);
      setUploadedArchS3(result.s3_path);
      setUploadStatus('ok');
      listArchitectures().then(setArchitectures).catch(() => {});
    } catch (err) {
      setUploadStatus('error');
      setUploadError(String(err));
    }
  };

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="mx-auto max-w-3xl px-4 py-6">
      {/* ── Page title ─────────────────────────────────────────────────────── */}
      <div className="mb-6 flex items-center gap-2">
        <Rocket size={18} className="text-blue-400" />
        <h1 className="text-base font-bold text-slate-100">Launch Wizard</h1>
        <span className="text-xs text-slate-500">— configure and launch a training or serving job</span>
      </div>

      {/* ── Step indicator ─────────────────────────────────────────────────── */}
      <div className="mb-6 flex items-center gap-1">
        {STEPS.map((label, i) => (
          <div key={i} className="flex items-center">
            <button
              type="button"
              onClick={() => { if (i < step) setStep(i); }}
              className={`flex h-6 w-6 items-center justify-center rounded-full text-[10px] font-bold transition-colors ${
                i === step
                  ? 'bg-blue-600 text-white'
                  : i < step
                  ? 'bg-blue-900 text-blue-300 cursor-pointer hover:bg-blue-800'
                  : 'bg-slate-700 text-slate-500 cursor-default'
              }`}
            >
              {i < step ? '✓' : i + 1}
            </button>
            <span
              className={`ml-1.5 text-xs font-medium ${
                i === step ? 'text-slate-100' : i < step ? 'text-blue-400' : 'text-slate-600'
              }`}
            >
              {label}
            </span>
            {i < STEPS.length - 1 && (
              <ChevronRight size={12} className="mx-1.5 text-slate-600" />
            )}
          </div>
        ))}
      </div>

      {/* ═══════════════════════════════════════════════════════════════════════
          STEP 1 — Model Selection
      ════════════════════════════════════════════════════════════════════════ */}
      {step === 0 && (
        <div className="flex flex-col gap-5 rounded border border-slate-700 bg-slate-900/50 p-5">
          {/* Job type */}
          <div className="flex flex-col gap-1.5">
            <span className={LABEL_CLS}>Job type</span>
            <div className="flex gap-2">
              {JOB_TYPES.map(({ id, label }) => (
                <button
                  key={id}
                  type="button"
                  onClick={() => setJobType(id)}
                  className={`rounded px-3 py-1 text-xs font-medium transition-colors ${
                    jobType === id
                      ? 'bg-blue-600 text-white'
                      : 'bg-slate-700 text-slate-400 hover:bg-slate-600'
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>

          {/* Model category */}
          <div className="flex flex-col gap-1.5">
            <span className={LABEL_CLS}>Model category</span>
            <div className="flex flex-col gap-1.5">
              {MODEL_CATEGORIES.map(({ id, label }) => (
                <label key={id} className="flex cursor-pointer items-center gap-2">
                  <input
                    type="radio"
                    name="modelCategory"
                    value={id}
                    checked={modelCategory === id}
                    onChange={() => setModelCategory(id)}
                    className="accent-blue-500"
                  />
                  <span className="text-xs text-slate-300">{label}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Tabular sub-selection */}
          {modelCategory === 'tabular' && (
            <div className="grid grid-cols-2 gap-4">
              <div className="flex flex-col gap-1.5">
                <span className={LABEL_CLS}>Model type</span>
                <select
                  className={SELECT_CLS}
                  value={tabularType}
                  onChange={(e) => setTabularType(e.target.value)}
                >
                  {TABULAR_TYPES.map((t) => (
                    <option key={t} value={t}>{t.toUpperCase()}</option>
                  ))}
                  {architectures
                    .filter((a) => !a.builtin)
                    .map((a) => (
                      <option key={a.id} value={a.id}>{a.name}</option>
                    ))}
                </select>
              </div>
              <div className="flex flex-col gap-1.5">
                <span className={LABEL_CLS}>Preprocess run ID</span>
                <input
                  type="text"
                  className={INPUT_CLS}
                  value={preprocessRunId}
                  placeholder="pre-dataset-xxx"
                  onChange={(e) => setPreprocessRunId(e.target.value)}
                />
              </div>
              <div className="flex flex-col gap-1.5">
                <span className={LABEL_CLS}>Dataset</span>
                <input
                  type="text"
                  className={INPUT_CLS}
                  value={dataset}
                  placeholder="network_traffic"
                  onChange={(e) => setDataset(e.target.value)}
                />
              </div>
            </div>
          )}

          {/* LLM sub-selection */}
          {modelCategory === 'llm' && (
            <div className="grid grid-cols-2 gap-4">
              <div className="flex flex-col gap-1.5">
                <span className={LABEL_CLS}>LLM model</span>
                <select
                  className={SELECT_CLS}
                  value={llmModel}
                  onChange={(e) => setLlmModel(e.target.value)}
                >
                  {llmCatalog.map((m) => (
                    <option key={m.model_id} value={m.model_id}>{m.model_id}</option>
                  ))}
                </select>
              </div>
              {selectedLLM && (
                <div className="flex flex-col justify-end gap-1 text-xs text-slate-400">
                  <span>Min VRAM: <strong className="text-slate-200">{selectedLLM.vram_gb} GB</strong></span>
                  <span>Min GPUs: <strong className="text-slate-200">{selectedLLM.min_gpus}</strong></span>
                  <span>Recommended: <strong className="text-slate-200">{selectedLLM.recommended_gpu}</strong></span>
                </div>
              )}
              <div className="flex flex-col gap-1.5">
                <span className={LABEL_CLS}>HuggingFace token</span>
                <input
                  type="password"
                  className={INPUT_CLS}
                  value={hfToken}
                  placeholder="hf_xxx (optional for Qwen)"
                  onChange={(e) => setHfToken(e.target.value)}
                />
              </div>
              <div className="flex flex-col gap-1.5">
                <span className={LABEL_CLS}>Dataset S3 path (JSONL)</span>
                <input
                  type="text"
                  className={INPUT_CLS}
                  value={datasetS3}
                  placeholder="s3://bucket/llm-data/train.jsonl"
                  onChange={(e) => setDatasetS3(e.target.value)}
                />
              </div>
              <div className="flex flex-col gap-1.5">
                <span className={LABEL_CLS}>Max steps</span>
                <input
                  type="number"
                  className={INPUT_CLS}
                  value={maxSteps}
                  min={1}
                  onChange={(e) => setMaxSteps(e.target.value)}
                />
              </div>
            </div>
          )}

          {/* Custom upload sub-selection */}
          {modelCategory === 'upload' && (
            <div className="flex flex-col gap-3">
              <div className="grid grid-cols-2 gap-4">
                <div className="flex flex-col gap-1.5">
                  <span className={LABEL_CLS}>Architecture name</span>
                  <input
                    type="text"
                    className={INPUT_CLS}
                    value={uploadName}
                    placeholder="MyCustomNet"
                    onChange={(e) => setUploadName(e.target.value)}
                  />
                </div>
                <div className="flex flex-col gap-1.5">
                  <span className={LABEL_CLS}>Description (optional)</span>
                  <input
                    type="text"
                    className={INPUT_CLS}
                    value={uploadDesc}
                    placeholder="Short description"
                    onChange={(e) => setUploadDesc(e.target.value)}
                  />
                </div>
              </div>

              <div className="flex items-center gap-3">
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className="flex items-center gap-1.5 rounded bg-slate-700 px-3 py-1 text-xs text-slate-300 hover:bg-slate-600"
                >
                  <UploadCloud size={12} />
                  {uploadFile ? uploadFile.name : 'Select .py file'}
                </button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".py"
                  className="hidden"
                  onChange={(e) => setUploadFile(e.target.files?.[0] ?? null)}
                />
                <button
                  type="button"
                  disabled={!uploadFile || !uploadName || uploadStatus === 'uploading'}
                  onClick={handleUpload}
                  className={BTN_PRIMARY}
                >
                  {uploadStatus === 'uploading' ? (
                    <Loader2 size={12} className="animate-spin inline mr-1" />
                  ) : null}
                  Validate & Upload
                </button>
              </div>

              {uploadStatus === 'ok' && (
                <div className="flex items-center gap-1.5 text-xs text-green-400">
                  <CheckCircle2 size={12} />
                  Uploaded — ID: {uploadedArchId}
                </div>
              )}
              {uploadStatus === 'error' && (
                <div className="text-xs text-red-400">
                  <XCircle size={12} className="inline mr-1" />
                  {uploadError}
                </div>
              )}
            </div>
          )}

          <div className="flex justify-end pt-2">
            <button
              type="button"
              className={BTN_PRIMARY}
              onClick={() => setStep(1)}
            >
              Next: Resources
            </button>
          </div>
        </div>
      )}

      {/* ═══════════════════════════════════════════════════════════════════════
          STEP 2 — Resource Configuration
      ════════════════════════════════════════════════════════════════════════ */}
      {step === 1 && (
        <div className="flex flex-col gap-5 rounded border border-slate-700 bg-slate-900/50 p-5">
          <GPUResourceSelector
            value={constraints}
            onChange={setConstraints}
          />

          {/* Num nodes + InfiniBand */}
          <div className="grid grid-cols-2 gap-4">
            <div className="flex flex-col gap-1.5">
              <span className={LABEL_CLS}>Nodes</span>
              <div className="flex gap-2">
                {[1, 2, 4, 8].map((n) => (
                  <button
                    key={n}
                    type="button"
                    onClick={() => setNumNodes(n)}
                    className={`rounded px-2.5 py-0.5 text-xs font-medium transition-colors ${
                      numNodes === n
                        ? 'bg-blue-700 text-white'
                        : 'bg-slate-700 text-slate-400 hover:bg-slate-600'
                    }`}
                  >
                    {n === 1 ? 'Single' : `${n}×`}
                  </button>
                ))}
              </div>
            </div>

            <div className="flex flex-col gap-1.5">
              <span className={LABEL_CLS}>Network</span>
              <label className="flex cursor-pointer items-center gap-2 text-xs text-slate-300">
                <input
                  type="checkbox"
                  checked={useInfiniband}
                  onChange={(e) => setUseInfiniband(e.target.checked)}
                  className="accent-blue-500"
                />
                Require InfiniBand / EFA
                {numNodes > 1 && (
                  <span className="text-amber-400 text-[10px]">(recommended for multi-node)</span>
                )}
              </label>
            </div>
          </div>

          <div className="flex justify-between pt-2">
            <button type="button" className={BTN_SECONDARY} onClick={() => setStep(0)}>
              Back
            </button>
            <button type="button" className={BTN_PRIMARY} onClick={() => setStep(2)}>
              Next: Review
            </button>
          </div>
        </div>
      )}

      {/* ═══════════════════════════════════════════════════════════════════════
          STEP 3 — Smart Recommendations
      ════════════════════════════════════════════════════════════════════════ */}
      {step === 2 && (
        <div className="flex flex-col gap-5 rounded border border-slate-700 bg-slate-900/50 p-5">
          <div className="flex items-center justify-between">
            <span className="text-sm font-semibold text-slate-200">Orchestration Recommendation</span>
            <button
              type="button"
              onClick={fetchRecommendation}
              disabled={reviewLoading}
              className="flex items-center gap-1 text-xs text-slate-400 hover:text-slate-200"
            >
              <RefreshCw size={12} className={reviewLoading ? 'animate-spin' : ''} />
              Refresh
            </button>
          </div>

          {/* Runtime presets for cost estimate */}
          <div className="flex flex-col gap-1.5">
            <span className={LABEL_CLS}>Est. runtime</span>
            <div className="flex gap-1.5">
              {RUNTIME_PRESETS.map(({ label, hours }) => (
                <button
                  key={label}
                  type="button"
                  onClick={() => setRuntimeHours(hours)}
                  className={`rounded px-2 py-0.5 text-xs font-medium transition-colors ${
                    runtimeHours === hours
                      ? 'bg-blue-700 text-white'
                      : 'bg-slate-700 text-slate-400 hover:bg-slate-600'
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>

          {reviewLoading && (
            <div className="flex items-center gap-2 text-xs text-slate-500">
              <Loader2 size={14} className="animate-spin" />
              Fetching recommendation…
            </div>
          )}

          {reviewError && (
            <p className="text-xs text-red-400">
              <XCircle size={12} className="inline mr-1" />
              {reviewError}
            </p>
          )}

          {recommendation && !reviewLoading && (
            <OrchestrationRecommendation
              recommendation={recommendation}
              yamlPreview={yamlPreview}
              runtimeHours={runtimeHours}
            />
          )}

          <div className="flex justify-between pt-2">
            <button type="button" className={BTN_SECONDARY} onClick={() => setStep(1)}>
              Back
            </button>
            <button
              type="button"
              className={BTN_PRIMARY}
              disabled={!recommendation}
              onClick={() => setStep(3)}
            >
              Next: Launch
            </button>
          </div>
        </div>
      )}

      {/* ═══════════════════════════════════════════════════════════════════════
          STEP 4 — Launch
      ════════════════════════════════════════════════════════════════════════ */}
      {step === 3 && (
        <div className="flex flex-col gap-5 rounded border border-slate-700 bg-slate-900/50 p-5">
          {/* Summary */}
          <div className="flex flex-col gap-2">
            <span className="text-sm font-semibold text-slate-200">Launch Summary</span>
            <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
              <dt className="text-slate-500">Job type</dt>
              <dd className="text-slate-200 capitalize">{jobType}</dd>
              <dt className="text-slate-500">Model</dt>
              <dd className="text-slate-200">
                {modelCategory === 'llm' ? llmModel : resolvedModelType.toUpperCase()}
              </dd>
              <dt className="text-slate-500">Providers</dt>
              <dd className="text-slate-200">{(constraints.providers ?? ['runpod']).join(', ')}</dd>
              <dt className="text-slate-500">Spot</dt>
              <dd className="text-slate-200">{constraints.prefer_spot !== false ? 'Preferred' : 'Off'}</dd>
              <dt className="text-slate-500">Nodes</dt>
              <dd className="text-slate-200">{numNodes}</dd>
              {recommendation && (
                <>
                  <dt className="text-slate-500">Orchestration</dt>
                  <dd className="text-slate-200">{recommendation.orchestration}</dd>
                </>
              )}
            </dl>
          </div>

          {!launchResult && (
            <button
              type="button"
              disabled={launching}
              onClick={handleLaunch}
              className="flex items-center justify-center gap-2 rounded bg-green-700 px-6 py-2 text-sm font-bold text-white hover:bg-green-600 transition-colors disabled:opacity-40"
            >
              {launching ? (
                <Loader2 size={14} className="animate-spin" />
              ) : (
                <Rocket size={14} />
              )}
              {launching ? 'Launching…' : 'Launch Job'}
            </button>
          )}

          {launchError && (
            <div className="rounded border border-red-700 bg-red-950/30 p-3 text-xs text-red-400">
              <XCircle size={12} className="inline mr-1" />
              {launchError}
            </div>
          )}

          {launchResult && (
            <div className="flex flex-col gap-3 rounded border border-green-700 bg-green-950/20 p-4">
              <div className="flex items-center gap-2 text-sm font-semibold text-green-400">
                <CheckCircle2 size={16} />
                Job launched successfully
              </div>
              {Object.entries(launchResult.job_ids).map(([type, runId]) => (
                <div key={type} className="text-xs text-slate-300">
                  <span className="capitalize text-slate-500">{type}: </span>
                  <span className="font-mono text-blue-300">{runId}</span>
                </div>
              ))}
              <p className="text-xs text-slate-500">
                Monitor progress in the Airflow UI or poll{' '}
                <span className="font-mono text-slate-400">GET /api/v2/jobs/&#123;id&#125;/status</span>
              </p>
            </div>
          )}

          <div className="flex justify-between pt-2">
            <button
              type="button"
              className={BTN_SECONDARY}
              onClick={() => {
                if (launchResult) {
                  // Reset for a new job
                  setLaunchResult(null);
                  setLaunchError('');
                  setStep(0);
                } else {
                  setStep(2);
                }
              }}
            >
              {launchResult ? 'New Job' : 'Back'}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
