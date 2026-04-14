/**
 * LaunchWizardPage — 4-step unified job launch wizard.
 *
 * Layout: 2-column (GPU pricing panel | wizard)
 *
 * Step 1 — Model Selection
 *   Tabular: full config (preprocess run, framework, model config, hyperparams, tuning)
 *            → reuses same logic/defaults as RunPage
 *   LLM:     model picker, HF token, training fields (if training), serving fields (if serving)
 *   Upload:  custom .py architecture upload
 *
 * Step 2 — Resource Configuration: GPUResourceSelector + num_nodes
 *
 * Step 3 — Review
 *   Tabular path: config summary (no dry_run)
 *   LLM path:     orchestration recommendation + YAML preview (dry_run)
 *
 * Step 4 — Launch
 *   Tabular/Upload: POST /api/v2/runs  (same as RunPage)
 *   LLM/Serving:    POST /api/v2/jobs/launch
 *
 * Serving is LLM-only (vLLM via SkyPilot). Tabular serving stays in the Serving tab.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import {
  ChevronRight,
  Rocket,
  UploadCloud,
  RefreshCw,
  CheckCircle2,
  XCircle,
  Loader2,
} from 'lucide-react';

import {
  launchJob,
  submitRun,
  listDatasets,
  getLLMCatalog,
  listArchitectures,
  uploadArchitecture,
  listPreprocessRunIds,
  getPreprocessParams,
  listTrainingRunIds,
  uploadRawSchema,
  submitServingConfig,
  triggerServingDeploy,
  getServingDeployStatus,
  getVllmEndpoint,
  type LaunchRequest,
  type LaunchResponse,
  type OrchestratorRecommendation,
  type LLMModelInfo,
  type DatasetInfo,
  type GPUFallbackEntry,
  type ResourceConstraints,
  type PreprocessRunId,
  type TrainingRunResult,
  type TrainingRunId,
  type SchemaUploadSingleResult,
  type ServingConfigResult,
  type ServingDeployResult,
} from '../api/platformClient';
import { generateRawYamlV2, type RawFieldEntry } from '../lib/schemaYaml';
import { RawYamlEditor } from '../components/schema/RawYamlEditor';
import {
  getDefaultsForTask,
  getTuneSettingsDefaults,
  getParamMetaForTask,
  getTuneSettingsMeta,
  getSearchSpace,
  getNonTunableKeys,
  getFinalTrainMeta,
  type ParamMeta,
  type TaskType,
} from '../types/hyperparams';
import { GPUResourceSelector } from '../components/GPUResourceSelector';
import { OrchestrationRecommendation } from '../components/OrchestrationRecommendation';
import { GPUPricingPanel } from '../components/GPUPricingPanel';
import { ChipInput } from '../components/forms/ChipInput';
import { NumericInput } from '../components/forms/NumericInput';
import { parseFiniteNumber, parseInteger, parseStringChip, stringifyNumberValue } from '../lib/formValues';

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
const BTN_TOGGLE_ON  = 'rounded px-3 py-1 text-xs font-medium bg-blue-600 text-white transition-colors';
const BTN_TOGGLE_OFF = 'rounded px-3 py-1 text-xs font-medium bg-slate-700 text-slate-400 hover:bg-slate-600 transition-colors';

const STEPS = ['Model', 'Resources', 'Review', 'Launch'];

const JOB_TYPES = [
  { id: 'training',  label: 'Training' },
  { id: 'serving',   label: 'Serving' },
  { id: 'both',      label: 'Training + Serving' },
] as const;

const MODEL_CATEGORIES = [
  { id: 'tabular', label: 'Tabular (ANN / SSM / BAE / XGBoost)' },
  { id: 'llm',     label: 'LLM (fine-tune)' },
  { id: 'upload',  label: 'Upload custom model (.py)' },
] as const;

const PYTORCH_MODEL_TYPES = ['mlp', 'ssm', 'bae'] as const;
type SkyLaunchMode = 'launch' | 'jobs_launch';

const RUNTIME_PRESETS = [
  { label: '0.5 h', hours: 0.5 },
  { label: '2 h',   hours: 2   },
  { label: '8 h',   hours: 8   },
  { label: '24 h',  hours: 24  },
];

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

  // ── Job type / category ──────────────────────────────────────────────────
  const [jobType, setJobType] = useState<'training' | 'serving' | 'both'>('training');
  const [modelCategory, setModelCategory] = useState<'tabular' | 'llm' | 'upload'>('tabular');

  const isServing  = jobType === 'serving' || jobType === 'both';
  const isTraining = jobType === 'training' || jobType === 'both';

  // (no auto-lock: tabular + serving is valid — deploys via SkyPilot sky serve)

  // ── Tabular training state ───────────────────────────────────────────────
  const [preprocessRunIds, setPreprocessRunIds] = useState<PreprocessRunId[]>([]);
  const [preprocessLoading, setPreprocessLoading] = useState(false);
  const [datasetFilter, setDatasetFilter] = useState('');
  const [availableDatasets, setAvailableDatasets] = useState<DatasetInfo[]>([]);
  const [preprocessRunId, setPreprocessRunId] = useState('');
  const [preprocessInfo, setPreprocessInfo] = useState<{
    dataset: string;
    dsl_version: number | null;
    schema_version: number | null;
  } | null>(null);

  const [framework, setFramework] = useState<'xgboost' | 'pytorch'>('xgboost');
  const [taskType, setTaskType] = useState<TaskType>('classification');
  const [pytorchModelType, setPytorchModelType] = useState('mlp');
  const [target, setTarget] = useState('');
  const [numClasses, setNumClasses] = useState<number | ''>(2);
  const [seed, setSeed] = useState<number | ''>(42);
  const [experimentName, setExperimentName] = useState('');
  const [registryName, setRegistryName] = useState('');
  const [mlflowTrackingUri, setMlflowTrackingUri] = useState('');
  const [mlflowArtifactLocation, setMlflowArtifactLocation] = useState(
    's3://k8s-mlops-platform-bucket/mlflow-artifacts/',
  );

  const [hyperparams, setHyperparams] = useState<Record<string, number | string | string[]>>(
    () => getDefaultsForTask('xgboost', 'classification'),
  );
  const [tuneEnabled, setTuneEnabled] = useState(true);
  const [numTrials, setNumTrials] = useState<number | ''>(3);
  const [sampleFraction, setSampleFraction] = useState(0.2);
  const [tuneSettings, setTuneSettings] = useState<Record<string, number | ''>>(
    () => getTuneSettingsDefaults('xgboost'),
  );
  const [searchSpaceOverrides, setSearchSpaceOverrides] = useState<
    Record<string, { min?: number; max?: number; options?: number[] }>
  >({});
  const [finalTrainOverrides, setFinalTrainOverrides] = useState<Record<string, number | ''>>({});
  const [showFinalTrainOverrides, setShowFinalTrainOverrides] = useState(true);
  const [showHyperparams, setShowHyperparams] = useState(false);

  const [tabularResult, setTabularResult] = useState<TrainingRunResult | null>(null);
  const [tabularError, setTabularError] = useState('');
  const [tabularLaunching, setTabularLaunching] = useState(false);
  const [skyLaunchMode, setSkyLaunchMode] = useState<SkyLaunchMode>('launch');

  // Reset hyperparams + tuning when framework or task changes
  useEffect(() => {
    setHyperparams(getDefaultsForTask(framework, taskType));
    setTuneSettings(getTuneSettingsDefaults(framework));
    setSearchSpaceOverrides({});
    setFinalTrainOverrides({});
  }, [framework, taskType]);

  // Load preprocess run IDs when category is tabular
  const loadPreprocessIds = useCallback(async () => {
    if (modelCategory !== 'tabular') return;
    setPreprocessLoading(true);
    try {
      const res = await listPreprocessRunIds(datasetFilter || undefined);
      setPreprocessRunIds(res.runs ?? []);
      const firstRun = res.runs?.[0];
      if (!preprocessRunId && firstRun) {
        setPreprocessRunId(firstRun.preprocess_run_id);
      }
    } catch {
      // ignore
    } finally {
      setPreprocessLoading(false);
    }
  }, [modelCategory, datasetFilter]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { loadPreprocessIds(); }, [loadPreprocessIds]);

  // Auto-load preprocessInfo when run ID changes (parse yaml_content same as RunPage)
  useEffect(() => {
    if (!preprocessRunId) { setPreprocessInfo(null); return; }
    getPreprocessParams(preprocessRunId)
      .then(async r => {
        const { parse } = await import('yaml');
        const parsed = parse(r.yaml_content) as {
          execution?: { dsl_s3_path?: string; dsl_version?: number; raw_dataset_name?: string };
          schema_ref?: { version?: number };
          run_metadata?: { dataset?: string; dsl_version?: number };
        };
        setPreprocessInfo({
          dataset:        parsed.run_metadata?.dataset ?? parsed.execution?.raw_dataset_name ?? '',
          dsl_version:    parsed.run_metadata?.dsl_version ?? parsed.execution?.dsl_version ?? null,
          schema_version: parsed.schema_ref?.version ?? null,
        });
      })
      .catch(() => setPreprocessInfo(null));
  }, [preprocessRunId]);

  // Unique datasets from loaded preprocess runs
  const datasetOptions = availableDatasets.length > 0
    ? availableDatasets.map((d) => d.name)
    : [...new Set(preprocessRunIds.map((r) => r.dataset).filter(Boolean))];

  const resolveIntegerDraft = (value: number | string, fallback: number) => {
    const parsed = typeof value === 'number' ? value : parseInteger(value);
    return parsed ?? fallback;
  };

  const resolveNumberDraft = (value: number | string, fallback: number) => {
    const parsed = typeof value === 'number' ? value : parseFiniteNumber(value);
    return parsed ?? fallback;
  };

  // ── LLM / serving state ──────────────────────────────────────────────────
  const [llmModel, setLlmModel] = useState('');
  const [llmCatalog, setLlmCatalog] = useState<LLMModelInfo[]>([]);
  const [hfToken, setHfToken] = useState('');
  const [datasetS3, setDatasetS3] = useState('');
  const [maxSteps, setMaxSteps] = useState('500');
  const [llmAdapterS3, setLlmAdapterS3] = useState('');
  const [maxModelLen, setMaxModelLen] = useState('4096');
  const [servingReplicas, setServingReplicas] = useState(1);

  // ── Tabular serving state ────────────────────────────────────────────────
  const [tabServTrainRunIds, setTabServTrainRunIds] = useState<TrainingRunId[]>([]);
  const [tabServRunIdsLoading, setTabServRunIdsLoading] = useState(false);
  const [tabServTrainRunId, setTabServTrainRunId] = useState('');
  const [tabServDataset, setTabServDataset] = useState('');
  const [tabServMode, setTabServMode] = useState<'ray_only' | 'kafka'>('ray_only');

  const [tabServRawFields, setTabServRawFields] = useState<RawFieldEntry[]>([]);
  const [tabServRawGroups, setTabServRawGroups] = useState<Record<string, string>>({ properties: 'properties' });
  const [tabServIdField, setTabServIdField] = useState('');
  const [tabServSavingSchema, setTabServSavingSchema] = useState(false);
  const [tabServSchemaResult, setTabServSchemaResult] = useState<SchemaUploadSingleResult | null>(null);
  const [tabServSchemaError, setTabServSchemaError] = useState('');
  const [tabServRawSchemaS3, setTabServRawSchemaS3] = useState('');

  const [tabServAlias, setTabServAlias] = useState('champion');
  const [tabServCanary, setTabServCanary] = useState(false);
  const [tabServCanaryAlias, setTabServCanaryAlias] = useState('challenger');
  const [tabServCanaryProb, setTabServCanaryProb] = useState(0.1);
  const [tabServInitReplicas, setTabServInitReplicas] = useState(0);
  const [tabServWebhookUrl, setTabServWebhookUrl] = useState('');
  const [tabServWebhookPath, setTabServWebhookPath] = useState('/infer/webhook');
  const [tabServWebhookMaxAge, setTabServWebhookMaxAge] = useState(300);

  const tabServDeployTarget: 'skypilot' = 'skypilot';
  const [tabServMinReplicas, setTabServMinReplicas] = useState(1);
  const [tabServMaxReplicas, setTabServMaxReplicas] = useState(3);
  const [tabServQps, setTabServQps] = useState(10);
  const [tabServEndpoint, setTabServEndpoint] = useState<string | null>(null);
  const tabServEndpointPollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const [tabServSubmitResult, setTabServSubmitResult] = useState<ServingConfigResult | null>(null);
  const [tabServSubmitError, setTabServSubmitError] = useState('');
  const [tabServDeployResult, setTabServDeployResult] = useState<ServingDeployResult | null>(null);
  const [tabServDeployState, setTabServDeployState] = useState('');
  const [tabServDeployError, setTabServDeployError] = useState('');
  const tabServDeployPollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Upload custom model state ────────────────────────────────────────────
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadName, setUploadName] = useState('');
  const [uploadDesc, setUploadDesc] = useState('');
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'uploading' | 'ok' | 'error'>('idle');
  const [uploadedArchS3, setUploadedArchS3] = useState('');
  const [uploadedArchId, setUploadedArchId] = useState('');
  const [uploadError, setUploadError] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  // ── Step 2 state ─────────────────────────────────────────────────────────
  const [constraints, setConstraints] = useState<ResourceConstraints>(DEFAULT_CONSTRAINTS);
  const [numNodes, setNumNodes] = useState(1);
  const [useInfiniband, setUseInfiniband] = useState(false);

  // Append a GPU entry from the pricing panel to the manual fallback list.
  // Only wired up when step === 1 and constraints.gpu_fallbacks is an array (manual mode).
  const handleAddFromPricing = useCallback((entry: GPUFallbackEntry) => {
    setConstraints(prev => ({
      ...prev,
      gpu_fallbacks: [...(prev.gpu_fallbacks ?? []), entry],
    }));
  }, []);

  // ── Step 3 state (LLM path) ──────────────────────────────────────────────
  const [recommendation, setRecommendation] = useState<OrchestratorRecommendation | null>(null);
  const [yamlPreview, setYamlPreview] = useState('');
  const [runtimeHours, setRuntimeHours] = useState(2);
  const [reviewLoading, setReviewLoading] = useState(false);
  const [reviewError, setReviewError] = useState('');

  // ── Step 4 state (LLM path) ──────────────────────────────────────────────
  const [launching, setLaunching] = useState(false);
  const [launchResult, setLaunchResult] = useState<LaunchResponse | null>(null);
  const [launchError, setLaunchError] = useState('');

  // ── DeepSpeed ─────────────────────────────────────────────────────────────
  const [useDeepspeed, setUseDeepspeed] = useState(false);
  useEffect(() => {
    setUseDeepspeed(modelCategory === 'llm');
  }, [modelCategory]);

  // ── Load catalogs on mount ────────────────────────────────────────────────
  useEffect(() => {
    getLLMCatalog().then(setLlmCatalog).catch(() => {});
    listArchitectures().catch(() => {});
    listDatasets()
      .then((datasets) => setAvailableDatasets(datasets))
      .catch(() => setAvailableDatasets([]));
  }, []);

  // ── Cleanup poll refs on unmount ──────────────────────────────────────────
  useEffect(() => {
    return () => {
      if (tabServDeployPollRef.current) clearInterval(tabServDeployPollRef.current);
      if (tabServEndpointPollRef.current) clearInterval(tabServEndpointPollRef.current);
    };
  }, []);

  useEffect(() => {
    const firstModel = llmCatalog[0];
    if (firstModel && !llmModel) setLlmModel(firstModel.model_id);
  }, [llmCatalog]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Derived ──────────────────────────────────────────────────────────────
  const selectedLLM = llmCatalog.find(m => m.model_id === llmModel);
  const isTabularServingPath = modelCategory === 'tabular' && isServing;
  const isLlmPath = modelCategory === 'llm' && !isTabularServingPath;
  const isTabularPath = (modelCategory === 'tabular' || modelCategory === 'upload') && !isTabularServingPath;

  const gpuSummary = (() => {
    const gpusPerNode = constraints.num_gpus_per_node ?? 1;
    const nodes = numNodes;
    const totalGpus = nodes * gpusPerNode;
    const isMulti = nodes > 1;
    return {
      totalGpus,
      numWorkers: totalGpus,
      numWorkersTune: isMulti ? gpusPerNode : 1,
      maxConcurrentTrials: isMulti ? nodes : totalGpus,
      mode: isMulti ? 'multi-node' : 'single-node',
    };
  })();

  const resolvedModelType =
    modelCategory === 'tabular' ? (framework === 'pytorch' ? pytorchModelType : 'xgboost')
    : modelCategory === 'llm'   ? 'llm'
    : 'user_uploaded';

  // ── Tabular serving: load training run IDs ──────────────────────────────
  const loadTabServTrainRunIds = useCallback(async () => {
    if (!isTabularServingPath) return;
    setTabServRunIdsLoading(true);
    try {
      const res = await listTrainingRunIds();
      setTabServTrainRunIds(res.runs ?? []);
      if (!tabServTrainRunId && res.runs?.[0]) {
        setTabServTrainRunId(res.runs[0].train_run_id);
      }
    } catch { /* ignore */ } finally {
      setTabServRunIdsLoading(false);
    }
  }, [isTabularServingPath]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { loadTabServTrainRunIds(); }, [loadTabServTrainRunIds]);

  // Resolve dataset from selected training run ID
  useEffect(() => {
    if (!tabServTrainRunId) { setTabServDataset(''); return; }
    const entry = tabServTrainRunIds.find(r => r.train_run_id === tabServTrainRunId);
    if (entry?.dataset) { setTabServDataset(entry.dataset); return; }
    const m = tabServTrainRunId.match(/^train-(.+)-[0-9a-f]{6}-\d{8}T\d{6}Z-[0-9a-f]{6}$/);
    setTabServDataset(m?.[1] ?? '');
  }, [tabServTrainRunId, tabServTrainRunIds]);

  // Reset schema save state when fields/dataset change
  useEffect(() => {
    setTabServSchemaResult(null);
    setTabServSchemaError('');
    setTabServRawSchemaS3('');
  }, [tabServDataset, tabServRawFields, tabServRawGroups, tabServIdField]);

  // ── Tabular serving raw.yaml content ────────────────────────────────────
  const tabServRawYamlContent = tabServRawFields.length > 0
    ? generateRawYamlV2({
        dataset: tabServDataset,
        dslName: '', dslVersion: '', processedTableName: '', generatedFrom: '',
        allColumns: [], rawFields: tabServRawFields, rawGroups: tabServRawGroups,
        idField: tabServIdField, fullFields: [], preprocessedFields: [],
        rawTopLevel: [], propertiesFields: [], fullColumns: [], preprocessedColumns: [],
        targetColumn: '', idColumn: '', typeOverrides: {},
      })
    : '# Add fields to generate raw.yaml';

  // ── Tabular serving handlers ─────────────────────────────────────────────
  const handleTabServSaveSchema = async () => {
    if (!tabServDataset) return;
    setTabServSavingSchema(true);
    setTabServSchemaError('');
    setTabServSchemaResult(null);
    setTabServRawSchemaS3('');
    try {
      const result = await uploadRawSchema(tabServDataset, tabServRawYamlContent);
      setTabServSchemaResult(result);
      setTabServRawSchemaS3(result.s3_path);
    } catch (e) {
      setTabServSchemaError(e instanceof Error ? e.message : String(e));
    } finally {
      setTabServSavingSchema(false);
    }
  };

  const handleTabServSubmit = async () => {
    setTabServSubmitError('');
    setTabServSubmitResult(null);
    try {
      const result = await submitServingConfig({
        train_run_id: tabServTrainRunId,
        serving_mode: tabServMode,
        ...(tabServMode === 'kafka' ? { raw_schema_s3_path: tabServRawSchemaS3 } : {}),
        alias: tabServAlias,
        canary: tabServCanary,
        canary_alias: tabServCanaryAlias,
        canary_probability: tabServCanaryProb,
        initial_replicas: tabServInitReplicas,
        webhook_public_base_url: tabServWebhookUrl,
        webhook_path: tabServWebhookPath,
        webhook_max_timestamp_age_seconds: tabServWebhookMaxAge,
        deployment_target: tabServDeployTarget,
        min_replicas: tabServMinReplicas,
        max_replicas: tabServMaxReplicas,
        target_qps_per_replica: tabServQps,
        resource_constraints: {
          ...constraints,
          num_nodes: numNodes,
          job_type: 'tabular',
        },
      });
      setTabServSubmitResult(result);
    } catch (e) {
      setTabServSubmitError(e instanceof Error ? e.message : String(e));
    }
  };

  const handleTabServDeploy = async () => {
    if (!tabServSubmitResult) return;
    setTabServDeployError('');
    setTabServDeployResult(null);
    setTabServDeployState('queued');
    setTabServEndpoint(null);
    if (tabServDeployPollRef.current) clearInterval(tabServDeployPollRef.current);
    if (tabServEndpointPollRef.current) clearInterval(tabServEndpointPollRef.current);
    try {
      const result = await triggerServingDeploy(tabServSubmitResult.serve_run_id);
      setTabServDeployResult(result);
      tabServDeployPollRef.current = setInterval(() => {
        void getServingDeployStatus(tabServSubmitResult.serve_run_id, result.dag_run_id, result.dag_id)
          .then(s => {
            setTabServDeployState(s.state);
            if (['success', 'failed', 'upstream_failed'].includes(s.state.toLowerCase())) {
              if (tabServDeployPollRef.current) clearInterval(tabServDeployPollRef.current);
            }
          })
          .catch(() => { /* keep polling */ });
      }, 15_000);
      tabServEndpointPollRef.current = setInterval(() => {
        void getVllmEndpoint(tabServSubmitResult.serve_run_id)
          .then(ep => {
            if (ep.status === 'healthy' && ep.endpoint_url) {
              setTabServEndpoint(ep.endpoint_url);
              if (tabServEndpointPollRef.current) clearInterval(tabServEndpointPollRef.current);
            }
          })
          .catch(() => { /* keep polling */ });
      }, 20_000);
    } catch (e) {
      setTabServDeployError(e instanceof Error ? e.message : String(e));
      setTabServDeployState('');
    }
  };

  // ── Step 3: fetch recommendation (LLM path only) ─────────────────────────
  const fetchRecommendation = useCallback(async () => {
    if (isTabularPath) return;
    setReviewLoading(true);
    setReviewError('');
    try {
      const req: LaunchRequest = {
        job_type: jobType,
        model: {
          model_type: resolvedModelType,
          model_id:   llmModel,
          architecture_s3: uploadedArchS3,
          vram_gb:    selectedLLM?.vram_gb ?? 0,
        },
        resource_constraints: {
          ...constraints,
          num_nodes:          numNodes,
          require_infiniband: useInfiniband,
          job_type:           'llm',
        },
        training: {
          dataset_s3_path: datasetS3,
          hf_token:        hfToken,
          max_steps:       parseInt(maxSteps, 10) || 500,
          num_nodes:       numNodes,
          use_deepspeed:   useDeepspeed,
          deepspeed_stage: useDeepspeed ? 3 : 1,
        },
        serving: {
          hf_token:               hfToken,
          llm_adapter_s3:         llmAdapterS3,
          max_model_len:          parseInt(maxModelLen) || 4096,
          num_nodes:              numNodes,
          tensor_parallel_size:   constraints.num_gpus_per_node ?? 1,
          pipeline_parallel_size: numNodes,
          replicas:               servingReplicas,
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
    isTabularPath, jobType, resolvedModelType, llmModel, uploadedArchS3, selectedLLM,
    constraints, numNodes, useInfiniband, datasetS3, hfToken, maxSteps,
    llmAdapterS3, maxModelLen, servingReplicas,
  ]);

  useEffect(() => {
    if (step === 2) fetchRecommendation();
  }, [step]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Launch handlers ───────────────────────────────────────────────────────

  const handleLaunchTabular = async () => {
    setTabularLaunching(true);
    setTabularError('');
    try {
      const useGpu = (constraints.providers ?? []).some(p =>
        ['runpod', 'vast', 'aws', 'gcp', 'azure'].includes(p),
      );
      const resolvedNumClasses = resolveIntegerDraft(numClasses, 2);
      const resolvedSeed = resolveIntegerDraft(seed, 42);
      const resolvedNumTrials = resolveIntegerDraft(numTrials, 3);
      const resolvedTuneSettings = Object.fromEntries(
        Object.entries(tuneSettings).map(([key, value]) => [
          key,
          resolveNumberDraft(value, getTuneSettingsDefaults(framework)[key] ?? 0),
        ]),
      ) as Record<string, number>;
      const resolvedFinalTrainOverrides = Object.fromEntries(
        Object.entries(finalTrainOverrides)
          .map(([key, value]) => [key, resolveNumberDraft(value, 0)])
          .filter(([, value]) => Number.isFinite(value as number)),
      ) as Record<string, number>;
      const result = await submitRun({
        preprocess_run_id: preprocessRunId,
        framework,
        use_gpu: useGpu,
        use_managed_jobs: skyLaunchMode === 'jobs_launch',
        resource_constraints: {
          ...constraints,
          num_nodes:          numNodes,
          require_infiniband: useInfiniband,
          job_type:           'tabular',
        },
        model: {
          experiment_name:    experimentName || `launch_${framework}`,
          registry_model_name: registryName || framework,
          mlflow_tracking_uri: mlflowTrackingUri,
          mlflow_artifact_location: mlflowArtifactLocation,
          target,
          num_classes:        resolvedNumClasses,
          seed:               resolvedSeed,
          task_type:          taskType,
          model_type:         framework === 'pytorch' ? pytorchModelType : undefined,
        },
        tuning: { enabled: tuneEnabled, number_of_trials: resolvedNumTrials },
        sample_fraction_for_tuning: sampleFraction,
        hyperparams: tuneEnabled ? resolvedFinalTrainOverrides : hyperparams,
        tune_settings: tuneEnabled ? resolvedTuneSettings : {},
        search_space: tuneEnabled ? (() => {
          const baseSpace = getSearchSpace(framework);
          const resolved: Record<string, { type: string; options?: number[]; min?: number; max?: number; value?: unknown }> = {};
          for (const [k, entry] of Object.entries(baseSpace)) {
            const ov = searchSpaceOverrides[k];
            type SSVal = { type: string; options?: number[]; min?: number; max?: number; value?: unknown };
            if (!ov) {
              resolved[k] = entry as SSVal;
            } else if (entry.type === 'choice' && ov.options && ov.options.length > 0) {
              resolved[k] = { type: 'choice', options: ov.options };
            } else if (entry.type !== 'choice' && (ov.min !== undefined || ov.max !== undefined)) {
              resolved[k] = {
                type: entry.type,
                min: ov.min ?? (entry as { min: number }).min,
                max: ov.max ?? (entry as { max: number }).max,
              };
            } else {
              resolved[k] = entry as SSVal;
            }
          }
          return resolved;
        })() : undefined,
      });
      setTabularResult(result);
    } catch (err) {
      setTabularError(String(err));
    } finally {
      setTabularLaunching(false);
    }
  };

  const handleLaunchLlm = async () => {
    setLaunching(true);
    setLaunchError('');
    try {
      const req: LaunchRequest = {
        job_type: jobType,
        model: {
          model_type:      resolvedModelType,
          model_id:        llmModel,
          architecture_s3: uploadedArchS3,
          vram_gb:         selectedLLM?.vram_gb ?? 0,
        },
        resource_constraints: {
          ...constraints,
          num_nodes:          numNodes,
          require_infiniband: useInfiniband,
          job_type:           'llm',
        },
        training: {
          dataset_s3_path: datasetS3,
          hf_token:        hfToken,
          max_steps:       parseInt(maxSteps, 10) || 500,
          num_nodes:       numNodes,
          use_deepspeed:   useDeepspeed,
          deepspeed_stage: useDeepspeed ? 3 : 1,
        },
        serving: {
          hf_token:               hfToken,
          llm_adapter_s3:         llmAdapterS3,
          max_model_len:          parseInt(maxModelLen) || 4096,
          num_nodes:              numNodes,
          tensor_parallel_size:   constraints.num_gpus_per_node ?? 1,
          pipeline_parallel_size: numNodes,
          replicas:               servingReplicas,
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

  // ── Upload handler ────────────────────────────────────────────────────────
  const handleUpload = async () => {
    if (!uploadFile || !uploadName) return;
    setUploadStatus('uploading');
    setUploadError('');
    try {
      const result = await uploadArchitecture(uploadFile, uploadName, uploadDesc);
      setUploadedArchId(result.id);
      setUploadedArchS3(result.s3_path);
      setUploadStatus('ok');
      listArchitectures().catch(() => {});
    } catch (err) {
      setUploadStatus('error');
      setUploadError(String(err));
    }
  };

  // ── Reset ─────────────────────────────────────────────────────────────────
  const resetForNewJob = () => {
    setTabularResult(null);
    setTabularError('');
    setLaunchResult(null);
    setLaunchError('');
    setRecommendation(null);
    setTabServSubmitResult(null);
    setTabServSubmitError('');
    setTabServDeployResult(null);
    setTabServDeployState('');
    setTabServDeployError('');
    setTabServEndpoint(null);
    if (tabServDeployPollRef.current) clearInterval(tabServDeployPollRef.current);
    if (tabServEndpointPollRef.current) clearInterval(tabServEndpointPollRef.current);
    setStep(0);
  };

  // ── Param grid helper (tabular hyperparams) ───────────────────────────────
  const paramMeta = getParamMetaForTask(framework, taskType);
  const tuneSettingsMeta = getTuneSettingsMeta(framework);
  const nonTunableKeys = getNonTunableKeys(framework);
  const searchSpaceSummary = Object.entries(getSearchSpace(framework)).map(([key, entry]) => {
    const override = searchSpaceOverrides[key];
    if (entry.type === 'choice') {
      const options = (override?.options ?? entry.options) as Array<string | number>;
      return [key, `choice: [${options.join(', ')}]`] as [string, string];
    }
    if (entry.type === 'fixed') {
      const value = Array.isArray(entry.value) ? entry.value.join(', ') : String(entry.value);
      return [key, `fixed: ${value}`] as [string, string];
    }
    const min = override?.min ?? entry.min;
    const max = override?.max ?? entry.max;
    return [key, `${entry.type}: min=${min}, max=${max}`] as [string, string];
  });

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="px-4 py-6">
      {/* Page title */}
      <div className="mb-6 flex items-center gap-2">
        <Rocket size={18} className="text-blue-400" />
        <h1 className="text-base font-bold text-slate-100">Launch Wizard</h1>
        <span className="text-xs text-slate-500">— configure and launch a training or serving job</span>
      </div>

      {/* 2-column layout */}
      <div className="grid grid-cols-[minmax(0,1fr)_minmax(0,1.6fr)] gap-6 items-start">

        {/* ── Left: GPU Pricing Panel ──────────────────────────────────────── */}
        <GPUPricingPanel
          onAddToFallback={
            step === 1 && Array.isArray(constraints.gpu_fallbacks)
              ? handleAddFromPricing
              : undefined
          }
        />

        {/* ── Right: Wizard ────────────────────────────────────────────────── */}
        <div className="flex flex-col gap-4">

          {/* Step indicator */}
          <div className="flex items-center gap-1">
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
                <span className={`ml-1.5 text-xs font-medium ${
                  i === step ? 'text-slate-100' : i < step ? 'text-blue-400' : 'text-slate-600'
                }`}>
                  {label}
                </span>
                {i < STEPS.length - 1 && <ChevronRight size={12} className="mx-1.5 text-slate-600" />}
              </div>
            ))}
          </div>

          {/* ═══════════════════════════════════════════════════════════════
              STEP 1 — Model Selection
          ══════════════════════════════════════════════════════════════════ */}
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
                      className={jobType === id ? BTN_TOGGLE_ON : BTN_TOGGLE_OFF}
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
                    <div key={id} className="flex flex-col gap-0.5">
                      <label className="flex cursor-pointer items-center gap-2">
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
                      {id === 'tabular' && modelCategory === 'tabular' && isServing && (
                        <p className="ml-5 text-[10px] text-slate-500 italic">
                          SkyPilot sky serve deployment on external VMs.
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              </div>

              {/* ── Tabular config ──────────────────────────────────────────── */}
              {modelCategory === 'tabular' && isTraining && (
                <div className="flex flex-col gap-4">

                  {/* Section A: Preprocessing run */}
                  <div className="flex flex-col gap-2 rounded border border-slate-700/60 p-3">
                    <span className={LABEL_CLS}>Preprocessing run</span>

                    {/* Dataset filter */}
                    <div className="grid grid-cols-2 gap-3">
                      <div className="flex flex-col gap-1">
                        <span className="text-[10px] text-slate-500">Dataset filter</span>
                        <select
                          className={SELECT_CLS}
                          value={datasetFilter}
                          onChange={e => setDatasetFilter(e.target.value)}
                        >
                          <option value="">All datasets</option>
                          {datasetOptions.map(d => (
                            <option key={d} value={d}>{d}</option>
                          ))}
                        </select>
                      </div>

                      {/* Run ID */}
                      <div className="flex flex-col gap-1">
                        <div className="flex items-center justify-between">
                          <span className="text-[10px] text-slate-500">Run ID</span>
                          <button
                            type="button"
                            onClick={loadPreprocessIds}
                            disabled={preprocessLoading}
                            className="text-[10px] text-slate-500 hover:text-slate-300 flex items-center gap-0.5"
                          >
                            <RefreshCw size={9} className={preprocessLoading ? 'animate-spin' : ''} />
                          </button>
                        </div>
                        <select
                          className={SELECT_CLS}
                          value={preprocessRunId}
                          onChange={e => setPreprocessRunId(e.target.value)}
                        >
                          <option value="">Select run…</option>
                          {preprocessRunIds.map(r => (
                            <option key={r.preprocess_run_id} value={r.preprocess_run_id}>
                              {r.preprocess_run_id}
                            </option>
                          ))}
                        </select>
                      </div>
                    </div>

                    {/* Auto-loaded info */}
                    {preprocessInfo && (
                      <div className="flex gap-4 text-[10px] text-slate-400">
                        <span>Dataset: <strong className="text-slate-200">{preprocessInfo.dataset}</strong></span>
                        {preprocessInfo.dsl_version != null && (
                          <span>DSL: <strong className="text-slate-200">v{preprocessInfo.dsl_version}</strong></span>
                        )}
                        {preprocessInfo.schema_version != null && (
                          <span>Schema: <strong className="text-slate-200">v{preprocessInfo.schema_version}</strong></span>
                        )}
                      </div>
                    )}
                  </div>

                  {/* Section B: Framework & Task */}
                  <div className="flex flex-col gap-2 rounded border border-slate-700/60 p-3">
                    <span className={LABEL_CLS}>Framework & task</span>
                    <div className="grid grid-cols-2 gap-3">
                      <div className="flex flex-col gap-1">
                        <span className="text-[10px] text-slate-500">Framework</span>
                        <div className="flex gap-1.5">
                          {(['xgboost', 'pytorch'] as const).map(f => (
                            <button
                              key={f}
                              type="button"
                              onClick={() => setFramework(f)}
                              className={framework === f ? BTN_TOGGLE_ON : BTN_TOGGLE_OFF}
                            >
                              {f === 'xgboost' ? 'XGBoost' : 'PyTorch'}
                            </button>
                          ))}
                        </div>
                      </div>
                      <div className="flex flex-col gap-1">
                        <span className="text-[10px] text-slate-500">Task type</span>
                        <div className="flex gap-1.5">
                          {(['classification', 'regression'] as const).map(t => (
                            <button
                              key={t}
                              type="button"
                              onClick={() => setTaskType(t)}
                              className={taskType === t ? BTN_TOGGLE_ON : BTN_TOGGLE_OFF}
                            >
                              {t.charAt(0).toUpperCase() + t.slice(1)}
                            </button>
                          ))}
                        </div>
                      </div>
                    </div>
                    {framework === 'pytorch' && (
                      <div className="flex flex-col gap-1">
                        <span className="text-[10px] text-slate-500">Model architecture</span>
                        <div className="flex gap-1.5">
                          {PYTORCH_MODEL_TYPES.map(m => (
                            <button
                              key={m}
                              type="button"
                              onClick={() => setPytorchModelType(m)}
                              className={pytorchModelType === m ? BTN_TOGGLE_ON : BTN_TOGGLE_OFF}
                            >
                              {m.toUpperCase()}
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Section C: Model config */}
                  <div className="flex flex-col gap-2 rounded border border-slate-700/60 p-3">
                    <span className={LABEL_CLS}>Model config</span>
                    <div className="grid grid-cols-2 gap-3">
                      <div className="flex flex-col gap-1">
                        <span className="text-[10px] text-slate-500">Target column</span>
                        <input
                          type="text"
                          className={INPUT_CLS}
                          value={target}
                          placeholder="e.g. attack"
                          onChange={e => setTarget(e.target.value)}
                        />
                      </div>
                      {taskType === 'classification' && (
                        <div className="flex flex-col gap-1">
                          <span className="text-[10px] text-slate-500">Num classes</span>
                          <NumericInput
                            className={INPUT_CLS}
                            value={stringifyNumberValue(numClasses)}
                            min={2}
                            onChange={next => {
                              const parsed = parseInteger(next);
                              setNumClasses(next === '' ? '' : parsed ?? numClasses);
                            }}
                          />
                        </div>
                      )}
                      <div className="flex flex-col gap-1">
                        <span className="text-[10px] text-slate-500">Experiment name</span>
                        <input
                          type="text"
                          className={INPUT_CLS}
                          value={experimentName}
                          placeholder={`launch_${framework}`}
                          onChange={e => setExperimentName(e.target.value)}
                        />
                      </div>
                      <div className="flex flex-col gap-1">
                        <span className="text-[10px] text-slate-500">Registry model name</span>
                        <input
                          type="text"
                          className={INPUT_CLS}
                          value={registryName}
                          placeholder={framework}
                          onChange={e => setRegistryName(e.target.value)}
                        />
                      </div>
                      <div className="flex flex-col gap-1">
                        <span className="text-[10px] text-slate-500">Seed</span>
                        <NumericInput
                          className={INPUT_CLS}
                          value={stringifyNumberValue(seed)}
                          onChange={next => {
                            const parsed = parseInteger(next);
                            setSeed(next === '' ? '' : parsed ?? seed);
                          }}
                        />
                      </div>
                      <div className="flex flex-col gap-1">
                        <span className="text-[10px] text-slate-500">MLflow tracking URI</span>
                        <input
                          type="text"
                          className={INPUT_CLS}
                          value={mlflowTrackingUri}
                          placeholder="Default: from Airflow MLFLOW_TRACKING_URI"
                          onChange={e => setMlflowTrackingUri(e.target.value)}
                        />
                      </div>
                      <div className="flex flex-col gap-1">
                        <span className="text-[10px] text-slate-500">MLflow artifact location</span>
                        <input
                          type="text"
                          className={INPUT_CLS}
                          value={mlflowArtifactLocation}
                          onChange={e => setMlflowArtifactLocation(e.target.value)}
                        />
                      </div>
                      <div className="flex flex-col gap-1">
                        <span className="text-[10px] text-slate-500">DeepSpeed ZeRO</span>
                        <div className="flex gap-1.5">
                          <button type="button" onClick={() => setUseDeepspeed(true)}
                            className={useDeepspeed ? BTN_TOGGLE_ON : BTN_TOGGLE_OFF}>
                            Enabled
                          </button>
                          <button type="button" onClick={() => setUseDeepspeed(false)}
                            className={!useDeepspeed ? BTN_TOGGLE_ON : BTN_TOGGLE_OFF}>
                            Disabled
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Section D: Hyperparams / Tuning (collapsible) */}
                  <div className="flex flex-col gap-2 rounded border border-slate-700/60 p-3">
                    <button
                      type="button"
                      className="flex items-center justify-between w-full"
                      onClick={() => setShowHyperparams(v => !v)}
                    >
                      <span className={LABEL_CLS}>Hyperparameters & tuning</span>
                      <span className="text-[10px] text-slate-500">{showHyperparams ? '▲' : '▼'}</span>
                    </button>

                    {showHyperparams && (
                      <div className="flex flex-col gap-3 pt-1">
                        {/* Tuning toggle */}
                        <div className="flex items-center gap-3">
                          <span className="text-[10px] text-slate-500">Ray Tune</span>
                          <div className="flex gap-1.5">
                            <button
                              type="button"
                              onClick={() => setTuneEnabled(true)}
                              className={tuneEnabled ? BTN_TOGGLE_ON : BTN_TOGGLE_OFF}
                            >
                              Enabled
                            </button>
                            <button
                              type="button"
                              onClick={() => setTuneEnabled(false)}
                              className={!tuneEnabled ? BTN_TOGGLE_ON : BTN_TOGGLE_OFF}
                            >
                              Disabled
                            </button>
                          </div>
                        </div>

                        {tuneEnabled ? (
                          <div className="flex flex-col gap-4">

                            {/* ── A: Search Space ──────────────────────────── */}
                            <div className="flex flex-col gap-2">
                              <span className={LABEL_CLS}>Search Space</span>
                              <div className="flex flex-col gap-2">
                                {Object.entries(getSearchSpace(framework)).map(([k, entry]) => {
                                  if (entry.type === 'fixed') {
                                    return (
                                      <div key={k} className="flex items-center justify-between">
                                        <span className="text-[10px] text-slate-500">{k}</span>
                                        <span className="text-[10px] text-slate-400 bg-slate-800 rounded px-2 py-0.5">
                                          {Array.isArray(entry.value) ? entry.value.join(', ') : String(entry.value)}
                                        </span>
                                      </div>
                                    );
                                  }
                                  if (entry.type === 'choice') {
                                    const ov = searchSpaceOverrides[k];
                                    const options = (ov?.options ?? entry.options) as number[];
                                    const hasFinalOverride = k in finalTrainOverrides;
                                    return (
                                      <div key={k} className="flex flex-col gap-1">
                                        <div className="flex items-center gap-1.5">
                                          <span className="text-[10px] text-slate-500">{k}</span>
                                          {hasFinalOverride && (
                                            <span className="text-[9px] bg-amber-900/50 text-amber-400 rounded px-1.5 py-px">override</span>
                                          )}
                                        </div>
                                        <ChipInput
                                          value={options}
                                          onChange={(next) => setSearchSpaceOverrides(prev => ({
                                            ...prev,
                                            [k]: { ...prev[k], options: next as number[] },
                                          }))}
                                          parseItem={(raw) => {
                                            const parsed = parseFiniteNumber(raw);
                                            return parsed === null ? null : parsed;
                                          }}
                                          formatItem={(item) => String(item)}
                                          placeholder="Type a value and press Enter"
                                          className="space-y-1"
                                        />
                                      </div>
                                    );
                                  }
                                  // randint | uniform | loguniform
                                  const ov = searchSpaceOverrides[k];
                                  const rangeEntry = entry as { type: string; min: number; max: number };
                                  const hasFinalOverride = k in finalTrainOverrides;
                                  return (
                                    <div key={k} className="flex flex-col gap-1">
                                      <div className="flex items-center gap-1.5">
                                        <span className="text-[10px] text-slate-500">{k}</span>
                                        <span className="text-[9px] text-slate-600">{entry.type}</span>
                                        {hasFinalOverride && (
                                          <span className="text-[9px] bg-amber-900/50 text-amber-400 rounded px-1.5 py-px">override</span>
                                        )}
                                      </div>
                                      <div className="grid grid-cols-2 gap-2">
                                        <div className="flex flex-col gap-0.5">
                                          <span className="text-[9px] text-slate-600">Min</span>
                                          <NumericInput
                                            className={INPUT_CLS}
                                            value={stringifyNumberValue(ov?.min ?? rangeEntry.min)}
                                            step="any"
                                            onChange={next => {
                                              const parsed = parseFiniteNumber(next);
                                              setSearchSpaceOverrides(prev => ({
                                                ...prev,
                                                [k]: {
                                                  ...prev[k],
                                                  min: next === '' ? undefined : (parsed ?? prev[k]?.min ?? rangeEntry.min),
                                                },
                                              }));
                                            }}
                                          />
                                        </div>
                                        <div className="flex flex-col gap-0.5">
                                          <span className="text-[9px] text-slate-600">Max</span>
                                          <NumericInput
                                            className={INPUT_CLS}
                                            value={stringifyNumberValue(ov?.max ?? rangeEntry.max)}
                                            step="any"
                                            onChange={next => {
                                              const parsed = parseFiniteNumber(next);
                                              setSearchSpaceOverrides(prev => ({
                                                ...prev,
                                                [k]: {
                                                  ...prev[k],
                                                  max: next === '' ? undefined : (parsed ?? prev[k]?.max ?? rangeEntry.max),
                                                },
                                              }));
                                            }}
                                          />
                                        </div>
                                      </div>
                                    </div>
                                  );
                                })}
                              </div>
                            </div>

                            {/* ── B: Scheduler ─────────────────────────────── */}
                            <div className="flex flex-col gap-2">
                              <span className={LABEL_CLS}>Scheduler</span>
                              <div className="grid grid-cols-2 gap-3">
                                <div className="flex flex-col gap-1">
                                  <span className="text-[10px] text-slate-500">Trials</span>
                                  <NumericInput
                                    className={INPUT_CLS}
                                    value={stringifyNumberValue(numTrials)}
                                    min={1}
                                    onChange={next => {
                                      const parsed = parseInteger(next);
                                      setNumTrials(next === '' ? '' : parsed ?? numTrials);
                                    }}
                                  />
                                </div>
                                <div className="flex flex-col gap-1">
                                  <span className="text-[10px] text-slate-500">Sample fraction</span>
                                  <NumericInput
                                    className={INPUT_CLS}
                                    value={stringifyNumberValue(sampleFraction)}
                                    min={0.01}
                                    max={1}
                                    step={0.01}
                                    onChange={next => {
                                      const parsed = parseFiniteNumber(next);
                                      if (parsed !== null) setSampleFraction(parsed);
                                    }}
                                  />
                                </div>
                                {Object.entries(tuneSettingsMeta).map(([k, meta]) => (
                                  <div key={k} className="flex flex-col gap-1">
                                    <span className="text-[10px] text-slate-500">{k}</span>
                                    <NumericInput
                                      className={INPUT_CLS}
                                      value={stringifyNumberValue(tuneSettings[k] ?? (meta.defaultValue as number))}
                                      min={meta.min}
                                      step={meta.step ?? 1}
                                      onChange={next => setTuneSettings(prev => ({
                                        ...prev, [k]: next === '' ? '' : (parseFiniteNumber(next) ?? ''),
                                      }))}
                                    />
                                  </div>
                                ))}
                              </div>
                            </div>

                            {nonTunableKeys.length > 0 && (
                              <div className="flex flex-col gap-2">
                                <span className={LABEL_CLS}>Non-Tunable Constants</span>
                                <p className="text-[10px] text-slate-500">
                                  Applied as fixed values in every trial.
                                </p>
                                <div className="grid grid-cols-2 gap-2">
                                  {nonTunableKeys.map((k) => {
                                    const meta: ParamMeta | undefined = paramMeta[k];
                                    if (!meta) return null;
                                    const val = hyperparams[k] ?? meta.defaultValue;
                                    return (
                                      <div key={k} className="flex flex-col gap-1">
                                        <span className="text-[10px] text-slate-500">{k}</span>
                                        {meta.options ? (
                                          <select
                                            className={SELECT_CLS}
                                            value={String(val)}
                                            onChange={e => setHyperparams(prev => ({ ...prev, [k]: e.target.value }))}
                                          >
                                            {meta.options.map(o => (
                                              <option key={o} value={o}>{o}</option>
                                            ))}
                                          </select>
                                        ) : meta.type === 'array' ? (
                                          <ChipInput
                                            value={Array.isArray(val) ? (val as string[]) : []}
                                            onChange={(next) => setHyperparams(prev => ({ ...prev, [k]: next }))}
                                            parseItem={parseStringChip}
                                            formatItem={(item) => item}
                                            className="space-y-1"
                                          />
                                        ) : (
                                          <NumericInput
                                            className={INPUT_CLS}
                                            value={stringifyNumberValue(val as number | string)}
                                            min={meta.min}
                                            max={meta.max}
                                            step={meta.step}
                                            onChange={next => {
                                              const parsed = parseFiniteNumber(next);
                                              if (parsed !== null) {
                                                setHyperparams(prev => ({ ...prev, [k]: parsed }));
                                              }
                                            }}
                                          />
                                        )}
                                      </div>
                                    );
                                  })}
                                </div>
                              </div>
                            )}

                            {/* ── C: Final Training Overrides (collapsible) ── */}
                            <div className="flex flex-col gap-2">
                              <button
                                type="button"
                                className="flex items-center justify-between w-full"
                                onClick={() => setShowFinalTrainOverrides(v => !v)}
                              >
                                <div className="flex items-center gap-2">
                                  <span className={LABEL_CLS}>Final Training Overrides</span>
                                  <span className="text-[9px] text-slate-600">optional — leave blank to use tuned value</span>
                                </div>
                                <span className="text-[10px] text-slate-500">{showFinalTrainOverrides ? '▲' : '▼'}</span>
                              </button>
                              {showFinalTrainOverrides && (
                                <div className="grid grid-cols-2 gap-2">
                                  {Object.entries(getFinalTrainMeta(framework)).map(([k, meta]) => (
                                    <div key={k} className="flex flex-col gap-1">
                                      <span className="text-[10px] text-slate-500">{k}</span>
                                      <NumericInput
                                        className={INPUT_CLS}
                                        placeholder={String(meta.defaultValue)}
                                        value={stringifyNumberValue(finalTrainOverrides[k] ?? '')}
                                        min={meta.min}
                                        step={meta.step ?? 1}
                                        onChange={next => {
                                          setFinalTrainOverrides(prev => {
                                            if (next === '') {
                                              const { [k]: _removed, ...rest } = prev;
                                              return rest;
                                            }
                                            const parsed = parseFiniteNumber(next);
                                            return { ...prev, [k]: parsed ?? '' };
                                          });
                                        }}
                                      />
                                    </div>
                                  ))}
                                </div>
                              )}
                            </div>
                          </div>
                        ) : (
                          /* Fixed hyperparams */
                          <div className="grid grid-cols-2 gap-2">
                            {Object.entries(paramMeta).map(([k, meta]) => {
                              const val = hyperparams[k] ?? meta.defaultValue;
                              return (
                                <div key={k} className="flex flex-col gap-1">
                                  <span className="text-[10px] text-slate-500">{k}</span>
                                  {meta.options ? (
                                    <select
                                      className={SELECT_CLS}
                                      value={String(val)}
                                      onChange={e => setHyperparams(prev => ({ ...prev, [k]: e.target.value }))}
                                    >
                                      {meta.options.map(o => (
                                        <option key={o} value={o}>{o}</option>
                                      ))}
                                    </select>
                                  ) : meta.type === 'array' ? (
                                    <ChipInput
                                      value={Array.isArray(val) ? (val as string[]) : []}
                                      onChange={(next) => setHyperparams(prev => ({ ...prev, [k]: next }))}
                                      parseItem={parseStringChip}
                                      formatItem={(item) => item}
                                      className="space-y-1"
                                    />
                                  ) : (
                                    <NumericInput
                                      className={INPUT_CLS}
                                      value={stringifyNumberValue(val as number | string)}
                                      min={meta.min}
                                      max={meta.max}
                                      step={meta.step}
                                      onChange={next => {
                                        const parsed = parseFiniteNumber(next);
                                        if (parsed !== null) {
                                          setHyperparams(prev => ({ ...prev, [k]: parsed }));
                                        }
                                      }}
                                    />
                                  )}
                                </div>
                              );
                            })}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* ── Tabular serving config ──────────────────────────────────── */}
              {isTabularServingPath && (
                <div className="flex flex-col gap-4">

                  {/* Section A: Training run */}
                  <div className="flex flex-col gap-2 rounded border border-slate-700/60 p-3">
                    <span className={LABEL_CLS}>Training run</span>
                    <div className="flex gap-1">
                      <select
                        className={SELECT_CLS}
                        value={tabServTrainRunId}
                        onChange={e => setTabServTrainRunId(e.target.value)}
                        disabled={tabServRunIdsLoading}
                      >
                        <option value="">{tabServRunIdsLoading ? '— loading… —' : '— select run —'}</option>
                        {tabServTrainRunIds.map(r => (
                          <option key={r.train_run_id} value={r.train_run_id}>
                            {r.train_run_id}{r.dataset ? ` · ${r.dataset}` : ''}
                          </option>
                        ))}
                      </select>
                      <button
                        type="button"
                        onClick={() => void loadTabServTrainRunIds()}
                        disabled={tabServRunIdsLoading}
                        className="rounded bg-slate-700 px-2 py-1 text-xs text-slate-300 hover:bg-slate-600 disabled:opacity-40"
                        title="Refresh"
                      >
                        <RefreshCw size={12} className={tabServRunIdsLoading ? 'animate-spin' : ''} />
                      </button>
                    </div>
                    {tabServTrainRunId && (
                      <div className="grid grid-cols-[80px_1fr] gap-x-2 gap-y-0.5 text-[11px]">
                        <span className="text-slate-500">Dataset</span>
                        <span className="font-mono text-slate-200">{tabServDataset || '—'}</span>
                      </div>
                    )}
                  </div>

                  {/* Section B: Serving mode */}
                  <div className="flex flex-col gap-2 rounded border border-slate-700/60 p-3">
                    <span className={LABEL_CLS}>Serving mode</span>
                    <div className="flex gap-2">
                      {(['ray_only', 'kafka'] as const).map(mode => (
                        <button
                          key={mode}
                          type="button"
                          onClick={() => setTabServMode(mode)}
                          className={tabServMode === mode ? BTN_TOGGLE_ON : BTN_TOGGLE_OFF}
                        >
                          {mode === 'ray_only' ? 'Ray Only' : 'Kafka'}
                        </button>
                      ))}
                    </div>
                    <p className="text-[10px] text-slate-500">
                      {tabServMode === 'ray_only'
                        ? 'Promotes model to @champion and patches the RayService. No Kafka connector.'
                        : 'Promotes model, patches RayService, and deploys a Spark Kafka streaming connector.'}
                    </p>
                  </div>

                  {/* Section C: Kafka schema (only in kafka mode) */}
                  {tabServMode === 'kafka' && (
                    <div className="flex flex-col gap-2 rounded border border-slate-700/60 p-3">
                      <span className={LABEL_CLS}>Kafka input schema (raw.yaml)</span>
                      <p className="text-[10px] text-slate-500">
                        Define the Kafka message schema for dataset{' '}
                        <span className="font-mono text-slate-300">{tabServDataset || '(dataset)'}</span>.
                      </p>
                      <RawYamlEditor
                        fields={tabServRawFields}
                        groups={tabServRawGroups}
                        idField={tabServIdField}
                        onChange={(fields, groups, idField) => {
                          setTabServRawFields(fields);
                          setTabServRawGroups(groups);
                          setTabServIdField(idField);
                        }}
                        dataset={tabServDataset}
                      />
                      <div className="flex items-center gap-3 pt-1">
                        <button
                          type="button"
                          onClick={() => void handleTabServSaveSchema()}
                          disabled={tabServSavingSchema || tabServRawFields.length === 0 || !tabServIdField || !tabServDataset}
                          className={`rounded bg-blue-600 px-3 py-1 text-xs font-medium text-white hover:bg-blue-500 disabled:opacity-40`}
                        >
                          {tabServSavingSchema ? 'Saving…' : 'Save raw.yaml to S3'}
                        </button>
                        {tabServSchemaResult && (
                          <span className="rounded-full bg-green-900/50 px-2 py-0.5 text-[10px] text-green-400">
                            ✓ Saved as v{tabServSchemaResult.version}
                          </span>
                        )}
                        {tabServRawSchemaS3 && (
                          <span className="max-w-[260px] truncate font-mono text-[10px] text-slate-500" title={tabServRawSchemaS3}>
                            {tabServRawSchemaS3}
                          </span>
                        )}
                      </div>
                      {tabServSchemaError && <p className="text-xs text-red-400">{tabServSchemaError}</p>}
                    </div>
                  )}

                  {/* Section C: Deployment target */}
                  <div className="flex flex-col gap-2 rounded border border-slate-700/60 p-3">
                    <span className={LABEL_CLS}>Deployment target</span>
                    <p className="text-[10px] text-slate-500">
                      Out-Cluster only: SkyPilot sky serve on external VMs with auto-scaled replicas.
                    </p>
                    <div className="grid grid-cols-3 gap-3 mt-1">
                      <div className="flex flex-col gap-1">
                        <span className="text-[10px] text-slate-500">Min replicas</span>
                        <input
                          type="number"
                          className={INPUT_CLS}
                          min={0}
                          value={tabServMinReplicas}
                          onChange={e => setTabServMinReplicas(parseInt(e.target.value) || 1)}
                        />
                      </div>
                      <div className="flex flex-col gap-1">
                        <span className="text-[10px] text-slate-500">Max replicas</span>
                        <input
                          type="number"
                          className={INPUT_CLS}
                          min={1}
                          value={tabServMaxReplicas}
                          onChange={e => setTabServMaxReplicas(parseInt(e.target.value) || 3)}
                        />
                      </div>
                      <div className="flex flex-col gap-1">
                        <span className="text-[10px] text-slate-500">Target QPS / replica</span>
                        <input
                          type="number"
                          className={INPUT_CLS}
                          min={1}
                          value={tabServQps}
                          onChange={e => setTabServQps(parseInt(e.target.value) || 10)}
                        />
                      </div>
                    </div>
                  </div>

                  {/* Section D: Serving config */}
                  <div className="flex flex-col gap-3 rounded border border-slate-700/60 p-3">
                    <span className={LABEL_CLS}>Serving config</span>

                    {/* Alias */}
                    <div className="grid grid-cols-2 gap-3">
                      <div className="flex flex-col gap-1">
                        <span className="text-[10px] text-slate-500">Model alias</span>
                        <input
                          type="text"
                          className={INPUT_CLS}
                          value={tabServAlias}
                          placeholder="champion"
                          onChange={e => setTabServAlias(e.target.value)}
                        />
                      </div>
                    </div>

                    {/* Canary */}
                    <div className="flex flex-col gap-2">
                      <div className="flex items-center gap-3">
                        <span className="text-[10px] text-slate-500">Canary deployment</span>
                        <div className="flex gap-1.5">
                          <button type="button" onClick={() => setTabServCanary(true)} className={tabServCanary ? BTN_TOGGLE_ON : BTN_TOGGLE_OFF}>Enabled</button>
                          <button type="button" onClick={() => setTabServCanary(false)} className={!tabServCanary ? BTN_TOGGLE_ON : BTN_TOGGLE_OFF}>Disabled</button>
                        </div>
                      </div>
                      {tabServCanary && (
                        <div className="grid grid-cols-3 gap-3">
                          <div className="flex flex-col gap-1">
                            <span className="text-[10px] text-slate-500">Canary alias</span>
                            <input type="text" className={INPUT_CLS} value={tabServCanaryAlias} placeholder="challenger" onChange={e => setTabServCanaryAlias(e.target.value)} />
                          </div>
                          <div className="flex flex-col gap-1">
                            <span className="text-[10px] text-slate-500">Probability (0–1)</span>
                            <input type="number" className={INPUT_CLS} min={0} max={1} step={0.01} value={tabServCanaryProb} onChange={e => setTabServCanaryProb(parseFloat(e.target.value) || 0)} />
                          </div>
                          <div className="flex flex-col gap-1">
                            <span className="text-[10px] text-slate-500">Initial replicas</span>
                            <input type="number" className={INPUT_CLS} min={0} value={tabServInitReplicas} onChange={e => setTabServInitReplicas(parseInt(e.target.value) || 0)} />
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Webhook */}
                    <div className="flex flex-col gap-2">
                      <span className="text-[10px] text-slate-500">Webhook (MLflow alias-change hot reloads)</span>
                      <div className="grid grid-cols-2 gap-3">
                        <div className="flex flex-col gap-1">
                          <span className="text-[10px] text-slate-500">Public base URL</span>
                          <input type="text" className={INPUT_CLS} value={tabServWebhookUrl} placeholder="http://model-serving-serve-svc.ray.svc.cluster.local:8000" onChange={e => setTabServWebhookUrl(e.target.value)} />
                        </div>
                        <div className="flex flex-col gap-1">
                          <span className="text-[10px] text-slate-500">Webhook path</span>
                          <input type="text" className={INPUT_CLS} value={tabServWebhookPath} placeholder="/infer/webhook" onChange={e => setTabServWebhookPath(e.target.value)} />
                        </div>
                        <div className="flex flex-col gap-1">
                          <span className="text-[10px] text-slate-500">Max timestamp age (s)</span>
                          <input type="number" className={INPUT_CLS} min={1} value={tabServWebhookMaxAge} onChange={e => setTabServWebhookMaxAge(parseInt(e.target.value) || 300)} />
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* ── LLM config ──────────────────────────────────────────────── */}
              {modelCategory === 'llm' && (
                <div className="grid grid-cols-2 gap-4">
                  {/* LLM model picker — always */}
                  <div className="flex flex-col gap-1.5">
                    <span className={LABEL_CLS}>LLM model</span>
                    <select
                      className={SELECT_CLS}
                      value={llmModel}
                      onChange={e => setLlmModel(e.target.value)}
                    >
                      {llmCatalog.map(m => (
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
                  {/* HF token — always */}
                  <div className="flex flex-col gap-1.5 col-span-2">
                    <span className={LABEL_CLS}>HuggingFace token</span>
                    <input
                      type="password"
                      className={INPUT_CLS}
                      value={hfToken}
                      placeholder="hf_xxx (optional for public models)"
                      onChange={e => setHfToken(e.target.value)}
                    />
                  </div>

                  {/* Training-specific fields */}
                  {isTraining && (
                    <>
                      <div className="flex flex-col gap-1.5">
                        <span className={LABEL_CLS}>Dataset S3 path (JSONL)</span>
                        <input
                          type="text"
                          className={INPUT_CLS}
                          value={datasetS3}
                          placeholder="s3://bucket/llm-data/train.jsonl"
                          onChange={e => setDatasetS3(e.target.value)}
                        />
                      </div>
                      <div className="flex flex-col gap-1.5">
                        <span className={LABEL_CLS}>Max steps</span>
                        <input
                          type="number"
                          className={INPUT_CLS}
                          value={maxSteps}
                          min={1}
                          onChange={e => setMaxSteps(e.target.value)}
                        />
                      </div>
                      <div className="flex flex-col gap-1.5">
                        <span className={LABEL_CLS}>DeepSpeed ZeRO</span>
                        <div className="flex gap-1.5">
                          <button type="button" onClick={() => setUseDeepspeed(true)}
                            className={useDeepspeed ? BTN_TOGGLE_ON : BTN_TOGGLE_OFF}>
                            Enabled
                          </button>
                          <button type="button" onClick={() => setUseDeepspeed(false)}
                            className={!useDeepspeed ? BTN_TOGGLE_ON : BTN_TOGGLE_OFF}>
                            Disabled
                          </button>
                        </div>
                      </div>
                    </>
                  )}

                  {/* Separator for both */}
                  {jobType === 'both' && (
                    <div className="col-span-2 border-t border-slate-700 pt-2">
                      <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-600">
                        Serving config
                      </span>
                    </div>
                  )}

                  {/* Serving-specific fields */}
                  {isServing && (
                    <>
                      <div className="flex flex-col gap-1.5">
                        <span className={LABEL_CLS}>LoRA adapter S3 path</span>
                        <input
                          type="text"
                          className={INPUT_CLS}
                          value={llmAdapterS3}
                          placeholder="s3://…/adapter/ (optional)"
                          onChange={e => setLlmAdapterS3(e.target.value)}
                        />
                      </div>
                      <div className="flex flex-col gap-1.5">
                        <span className={LABEL_CLS}>Max context length</span>
                        <input
                          type="number"
                          className={INPUT_CLS}
                          value={maxModelLen}
                          min={512}
                          step={512}
                          onChange={e => setMaxModelLen(e.target.value)}
                        />
                      </div>
                      <div className="flex flex-col gap-1.5">
                        <span className={LABEL_CLS}>Replicas</span>
                        <input
                          type="number"
                          className={INPUT_CLS}
                          value={servingReplicas}
                          min={1}
                          max={10}
                          onChange={e => setServingReplicas(Math.max(1, parseInt(e.target.value) || 1))}
                        />
                        <span className="text-[10px] text-slate-500">
                          SkyServe replicas — each replica = {numNodes}-node Ray cluster
                        </span>
                      </div>
                    </>
                  )}
                </div>
              )}

              {/* ── Upload custom model ─────────────────────────────────────── */}
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
                        onChange={e => setUploadName(e.target.value)}
                      />
                    </div>
                    <div className="flex flex-col gap-1.5">
                      <span className={LABEL_CLS}>Description (optional)</span>
                      <input
                        type="text"
                        className={INPUT_CLS}
                        value={uploadDesc}
                        placeholder="Short description"
                        onChange={e => setUploadDesc(e.target.value)}
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
                      onChange={e => setUploadFile(e.target.files?.[0] ?? null)}
                    />
                    <button
                      type="button"
                      disabled={!uploadFile || !uploadName || uploadStatus === 'uploading'}
                      onClick={handleUpload}
                      className={BTN_PRIMARY}
                    >
                      {uploadStatus === 'uploading' && <Loader2 size={12} className="animate-spin inline mr-1" />}
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
                      <XCircle size={12} className="inline mr-1" />{uploadError}
                    </div>
                  )}
                  {/* Preprocess run for upload */}
                  <div className="grid grid-cols-2 gap-3 mt-1">
                    <div className="flex flex-col gap-1.5">
                      <span className={LABEL_CLS}>Preprocess run ID</span>
                      <input
                        type="text"
                        className={INPUT_CLS}
                        value={preprocessRunId}
                        placeholder="pre-dataset-xxx"
                        onChange={e => setPreprocessRunId(e.target.value)}
                      />
                    </div>
                  </div>
                </div>
              )}

              <div className="flex justify-end pt-2">
                <button type="button" className={BTN_PRIMARY} onClick={() => setStep(1)}>
                  Next: Resources
                </button>
              </div>
            </div>
          )}

          {/* ═══════════════════════════════════════════════════════════════
              STEP 2 — Resource Configuration
          ══════════════════════════════════════════════════════════════════ */}
          {step === 1 && (
            <div className="flex flex-col gap-5 rounded border border-slate-700 bg-slate-900/50 p-5">
              {isTabularServingPath ? (
                <>
                  <GPUResourceSelector value={constraints} onChange={setConstraints} />
                  <div className="grid grid-cols-2 gap-4">
                    <div className="flex flex-col gap-1.5">
                      <span className={LABEL_CLS}>Nodes per replica</span>
                      <div className="flex gap-2">
                        {[1, 2, 4].map(n => (
                          <button
                            key={n}
                            type="button"
                            onClick={() => setNumNodes(n)}
                            className={`rounded px-2.5 py-0.5 text-xs font-medium transition-colors ${
                              numNodes === n ? 'bg-blue-700 text-white' : 'bg-slate-700 text-slate-400 hover:bg-slate-600'
                            }`}
                          >
                            {n === 1 ? 'Single' : `${n}×`}
                          </button>
                        ))}
                      </div>
                      {numNodes > 1 && (
                        <p className="text-[10px] text-slate-500">
                          Each sky serve replica spawns a {numNodes}-node Ray cluster (head + {numNodes - 1} worker{numNodes > 2 ? 's' : ''}).
                        </p>
                      )}
                    </div>
                    <div className="rounded border border-amber-700/40 bg-amber-900/10 px-3 py-2 self-start">
                      <p className="text-[10px] text-amber-400">
                        Serving — on-demand recommended. SkyServe handles replica restarts on spot preemption.
                      </p>
                    </div>
                  </div>
                </>
              ) : (
                <GPUResourceSelector value={constraints} onChange={setConstraints} />
              )}

              {!isTabularServingPath && isTabularPath && (
                <div className="flex flex-col gap-1.5 rounded border border-slate-700/60 p-3">
                  <span className={LABEL_CLS}>SkyPilot launch mode</span>
                  <div className="flex flex-wrap gap-1.5">
                    <button
                      type="button"
                      onClick={() => setSkyLaunchMode('launch')}
                      className={skyLaunchMode === 'launch' ? BTN_TOGGLE_ON : BTN_TOGGLE_OFF}
                    >
                      sky launch --retry-until-up
                    </button>
                    <button
                      type="button"
                      onClick={() => setSkyLaunchMode('jobs_launch')}
                      className={skyLaunchMode === 'jobs_launch' ? BTN_TOGGLE_ON : BTN_TOGGLE_OFF}
                    >
                      sky jobs launch (managed)
                    </button>
                  </div>
                  <p className="text-[10px] text-slate-500">
                    Direct launch keeps retry-until-up enabled. Managed jobs skip that flag.
                  </p>
                </div>
              )}

              {/* Serving note (LLM only) */}
              {!isTabularServingPath && isServing && !isTraining && (
                <div className="rounded border border-amber-700/40 bg-amber-900/10 px-3 py-2 text-[10px] text-amber-400">
                  Serving clusters are persistent — on-demand is recommended over spot.
                </div>
              )}

              {/* Nodes + InfiniBand (SkyPilot jobs only — excluded for tabular serving which has its own selector above) */}
              {!isTabularServingPath && (
                <div className="grid grid-cols-2 gap-4">
                  <div className="flex flex-col gap-1.5">
                    <span className={LABEL_CLS}>Nodes</span>
                    <div className="flex gap-2">
                      {[1, 2, 4, 8].map(n => (
                        <button
                          key={n}
                          type="button"
                          onClick={() => setNumNodes(n)}
                          className={`rounded px-2.5 py-0.5 text-xs font-medium transition-colors ${
                            numNodes === n ? 'bg-blue-700 text-white' : 'bg-slate-700 text-slate-400 hover:bg-slate-600'
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
                        onChange={e => setUseInfiniband(e.target.checked)}
                        className="accent-blue-500"
                      />
                      Require InfiniBand / EFA
                      {numNodes > 1 && (
                        <span className="text-amber-400 text-[10px]">(recommended for multi-node)</span>
                      )}
                    </label>
                  </div>
                </div>
              )}

              <div className="flex justify-between pt-2">
                <button type="button" className={BTN_SECONDARY} onClick={() => setStep(0)}>Back</button>
                <button type="button" className={BTN_PRIMARY} onClick={() => setStep(2)}>Next: Review</button>
              </div>
            </div>
          )}

          {/* ═══════════════════════════════════════════════════════════════
              STEP 3 — Review
          ══════════════════════════════════════════════════════════════════ */}
          {step === 2 && (
            <div className="flex flex-col gap-5 rounded border border-slate-700 bg-slate-900/50 p-5">

              {isTabularServingPath ? (
                /* Tabular serving: config summary cards */
                <>
                  <span className="text-sm font-semibold text-slate-200">Serving Configuration Summary</span>

                  {/* Training Run card */}
                  <div className="rounded border border-slate-700 bg-slate-800/40 px-4 py-3">
                    <div className="mb-2 flex items-center justify-between">
                      <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">Training Run</span>
                      <button onClick={() => setStep(0)} className="text-[10px] text-blue-400 hover:text-blue-300">Edit</button>
                    </div>
                    <div className="grid grid-cols-[120px_1fr] gap-x-3 gap-y-1">
                      {([
                        ['Train run ID', tabServTrainRunId],
                        ['Dataset', tabServDataset],
                        ['Mode', tabServMode],
                        ['Deployment', 'Out-Cluster (SkyPilot sky serve)'],
                        ['Replicas', `${tabServMinReplicas}–${tabServMaxReplicas} (QPS target: ${tabServQps})`],
                        ['Nodes/replica', String(numNodes)],
                        ['Providers', (constraints.providers ?? ['runpod']).join(', ')],
                      ] as [string, string][]).map(([k, v]) => (
                        <div key={k} className="contents">
                          <span className="text-[11px] text-slate-500">{k}</span>
                          <span className="break-all font-mono text-[11px] text-slate-200">{v || '—'}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Kafka Schema card */}
                  {tabServMode === 'kafka' && (
                    <div className="rounded border border-slate-700 bg-slate-800/40 px-4 py-3">
                      <div className="mb-2 flex items-center justify-between">
                        <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">Kafka Schema</span>
                        <button onClick={() => setStep(0)} className="text-[10px] text-blue-400 hover:text-blue-300">Edit</button>
                      </div>
                      <div className="grid grid-cols-[120px_1fr] gap-x-3 gap-y-1">
                        {[
                          ['Fields', String(tabServRawFields.length)],
                          ['id_field', tabServIdField],
                          ['Schema version', tabServSchemaResult ? `v${tabServSchemaResult.version}` : '—'],
                          ['S3 path', tabServRawSchemaS3],
                        ].map(([k, v]) => (
                          <div key={k} className="contents">
                            <span className="text-[11px] text-slate-500">{k}</span>
                            <span className="break-all font-mono text-[11px] text-slate-200">{v || '—'}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Serving Config card */}
                  <div className="rounded border border-slate-700 bg-slate-800/40 px-4 py-3">
                    <div className="mb-2 flex items-center justify-between">
                      <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">Serving Config</span>
                      <button onClick={() => setStep(0)} className="text-[10px] text-blue-400 hover:text-blue-300">Edit</button>
                    </div>
                    <div className="grid grid-cols-[120px_1fr] gap-x-3 gap-y-1">
                      {[
                        ['Alias', tabServAlias],
                        ['Canary', tabServCanary ? `Yes (${tabServCanaryAlias}, p=${tabServCanaryProb})` : 'No'],
                        ['Webhook path', tabServWebhookPath || '—'],
                        ['Max timestamp age', `${tabServWebhookMaxAge}s`],
                      ].map(([k, v]) => (
                        <div key={k} className="contents">
                          <span className="text-[11px] text-slate-500">{k}</span>
                          <span className="break-all font-mono text-[11px] text-slate-200">{v || '—'}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Validation */}
                  {!tabServTrainRunId ? (
                    <div className="rounded border border-amber-800/40 bg-amber-900/10 px-3 py-2">
                      <p className="text-xs text-amber-400">Select a training run before launching.</p>
                    </div>
                  ) : tabServMode === 'kafka' && !tabServRawSchemaS3 ? (
                    <div className="rounded border border-amber-800/40 bg-amber-900/10 px-3 py-2">
                      <p className="text-xs text-amber-400">Save raw.yaml to S3 before launching (Kafka mode).</p>
                    </div>
                  ) : (
                    <div className="rounded border border-green-800/30 bg-green-900/10 px-3 py-2">
                      <p className="text-xs text-green-400">✓ Configuration valid — ready to launch.</p>
                    </div>
                  )}
                </>
              ) : isTabularPath ? (
                /* Tabular training: config summary — no dry_run */
                <>
                  <span className="text-sm font-semibold text-slate-200">Configuration Summary</span>
                  <dl className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs">
                    <dt className="text-slate-500">Preprocess run</dt>
                    <dd className="text-slate-200 font-mono text-[10px]">{preprocessRunId || '—'}</dd>
                    {preprocessInfo && (
                      <>
                        <dt className="text-slate-500">Dataset</dt>
                        <dd className="text-slate-200">{preprocessInfo.dataset}</dd>
                      </>
                    )}
                    <dt className="text-slate-500">Framework</dt>
                    <dd className="text-slate-200">{framework === 'xgboost' ? 'XGBoost' : `PyTorch (${pytorchModelType.toUpperCase()})`}</dd>
                    <dt className="text-slate-500">Task</dt>
                    <dd className="text-slate-200 capitalize">{taskType}</dd>
                    <dt className="text-slate-500">Target</dt>
                    <dd className="text-slate-200">{target || '—'}</dd>
                    <dt className="text-slate-500">MLflow tracking URI</dt>
                    <dd className="text-slate-200 break-all">{mlflowTrackingUri || 'default from Airflow env'}</dd>
                    <dt className="text-slate-500">MLflow artifact location</dt>
                    <dd className="text-slate-200 break-all">{mlflowArtifactLocation}</dd>
                    {taskType === 'classification' && (
                      <>
                        <dt className="text-slate-500">Num classes</dt>
                        <dd className="text-slate-200">{numClasses}</dd>
                      </>
                    )}
                    <dt className="text-slate-500">Ray Tune</dt>
                    <dd className="text-slate-200">{tuneEnabled ? `Enabled (${numTrials} trials)` : 'Disabled'}</dd>
                    <dt className="text-slate-500">Providers</dt>
                    <dd className="text-slate-200">{(constraints.providers ?? ['runpod']).join(', ')}</dd>
                    <dt className="text-slate-500">Nodes</dt>
                    <dd className="text-slate-200">{numNodes}</dd>
                    <dt className="text-slate-500">Spot</dt>
                    <dd className="text-slate-200">{constraints.prefer_spot !== false ? 'Preferred' : 'Off'}</dd>
                    <dt className="text-slate-500">Launch mode</dt>
                    <dd className="text-slate-200">
                      {skyLaunchMode === 'jobs_launch'
                        ? 'sky jobs launch (managed)'
                        : 'sky launch --retry-until-up'}
                    </dd>
                  </dl>

                  {tuneEnabled && (
                    <div className="rounded border border-slate-700 bg-slate-800/60 p-3">
                      <span className="text-xs font-semibold text-slate-300">Tuning Details</span>
                      <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
                        <dt className="text-slate-500">Trials</dt>
                        <dd className="text-slate-200">{stringifyNumberValue(numTrials) || '—'}</dd>
                        <dt className="text-slate-500">Sample fraction</dt>
                        <dd className="text-slate-200">{sampleFraction}</dd>
                        {Object.entries(tuneSettingsMeta).map(([key, meta]) => (
                          <div key={`tune-setting-${key}`} className="contents">
                            <dt className="text-slate-500">{key}</dt>
                            <dd className="text-slate-200">{String(tuneSettings[key] ?? meta.defaultValue)}</dd>
                          </div>
                        ))}
                        {searchSpaceSummary.map(([key, value]) => (
                          <div key={`search-space-${key}`} className="contents">
                            <dt className="text-slate-500">{key}</dt>
                            <dd className="text-slate-200 break-all">{value}</dd>
                          </div>
                        ))}
                        <dt className="text-slate-500">Final overrides</dt>
                        <dd className="text-slate-200 break-all">
                          {Object.keys(finalTrainOverrides).length > 0
                            ? Object.entries(finalTrainOverrides)
                                .map(([key, value]) => `${key}=${value}`)
                                .join(', ')
                            : 'none'}
                        </dd>
                      </dl>
                    </div>
                  )}

                  {/* GPU Resource Plan */}
                  <div className="rounded border border-slate-700 bg-slate-800/60 p-3">
                    <span className="text-xs font-semibold text-slate-300">GPU Resource Plan</span>
                    <p className="mt-1 text-xs text-slate-400">
                      {numNodes} node{numNodes > 1 ? 's' : ''} × {constraints.num_gpus_per_node ?? 1} GPU{(constraints.num_gpus_per_node ?? 1) > 1 ? 's' : ''} = <span className="text-slate-200 font-medium">{gpuSummary.totalGpus} total GPU{gpuSummary.totalGpus > 1 ? 's' : ''}</span>
                      {' '}({gpuSummary.mode})
                    </p>
                    <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
                      <dt className="text-slate-500">Training workers</dt>
                      <dd className="text-slate-200">{gpuSummary.numWorkers} <span className="text-slate-500">(1 GPU each, DDP)</span></dd>
                      {tuneEnabled && (
                        <>
                          <dt className="text-slate-500">Concurrent trials</dt>
                          <dd className="text-slate-200">{gpuSummary.maxConcurrentTrials}</dd>
                          <dt className="text-slate-500">Workers / trial</dt>
                          <dd className="text-slate-200">{gpuSummary.numWorkersTune} <span className="text-slate-500">(1 GPU each)</span></dd>
                        </>
                      )}
                    </dl>
                  </div>
                </>
              ) : (
                /* LLM path: orchestration recommendation */
                <>
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

                  <div className="flex flex-col gap-1.5">
                    <span className={LABEL_CLS}>Est. runtime</span>
                    <div className="flex gap-1.5">
                      {RUNTIME_PRESETS.map(({ label, hours }) => (
                        <button
                          key={label}
                          type="button"
                          onClick={() => setRuntimeHours(hours)}
                          className={`rounded px-2 py-0.5 text-xs font-medium transition-colors ${
                            runtimeHours === hours ? 'bg-blue-700 text-white' : 'bg-slate-700 text-slate-400 hover:bg-slate-600'
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
                      <XCircle size={12} className="inline mr-1" />{reviewError}
                    </p>
                  )}
                  {recommendation && !reviewLoading && (
                    <OrchestrationRecommendation
                      recommendation={recommendation}
                      yamlPreview={yamlPreview}
                      runtimeHours={runtimeHours}
                    />
                  )}
                </>
              )}

              <div className="flex justify-between pt-2">
                <button type="button" className={BTN_SECONDARY} onClick={() => setStep(1)}>Back</button>
                <button
                  type="button"
                  className={BTN_PRIMARY}
                  disabled={
                    (isLlmPath && !recommendation) ||
                    (isTabularServingPath && (!tabServTrainRunId || (tabServMode === 'kafka' && !tabServRawSchemaS3)))
                  }
                  onClick={() => setStep(3)}
                >
                  Next: Launch
                </button>
              </div>
            </div>
          )}

          {/* ═══════════════════════════════════════════════════════════════
              STEP 4 — Launch
          ══════════════════════════════════════════════════════════════════ */}
          {step === 3 && (
            <div className="flex flex-col gap-5 rounded border border-slate-700 bg-slate-900/50 p-5">
              {/* Summary */}
              {!isTabularServingPath && (
              <div className="flex flex-col gap-2">
                <span className="text-sm font-semibold text-slate-200">Launch Summary</span>
                <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
                  <dt className="text-slate-500">Job type</dt>
                  <dd className="text-slate-200 capitalize">{jobType}</dd>
                  <dt className="text-slate-500">Model</dt>
                  <dd className="text-slate-200">
                    {modelCategory === 'llm'
                      ? llmModel
                      : modelCategory === 'tabular'
                      ? `${framework === 'pytorch' ? pytorchModelType.toUpperCase() : 'XGBoost'}`
                      : resolvedModelType.toUpperCase()}
                  </dd>
                  <dt className="text-slate-500">Providers</dt>
                  <dd className="text-slate-200">{(constraints.providers ?? ['runpod']).join(', ')}</dd>
                  <dt className="text-slate-500">Nodes</dt>
                  <dd className="text-slate-200">{numNodes}</dd>
                  {isTabularPath && (
                    <>
                      <dt className="text-slate-500">Launch mode</dt>
                      <dd className="text-slate-200">
                        {skyLaunchMode === 'jobs_launch'
                          ? 'sky jobs launch (managed)'
                          : 'sky launch --retry-until-up'}
                      </dd>
                    </>
                  )}
                  {isServing && llmAdapterS3 && (
                    <>
                      <dt className="text-slate-500">Adapter</dt>
                      <dd className="text-slate-200 font-mono text-[10px] truncate">{llmAdapterS3}</dd>
                    </>
                  )}
                  {recommendation && (
                    <>
                      <dt className="text-slate-500">Orchestration</dt>
                      <dd className="text-slate-200">{recommendation.orchestration}</dd>
                    </>
                  )}
                </dl>
              </div>
              )}

              {/* ── Tabular serving launch ── */}
              {isTabularServingPath && (
                <div className="flex flex-col gap-3">
                  {tabServSubmitResult ? (
                    <div className="flex flex-col gap-3 rounded border border-green-700 bg-green-950/20 p-4">
                      <div className="flex items-center gap-2 text-sm font-semibold text-green-400">
                        <CheckCircle2 size={16} />Serving config saved
                      </div>
                      <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-xs">
                        <dt className="text-slate-400">Serve run ID</dt>
                        <dd className="break-all font-mono text-slate-200">{tabServSubmitResult.serve_run_id}</dd>
                        <dt className="text-slate-400">Mode</dt>
                        <dd className="text-slate-200">{tabServSubmitResult.serving_mode}</dd>
                        <dt className="text-slate-400">Dataset</dt>
                        <dd className="text-slate-200">{tabServSubmitResult.dataset}</dd>
                        <dt className="text-slate-400">Params S3 path</dt>
                        <dd className="break-all font-mono text-slate-200">{tabServSubmitResult.params_s3_path}</dd>
                      </dl>

                      {!tabServDeployResult ? (
                        <div className="flex flex-col gap-2 border-t border-slate-700 pt-3">
                          {tabServDeployError && <p className="text-xs text-red-400">{tabServDeployError}</p>}
                          <button
                            type="button"
                            onClick={() => void handleTabServDeploy()}
                            className="rounded bg-green-700 py-2 text-sm font-semibold text-white hover:bg-green-600"
                          >
                            Deploy Now
                          </button>
                          <p className="text-center text-[10px] text-slate-500">
                            Triggers: tabular_serving_skypilot_pipeline DAG → sky serve up → endpoint registered
                          </p>
                        </div>
                      ) : (
                        <div className="rounded border border-slate-700 bg-slate-800/40 px-3 py-2">
                          <p className="text-xs font-medium text-slate-300 mb-1">Deployment triggered</p>
                          <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
                            <span className="text-slate-500">DAG run ID</span>
                            <span className="break-all font-mono text-slate-200">{tabServDeployResult.dag_run_id}</span>
                            <span className="text-slate-500">State</span>
                            <span className={`font-mono ${
                              tabServDeployState === 'success' ? 'text-green-400'
                              : tabServDeployState === 'failed' || tabServDeployState === 'upstream_failed' ? 'text-red-400'
                              : 'text-amber-400'
                            }`}>
                              {tabServDeployState || 'queued'}
                            </span>
                            {tabServDeployTarget === 'skypilot' && (
                              <>
                                <span className="text-slate-500">Endpoint</span>
                                <span className={`font-mono ${tabServEndpoint ? 'text-green-400' : 'text-amber-400'}`}>
                                  {tabServEndpoint ?? 'provisioning…'}
                                </span>
                              </>
                            )}
                          </div>
                        </div>
                      )}

                      <button
                        type="button"
                        onClick={() => {
                          setTabServSubmitResult(null);
                          setTabServDeployResult(null);
                          setTabServDeployState('');
                          setTabServDeployError('');
                          setTabServEndpoint(null);
                          if (tabServDeployPollRef.current) clearInterval(tabServDeployPollRef.current);
                          if (tabServEndpointPollRef.current) clearInterval(tabServEndpointPollRef.current);
                        }}
                        className="rounded bg-slate-700 px-3 py-1 text-xs text-slate-300 hover:bg-slate-600"
                      >
                        New Serving Config
                      </button>
                    </div>
                  ) : (
                    <div className="flex flex-col gap-2">
                      {tabServSubmitError && (
                        <div className="rounded border border-red-800/40 bg-red-900/20 px-3 py-2">
                          <p className="text-xs text-red-400">{tabServSubmitError}</p>
                        </div>
                      )}
                      <button
                        type="button"
                        onClick={() => void handleTabServSubmit()}
                        disabled={!tabServTrainRunId}
                        className="flex items-center justify-center gap-2 rounded bg-blue-600 px-6 py-2 text-sm font-bold text-white hover:bg-blue-500 transition-colors disabled:opacity-40"
                      >
                        <Rocket size={14} />
                        Save Serving Config
                      </button>
                    </div>
                  )}
                </div>
              )}

              {/* ── Tabular training launch ── */}
              {isTabularPath && !tabularResult && (
                <button
                  type="button"
                  disabled={tabularLaunching}
                  onClick={handleLaunchTabular}
                  className="flex items-center justify-center gap-2 rounded bg-green-700 px-6 py-2 text-sm font-bold text-white hover:bg-green-600 transition-colors disabled:opacity-40"
                >
                  {tabularLaunching ? <Loader2 size={14} className="animate-spin" /> : <Rocket size={14} />}
                  {tabularLaunching ? 'Launching…' : 'Launch Training Job'}
                </button>
              )}

              {tabularError && (
                <div className="rounded border border-red-700 bg-red-950/30 p-3 text-xs text-red-400">
                  <XCircle size={12} className="inline mr-1" />{tabularError}
                </div>
              )}

              {tabularResult && (
                <div className="flex flex-col gap-2 rounded border border-green-700 bg-green-950/20 p-4">
                  <div className="flex items-center gap-2 text-sm font-semibold text-green-400">
                    <CheckCircle2 size={16} />Job launched successfully
                  </div>
                  <div className="text-xs text-slate-300">
                    <span className="text-slate-500">Train run: </span>
                    <span className="font-mono text-blue-300">{tabularResult.train_run_id}</span>
                  </div>
                  <div className="text-xs text-slate-300">
                    <span className="text-slate-500">DAG run: </span>
                    <span className="font-mono text-blue-300">{tabularResult.dag_run_id}</span>
                  </div>
                  <div className="text-xs text-slate-500">
                    Monitor in Airflow UI or poll{' '}
                    <span className="font-mono text-slate-400">GET /api/v2/runs/{tabularResult.dag_run_id}/status</span>
                  </div>
                </div>
              )}

              {/* ── LLM / serving launch ── */}
              {isLlmPath && !launchResult && (
                <button
                  type="button"
                  disabled={launching}
                  onClick={handleLaunchLlm}
                  className="flex items-center justify-center gap-2 rounded bg-green-700 px-6 py-2 text-sm font-bold text-white hover:bg-green-600 transition-colors disabled:opacity-40"
                >
                  {launching ? <Loader2 size={14} className="animate-spin" /> : <Rocket size={14} />}
                  {launching ? 'Launching…' : 'Launch Job'}
                </button>
              )}

              {launchError && (
                <div className="rounded border border-red-700 bg-red-950/30 p-3 text-xs text-red-400">
                  <XCircle size={12} className="inline mr-1" />{launchError}
                </div>
              )}

              {launchResult && (
                <div className="flex flex-col gap-3 rounded border border-green-700 bg-green-950/20 p-4">
                  <div className="flex items-center gap-2 text-sm font-semibold text-green-400">
                    <CheckCircle2 size={16} />Job launched successfully
                  </div>
                  {Object.entries(launchResult.job_ids).map(([type, runId]) => (
                    <div key={type} className="text-xs text-slate-300">
                      <span className="capitalize text-slate-500">{type}: </span>
                      <span className="font-mono text-blue-300">{runId}</span>
                    </div>
                  ))}
                  <p className="text-xs text-slate-500">
                    Monitor in Airflow UI or poll{' '}
                    <span className="font-mono text-slate-400">GET /api/v2/jobs/&#123;id&#125;/status</span>
                  </p>
                </div>
              )}

              <div className="flex justify-between pt-2">
                <button
                  type="button"
                  className={BTN_SECONDARY}
                  onClick={() => {
                    if (tabularResult || launchResult || tabServDeployResult) resetForNewJob();
                    else setStep(2);
                  }}
                >
                  {tabularResult || launchResult || tabServDeployResult ? 'New Job' : 'Back'}
                </button>
              </div>
            </div>
          )}

        </div>{/* end wizard column */}
      </div>{/* end grid */}
    </div>
  );
}
