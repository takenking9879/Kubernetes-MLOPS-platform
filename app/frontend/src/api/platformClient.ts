/**
 * API client for the dataset-oriented platform endpoints (v2).
 *
 * All functions throw on non-2xx responses.
 */

import type { ColumnMeta } from '../types/schema';

export type { ColumnMeta };

// Use relative paths so the browser sends requests to the frontend server,
// which then proxies them to the backend inside the pod.
const API_BASE = '';

async function _fetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...init?.headers },
    ...init,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
  return res.json() as Promise<T>;
}

// ─── Dataset types ────────────────────────────────────────────────────────────

export interface DatasetInfo {
  name: string;
  file_count: number;
  size_bytes: number;
}

export interface UploadResult {
  uploaded: Array<{ key: string; size_bytes: number }>;
}

export interface IngestResult {
  job_name: string;
}

export interface IngestStatus {
  state: string;
}

export interface SampleResult {
  columns: ColumnMeta[];
  rows: Record<string, unknown>[];
  row_count: number;
}

// ─── DSL types ────────────────────────────────────────────────────────────────

export interface DslVersion {
  version: number;
  slug: string;
  key: string;
  last_modified: string;
}

export interface DslListResult {
  dataset: string;
  dsls: DslVersion[];
}

export interface DslContent {
  version: number;
  slug: string;
  key: string;
  yaml_content: string;
}

export interface SaveDslResult {
  version: number;
  slug: string;
  key: string;
}

export interface DslFinalFeatures {
  features: string[];
  target: string[];
  metadata: string[];
  passthrough: string[];
}

export interface DslFeaturesResult {
  version: number;
  slug: string;
  finalFeatures: DslFinalFeatures;
  /** Original yaml_content for reference. */
  rawYaml: string;
}

// ─── Run types ────────────────────────────────────────────────────────────────

export interface ArtifactCheckResult {
  exists: boolean;
  processed_table: string | null;
}

/** @deprecated Use ProcessingRunResult or TrainingRunResult for specific run types. */
export interface RunResult {
  dag_run_id: string;
  execution_id: string;
  params_s3_path: string;
  dsl_s3_path?: string;
}

/** Result from POST /api/v2/processing-runs */
export interface ProcessingRunResult {
  dag_run_id: string;
  preprocess_run_id: string;
  artifact_set_id: string;
  preprocess_params_s3_path: string;
  dsl_s3_path: string;
  schema_version: number;
}

/** Result from POST /api/v2/runs */
export interface TrainingRunResult {
  dag_run_id: string;
  dag_id?: string;
  train_run_id: string;
  preprocess_run_id: string;
  train_params_s3_path: string;
  processed_table: string;
  skypilot?: boolean;
  use_managed_jobs?: boolean;
}

export interface RunStatus {
  dag_run_id: string;
  state: string;
  start_date: string | null;
  end_date: string | null;
}

export interface SplitRange {
  start: string;
  end: string;
}

export interface TuningConfig {
  enabled: boolean;
  number_of_trials: number;
}

export interface ModelConfig {
  experiment_name?: string;
  registry_model_name?: string;
  mlflow_tracking_uri?: string;
  mlflow_artifact_location?: string;
  target?: string;
  num_classes?: number;
  seed?: number;
  task_type?: 'classification' | 'regression';
  model_type?: string;
}

/** One entry in a manually-ranked GPU fallback list (mirrors SkyPilot any_of format). */
export interface GPUFallbackEntry {
  /** SkyPilot infra path: "runpod", "vast", "aws/us-east-1", etc. */
  infra: string;
  /** Accelerator spec, e.g. "A100-80GB:1", "A100:2", "H100:8" */
  accelerators: string;
  use_spot: boolean;
}

export interface ResourceConstraints {
  providers?: string[];
  gpu_types?: string[] | null;
  min_vram_gb?: number;
  max_price_per_hour?: number;
  prefer_spot?: boolean;
  require_infiniband?: boolean;
  preferred_regions?: string[];
  num_nodes?: number;
  num_gpus_per_node?: number;
  job_type?: string;
  /** Explicit ranked fallback list; when set, overrides the auto-catalog selector. */
  gpu_fallbacks?: GPUFallbackEntry[] | null;
}

export interface RunRequest {
  preprocess_run_id: string;   // replaces processed_table
  execution_id?: string;
  framework: 'xgboost' | 'pytorch';
  use_gpu?: boolean;
  // true -> sky jobs launch (managed), false -> sky launch --retry-until-up (direct)
  use_managed_jobs?: boolean;
  resource_constraints?: ResourceConstraints | null;
  tuning?: TuningConfig;
  model?: ModelConfig;
  sample_fraction_for_tuning?: number;
  hyperparams?: Record<string, unknown>;
  tune_settings?: Record<string, unknown>;
  /** Per-run search space overrides sent when Override mode is active in the UI.
   *  Shape matches the SearchSpaceEntry union in hyperparams.ts serialised to JSON. */
  search_space?: Record<string, { type: string; options?: number[]; min?: number; max?: number; value?: unknown }>;
}


/** Entry from GET /api/v2/processing-runs/ids */
export interface PreprocessRunId {
  preprocess_run_id: string;
  dataset: string;
}

/** Entry from GET /api/v2/runs/ids */
export interface TrainingRunId {
  train_run_id: string;
  dataset: string;
}

export interface SchemaUploadSingleResult {
  version: number;
  s3_path: string;
}

export interface SkyServeControllerConfig {
  high_availability?: boolean;
  resources?: {
    infra?: string;
    cpus?: string;
    disk_size?: number;
  };
}

export interface ServingConfigRequest {
  train_run_id: string;
  serving_mode: 'ray_only' | 'kafka';
  raw_schema_s3_path?: string;   // required when serving_mode === 'kafka'
  alias?: string;
  canary?: boolean;
  canary_alias?: string;
  canary_probability?: number;
  initial_replicas?: number;
  webhook_public_base_url?: string;
  webhook_path?: string;
  webhook_max_timestamp_age_seconds?: number;
  deployment_target?: 'skypilot';
  min_replicas?: number;
  max_replicas?: number;
  target_qps_per_replica?: number;
  resource_constraints?: ResourceConstraints;
  serve_controller?: SkyServeControllerConfig;
}

export interface ServingConfigResult {
  serve_run_id: string;
  params_s3_path: string;
  dataset: string;
  train_run_id: string;
  serving_mode: 'ray_only' | 'kafka';
  registry_model_name: string;
  deployment_target?: 'skypilot';
}

export interface ServingDeployResult {
  dag_run_id: string;
  serve_run_id: string;
  dag_id?: string;
}

export interface ServingDeployStatus {
  dag_run_id: string;
  serve_run_id: string;
  dag_id?: string;
  state: string;
  start_date: string | null;
  end_date: string | null;
}

// ─── Processing run types ─────────────────────────────────────────────────────

export interface ProcessingRunRequest {
  dataset: string;
  dsl_version: number;
  execution_id?: string;
  splits: {
    train: SplitRange;
    val: SplitRange;
    test: SplitRange;
  };
}

export interface ProcessedTableEntry {
  execution_id: string;
  dataset: string;           // alias de raw_dataset_name (backward-compat)
  processed_table_name: string;
  pipeline_hash: string;
  dsl_name: string;          // e.g. "v1__network_traffic.yaml"
  created_at: string;
  raw_dataset_name: string;  // nombre explícito del dataset fuente
}

export interface PreprocessParamsResult {
  execution_id: string;
  yaml_content: string;
}

export interface ProcessingRunsResult {
  runs: ProcessedTableEntry[];
}

// ─── Schema types ─────────────────────────────────────────────────────────────

export interface SchemaUploadRequest {
  raw: string;
  full: string;
  preprocessed: string;
}

export interface SchemaUploadResult {
  version: number;
  uploaded: {
    raw: string;
    full: string;
    preprocessed: string;
  };
}

/** Lightweight schema response from /iceberg-schema — column list only, no rows. */
export interface IcebergSchemaResult {
  columns: ColumnMeta[];
}

// ─── Dataset API ──────────────────────────────────────────────────────────────

export async function listDatasets(): Promise<DatasetInfo[]> {
  return _fetch<DatasetInfo[]>('/api/v2/datasets');
}

export async function createDataset(name: string): Promise<DatasetInfo> {
  return _fetch<DatasetInfo>('/api/v2/datasets', {
    method: 'POST',
    body: JSON.stringify({ name }),
  });
}

export async function uploadParquets(
  datasetName: string,
  files: File[],
): Promise<UploadResult> {
  const form = new FormData();
  for (const f of files) form.append('files', f);
  const res = await fetch(`${API_BASE}/api/v2/datasets/${datasetName}/upload`, {
    method: 'POST',
    body: form,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
  return res.json() as Promise<UploadResult>;
}

export async function submitIngest(datasetName: string): Promise<IngestResult> {
  return _fetch<IngestResult>(`/api/v2/datasets/${datasetName}/ingest`, {
    method: 'POST',
  });
}

export async function getIngestStatus(
  datasetName: string,
  jobName: string,
): Promise<IngestStatus> {
  return _fetch<IngestStatus>(`/api/v2/datasets/${datasetName}/ingest/${jobName}/status`);
}

export async function getIcebergSample(
  datasetName: string,
  limit = 3000,
): Promise<SampleResult> {
  return _fetch<SampleResult>(
    `/api/v2/datasets/${datasetName}/sample?limit=${limit}`,
  );
}

/**
 * Fetch column schema from iceberg.raw.{datasetName} without loading any rows.
 * Used by the full.yaml editor "Load from Iceberg" button.
 */
export async function getIcebergSchema(datasetName: string): Promise<IcebergSchemaResult> {
  return _fetch<IcebergSchemaResult>(`/api/v2/datasets/${datasetName}/iceberg-schema`);
}

// ─── DSL API ──────────────────────────────────────────────────────────────────

export async function listDsls(datasetName: string): Promise<DslListResult> {
  return _fetch<DslListResult>(`/api/v2/datasets/${datasetName}/dsls`);
}

export async function getDsl(
  datasetName: string,
  version: number,
): Promise<DslContent> {
  return _fetch<DslContent>(`/api/v2/datasets/${datasetName}/dsls/${version}`);
}

/**
 * Fetch a DSL by version and parse its final_features block.
 * Returns the structured feature lists ready for the schema builder.
 * Throws if the DSL YAML does not contain a final_features key.
 */
export async function getDslFeatures(
  datasetName: string,
  version: number,
): Promise<DslFeaturesResult> {
  const dsl = await getDsl(datasetName, version);
  const { parse } = await import('yaml');
  const parsed = parse(dsl.yaml_content) as {
    final_features?: {
      features?: string[];
      target?: string[];
      metadata?: string[];
      passthrough?: string[];
    };
  };
  const ff = parsed.final_features;
  if (!ff) {
    throw new Error(
      `DSL v${version} ('${dsl.slug}') has no final_features block`,
    );
  }
  return {
    version: dsl.version,
    slug: dsl.slug,
    finalFeatures: {
      features: ff.features ?? [],
      target: ff.target ?? [],
      metadata: ff.metadata ?? [],
      passthrough: ff.passthrough ?? [],
    },
    rawYaml: dsl.yaml_content,
  };
}

export async function saveDsl(
  datasetName: string,
  name: string,
  yamlContent: string,
): Promise<SaveDslResult> {
  return _fetch<SaveDslResult>(`/api/v2/datasets/${datasetName}/dsls`, {
    method: 'POST',
    body: JSON.stringify({ name, yaml_content: yamlContent }),
  });
}

// ─── Runs API ────────────────────────────────────────────────────────────────

export async function checkArtifact(
  executionId: string,
  dataset: string,
): Promise<ArtifactCheckResult> {
  return _fetch<ArtifactCheckResult>(
    `/api/v2/runs/check?execution_id=${encodeURIComponent(executionId)}&dataset=${encodeURIComponent(dataset)}`,
  );
}

export async function submitRun(request: RunRequest): Promise<TrainingRunResult> {
  return _fetch<TrainingRunResult>('/api/v2/runs', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

export async function listTrainingRunIds(): Promise<{ runs: TrainingRunId[] }> {
  return _fetch<{ runs: TrainingRunId[] }>('/api/v2/runs/ids');
}

// ─── Processing Runs API ─────────────────────────────────────────────────────

export async function submitProcessingRun(
  request: ProcessingRunRequest,
): Promise<ProcessingRunResult> {
  return _fetch<ProcessingRunResult>('/api/v2/processing-runs', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

export async function listPreprocessRunIds(dataset?: string): Promise<{ runs: PreprocessRunId[] }> {
  const qs = dataset ? `?dataset=${encodeURIComponent(dataset)}` : '';
  return _fetch<{ runs: PreprocessRunId[] }>(`/api/v2/processing-runs/ids${qs}`);
}

export async function listProcessingRuns(rawDataset?: string): Promise<ProcessingRunsResult> {
  const qs = rawDataset ? `?dataset=${encodeURIComponent(rawDataset)}` : '';
  return _fetch<ProcessingRunsResult>(`/api/v2/processing-runs${qs}`);
}

export async function getPreprocessParams(executionId: string): Promise<PreprocessParamsResult> {
  return _fetch<PreprocessParamsResult>(
    `/api/v2/processing-runs/${encodeURIComponent(executionId)}/params`,
  );
}

export async function getProcessingRunStatus(dagRunId: string): Promise<RunStatus> {
  return _fetch<RunStatus>(
    `/api/v2/processing-runs/${encodeURIComponent(dagRunId)}/status`,
  );
}

// ─── Schema API ───────────────────────────────────────────────────────────────

export async function uploadSchemas(
  datasetName: string,
  schemas: SchemaUploadRequest,
): Promise<SchemaUploadResult> {
  return _fetch<SchemaUploadResult>(`/api/v2/datasets/${encodeURIComponent(datasetName)}/schemas`, {
    method: 'POST',
    body: JSON.stringify(schemas),
  });
}

export async function uploadFullSchema(
  dataset: string,
  yamlContent: string,
): Promise<SchemaUploadSingleResult> {
  return _fetch<SchemaUploadSingleResult>(`/api/v2/schemas/${encodeURIComponent(dataset)}/full`, {
    method: 'POST',
    body: JSON.stringify({ yaml_content: yamlContent }),
  });
}

export async function uploadRawSchema(
  dataset: string,
  yamlContent: string,
): Promise<SchemaUploadSingleResult> {
  return _fetch<SchemaUploadSingleResult>(`/api/v2/schemas/${encodeURIComponent(dataset)}/raw`, {
    method: 'POST',
    body: JSON.stringify({ yaml_content: yamlContent }),
  });
}

// ─── Serving Configs API ──────────────────────────────────────────────────────

export async function submitServingConfig(
  request: ServingConfigRequest,
): Promise<ServingConfigResult> {
  return _fetch<ServingConfigResult>('/api/v2/serving-configs', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

export async function triggerServingDeploy(
  serve_run_id: string,
): Promise<ServingDeployResult> {
  return _fetch<ServingDeployResult>(
    `/api/v2/serving-configs/${encodeURIComponent(serve_run_id)}/deploy`,
    { method: 'POST' },
  );
}

export async function getServingDeployStatus(
  serve_run_id: string,
  dag_run_id: string,
  dag_id?: string,
): Promise<ServingDeployStatus> {
  const q = dag_id ? `?dag_id=${encodeURIComponent(dag_id)}` : '';
  return _fetch<ServingDeployStatus>(
    `/api/v2/serving-configs/${encodeURIComponent(serve_run_id)}/deploy/${encodeURIComponent(dag_run_id)}/status${q}`,
  );
}

// ─── GPU Resources API (Phase 2/3) ────────────────────────────────────────────

export interface GPUOffer {
  provider: string;
  gpu_type: string;
  gpu_count: number;
  vram_gb: number;
  vcpus: number;
  ram_gb: number;
  price_on_demand: number;
  price_spot: number | null;
  spot_available: boolean;
  available_count: number;  // -1 = not tracked/unlimited (AWS); 0 = confirmed no stock (RunPod); >0 = units available
  region: string;
  infiniband: boolean;
  skypilot_supported: boolean;
  skypilot_accelerator: string;
  skypilot_cloud: string;
  provider_region_id: string;
  skypilot_region: string;
  skypilot_zone: string;
  skypilot_infra: string;
}

export interface RunPodRegionOption {
  provider: string;
  provider_region_id: string;
  name: string;
  location: string;
  skypilot_region: string;
  skypilot_zone: string;
  skypilot_infra: string;
}

export interface RunPodRegionAvailability {
  gpu_type: string;
  provider: string;
  provider_region_id: string;
  skypilot_region: string;
  skypilot_zone: string;
  skypilot_infra: string;
  available: boolean;
  available_counts: number[];
  max_available: number;
}

export interface GPUSelectResult {
  any_of: Array<{ infra: string; accelerators: string; use_spot: boolean }>;
  spot_entries: number;
  ondemand_entries: number;
  estimated_cost_spot: number | null;
  estimated_cost_ondemand: number | null;
}

export interface LLMModelInfo {
  model_id: string;
  vram_gb: number;
  min_gpus: number;
  recommended_gpu: string;
}

export async function queryGPUCatalog(params?: {
  providers?: string;
  min_vram?: number;
}): Promise<GPUOffer[]> {
  const q = new URLSearchParams();
  if (params?.providers) q.set('providers', params.providers);
  if (params?.min_vram != null) q.set('min_vram', String(params.min_vram));
  const qs = q.toString() ? `?${q.toString()}` : '';
  return _fetch<GPUOffer[]>(`/api/v2/gpu-resources/catalog${qs}`);
}

export async function selectGPUResources(
  constraints: ResourceConstraints,
): Promise<GPUSelectResult> {
  return _fetch<GPUSelectResult>('/api/v2/gpu-resources/select', {
    method: 'POST',
    body: JSON.stringify(constraints),
  });
}

export async function getLLMCatalog(): Promise<LLMModelInfo[]> {
  return _fetch<LLMModelInfo[]>('/api/v2/gpu-resources/llm-catalog');
}

export async function getRunpodSupportedRegions(): Promise<RunPodRegionOption[]> {
  return _fetch<RunPodRegionOption[]>('/api/v2/gpu-resources/runpod/regions');
}

export async function queryRunpodRegionAvailability(payload: {
  gpu_types: string[];
  regions: string[];
}): Promise<RunPodRegionAvailability[]> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 25_000);
  try {
    const res = await fetch(`${API_BASE}/api/v2/gpu-resources/runpod/availability`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    if (!res.ok) {
      const body = await res.text();
      throw new Error(`${res.status} ${res.statusText}: ${body}`);
    }
    return res.json() as Promise<RunPodRegionAvailability[]>;
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') {
      throw new Error('RunPod availability request timed out');
    }
    throw err;
  } finally {
    clearTimeout(timeoutId);
  }
}

// ─── vLLM Serving API (Phase 6) ───────────────────────────────────────────────

export interface VllmEndpointResult {
  endpoint_url: string;
  model_id: string;
  status: 'healthy' | 'pending' | 'not_found';
}

export async function getVllmEndpoint(
  serveRunId: string,
): Promise<VllmEndpointResult> {
  return _fetch<VllmEndpointResult>(
    `/api/v2/serving-configs/${encodeURIComponent(serveRunId)}/endpoint`,
  );
}

// ─── Model Architectures API (Phase 8) ────────────────────────────────────────

export interface ArchitectureInfo {
  id: string;
  name: string;
  description: string;
  builtin: boolean;
  s3_path?: string | null;
  uploaded_at?: string | null;
}

export interface ArchitectureUploadResult {
  id: string;
  name: string;
  s3_path: string;
  status: string;
}

export async function listArchitectures(): Promise<ArchitectureInfo[]> {
  return _fetch<ArchitectureInfo[]>('/api/v2/model-architectures/');
}

export async function uploadArchitecture(
  file: File,
  name: string,
  description?: string,
): Promise<ArchitectureUploadResult> {
  const form = new FormData();
  form.append('file', file);
  form.append('name', name);
  if (description) form.append('description', description);
  const res = await fetch(`${API_BASE}/api/v2/model-architectures/upload`, {
    method: 'POST',
    body: form,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
  return res.json() as Promise<ArchitectureUploadResult>;
}

// ─── Jobs API (Phase 8) ───────────────────────────────────────────────────────

export interface ModelConfigIn {
  model_type: string;
  model_id?: string;
  architecture_s3?: string;
  vram_gb?: number;
}

export interface TrainingConfigIn {
  preprocess_run_id?: string;
  dataset?: string;
  processed_table?: string;
  dataset_s3_path?: string;
  use_gpu?: boolean;
  use_managed_jobs?: boolean;
  use_deepspeed?: boolean;
  deepspeed_stage?: number;
  lora_enabled?: boolean;
  lora_rank?: number;
  max_steps?: number;
  save_steps?: number;
  hf_token?: string;
  hyperparams?: Record<string, unknown>;
  model_cfg?: Record<string, unknown>;
  num_nodes?: number;
}

export interface ServingConfigIn {
  hf_token?: string;
  llm_adapter_s3?: string;
  vllm_port?: number;
  max_model_len?: number;
  tensor_parallel_size?: number;
  pipeline_parallel_size?: number;
  num_nodes?: number;
  replicas?: number;
}

export interface LaunchRequest {
  job_type: 'training' | 'serving' | 'both';
  model: ModelConfigIn;
  resource_constraints?: ResourceConstraints | null;
  training?: TrainingConfigIn;
  serving?: ServingConfigIn;
  dry_run?: boolean;
}

export interface OrchestratorRecommendation {
  orchestration: string;
  dag_id: string;
  sky_yaml_template: string;
  reason: string;
  estimated_cost_spot: number | null;
  estimated_cost_ondemand: number | null;
  warnings: string[];
}

export interface LaunchResponse {
  job_ids: Record<string, string>;
  orchestration: string;
  recommendation: OrchestratorRecommendation;
  sky_yaml_preview: string;
  dry_run: boolean;
}

export interface JobStatus {
  job_id: string;
  dag_run_id: string;
  dag_id: string;
  state: string;
  start_date: string | null;
  end_date: string | null;
}

export async function launchJob(request: LaunchRequest): Promise<LaunchResponse> {
  return _fetch<LaunchResponse>('/api/v2/jobs/launch', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

export async function getJobStatus(
  jobId: string,
  dagId?: string,
): Promise<JobStatus> {
  const q = dagId ? `?dag_id=${encodeURIComponent(dagId)}` : '';
  return _fetch<JobStatus>(`/api/v2/jobs/${encodeURIComponent(jobId)}/status${q}`);
}

export async function listJobs(dagId?: string, limit = 20): Promise<JobStatus[]> {
  const q = new URLSearchParams();
  if (dagId) q.set('dag_id', dagId);
  q.set('limit', String(limit));
  return _fetch<JobStatus[]>(`/api/v2/jobs/?${q.toString()}`);
}

export async function cancelJob(jobId: string, dagId?: string): Promise<{ job_id: string; state: string }> {
  const q = dagId ? `?dag_id=${encodeURIComponent(dagId)}` : '';
  return _fetch(`/api/v2/jobs/${encodeURIComponent(jobId)}${q}`, {
    method: 'DELETE',
  });
}
