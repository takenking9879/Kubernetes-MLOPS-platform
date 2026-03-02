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

export interface RunResult {
  dag_run_id: string;
  execution_id: string;
  params_s3_path: string;
  dsl_s3_path?: string;
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
  target?: string;
  num_classes?: number;
  seed?: number;
}

export interface RunRequest {
  processed_table: string;
  execution_id?: string;
  framework: 'xgboost' | 'pytorch';
  tuning?: TuningConfig;
  model?: ModelConfig;
  sample_fraction_for_tuning?: number;
  hyperparams?: Record<string, unknown>;
  tune_settings?: Record<string, unknown>;
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

export async function submitRun(request: RunRequest): Promise<RunResult> {
  return _fetch<RunResult>('/api/v2/runs', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

export async function getRunStatus(dagRunId: string): Promise<RunStatus> {
  return _fetch<RunStatus>(`/api/v2/runs/${encodeURIComponent(dagRunId)}/status`);
}

// ─── Processing Runs API ─────────────────────────────────────────────────────

export async function submitProcessingRun(
  request: ProcessingRunRequest,
): Promise<RunResult> {
  return _fetch<RunResult>('/api/v2/processing-runs', {
    method: 'POST',
    body: JSON.stringify(request),
  });
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
