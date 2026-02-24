/**
 * API client for the dataset-oriented platform endpoints (v2).
 *
 * All functions throw on non-2xx responses.
 */

import type { ColumnMeta } from '../types/schema';

export type { ColumnMeta };

const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

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

// ─── Run types ────────────────────────────────────────────────────────────────

export interface ArtifactCheckResult {
  exists: boolean;
  processed_table: string | null;
}

export interface RunResult {
  dag_run_id: string;
  execution_id: string;
  params_s3_path: string;
  dsl_s3_path: string;
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

export interface RunRequest {
  dataset: string;
  dsl_version: number;
  execution_id?: string;
  framework: 'xgboost' | 'pytorch';
  splits: {
    train: SplitRange;
    val: SplitRange;
    test: SplitRange;
  };
  hyperparams?: Record<string, unknown>;
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
