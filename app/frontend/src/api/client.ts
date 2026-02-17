/**
 * API Client for backend communication.
 *
 * All functions throw on failure (fail-fast). Errors are propagated
 * to the store which logs them to the console panel.
 */

import type { SparkSchema } from '../types/schema';
import type { DryRunResult, DryRunRequest } from '../types/dryrun';

const BASE_URL = '/api';

async function fetchJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...init?.headers,
    },
  });

  if (!response.ok) {
    const text = await response.text().catch(() => response.statusText);
    throw new Error(`HTTP ${response.status}: ${text}`);
  }

  return response.json() as Promise<T>;
}

// ─── Schema endpoints ───────────────────────────────────────────────

export interface SchemaFromCSVRequest {
  path: string;
  delimiter?: string;
  header?: boolean;
  sample_rows?: number;
}

export async function fetchSchemaFromCSV(req: SchemaFromCSVRequest): Promise<SparkSchema> {
  return fetchJSON<SparkSchema>(`${BASE_URL}/schema/from-csv`, {
    method: 'POST',
    body: JSON.stringify(req),
  });
}

export async function uploadSchemaFile(file: File): Promise<SparkSchema> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${BASE_URL}/schema/upload-file`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const text = await response.text().catch(() => response.statusText);
    throw new Error(`HTTP ${response.status}: ${text}`);
  }

  return response.json() as Promise<SparkSchema>;
}

// Backwards-compatible alias
export { uploadSchemaFile as uploadSchemaFromCSV };

export async function fetchSchemaFromIceberg(table: string): Promise<SparkSchema> {
  return fetchJSON<SparkSchema>(
    `${BASE_URL}/schema/from-iceberg?table=${encodeURIComponent(table)}`,
  );
}

// ─── Dry-run endpoint ───────────────────────────────────────────────

export async function executeDryRun(req: DryRunRequest): Promise<DryRunResult> {
  return fetchJSON<DryRunResult>(`${BASE_URL}/dry-run`, {
    method: 'POST',
    body: JSON.stringify(req),
  });
}

// ─── YAML validation endpoint ──────────────────────────────────────

export interface ValidateYAMLResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
  stageCount?: number;
}

export async function validateYAMLOnBackend(yamlContent: string): Promise<ValidateYAMLResult> {
  return fetchJSON<ValidateYAMLResult>(`${BASE_URL}/validate-yaml`, {
    method: 'POST',
    body: JSON.stringify({ yaml: yamlContent }),
  });
}
