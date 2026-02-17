import type { SparkSchema } from './schema';

export interface DryRunRequest {
  readonly yaml: string;
  readonly datasetPath: string;
  readonly sampleLimit?: number;
}

/** Successful dry-run response from backend */
export interface DryRunSuccess {
  readonly success: true;
  readonly schema: SparkSchema;
  readonly preview: readonly Record<string, unknown>[];
  readonly metrics: {
    readonly executionTime: number;
    readonly stageCount: number;
  };
}

/** Failed dry-run response from backend */
export interface DryRunFailure {
  readonly success: false;
  readonly error: {
    readonly message: string;
    readonly sparkTrace?: string;
  };
}

/** Discriminated union for dry-run result */
export type DryRunResult = DryRunSuccess | DryRunFailure;
