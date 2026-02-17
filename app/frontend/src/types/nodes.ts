import type { SparkSchema } from './schema';
import type { OutputNamingStrategy } from './naming';

// ─── Utility ────────────────────────────────────────────────────────

/** Exhaustive check helper. Causes a compile error if a switch is not exhaustive. */
export function assertNever(x: never): never {
  throw new Error(`Unexpected discriminant: ${JSON.stringify(x)}`);
}

// ─── Stage type literal union ───────────────────────────────────────

export type TransformerType =
  | 'cast_transformer'
  | 'temporal_extractor'
  | 'arithmetic_transformer'
  | 'cyclic_transformer'
  | 'log_transformer'
  | 'binning_transformer'
  | 'conditional_transformer'
  | 'concat_transformer'
  | 'ratio_transformer';

export type EstimatorType =
  | 'string_indexer'
  | 'standard_scaler'
  | 'minmax_scaler'
  | 'imputer'
  | 'frequency_encoder';

export type StageType = TransformerType | EstimatorType;

export type FlowNodeType = 'dataset' | StageType | 'final_features';

// ─── Base interfaces ────────────────────────────────────────────────

export interface BaseNodeData {
  readonly id: string;
  readonly label: string;
}

export interface BaseStageNodeData extends BaseNodeData {
  readonly stageType: 'transformer' | 'estimator';
  inputs: string[];
  outputs: string[];
  params: Record<string, unknown>;
  namingStrategy: OutputNamingStrategy;
  resolvedInputs: string[];
  resolvedOutputs: string[];
  outputSchema: SparkSchema | null;
  compilation: NodeCompilationState;
}

export type NodeCompilationState =
  | { status: 'draft' }
  | {
      status: 'compiled';
      inputSchema: SparkSchema;
      outputSchema: SparkSchema;
    }
  | {
      status: 'error';
      errors: string[];
    };

// ─── Dataset node ───────────────────────────────────────────────────

export interface DatasetNodeData extends BaseNodeData {
  readonly type: 'dataset';
  source: 'csv' | 'iceberg';
  path: string;
  schema: SparkSchema | null;
}

// ─── Transformer nodes ──────────────────────────────────────────────

export interface CastTransformerData extends BaseStageNodeData {
  readonly type: 'cast_transformer';
  readonly stageType: 'transformer';
  params: { target_type: 'timestamp' | 'double' | 'integer' | 'string' | 'long' };
}

export interface TemporalExtractorData extends BaseStageNodeData {
  readonly type: 'temporal_extractor';
  readonly stageType: 'transformer';
  params: {
    component:
      | 'hour' | 'minute' | 'second' | 'dayofweek' | 'dayofmonth'
      | 'month' | 'year' | 'quarter' | 'weekofyear';
  };
}

export interface ArithmeticTransformerData extends BaseStageNodeData {
  readonly type: 'arithmetic_transformer';
  readonly stageType: 'transformer';
  params: {
    operation: 'add' | 'subtract' | 'multiply' | 'divide' | 'power' | 'absolute';
    value: number;
  };
}

export interface CyclicTransformerData extends BaseStageNodeData {
  readonly type: 'cyclic_transformer';
  readonly stageType: 'transformer';
  params: { function: 'sin' | 'cos'; period: number };
}

export interface LogTransformerData extends BaseStageNodeData {
  readonly type: 'log_transformer';
  readonly stageType: 'transformer';
  params: { log_type: 'log1p' | 'log' | 'log10' };
}

export interface BinningTransformerData extends BaseStageNodeData {
  readonly type: 'binning_transformer';
  readonly stageType: 'transformer';
  params: { bins: number[]; labels?: string[] };
}

export interface ConditionalTransformerData extends BaseStageNodeData {
  readonly type: 'conditional_transformer';
  readonly stageType: 'transformer';
  params: {
    condition: 'isin' | 'greater_than' | 'less_than' | 'equals';
    values: unknown[];
    true_value: unknown;
    false_value: unknown;
  };
}

export interface ConcatTransformerData extends BaseStageNodeData {
  readonly type: 'concat_transformer';
  readonly stageType: 'transformer';
  params: { separator: string };
}

export interface RatioTransformerData extends BaseStageNodeData {
  readonly type: 'ratio_transformer';
  readonly stageType: 'transformer';
  params: { default_value?: number };
}

// ─── Estimator nodes ────────────────────────────────────────────────

export interface StringIndexerData extends BaseStageNodeData {
  readonly type: 'string_indexer';
  readonly stageType: 'estimator';
  params: { handle_invalid: 'keep' | 'skip' | 'error' };
}

export interface StandardScalerData extends BaseStageNodeData {
  readonly type: 'standard_scaler';
  readonly stageType: 'estimator';
  params: { with_mean: boolean; with_std: boolean };
}

export interface MinMaxScalerData extends BaseStageNodeData {
  readonly type: 'minmax_scaler';
  readonly stageType: 'estimator';
  params: Record<string, never>;
}

export interface ImputerData extends BaseStageNodeData {
  readonly type: 'imputer';
  readonly stageType: 'estimator';
  params: { strategy: 'mean' | 'median' | 'mode' };
}

export interface FrequencyEncoderData extends BaseStageNodeData {
  readonly type: 'frequency_encoder';
  readonly stageType: 'estimator';
  params: { encoding: 'freq' | 'count'; top_k?: number; smoothing?: number };
}

// ─── Final Feature Selector ─────────────────────────────────────────

export interface FinalFeatureSelectorData extends BaseNodeData {
  readonly type: 'final_features';
  features: string[];
  target: string[];
  metadata: string[];
  passthrough: string[];
  manualOverrides: Record<string, 'feature' | 'target' | 'metadata' | 'passthrough'>;
  confirmedColumns: string[];
}

// ─── Discriminated union ────────────────────────────────────────────

export type StageNodeData =
  | CastTransformerData
  | TemporalExtractorData
  | ArithmeticTransformerData
  | CyclicTransformerData
  | LogTransformerData
  | BinningTransformerData
  | ConditionalTransformerData
  | ConcatTransformerData
  | RatioTransformerData
  | StringIndexerData
  | StandardScalerData
  | MinMaxScalerData
  | ImputerData
  | FrequencyEncoderData;

export type NodeData =
  | DatasetNodeData
  | StageNodeData
  | FinalFeatureSelectorData;

// ─── Type guards ────────────────────────────────────────────────────

export function isStageNode(data: NodeData): data is StageNodeData {
  return 'stageType' in data;
}

export function isDatasetNode(data: NodeData): data is DatasetNodeData {
  return data.type === 'dataset';
}

export function isFinalFeatures(data: NodeData): data is FinalFeatureSelectorData {
  return data.type === 'final_features';
}
