/**
 * Spark data types supported by the DSL.
 * Maps 1:1 to PySpark types used in src/dsl/transformers.py
 */
export type SparkDataType =
  | 'string'
  | 'integer'
  | 'double'
  | 'timestamp'
  | 'long'
  | 'boolean';

/** Numeric Spark types used for group selectors */
export const NUMERIC_SPARK_TYPES: ReadonlySet<SparkDataType> = new Set([
  'integer',
  'long',
  'double',
]);

/** Categorical Spark types used for group selectors */
export const CATEGORICAL_SPARK_TYPES: ReadonlySet<SparkDataType> = new Set([
  'string',
]);

/**
 * Column metadata from Spark schema.
 * cardinality and nullRatio are populated by the backend's schema endpoint.
 */
export interface ColumnMeta {
  readonly name: string;
  readonly sparkType: SparkDataType;
  readonly nullable: boolean;
  readonly cardinality?: number;
  readonly nullRatio?: number;
}

/**
 * Cardinality bin boundaries for SchemaPanel visualization.
 * Used for grouping columns by their distinct value count.
 */
export const CARDINALITY_BINS = [
  { label: '0-10', min: 0, max: 10 },
  { label: '10-50', min: 10, max: 50 },
  { label: '50-500', min: 50, max: 500 },
  { label: '500-1k', min: 500, max: 1000 },
  { label: '1k+', min: 1000, max: Infinity },
] as const;

/**
 * Complete DataFrame schema returned by backend endpoints.
 */
export interface SparkSchema {
  readonly columns: readonly ColumnMeta[];
  readonly sampleRows?: readonly Record<string, unknown>[];
}

/**
 * Schema evolution result after a single stage transformation.
 * Used by the LAF compiler during schema simulation.
 */
export interface SchemaEvolution {
  readonly stageId: string;
  readonly inputSchema: SparkSchema;
  readonly outputSchema: SparkSchema;
  readonly columnsAdded: readonly string[];
}

/**
 * Classify a column as numeric or categorical based on its Spark type.
 */
export function classifyColumn(col: ColumnMeta): 'numeric' | 'categorical' | 'other' {
  if (NUMERIC_SPARK_TYPES.has(col.sparkType)) return 'numeric';
  if (CATEGORICAL_SPARK_TYPES.has(col.sparkType)) return 'categorical';
  return 'other';
}

/**
 * Get the cardinality bin label for a given cardinality value.
 */
export function getCardinalityBin(cardinality: number): string {
  for (const bin of CARDINALITY_BINS) {
    if (cardinality >= bin.min && cardinality < bin.max) {
      return bin.label;
    }
  }
  return '1k+';
}
