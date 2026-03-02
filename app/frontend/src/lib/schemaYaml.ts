/**
 * Dataset schema YAML generators.
 *
 * Produces YAML strings matching the format in:
 *   k3s/schemas/datasets/network_traffic_raw/raw.yaml
 *   k3s/schemas/datasets/network_traffic_raw/full.yaml
 *   k3s/schemas/datasets/network_traffic_raw/preprocessed.yaml
 *
 * Uses a custom string builder (not pure stringify) to reproduce the exact
 * inline flow-style format: {name: x, type: y, nullable: z}
 */

import type { SparkDataType } from '../types/schema';

// ─── Types ────────────────────────────────────────────────────────────────────

export interface SchemaColumn {
  name: string;
  sparkType: string;
  nullable: boolean;
}

export interface SchemaBuilderState {
  /** All columns loaded from Iceberg (raw sample) */
  allColumns: SchemaColumn[];
  /** Names of columns to include as top-level fields in raw.yaml */
  rawTopLevel: string[];
  /** Names of columns to include as sub-fields of the 'properties' struct in raw.yaml */
  propertiesFields: string[];
  /** Names of columns included in full.yaml (features + target) */
  fullColumns: string[];
  /** Names of DSL output features in preprocessed.yaml */
  preprocessedColumns: string[];
  /** Name of the target column */
  targetColumn: string;
  /** Name of the id column — written as id_field in raw.yaml */
  idColumn: string;
  /**
   * Per-column type overrides for preprocessed.yaml.
   * By default all preprocessed columns are 'double'.
   * Use this to override specific columns to 'long', 'integer', etc.
   */
  typeOverrides?: Record<string, SparkDataType>;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function lookupColumn(
  name: string,
  allColumns: SchemaColumn[],
): SchemaColumn {
  return (
    allColumns.find((c) => c.name === name) ?? {
      name,
      sparkType: 'string',
      nullable: true,
    }
  );
}

/** Renders a flat field as inline YAML: "  - {name: x, type: y, nullable: z}" */
function flatFieldLine(col: SchemaColumn): string {
  return `  - {name: ${col.name}, type: ${col.sparkType}, nullable: ${col.nullable}}`;
}

/** Renders a struct field block (properties sub-field), indented 6 spaces */
function structSubFieldLine(col: SchemaColumn): string {
  return `      - {name: ${col.name}, type: ${col.sparkType}, nullable: ${col.nullable}}`;
}

// ─── Generators ───────────────────────────────────────────────────────────────

/**
 * Generate raw.yaml — raw Kafka input schema.
 *
 * Includes id_field metadata at the top so kafka_main.py can derive
 * the event identifier without per-run params configuration.
 *
 * Top-level fields are flat. The 'properties' struct wraps its sub-fields
 * using block YAML style. This matches the existing raw.yaml format.
 */
export function generateRawYaml(
  state: SchemaBuilderState,
  datasetName = '',
): string {
  const { allColumns, rawTopLevel, propertiesFields, idColumn } = state;

  const lines: string[] = [
    `# Schema: ${datasetName || '<dataset>'} — raw Kafka input`,
  ];

  // id_field annotation: read by kafka_main.py to derive id_column + Kafka message key
  if (idColumn) {
    lines.push(`id_field: ${idColumn}`);
    lines.push('');
  }

  lines.push('fields:');

  for (const name of rawTopLevel) {
    const col = lookupColumn(name, allColumns);
    lines.push(flatFieldLine(col));
  }

  if (propertiesFields.length > 0) {
    lines.push('  - name: properties');
    lines.push('    type: struct');
    lines.push('    nullable: false');
    lines.push('    fields:');
    for (const name of propertiesFields) {
      const col = lookupColumn(name, allColumns);
      lines.push(structSubFieldLine(col));
    }
  }

  return lines.join('\n') + '\n';
}

/**
 * Generate full.yaml — features + target label (flat schema).
 */
export function generateFullYaml(
  state: SchemaBuilderState,
  datasetName = '',
): string {
  const { allColumns, fullColumns } = state;

  const lines: string[] = [
    `# Schema: ${datasetName || '<dataset>'} — full (features + target label)`,
    'fields:',
  ];

  for (const name of fullColumns) {
    const col = lookupColumn(name, allColumns);
    lines.push(flatFieldLine(col));
  }

  return lines.join('\n') + '\n';
}

/**
 * Generate preprocessed.yaml — DSL output features.
 *
 * All preprocessed columns default to type: double, nullable: true.
 * Individual columns can be overridden via state.typeOverrides.
 * The column list is DSL-derived and must NOT be manually edited by users.
 */
export function generatePreprocessedYaml(
  state: SchemaBuilderState,
  datasetName = '',
): string {
  const { preprocessedColumns, typeOverrides = {} } = state;

  const lines: string[] = [
    `# Schema: ${datasetName || '<dataset>'} — preprocessed (${preprocessedColumns.length} DSL output features)`,
    '# Column list is derived from the DSL pipeline — do not edit manually.',
    'fields:',
  ];

  for (const name of preprocessedColumns) {
    // Apply user override if present; default to double (DSL output convention)
    const type: string = typeOverrides[name] ?? 'double';
    lines.push(`  - {name: ${name}, type: ${type}, nullable: true}`);
  }

  return lines.join('\n') + '\n';
}

/**
 * Validate schema builder state for consistency.
 * Returns a list of validation error strings (empty = valid).
 */
export function validateSchemaBuilder(state: SchemaBuilderState): string[] {
  const errors: string[] = [];
  const { fullColumns, preprocessedColumns, targetColumn, propertiesFields, idColumn, rawTopLevel } = state;

  // Target column checks
  if (!targetColumn) {
    errors.push('Schema: target column must be selected');
  }
  if (targetColumn && !fullColumns.includes(targetColumn)) {
    errors.push('Schema: target column must be included in full.yaml columns');
  }
  if (targetColumn && preprocessedColumns.includes(targetColumn)) {
    errors.push('Schema: target column must not be in preprocessed features');
  }

  // Duplicate checks
  const fullSet = new Set(fullColumns);
  if (fullSet.size !== fullColumns.length) {
    errors.push('Schema: full.yaml has duplicate columns');
  }
  const prepSet = new Set(preprocessedColumns);
  if (prepSet.size !== preprocessedColumns.length) {
    errors.push('Schema: preprocessed.yaml has duplicate columns');
  }

  // id_field check
  if (idColumn && !rawTopLevel.includes(idColumn)) {
    errors.push(`Schema: id_field '${idColumn}' must be a top-level field in raw.yaml`);
  }

  // Cross-schema: all properties.* fields must exist in full.yaml (non-target)
  const fullNonTarget = new Set(fullColumns.filter((c) => c !== targetColumn));
  for (const propField of propertiesFields) {
    if (!fullNonTarget.has(propField)) {
      errors.push(
        `Schema: raw.yaml properties field '${propField}' is missing from full.yaml`,
      );
    }
  }

  return errors;
}
