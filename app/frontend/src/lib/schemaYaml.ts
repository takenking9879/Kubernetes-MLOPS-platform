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
  /** Name of the id column */
  idColumn: string;
  /** Name of the prediction column */
  predictionColumn: string;
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
 * Top-level fields are flat. The 'properties' struct wraps its sub-fields
 * using block YAML style. This matches the existing raw.yaml format.
 */
export function generateRawYaml(
  state: SchemaBuilderState,
  datasetName = '',
): string {
  const { allColumns, rawTopLevel, propertiesFields } = state;

  const lines: string[] = [
    `# Schema: ${datasetName || '<dataset>'} — raw Kafka input`,
    'fields:',
  ];

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
 * All preprocessed columns are type: double, nullable: true (DSL output convention).
 * If a column has a different type in allColumns, its actual type is used.
 */
export function generatePreprocessedYaml(
  state: SchemaBuilderState,
  datasetName = '',
): string {
  const { allColumns, preprocessedColumns } = state;

  const lines: string[] = [
    `# Schema: ${datasetName || '<dataset>'} — preprocessed (${preprocessedColumns.length} DSL output features)`,
    'fields:',
  ];

  for (const name of preprocessedColumns) {
    const col = lookupColumn(name, allColumns);
    // DSL output features are always double by convention; preserve actual type if loaded
    const type = col.sparkType !== 'string' ? col.sparkType : 'double';
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
  const { fullColumns, preprocessedColumns, targetColumn } = state;

  if (!targetColumn) {
    errors.push('Schema: target column must be selected');
  }

  if (targetColumn && fullColumns.includes(targetColumn)) {
    // The target must be in fullColumns, but must NOT be in preprocessedColumns
    if (preprocessedColumns.includes(targetColumn)) {
      errors.push('Schema: target column must not be in preprocessed features');
    }
  }

  const fullSet = new Set(fullColumns);
  if (fullSet.size !== fullColumns.length) {
    errors.push('Schema: full.yaml has duplicate columns');
  }

  const prepSet = new Set(preprocessedColumns);
  if (prepSet.size !== preprocessedColumns.length) {
    errors.push('Schema: preprocessed.yaml has duplicate columns');
  }

  if (targetColumn && !fullColumns.includes(targetColumn)) {
    errors.push('Schema: target column must be included in full.yaml columns');
  }

  return errors;
}
