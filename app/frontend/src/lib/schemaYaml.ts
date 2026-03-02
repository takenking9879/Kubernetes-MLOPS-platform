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
 *
 * V1 generators (generateRawYaml, generateFullYaml, generatePreprocessedYaml)
 * are preserved for backward compatibility. V2 generators read from the new
 * structured field types and embed provenance comments.
 */

import type { SparkDataType } from '../types/schema';

// ─── Types ────────────────────────────────────────────────────────────────────

export interface SchemaColumn {
  name: string;
  sparkType: string;
  nullable: boolean;
}

/** Origin of a column in the schema editor. */
export type ColumnSource =
  | 'iceberg'       // loaded directly from Iceberg table schema
  | 'dsl_derived'   // derived from DSL final_features block
  | 'custom'        // manually added by the user
  | 'user_override'; // started as iceberg/dsl but user changed name/type

/** Role of a field in the raw.yaml logical input schema. */
export type RawFieldRole =
  | 'feature'
  | 'target'
  | 'id'
  | 'metadata'
  | 'unassigned';

/** A field in the raw.yaml tree editor (flat or grouped). */
export interface RawFieldEntry {
  name: string;
  sparkType: SparkDataType;
  nullable: boolean;
  role: RawFieldRole;
  source: ColumnSource;
  /** Group key this field belongs to (e.g. 'properties'). undefined = top-level. */
  group?: string;
  /** Display-only note — not written to YAML. */
  note?: string;
}

/** A column in the full.yaml editor (Iceberg-based, editable). */
export interface FullFieldEntry {
  name: string;
  sparkType: SparkDataType;
  nullable: boolean;
  source: ColumnSource;
  /** User-overridden name — written to YAML if set. */
  editedName?: string;
  /** User-overridden type — written to YAML if set. */
  editedType?: SparkDataType;
  note?: string;
}

/** A feature in the preprocessed.yaml editor (DSL-derived, type-overridable). */
export interface PreprocessedFieldEntry {
  name: string;
  /** Type override — default output type is 'double'. */
  typeOverride?: SparkDataType;
  source: ColumnSource; // 'dsl_derived' | 'custom'
  /** DSL-derived entries cannot be removed, only type-overridden. */
  dslLocked: boolean;
}

export interface SchemaBuilderState {
  // ── DSL context ──────────────────────────────────────────────────────────
  /** Dataset name inferred from the selected DSL's associated processing run. */
  dataset: string;
  /** Selected DSL slug — the primary selector for all schema generation. */
  dslName: string;
  /** DSL version number matching the selected dslName. */
  dslVersion: number | '';
  /** Processed table name inferred from processing runs (editable). */
  processedTableName: string;
  /** Provenance string embedded in YAML comments. */
  generatedFrom: string;

  // ── Column data ──────────────────────────────────────────────────────────
  /** All columns loaded from Iceberg (type reference for raw/full editors). */
  allColumns: SchemaColumn[];

  // ── raw.yaml state ───────────────────────────────────────────────────────
  /** Ordered list of fields (includes fields assigned to groups). */
  rawFields: RawFieldEntry[];
  /** Named groups: key → display label. Default: { properties: 'properties' } */
  rawGroups: Record<string, string>;
  /** id_field written to raw.yaml header. Empty = no id_field. */
  idField: string;

  // ── full.yaml state ──────────────────────────────────────────────────────
  fullFields: FullFieldEntry[];

  // ── preprocessed.yaml state ──────────────────────────────────────────────
  preprocessedFields: PreprocessedFieldEntry[];

  // ── Legacy fields (V1 compat — kept in sync by syncLegacyFields) ─────────
  /** @deprecated use rawFields */
  rawTopLevel: string[];
  /** @deprecated use rawFields with group='properties' */
  propertiesFields: string[];
  /** @deprecated use fullFields */
  fullColumns: string[];
  /** @deprecated use preprocessedFields */
  preprocessedColumns: string[];
  targetColumn: string;
  idColumn: string;
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

// ─── V1 Generators (preserved for backward compat) ────────────────────────────

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

// ─── V2 Generators (use structured field types) ───────────────────────────────

/**
 * Generate raw.yaml from RawFieldEntry[].
 *
 * Supports arbitrary named groups (not just 'properties').
 * Top-level fields come first, then each group as a struct block.
 * Embeds provenance in a header comment.
 */
export function generateRawYamlV2(state: SchemaBuilderState): string {
  const { rawFields, rawGroups, idField, dataset, generatedFrom } = state;

  const provenance = generatedFrom
    ? `\n# generated_from: ${generatedFrom}`
    : '';

  const lines: string[] = [
    `# Schema: ${dataset || '<dataset>'} — raw Kafka input${provenance}`,
  ];

  if (idField) {
    lines.push(`id_field: ${idField}`);
    lines.push('');
  }

  lines.push('fields:');

  // Top-level fields (no group)
  for (const f of rawFields) {
    if (f.group) continue;
    lines.push(
      `  - {name: ${f.name}, type: ${f.sparkType}, nullable: ${f.nullable}}`,
    );
  }

  // Group blocks
  for (const groupKey of Object.keys(rawGroups)) {
    const members = rawFields.filter((f) => f.group === groupKey);
    if (members.length === 0) continue;
    lines.push(`  - name: ${groupKey}`);
    lines.push('    type: struct');
    lines.push('    nullable: false');
    lines.push('    fields:');
    for (const f of members) {
      lines.push(
        `      - {name: ${f.name}, type: ${f.sparkType}, nullable: ${f.nullable}}`,
      );
    }
  }

  return lines.join('\n') + '\n';
}

/**
 * Generate full.yaml from FullFieldEntry[].
 *
 * Uses editedName / editedType when present (user overrides).
 * Embeds provenance in a header comment.
 */
export function generateFullYamlV2(state: SchemaBuilderState): string {
  const { fullFields, dataset, generatedFrom } = state;

  const provenance = generatedFrom
    ? `\n# generated_from: ${generatedFrom}`
    : '';

  const lines: string[] = [
    `# Schema: ${dataset || '<dataset>'} — full (features + target label)${provenance}`,
    'fields:',
  ];

  for (const f of fullFields) {
    const name = f.editedName ?? f.name;
    const type = f.editedType ?? f.sparkType;
    lines.push(`  - {name: ${name}, type: ${type}, nullable: ${f.nullable}}`);
  }

  return lines.join('\n') + '\n';
}

/**
 * Generate preprocessed.yaml from PreprocessedFieldEntry[].
 *
 * All fields default to type: double unless typeOverride is set.
 * Embeds provenance in a header comment.
 */
export function generatePreprocessedYamlV2(state: SchemaBuilderState): string {
  const { preprocessedFields, dataset, generatedFrom, dslName } = state;

  const provenance = generatedFrom
    ? `\n# generated_from: ${generatedFrom}`
    : '';

  const lines: string[] = [
    `# Schema: ${dataset || '<dataset>'} — preprocessed (${preprocessedFields.length} DSL output features)${provenance}`,
    `# Column list derived from DSL pipeline${dslName ? ` '${dslName}'` : ''} — do not edit manually.`,
    'fields:',
  ];

  for (const f of preprocessedFields) {
    const type = f.typeOverride ?? 'double';
    lines.push(`  - {name: ${f.name}, type: ${type}, nullable: true}`);
  }

  return lines.join('\n') + '\n';
}

// ─── Adapter: sync legacy fields from new structured fields ──────────────────

/**
 * Projects new structured fields back to V1 legacy shape so that V1
 * generators and validateSchemaBuilder continue to work during transition.
 *
 * Call this inside SchemaBuilderSection's onChange before propagating state
 * up to RunPage.
 */
export function syncLegacyFields(
  state: SchemaBuilderState,
): SchemaBuilderState {
  const rawTopLevel = state.rawFields
    .filter((f) => !f.group)
    .map((f) => f.name);

  const propertiesFields = state.rawFields
    .filter((f) => f.group === 'properties')
    .map((f) => f.name);

  const fullColumns = state.fullFields.map((f) => f.editedName ?? f.name);

  const preprocessedColumns = state.preprocessedFields.map((f) => f.name);

  const targetColumn =
    state.rawFields.find((f) => f.role === 'target')?.name ??
    state.targetColumn;

  const idColumn = state.idField || state.idColumn;

  const typeOverrides: Record<string, SparkDataType> = {};
  for (const f of state.preprocessedFields) {
    if (f.typeOverride) typeOverrides[f.name] = f.typeOverride;
  }

  // allColumns: prefer existing or build from rawFields
  const allColumns =
    state.allColumns.length > 0
      ? state.allColumns
      : state.rawFields.map((f) => ({
          name: f.name,
          sparkType: f.sparkType as string,
          nullable: f.nullable,
        }));

  return {
    ...state,
    allColumns,
    rawTopLevel,
    propertiesFields,
    fullColumns,
    preprocessedColumns,
    targetColumn,
    idColumn,
    typeOverrides,
  };
}

// ─── Validation ───────────────────────────────────────────────────────────────

/**
 * Validate schema builder state for consistency.
 * Returns a list of validation error strings (empty = valid).
 * Checks both legacy and new structured fields.
 */
export function validateSchemaBuilder(state: SchemaBuilderState): string[] {
  const errors: string[] = [];

  // ── V2 validation (when new fields are populated) ─────────────────────────

  if (state.fullFields.length > 0) {
    // Duplicate name check in fullFields
    const names = state.fullFields.map((f) => f.editedName ?? f.name);
    if (new Set(names).size !== names.length) {
      errors.push('Schema: full.yaml has duplicate column names');
    }
  }

  if (state.preprocessedFields.length > 0) {
    // Duplicate name check in preprocessedFields
    const names = state.preprocessedFields.map((f) => f.name);
    if (new Set(names).size !== names.length) {
      errors.push('Schema: preprocessed.yaml has duplicate feature names');
    }
  }

  if (state.rawFields.length > 0) {
    // id_field must be a top-level field
    if (state.idField) {
      const isTopLevel = state.rawFields.some(
        (f) => f.name === state.idField && !f.group,
      );
      if (!isTopLevel) {
        errors.push(
          `Schema: id_field '${state.idField}' must be a top-level field in raw.yaml`,
        );
      }
    }
  }

  // ── V1 / legacy validation ────────────────────────────────────────────────

  const {
    fullColumns,
    preprocessedColumns,
    targetColumn,
    propertiesFields,
    idColumn,
    rawTopLevel,
  } = state;

  // Only run V1 checks when V2 fields are NOT populated (avoid double-reporting)
  if (state.fullFields.length === 0) {
    if (!targetColumn) {
      errors.push('Schema: target column must be selected');
    }
    if (targetColumn && !fullColumns.includes(targetColumn)) {
      errors.push('Schema: target column must be included in full.yaml columns');
    }
    if (targetColumn && preprocessedColumns.includes(targetColumn)) {
      errors.push('Schema: target column must not be in preprocessed features');
    }

    const fullSet = new Set(fullColumns);
    if (fullSet.size !== fullColumns.length) {
      errors.push('Schema: full.yaml has duplicate columns');
    }
    const prepSet = new Set(preprocessedColumns);
    if (prepSet.size !== preprocessedColumns.length) {
      errors.push('Schema: preprocessed.yaml has duplicate columns');
    }

    if (idColumn && !rawTopLevel.includes(idColumn)) {
      errors.push(
        `Schema: id_field '${idColumn}' must be a top-level field in raw.yaml`,
      );
    }

    const fullNonTarget = new Set(fullColumns.filter((c) => c !== targetColumn));
    for (const propField of propertiesFields) {
      if (!fullNonTarget.has(propField)) {
        errors.push(
          `Schema: raw.yaml properties field '${propField}' is missing from full.yaml`,
        );
      }
    }
  }

  return errors;
}
