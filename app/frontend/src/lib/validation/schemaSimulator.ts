/**
 * Stage-level Schema Simulator.
 *
 * Simulates a single stage against its current input schema and resolved input
 * columns. Used by the incremental compiler in lafValidator.ts.
 *
 * FAIL-FAST:
 *  - Throws if any resolved input does not exist in input schema.
 *  - Throws if output columns are duplicated or collide with existing columns.
 */

import type { SparkSchema, ColumnMeta, SparkDataType } from '../../types/schema';
import type { StageNodeData } from '../../types/nodes';
import { getNodeDefinition } from '../../registry/NodeRegistry';
import type { OutputNamingStrategy } from '../../types/naming';

export interface StageSimulationResult {
  readonly resolvedOutputs: readonly string[];
  readonly outputSchema: SparkSchema;
  readonly columnsAdded: readonly string[];
}

export function mergeSchemas(schemas: readonly SparkSchema[]): SparkSchema {
  const merged = new Map<string, ColumnMeta>();

  for (const schema of schemas) {
    for (const col of schema.columns) {
      const existing = merged.get(col.name);
      if (!existing) {
        merged.set(col.name, col);
        continue;
      }

      if (existing.sparkType !== col.sparkType) {
        throw new Error(
          `Conflicting Spark types for column '${col.name}': '${existing.sparkType}' vs '${col.sparkType}'`,
        );
      }
    }
  }

  return { columns: Array.from(merged.values()) };
}

export function simulateStage(
  data: StageNodeData,
  inputSchema: SparkSchema,
  resolvedInputs: readonly string[],
): StageSimulationResult {
  const virtualColumns = new Map<string, ColumnMeta>();
  for (const col of inputSchema.columns) {
    virtualColumns.set(col.name, col);
  }

  for (const colName of resolvedInputs) {
    if (!virtualColumns.has(colName)) {
      throw new Error(
        `Stage '${data.label}' (${data.type}) requires column '${colName}' but it does not exist in schema`,
      );
    }
  }

  // Type constraint check
  const def = getNodeDefinition(data.type);
  if (def.inputTypeConstraint) {
    const { allowedTypes, errorMessage } = def.inputTypeConstraint;
    for (const colName of resolvedInputs) {
      const col = virtualColumns.get(colName);
      if (col && !allowedTypes.includes(col.sparkType)) {
        throw new TypeError(
          `${errorMessage}: column '${colName}' is ${col.sparkType}`,
        );
      }
    }
  }

  const outputCols = inferOutputColumns(data, resolvedInputs, virtualColumns);

  for (const out of outputCols) {
    if (virtualColumns.has(out.name)) {
      throw new Error(
        `Stage '${data.label}' (${data.type}) would produce duplicate output column '${out.name}'`,
      );
    }
    virtualColumns.set(out.name, out);
  }

  return {
    resolvedOutputs: outputCols.map((c) => c.name),
    outputSchema: { columns: Array.from(virtualColumns.values()) },
    columnsAdded: outputCols.map((c) => c.name),
  };
}

function inferOutputColumns(
  data: StageNodeData,
  inputColumns: readonly string[],
  virtualColumns: ReadonlyMap<string, ColumnMeta>,
): ColumnMeta[] {
  const def = getNodeDefinition(data.type);
  const outputs = deriveOutputNames(inputColumns, data.namingStrategy, data.type, def.transformSignature);

  if (outputs.length === 0) {
    throw new Error(
      `Stage '${data.label}' (${data.type}) has no derived output columns`,
    );
  }

  const seenOutputs = new Set<string>();
  for (const outputName of outputs) {
    if (seenOutputs.has(outputName)) {
      throw new Error(
        `Stage '${data.label}' (${data.type}) has duplicate output column '${outputName}'`,
      );
    }
    seenOutputs.add(outputName);
  }

  // If outputMatchesInput, validate lengths match
  if (def.outputMatchesInput && outputs.length !== inputColumns.length) {
    throw new Error(
      `Stage '${data.label}' (${data.type}) has ${inputColumns.length} inputs ` +
      `but ${outputs.length} outputs. They must match for this stage type.`,
    );
  }

  // Determine output type
  const outType: SparkDataType = resolveOutputType(data, def.outputType, inputColumns, virtualColumns);

  return outputs.map((name) => ({
    name,
    sparkType: outType,
    nullable: false,
  }));
}

function deriveOutputNames(
  inputColumns: readonly string[],
  namingStrategy: OutputNamingStrategy,
  nodeType: string,
  signature: 'one_to_one' | 'many_to_one' | 'many_to_many',
): string[] {
  if (inputColumns.length === 0) return [];

  if (namingStrategy.mode === 'custom') {
    return [...namingStrategy.columns];
  }

  if (signature === 'many_to_one') {
    const base = inputColumns[0] ?? nodeType;
    return [applyNaming(base, namingStrategy, nodeType)];
  }

  if (signature === 'many_to_many') {
    return inputColumns.map((input, index) => applyNaming(`${input}_${index + 1}`, namingStrategy, nodeType));
  }

  return inputColumns.map((input) => applyNaming(input, namingStrategy, nodeType));
}

function applyNaming(
  inputName: string,
  namingStrategy: Exclude<OutputNamingStrategy, { mode: 'custom' }>,
  nodeType: string,
): string {
  switch (namingStrategy.mode) {
    case 'auto':
      return `${inputName}_${nodeType}`;
    case 'prefix':
      return `${namingStrategy.value}${inputName}`;
    case 'suffix':
      return `${inputName}${namingStrategy.value}`;
    case 'passthrough':
      return inputName;
  }
}

function resolveOutputType(
  data: StageNodeData,
  registryOutputType: SparkDataType | null,
  inputColumns: readonly string[],
  virtualColumns: ReadonlyMap<string, ColumnMeta>,
): SparkDataType {
  // Special case: cast_transformer output type comes from the param
  if (data.type === 'cast_transformer') {
    const targetType = data.params.target_type;
    // 'long' maps to SparkDataType 'long'
    return targetType as SparkDataType;
  }

  // Special case: imputer preserves input type
  const firstInputName = inputColumns[0];
  if (data.type === 'imputer' && firstInputName !== undefined) {
    const firstInput = virtualColumns.get(firstInputName);
    return firstInput?.sparkType ?? 'double';
  }

  // Use registry default
  if (registryOutputType !== null) return registryOutputType;

  // Fallback: preserve first input type
  if (firstInputName !== undefined) {
    const firstInput = virtualColumns.get(firstInputName);
    return firstInput?.sparkType ?? 'double';
  }

  return 'double';
}
