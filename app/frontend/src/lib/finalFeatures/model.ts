import type { Node } from 'reactflow';
import type { FlowEdge } from '../../types/edges';
import type { NodeData, StageNodeData } from '../../types/nodes';
import { isStageNode } from '../../types/nodes';
import type { SparkSchema } from '../../types/schema';
import type { ValidationResult } from '../../types/validation';

export type FeatureBucket = 'feature' | 'target' | 'metadata' | 'passthrough';

export interface ColumnProvenance {
  readonly stageName: string;
  readonly stageType: string;
  readonly outputColName: string;
}

export interface FinalFeatureColumn {
  readonly name: string;
  readonly sparkType: string;
  readonly sourceNodeId: string;
  readonly provenance?: ColumnProvenance;
  readonly autoBucket: Exclude<FeatureBucket, 'target' | 'metadata'> | 'uncertain';
  readonly conflictReason?: string;
  readonly uncertainReason?: string;
}

const CATEGORICAL_STAGE_TYPES = new Set([
  'string_indexer',
  'frequency_encoder',
  'one_hot_encoder',
  'category_encoder',
]);

const NUMERICAL_STAGE_TYPES = new Set([
  'standard_scaler',
  'minmax_scaler',
  'imputer',
  'arithmetic_transformer',
  'log_transformer',
  'clip_transformer',
  'cyclic_transformer',
]);

const NUMERIC_SPARK_TYPES = new Set([
  'double',
  'float',
  'integer',
  'long',
  'short',
  'decimal',
  'DoubleType',
  'FloatType',
  'IntegerType',
  'LongType',
  'ShortType',
  'DecimalType',
]);

export function isNumericSparkType(sparkType: string): boolean {
  return NUMERIC_SPARK_TYPES.has(sparkType);
}

function fallbackBySparkType(sparkType: string): 'feature' | 'uncertain' {
  if (isNumericSparkType(sparkType)) return 'feature';
  return 'uncertain';
}

function classifyByProvenance(
  stageType: string | undefined,
  sparkType: string,
): { bucket: 'feature' | 'uncertain'; reason?: string } {
  if (stageType && CATEGORICAL_STAGE_TYPES.has(stageType)) {
    if (!isNumericSparkType(sparkType)) {
      return {
        bucket: 'uncertain',
        reason: `Stage '${stageType}' implies encoded feature but column type '${sparkType}' is non-numeric`,
      };
    }
    return { bucket: 'feature' };
  }

  if (stageType && NUMERICAL_STAGE_TYPES.has(stageType)) {
    if (!isNumericSparkType(sparkType)) {
      return {
        bucket: 'uncertain',
        reason: `Stage '${stageType}' implies numerical but column type '${sparkType}' is non-numeric`,
      };
    }
    return { bucket: 'feature' };
  }

  return { bucket: fallbackBySparkType(sparkType) };
}

function getSourceSchema(
  nodeId: string,
  datasetNodeId: string,
  datasetSchema: SparkSchema | null,
  validationResult: ValidationResult | null,
): SparkSchema | null {
  if (nodeId === datasetNodeId) {
    return datasetSchema;
  }
  const compiled = validationResult?.compiledNodes.get(nodeId);
  return compiled?.outputSchema ?? null;
}

export function collectFinalFeatureColumns(
  nodes: readonly Node<NodeData>[],
  edges: readonly FlowEdge[],
  finalNodeId: string,
  validationResult: ValidationResult | null,
  datasetSchema: SparkSchema | null,
): FinalFeatureColumn[] {
  const nodeMap = new Map(nodes.map((n) => [n.id, n]));
  const incoming = edges.filter((e) => e.target === finalNodeId);
  const topoOrder = validationResult?.topologicalOrder ?? [];
  const topoRank = new Map(topoOrder.map((id, idx) => [id, idx]));
  const datasetNode = nodes.find((n) => n.data.type === 'dataset');
  const datasetNodeId = datasetNode?.id;

  if (!datasetNodeId || !datasetSchema || incoming.length === 0) {
    return [];
  }

  const latestByColumn = new Map<string, FinalFeatureColumn>();

  for (const edge of incoming) {
    const sourceNode = nodeMap.get(edge.source);
    if (!sourceNode) continue;

    const sourceSchema = getSourceSchema(edge.source, datasetNodeId, datasetSchema, validationResult);
    if (!sourceSchema) continue;

    const resolvedFromValidation = validationResult?.compiledEdges.get(edge.id)?.resolvedColumns;
    const candidateCols = resolvedFromValidation
      ? [...resolvedFromValidation]
      : edge.selector.type === 'manual'
        ? edge.selector.columns
        : sourceSchema.columns.map((c) => c.name);

    const sourceRank = topoRank.get(edge.source) ?? Number.MAX_SAFE_INTEGER;

    for (const columnName of candidateCols) {
      const col = sourceSchema.columns.find((c) => c.name === columnName);
      if (!col) continue;

      const stageData = isStageNode(sourceNode.data) ? sourceNode.data as StageNodeData : null;
      const provenance = stageData
        ? {
            stageName: stageData.label,
            stageType: stageData.type,
            outputColName: columnName,
          }
        : undefined;

      const classified = classifyByProvenance(provenance?.stageType, col.sparkType);
      const fallback = fallbackBySparkType(col.sparkType);
      const conflictReason = provenance && classified.bucket !== fallback
        ? `Provenance (${provenance.stageType}) overrides spark type inference (${fallback})`
        : undefined;

      const candidate: FinalFeatureColumn = {
        name: col.name,
        sparkType: col.sparkType,
        sourceNodeId: edge.source,
        provenance,
        autoBucket: classified.bucket,
        conflictReason,
        uncertainReason: classified.bucket === 'uncertain' ? classified.reason ?? 'Cannot classify confidently' : undefined,
      };

      const prev = latestByColumn.get(col.name);
      if (!prev) {
        latestByColumn.set(col.name, candidate);
        continue;
      }

      const prevRank = topoRank.get(prev.sourceNodeId) ?? Number.MIN_SAFE_INTEGER;
      if (sourceRank >= prevRank) {
        latestByColumn.set(col.name, candidate);
      }
    }
  }

  return Array.from(latestByColumn.values()).sort((a, b) => a.name.localeCompare(b.name));
}

export function dedupeBuckets(
  selections: Record<FeatureBucket, string[]>,
): Record<FeatureBucket, string[]> {
  const used = new Set<string>();
  const order: FeatureBucket[] = ['feature', 'target', 'metadata', 'passthrough'];
  const out: Record<FeatureBucket, string[]> = {
    feature: [],
    target: [],
    metadata: [],
    passthrough: [],
  };

  for (const bucket of order) {
    if (!selections[bucket]) continue;
    for (const col of selections[bucket]) {
      if (used.has(col)) continue;
      used.add(col);
      out[bucket].push(col);
    }
  }

  return out;
}

export interface FinalFeatureValidationIssue {
  readonly code: string;
  readonly message: string;
}

export function validateFinalFeatureSelections(
  availableColumns: readonly FinalFeatureColumn[],
  selections: Record<FeatureBucket, string[]>,
  confirmedUncertain: readonly string[],
): FinalFeatureValidationIssue[] {
  const issues: FinalFeatureValidationIssue[] = [];
  const availableByName = new Map(availableColumns.map((c) => [c.name, c]));
  const allSelected = [
    ...(selections.feature || []),
    ...(selections.target || []),
    ...(selections.metadata || []),
    ...(selections.passthrough || []),
  ];

  const unique = new Set(allSelected);
  if (unique.size !== allSelected.length) {
    issues.push({
      code: 'E_DUPLICATE_SELECTION',
      message: 'Columns cannot appear in more than one final_features pane',
    });
  }

  if ((selections.target?.length || 0) < 1) {
    issues.push({
      code: 'E_TARGET_REQUIRED',
      message: 'At least one target column is required',
    });
  }

  if ((selections.feature?.length || 0) < 1) {
    issues.push({
      code: 'E_FEATURE_REQUIRED',
      message: 'At least one feature is required',
    });
  }

  for (const colName of allSelected) {
    const candidate = availableByName.get(colName);
    if (!candidate) {
      // If it is in passthrough, we allow it to be missing from the current schema
      if (selections.passthrough?.includes(colName)) {
        continue;
      }
      issues.push({
        code: 'E_SCHEMA_MISMATCH',
        message: `Selected column '${colName}' does not exist in terminal schema`,
      });
      continue;
    }

    if (
      selections.feature?.includes(colName)
      && !isNumericSparkType(candidate.sparkType)
    ) {
      issues.push({
        code: 'E_NON_NUMERIC_SELECTED',
        message: `Column '${colName}' must be numeric for final_features (current type: ${candidate.sparkType})`,
      });
    }

    if ((candidate.autoBucket === 'uncertain' || candidate.conflictReason) && !confirmedUncertain.includes(colName)) {
      issues.push({
        code: 'E_UNCERTAIN_NOT_CONFIRMED',
        message: `Column '${colName}' is uncertain/conflicting and requires explicit confirmation`,
      });
    }
  }

  return issues;
}
