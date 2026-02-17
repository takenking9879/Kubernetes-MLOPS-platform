import type { SchemaEvolution, SparkSchema } from './schema';
import type { NodeCompilationState } from './nodes';

export type ValidationErrorType =
  | 'cycle_detected'
  | 'missing_dataset'
  | 'multiple_dataset'
  | 'missing_schema'
  | 'missing_final_features'
  | 'multiple_final_features'
  | 'final_features_disconnected'
  | 'final_features_outgoing_edge'
  | 'missing_input_column'
  | 'duplicate_output_column'
  | 'invalid_params'
  | 'selector_empty'
  | 'selector_resolution_failed'
  | 'cardinality_mismatch'
  | 'type_mismatch'
  | 'disconnected_node';

export type ValidationSeverity = 'error' | 'warning';

export interface ValidationError {
  readonly type: ValidationErrorType;
  readonly severity: ValidationSeverity;
  readonly nodeId?: string;
  readonly edgeId?: string;
  readonly field?: string;
  readonly message: string;
}

export interface CompiledNodeState {
  readonly compilation: NodeCompilationState;
  readonly resolvedInputs: readonly string[];
  readonly resolvedOutputs: readonly string[];
  readonly outputSchema: SparkSchema | null;
}

export interface CompiledEdgeState {
  readonly resolvedColumns: readonly string[];
}

export interface ValidationResult {
  readonly valid: boolean;
  readonly errors: readonly ValidationError[];
  readonly warnings: readonly ValidationError[];
  readonly schemaEvolutions: ReadonlyMap<string, SchemaEvolution>;
  readonly topologicalOrder: readonly string[];
  readonly compiledNodes: ReadonlyMap<string, CompiledNodeState>;
  readonly compiledEdges: ReadonlyMap<string, CompiledEdgeState>;
}

/** Build a validation error (convenience constructor). */
export function validationError(
  type: ValidationErrorType,
  message: string,
  opts?: { nodeId?: string; edgeId?: string; field?: string },
): ValidationError {
  return {
    type,
    severity: 'error',
    message,
    ...opts,
  };
}

/** Build a validation warning (convenience constructor). */
export function validationWarning(
  type: ValidationErrorType,
  message: string,
  opts?: { nodeId?: string; edgeId?: string; field?: string },
): ValidationError {
  return {
    type,
    severity: 'warning',
    message,
    ...opts,
  };
}
