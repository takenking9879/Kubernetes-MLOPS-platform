import type { SparkDataType } from '../types/schema';
import type { OutputNamingStrategy } from '../types/naming';
import type { StageType } from '../types/nodes';

export type GraphNodeKind = 'dataset' | 'transformer' | 'estimator' | 'terminal';

export interface NodeContract {
  readonly id: string;
  readonly label: string;
  readonly category: 'root' | 'transformers' | 'estimators' | 'terminal';
  readonly kind: GraphNodeKind;
  readonly draggableInCatalog: boolean;
  readonly creatableVia: 'catalog' | 'button';
  readonly maxInstances: number;
  readonly required?: boolean;
  readonly color?: 'default' | 'red';
  readonly allowedInputs: readonly GraphNodeKind[];
  readonly allowedOutputs: readonly GraphNodeKind[];
  readonly exportBehavior: 'stage' | 'none' | 'final_features';
  readonly provenanceHints?: readonly string[];
}

// ─── Parameter Schema (drives property panel UI) ────────────────────

export interface ParamOption {
  readonly value: string | number | boolean;
  readonly label: string;
}

export interface ParamSchema {
  readonly name: string;
  readonly type: 'string' | 'number' | 'boolean' | 'select' | 'array';
  readonly label: string;
  readonly required: boolean;
  readonly default?: unknown;
  readonly options?: readonly ParamOption[];
  readonly min?: number;
  readonly max?: number;
  readonly elementType?: 'string' | 'number' | 'mixed';
}

// ─── Input cardinality constraint ───────────────────────────────────

export type InputCardinality =
  | { readonly kind: 'exactly'; readonly n: number }
  | { readonly kind: 'atLeast'; readonly n: number }
  | { readonly kind: 'any' };

// ─── Node Definition ────────────────────────────────────────────────

/** Optional type constraint on input columns. */
export interface InputTypeConstraint {
  readonly allowedTypes: readonly SparkDataType[];
  readonly errorMessage: string;
}

export interface NodeDefinition {
  readonly type: string;
  readonly label: string;
  readonly category: 'transformer' | 'estimator';
  readonly description: string;
  readonly accentColor: string;
  readonly iconName: string;
  readonly inputCardinality: InputCardinality;
  /** Whether outputs list must have same length as inputs list */
  readonly outputMatchesInput: boolean;
  /** Inferred Spark output type (null = same as input) */
  readonly outputType: SparkDataType | null;
  readonly defaultParams: Record<string, unknown>;
  readonly paramSchema: readonly ParamSchema[];
  /** Optional constraint on input column Spark types */
  readonly inputTypeConstraint?: InputTypeConstraint;
  readonly transformSignature: 'one_to_one' | 'many_to_one' | 'many_to_many';
  readonly generatesNewColumns: boolean;
  readonly mutatesExistingColumns: boolean;
  readonly defaultNamingStrategy: OutputNamingStrategy;
  readonly outputExample: {
    readonly input: readonly string[];
    readonly output: readonly string[];
  };
}

export interface TerminalNodeDefinition {
  readonly type: 'final_features';
  readonly label: string;
  readonly category: 'terminal';
  readonly description: string;
  readonly accentColor: string;
  readonly iconName: string;
}

type RegistryNodeDefinition = Omit<
  NodeDefinition,
  | 'inputTypeConstraint'
  | 'transformSignature'
  | 'generatesNewColumns'
  | 'mutatesExistingColumns'
  | 'defaultNamingStrategy'
  | 'outputExample'
> & {
  readonly inputTypeConstraint?: InputTypeConstraint;
};

// ─── Registry ───────────────────────────────────────────────────────

export const NODE_REGISTRY: Readonly<Record<string, RegistryNodeDefinition>> = {
  // ═══════════════════════════════════════════════════════════════════
  //  TRANSFORMERS
  // ═══════════════════════════════════════════════════════════════════

  cast_transformer: {
    type: 'cast_transformer',
    label: 'Cast Type',
    category: 'transformer',
    description: 'Convert column to a different Spark data type',
    accentColor: '#3b82f6',
    iconName: 'RefreshCw',
    inputCardinality: { kind: 'atLeast', n: 1 },
    outputMatchesInput: true,
    outputType: null, // determined by target_type param
    defaultParams: { target_type: 'double' },
    paramSchema: [
      {
        name: 'target_type',
        type: 'select',
        label: 'Target Type',
        required: true,
        default: 'double',
        options: [
          { value: 'timestamp', label: 'Timestamp' },
          { value: 'double', label: 'Double' },
          { value: 'integer', label: 'Integer' },
          { value: 'string', label: 'String' },
          { value: 'long', label: 'Long' },
        ],
      },
    ],
  },

  temporal_extractor: {
    type: 'temporal_extractor',
    label: 'Temporal Extractor',
    category: 'transformer',
    description: 'Extract hour, day, month, etc. from timestamp column',
    accentColor: '#3b82f6',
    iconName: 'Clock',
    inputCardinality: { kind: 'atLeast', n: 1 },
    outputMatchesInput: true,
    outputType: 'integer',
    inputTypeConstraint: {
      allowedTypes: ['timestamp'],
      errorMessage: 'Temporal Extractor requires timestamp columns',
    },
    defaultParams: { component: 'hour' },
    paramSchema: [
      {
        name: 'component',
        type: 'select',
        label: 'Component',
        required: true,
        default: 'hour',
        options: [
          { value: 'hour', label: 'Hour' },
          { value: 'minute', label: 'Minute' },
          { value: 'second', label: 'Second' },
          { value: 'dayofweek', label: 'Day of Week' },
          { value: 'dayofmonth', label: 'Day of Month' },
          { value: 'month', label: 'Month' },
          { value: 'year', label: 'Year' },
          { value: 'quarter', label: 'Quarter' },
          { value: 'weekofyear', label: 'Week of Year' },
        ],
      },
    ],
  },

  arithmetic_transformer: {
    type: 'arithmetic_transformer',
    label: 'Arithmetic',
    category: 'transformer',
    description: 'Apply arithmetic operation (add, subtract, multiply, divide, power)',
    accentColor: '#3b82f6',
    iconName: 'Calculator',
    inputCardinality: { kind: 'atLeast', n: 1 },
    outputMatchesInput: true,
    outputType: 'double',
    inputTypeConstraint: {
      allowedTypes: ['integer', 'double', 'long'],
      errorMessage: 'Arithmetic operations require numeric columns (integer, double, long)',
    },
    defaultParams: { operation: 'add', value: 0 },
    paramSchema: [
      {
        name: 'operation',
        type: 'select',
        label: 'Operation',
        required: true,
        default: 'add',
        options: [
          { value: 'add', label: 'Add' },
          { value: 'subtract', label: 'Subtract' },
          { value: 'multiply', label: 'Multiply' },
          { value: 'divide', label: 'Divide' },
          { value: 'power', label: 'Power' },
          { value: 'absolute', label: 'Absolute' },
        ],
      },
      { name: 'value', type: 'number', label: 'Value', required: true, default: 0 },
    ],
  },

  cyclic_transformer: {
    type: 'cyclic_transformer',
    label: 'Cyclic Encoder',
    category: 'transformer',
    description: 'Sin/cos cyclic encoding for periodic features (hours, days)',
    accentColor: '#3b82f6',
    iconName: 'Orbit',
    inputCardinality: { kind: 'atLeast', n: 1 },
    outputMatchesInput: true,
    outputType: 'double',
    inputTypeConstraint: {
      allowedTypes: ['integer', 'double', 'long'],
      errorMessage: 'Cyclic encoding requires numeric columns (integer, double, long)',
    },
    defaultParams: { function: 'sin', period: 24.0 },
    paramSchema: [
      {
        name: 'function',
        type: 'select',
        label: 'Function',
        required: true,
        default: 'sin',
        options: [
          { value: 'sin', label: 'Sine' },
          { value: 'cos', label: 'Cosine' },
        ],
      },
      { name: 'period', type: 'number', label: 'Period', required: true, default: 24.0, min: 0.001 },
    ],
  },

  log_transformer: {
    type: 'log_transformer',
    label: 'Log Transform',
    category: 'transformer',
    description: 'Apply logarithmic transformation (log, log1p, log10)',
    accentColor: '#3b82f6',
    iconName: 'TrendingDown',
    inputCardinality: { kind: 'atLeast', n: 1 },
    outputMatchesInput: true,
    outputType: 'double',
    inputTypeConstraint: {
      allowedTypes: ['integer', 'double', 'long'],
      errorMessage: 'Log transformation requires numeric columns (integer, double, long)',
    },
    defaultParams: { log_type: 'log1p' },
    paramSchema: [
      {
        name: 'log_type',
        type: 'select',
        label: 'Log Type',
        required: true,
        default: 'log1p',
        options: [
          { value: 'log1p', label: 'log1p (safe for zeros)' },
          { value: 'log', label: 'Natural log' },
          { value: 'log10', label: 'Log base 10' },
        ],
      },
    ],
  },

  binning_transformer: {
    type: 'binning_transformer',
    label: 'Binning',
    category: 'transformer',
    description: 'Discretize continuous values into bins',
    accentColor: '#3b82f6',
    iconName: 'BarChart3',
    inputCardinality: { kind: 'atLeast', n: 1 },
    outputMatchesInput: true,
    outputType: 'integer',
    defaultParams: { bins: [0, 10, 50, 100] },
    paramSchema: [
      { name: 'bins', type: 'array', label: 'Bin Edges (numbers)', required: true, default: [0, 10, 50, 100], elementType: 'number' },
      { name: 'labels', type: 'array', label: 'Labels (optional)', required: false, elementType: 'string' },
    ],
  },

  conditional_transformer: {
    type: 'conditional_transformer',
    label: 'Conditional',
    category: 'transformer',
    description: 'If-then-else logic based on column values',
    accentColor: '#3b82f6',
    iconName: 'GitBranch',
    inputCardinality: { kind: 'atLeast', n: 1 },
    outputMatchesInput: true,
    outputType: 'integer',
    defaultParams: { condition: 'isin', values: [], true_value: 1, false_value: 0 },
    paramSchema: [
      {
        name: 'condition',
        type: 'select',
        label: 'Condition',
        required: true,
        default: 'isin',
        options: [
          { value: 'isin', label: 'Is In' },
          { value: 'greater_than', label: 'Greater Than' },
          { value: 'less_than', label: 'Less Than' },
          { value: 'equals', label: 'Equals' },
        ],
      },
      { name: 'values', type: 'array', label: 'Values', required: true, elementType: 'mixed' },
      { name: 'true_value', type: 'number', label: 'True Value', required: true, default: 1 },
      { name: 'false_value', type: 'number', label: 'False Value', required: true, default: 0 },
    ],
  },

  concat_transformer: {
    type: 'concat_transformer',
    label: 'Concatenate',
    category: 'transformer',
    description: 'Concatenate multiple string columns with a separator',
    accentColor: '#3b82f6',
    iconName: 'Link',
    inputCardinality: { kind: 'atLeast', n: 2 },
    outputMatchesInput: false,
    outputType: 'string',
    inputTypeConstraint: {
      allowedTypes: ['string'],
      errorMessage: 'Concatenation requires string columns',
    },
    defaultParams: { separator: '_' },
    paramSchema: [
      { name: 'separator', type: 'string', label: 'Separator', required: true, default: '_' },
    ],
  },

  ratio_transformer: {
    type: 'ratio_transformer',
    label: 'Ratio',
    category: 'transformer',
    description: 'Calculate ratio between two numeric columns (numerator/denominator)',
    accentColor: '#3b82f6',
    iconName: 'Divide',
    inputCardinality: { kind: 'exactly', n: 2 },
    outputMatchesInput: false,
    outputType: 'double',
    inputTypeConstraint: {
      allowedTypes: ['integer', 'double', 'long'],
      errorMessage: 'Ratio calculation requires numeric columns (integer, double, long)',
    },
    defaultParams: {},
    paramSchema: [],
  },

  // ═══════════════════════════════════════════════════════════════════
  //  ESTIMATORS
  // ═══════════════════════════════════════════════════════════════════

  string_indexer: {
    type: 'string_indexer',
    label: 'String Indexer',
    category: 'estimator',
    description: 'Encode categorical strings to integer indices (learns vocabulary)',
    accentColor: '#8b5cf6',
    iconName: 'Hash',
    inputCardinality: { kind: 'atLeast', n: 1 },
    outputMatchesInput: true,
    outputType: 'integer',
    inputTypeConstraint: {
      allowedTypes: ['string'],
      errorMessage: 'String Indexer requires string columns',
    },
    defaultParams: { handle_invalid: 'keep' },
    paramSchema: [
      {
        name: 'handle_invalid',
        type: 'select',
        label: 'Handle Invalid',
        required: true,
        default: 'keep',
        options: [
          { value: 'keep', label: 'Keep (map to -1)' },
          { value: 'skip', label: 'Skip rows' },
          { value: 'error', label: 'Raise error' },
        ],
      },
    ],
  },

  standard_scaler: {
    type: 'standard_scaler',
    label: 'Standard Scaler',
    category: 'estimator',
    description: 'Standardize features by removing mean and scaling to unit variance',
    accentColor: '#8b5cf6',
    iconName: 'Scale',
    inputCardinality: { kind: 'atLeast', n: 1 },
    outputMatchesInput: true,
    outputType: 'double',
    defaultParams: { with_mean: true, with_std: true },
    paramSchema: [
      { name: 'with_mean', type: 'boolean', label: 'Center (subtract mean)', required: true, default: true },
      { name: 'with_std', type: 'boolean', label: 'Scale (divide by std)', required: true, default: true },
    ],
  },

  minmax_scaler: {
    type: 'minmax_scaler',
    label: 'MinMax Scaler',
    category: 'estimator',
    description: 'Scale features to [0, 1] range (learns min/max)',
    accentColor: '#8b5cf6',
    iconName: 'ArrowUpDown',
    inputCardinality: { kind: 'atLeast', n: 1 },
    outputMatchesInput: true,
    outputType: 'double',
    defaultParams: {},
    paramSchema: [],
  },

  imputer: {
    type: 'imputer',
    label: 'Imputer',
    category: 'estimator',
    description: 'Fill missing values with learned statistics (mean/median/mode)',
    accentColor: '#8b5cf6',
    iconName: 'Eraser',
    inputCardinality: { kind: 'atLeast', n: 1 },
    outputMatchesInput: true,
    outputType: null, // preserves input type
    defaultParams: { strategy: 'mean' },
    paramSchema: [
      {
        name: 'strategy',
        type: 'select',
        label: 'Strategy',
        required: true,
        default: 'mean',
        options: [
          { value: 'mean', label: 'Mean' },
          { value: 'median', label: 'Median' },
          { value: 'mode', label: 'Mode' },
        ],
      },
    ],
  },

  frequency_encoder: {
    type: 'frequency_encoder',
    label: 'Frequency Encoder',
    category: 'estimator',
    description: 'Encode categorical values by their frequency or count',
    accentColor: '#8b5cf6',
    iconName: 'BarChart',
    inputCardinality: { kind: 'atLeast', n: 1 },
    outputMatchesInput: true,
    outputType: 'double',
    inputTypeConstraint: {
      allowedTypes: ['string'],
      errorMessage: 'Frequency Encoder requires string columns',
    },
    defaultParams: { encoding: 'freq' },
    paramSchema: [
      {
        name: 'encoding',
        type: 'select',
        label: 'Encoding',
        required: true,
        default: 'freq',
        options: [
          { value: 'freq', label: 'Frequency (normalized)' },
          { value: 'count', label: 'Count (absolute)' },
        ],
      },
      { name: 'smoothing', type: 'number', label: 'Smoothing', required: false, default: 0, min: 0 },
      { name: 'top_k', type: 'number', label: 'Top K', required: false, min: 1 },
    ],
  },
} as const;

const NODE_BEHAVIOR: Readonly<Record<StageType, Pick<NodeDefinition,
  'transformSignature'
  | 'generatesNewColumns'
  | 'mutatesExistingColumns'
  | 'defaultNamingStrategy'
  | 'outputExample'
>>> = {
  cast_transformer: {
    transformSignature: 'one_to_one',
    generatesNewColumns: true,
    mutatesExistingColumns: false,
    defaultNamingStrategy: { mode: 'suffix', value: '_cast' },
    outputExample: { input: ['amount'], output: ['amount_cast'] },
  },
  temporal_extractor: {
    transformSignature: 'one_to_one',
    generatesNewColumns: true,
    mutatesExistingColumns: false,
    defaultNamingStrategy: { mode: 'suffix', value: '_time' },
    outputExample: { input: ['event_ts'], output: ['event_ts_time'] },
  },
  arithmetic_transformer: {
    transformSignature: 'one_to_one',
    generatesNewColumns: true,
    mutatesExistingColumns: false,
    defaultNamingStrategy: { mode: 'suffix', value: '_arith' },
    outputExample: { input: ['amount'], output: ['amount_arith'] },
  },
  cyclic_transformer: {
    transformSignature: 'one_to_one',
    generatesNewColumns: true,
    mutatesExistingColumns: false,
    defaultNamingStrategy: { mode: 'suffix', value: '_cyc' },
    outputExample: { input: ['hour'], output: ['hour_cyc'] },
  },
  log_transformer: {
    transformSignature: 'one_to_one',
    generatesNewColumns: true,
    mutatesExistingColumns: false,
    defaultNamingStrategy: { mode: 'suffix', value: '_log' },
    outputExample: { input: ['amount'], output: ['amount_log'] },
  },
  binning_transformer: {
    transformSignature: 'one_to_one',
    generatesNewColumns: true,
    mutatesExistingColumns: false,
    defaultNamingStrategy: { mode: 'suffix', value: '_bin' },
    outputExample: { input: ['amount'], output: ['amount_bin'] },
  },
  conditional_transformer: {
    transformSignature: 'one_to_one',
    generatesNewColumns: true,
    mutatesExistingColumns: false,
    defaultNamingStrategy: { mode: 'suffix', value: '_cond' },
    outputExample: { input: ['flag_source'], output: ['flag_source_cond'] },
  },
  concat_transformer: {
    transformSignature: 'many_to_one',
    generatesNewColumns: true,
    mutatesExistingColumns: false,
    defaultNamingStrategy: { mode: 'suffix', value: '_concat' },
    outputExample: { input: ['city', 'country'], output: ['city_concat'] },
  },
  ratio_transformer: {
    transformSignature: 'many_to_one',
    generatesNewColumns: true,
    mutatesExistingColumns: false,
    defaultNamingStrategy: { mode: 'suffix', value: '_ratio' },
    outputExample: { input: ['num_a', 'num_b'], output: ['num_a_ratio'] },
  },
  string_indexer: {
    transformSignature: 'one_to_one',
    generatesNewColumns: true,
    mutatesExistingColumns: false,
    defaultNamingStrategy: { mode: 'suffix', value: '_idx' },
    outputExample: { input: ['category'], output: ['category_idx'] },
  },
  standard_scaler: {
    transformSignature: 'one_to_one',
    generatesNewColumns: true,
    mutatesExistingColumns: false,
    defaultNamingStrategy: { mode: 'suffix', value: '_std' },
    outputExample: { input: ['amount'], output: ['amount_std'] },
  },
  minmax_scaler: {
    transformSignature: 'one_to_one',
    generatesNewColumns: true,
    mutatesExistingColumns: false,
    defaultNamingStrategy: { mode: 'suffix', value: '_minmax' },
    outputExample: { input: ['amount'], output: ['amount_minmax'] },
  },
  imputer: {
    transformSignature: 'one_to_one',
    generatesNewColumns: true,
    mutatesExistingColumns: false,
    defaultNamingStrategy: { mode: 'suffix', value: '_imp' },
    outputExample: { input: ['amount'], output: ['amount_imp'] },
  },
  frequency_encoder: {
    transformSignature: 'one_to_one',
    generatesNewColumns: true,
    mutatesExistingColumns: false,
    defaultNamingStrategy: { mode: 'suffix', value: '_freq' },
    outputExample: { input: ['merchant'], output: ['merchant_freq'] },
  },
};

export const DATASET_NODE_CONTRACT: NodeContract = {
  id: 'dataset_root',
  label: 'Dataset',
  category: 'root',
  kind: 'dataset',
  draggableInCatalog: false,
  creatableVia: 'button',
  maxInstances: 1,
  allowedInputs: [],
  allowedOutputs: ['transformer', 'estimator'],
  exportBehavior: 'none',
};

export const FINAL_FEATURES_NODE_CONTRACT: NodeContract = {
  id: 'final_features',
  label: 'Final Features',
  category: 'terminal',
  kind: 'terminal',
  draggableInCatalog: true,
  creatableVia: 'catalog',
  maxInstances: 1,
  required: true,
  color: 'red',
  allowedInputs: ['transformer', 'estimator'],
  allowedOutputs: [],
  exportBehavior: 'final_features',
  provenanceHints: ['hybrid-classification', 'manual-overrides', 'conflict-checks'],
};

const TERMINAL_DEFINITIONS: readonly TerminalNodeDefinition[] = [
  {
    type: 'final_features',
    label: 'Final Features',
    category: 'terminal',
    description: 'Required sink node. Exports final_features contract (not a stage).',
    accentColor: '#ef4444',
    iconName: 'Flag',
  },
];

// ─── Accessors ──────────────────────────────────────────────────────

export function getNodeDefinition(type: string): NodeDefinition {
  const def = NODE_REGISTRY[type];
  if (!def) throw new Error(`Unknown node type in registry: ${type}`);
  const behavior = NODE_BEHAVIOR[type as StageType];
  if (!behavior) throw new Error(`Missing node behavior metadata for: ${type}`);
  return {
    ...def,
    ...behavior,
  };
}

export function getTransformers(): NodeDefinition[] {
  return Object.keys(NODE_REGISTRY)
    .filter((type) => NODE_REGISTRY[type]?.category === 'transformer')
    .map((type) => getNodeDefinition(type));
}

export function getEstimators(): NodeDefinition[] {
  return Object.keys(NODE_REGISTRY)
    .filter((type) => NODE_REGISTRY[type]?.category === 'estimator')
    .map((type) => getNodeDefinition(type));
}

export function getTerminalNodes(): readonly TerminalNodeDefinition[] {
  return TERMINAL_DEFINITIONS;
}

export function getNodeContract(type: string): NodeContract {
  if (type === 'dataset') return DATASET_NODE_CONTRACT;
  if (type === 'final_features') return FINAL_FEATURES_NODE_CONTRACT;

  const def = NODE_REGISTRY[type];
  if (!def) {
    throw new Error(`Unknown node contract type: ${type}`);
  }

  return {
    id: type,
    label: def.label,
    category: def.category === 'transformer' ? 'transformers' : 'estimators',
    kind: def.category,
    draggableInCatalog: true,
    creatableVia: 'catalog',
    maxInstances: Number.MAX_SAFE_INTEGER,
    allowedInputs: ['dataset', 'transformer', 'estimator'],
    allowedOutputs: ['transformer', 'estimator', 'terminal'],
    exportBehavior: 'stage',
  };
}

/** Validate params against schema. Returns list of error messages (empty = valid). */
export function validateNodeParams(type: string, params: Record<string, unknown>): string[] {
  const def = getNodeDefinition(type);
  const errors: string[] = [];

  for (const schema of def.paramSchema) {
    const value = params[schema.name];

    if (schema.required && (value === undefined || value === null || value === '')) {
      errors.push(`Parameter '${schema.label}' is required`);
      continue;
    }

    if (value === undefined || value === null) continue;

    if (schema.type === 'number' && typeof value !== 'number') {
      errors.push(`Parameter '${schema.label}' must be a number`);
    }

    if (schema.type === 'number' && typeof value === 'number' && !Number.isFinite(value)) {
      errors.push(`Parameter '${schema.label}' must be a finite number`);
    }

    if (schema.type === 'boolean' && typeof value !== 'boolean') {
      errors.push(`Parameter '${schema.label}' must be a boolean`);
    }

    if (schema.type === 'array') {
      if (!Array.isArray(value)) {
        errors.push(`Parameter '${schema.label}' must be an array`);
        continue;
      }
      if (schema.required && value.length === 0) {
        errors.push(`Parameter '${schema.label}' must contain at least one item`);
        continue;
      }

      if (schema.elementType === 'number') {
        for (const item of value) {
          if (typeof item !== 'number' || !Number.isFinite(item)) {
            errors.push(`Parameter '${schema.label}' must contain only numbers`);
            break;
          }
        }
      }

      if (schema.elementType === 'string') {
        for (const item of value) {
          if (typeof item !== 'string') {
            errors.push(`Parameter '${schema.label}' must contain only strings`);
            break;
          }
        }
      }
    }

    if (schema.type === 'number' && typeof value === 'number') {
      if (schema.min !== undefined && value < schema.min) {
        errors.push(`Parameter '${schema.label}' must be >= ${schema.min}`);
      }
      if (schema.max !== undefined && value > schema.max) {
        errors.push(`Parameter '${schema.label}' must be <= ${schema.max}`);
      }
    }

    if (schema.type === 'select' && schema.options) {
      const validValues = schema.options.map((o) => o.value);
      if (!validValues.includes(value as string | number | boolean)) {
        errors.push(`Parameter '${schema.label}' must be one of: ${validValues.join(', ')}`);
      }
    }
  }

  return errors;
}
