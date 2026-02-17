# Adding Transformers and Estimators

This guide covers how to add a new transformer or estimator to both the Python DSL backend and the React frontend visual designer.

---

## Section A: Adding a Transformer in the Python DSL

### 1. Create the transformer class

In `src/dsl/transformers.py`, create a class extending `Transformer`:

```python
from src.dsl.base import Transformer

class ClipTransformer(Transformer):
    """Clip (clamp) column values to a min/max range."""

    def transform(self, df):
        from pyspark.sql import functions as F
        result = df
        min_val = self.params.get("min_value", float("-inf"))
        max_val = self.params.get("max_value", float("inf"))

        for inp, out in zip(self.inputs, self.outputs):
            result = result.withColumn(
                out, F.greatest(F.lit(min_val), F.least(F.lit(max_val), F.col(inp)))
            )
        return result
```

Key rules:
- Always read from `self.inputs` and write to `self.outputs` (both are lists)
- Access config via `self.params`
- Return the modified DataFrame

### 2. Register the transformer

In `src/dsl/state_registry.py`, add it to the registry:

```python
from src.dsl.transformers import ClipTransformer

registry = StageRegistry()
registry.register_transformer("clip_transformer", ClipTransformer)
```

### 3. Add to `__init__.py`

In `src/dsl/__init__.py`, export the new class:

```python
from src.dsl.transformers import ClipTransformer
```

### 4. Test with YAML

```yaml
- type: "clip_transformer"
  name: "clip_amount"
  inputs: ["amount"]
  outputs: "amount_clipped"
  params:
    min_value: 0
    max_value: 10000
```

---

## Section B: Adding to the Frontend Visual Designer

### 1. Add the type to the union

In `app/frontend/src/types/nodes.ts`:

```typescript
// Add to TransformerType union
export type TransformerType =
  | 'cast_transformer'
  | ... existing types ...
  | 'clip_transformer';   // <-- add here

// Add the interface
export interface ClipTransformerData extends BaseStageNodeData {
  readonly type: 'clip_transformer';
  readonly stageType: 'transformer';
  params: { min_value?: number; max_value?: number };
}

// Add to StageNodeData union
export type StageNodeData =
  | ... existing types ...
  | ClipTransformerData;
```

### 2. Add to the Node Registry

In `app/frontend/src/registry/NodeRegistry.ts`, add entries to both `NODE_REGISTRY` and `NODE_BEHAVIOR`:

```typescript
// In NODE_REGISTRY
clip_transformer: {
  type: 'clip_transformer',
  label: 'Clip',
  category: 'transformer',
  description: 'Clamp values to a min/max range',
  accentColor: '#3b82f6',
  iconName: 'Scissors',
  inputCardinality: { kind: 'exactly', n: 1 },
  outputMatchesInput: false,
  outputType: 'double',
  // Optional: add type constraint if the stage only accepts specific types
  inputTypeConstraint: {
    allowedTypes: ['integer', 'double', 'long'],
    errorMessage: 'Clip requires numeric columns',
  },
  defaultParams: { min_value: 0, max_value: 100 },
  paramSchema: [
    { name: 'min_value', type: 'number', label: 'Min Value', required: false },
    { name: 'max_value', type: 'number', label: 'Max Value', required: false },
  ],
},

// In NODE_BEHAVIOR
clip_transformer: {
  transformSignature: 'one_to_one',
  generatesNewColumns: true,
  mutatesExistingColumns: false,
  defaultNamingStrategy: { mode: 'suffix', value: '_clip' },
  outputExample: { input: ['amount'], output: ['amount_clip'] },
},
```

### Key fields explained

| Field | Purpose |
|-------|---------|
| `inputCardinality` | How many input columns: `exactly(n)`, `atLeast(n)`, or `any` |
| `outputMatchesInput` | If `true`, output count must equal input count (scalers, imputer) |
| `outputType` | Spark type of outputs. `null` = determined at runtime (e.g., `cast_transformer` uses `params.target_type`) |
| `inputTypeConstraint` | Optional. Validated during schema simulation — produces a `type_mismatch` error |
| `transformSignature` | `one_to_one` (each input → one output), `many_to_one` (all inputs → one output), `many_to_many` |
| `defaultNamingStrategy` | Initial naming mode when the stage is dropped onto the canvas |
| `paramSchema` | Drives the auto-generated Properties panel form |

---

## Section C: YAML Compatibility

### v2.0 Format

The frontend now generates YAML with a `meta` block:

```yaml
pipeline:
  name: "my_pipeline"
  version: "2.0"
  stages:
    - type: "clip_transformer"
      name: "clip_1"
      inputs: "amount"
      outputs: "amount_clip"
      params:
        min_value: 0
        max_value: 100
meta:
  dslVersion: 2
  schemaHash: "a1b2c3d4e5f6g7h8"
  generatedAt: "2025-01-15T10:30:00.000Z"
```

### Schema Hash

The `schemaHash` is computed from the dataset schema using SHA-256 (backend) or FNV-1a (frontend sync). It allows detecting when a pipeline's YAML was generated against a different dataset schema.

Algorithm:
1. Sort columns by name
2. Format each as `name:sparkType:nullable` (e.g., `amount:double:true`)
3. Join with `|`
4. Hash the resulting string

### Migration from v1

The YAML importer (`lib/yaml/importer.ts`) handles both v1 (no `meta` block) and v2 formats. Old pipelines import cleanly — the importer infers naming strategies from the explicit `inputs`/`outputs` columns.

---

## Section D: Best Practices

### Deterministic Outputs

Every stage must produce **deterministic** output column names from its inputs + naming strategy. The schema simulator (`schemaSimulator.ts`) must be able to derive the exact output column names without running PySpark.

### Schema-First Development

1. Define the `outputType` in the registry
2. If the output type depends on a param (like `cast_transformer`), handle it in `resolveOutputType()` in `schemaSimulator.ts`
3. If the stage preserves input types (like `imputer`), set `outputType: null` and handle it in the special case

### Input Type Constraints

Add `inputTypeConstraint` to the registry if your stage only accepts specific Spark types. This provides immediate feedback in the UI (red error badge) instead of a cryptic PySpark runtime error.

### Test Checklist

When adding a new stage, write tests for:

- [ ] `NodeRegistry.test.ts` — stage is registered, params validate correctly
- [ ] `schemaSimulator.test.ts` — output columns derived correctly for each naming mode
- [ ] `importer.test.ts` — YAML with the new stage imports correctly
- [ ] `generator.test.ts` — stage serializes to correct YAML structure
- [ ] Type constraint tests (if `inputTypeConstraint` is set)

Run tests: `cd app/frontend && npm test`
