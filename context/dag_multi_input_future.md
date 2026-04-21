# DAG Multi-Input & MergeNode — Future Implementation

## Why deferred

Current Spark preprocessing DAGs (`preprocessing_pipeline`) are **single-dataset per run**.  
The backend has no merge/join endpoint. Adding multi-input in the frontend without backend support would create a broken UX.

MVP enforces **single upstream input per node**. MergeNode and multi-handle nodes are shown in the `NodePalette` as disabled stubs with "Coming soon" tooltip.

---

## Future MergeNode type

```typescript
// To add to src/types/dag.ts when backend is ready
interface MergeOrchNodeData {
  nodeKind: 'merge';
  label: string;
  status: NodeStatus;
  errors: string[];
  inputs: string[];             // list of upstream datasetNames or tableIds
  operation: 'join' | 'union' | 'filter';
  joinKey?: string;             // for join
  filterExpression?: string;    // SQL-like filter
  dropColumns?: string[];
  outputTableName: string;
  mergeRunId: string | null;
  dagRunId: string | null;
}
```

Node would have **multiple target handles** (one per input). ReactFlow supports multiple handles with `id` prop on `<Handle>` components.

---

## Expected backend endpoint

```
POST /api/v2/merge-runs
{
  input_tables: ["network_traffic_v2", "feature_store_v1"],
  operation: "join",
  join_key: "event_id",
  output_table_name: "merged_traffic"
}
→ { merge_run_id, dag_run_id }
```

This triggers a new Airflow DAG `merge_pipeline` using `SparkOperator`, similar pattern to `preprocessing_pipeline`:

```
submit_spark_merge >> poll_spark_merge
```

The Spark job reads multiple Iceberg tables, applies the operation, and writes a new Iceberg table.

---

## UI changes when MergeNode is enabled

1. Enable `MergeNode` in `NodePalette` (remove `disabled: true`).
2. Add `MergeOrchNodeData` to `OrchestrationNodeData` union in `src/types/dag.ts`.
3. Create `MergeOrchNode.tsx` visual (multi-handle: multiple `target` handles on left, single `source` handle on right).
4. Create `MergeInspector.tsx` with: input table list, operation selector, join key / filter expression inputs.
5. Add `'merge'` case to `dagStore.addNode()` factory.
6. Add `'merge'` case to `dagStore.runNode()` — calls `POST /api/v2/merge-runs`.
7. Add `'merge'` to artifact propagation in `dagStore.propagateArtifacts()` — propagates `outputTableName` downstream as `dataset_name`.
8. Add to `nodeTypes` map in `OrchestrationCanvas.tsx`.
9. Add `MergeInspector` routing case in `NodeInspector.tsx`.

---

## Code reference (existing patterns to follow)

- SparkOperator pattern: `k3s/airflow/dags/preprocessing_dag.py`
- Backend router pattern: `app/backend/routers/processing_runs.py`
- Frontend inspector pattern: `src/components/OrchestrationCanvas/inspectors/ProcessingInspector.tsx`
- Node visual pattern: `src/components/OrchestrationCanvas/nodes/ProcessingOrchNode.tsx`

---

## Decision nodes (post-MVP)

DecisionNode allows conditional routing after training or evaluation:

```typescript
interface DecisionOrchNodeData {
  nodeKind: 'decision';
  condition: 'accuracy_above' | 'loss_below' | 'manual';
  threshold: number;
  metricKey: string;    // e.g. 'val_accuracy', 'val_f1_macro'
  trueTarget: string | null;   // downstream node ID
  falseTarget: string | null;
}
```

Requires `GET /api/v2/runs/{run_id}/metrics` endpoint (reads MLflow run metrics).  
Shape: orange diamond visual, two `source` handles (true / false).

---

## Multi-dataset training (not yet supported)

Current training backend expects a single `preprocess_run_id`.  
Multi-input training (ensemble from multiple datasets) is not supported in `src/serve/runtime.py` or any existing training DAG.  
Do not assume this exists. Document clearly as a Phase 3 item requiring changes to:
- `k3s/airflow/dags/training_dag_skypilot.py`
- `app/backend/routers/runs.py`
- `src/serve/runtime.py` (multi-table inference)
