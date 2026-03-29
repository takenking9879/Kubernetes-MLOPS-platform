# How to Implement New Changes

## Step 0 — Read before writing

1. Read `context/overview.md` to orient.
2. Use `context/routing_global.md` to identify the affected files.
3. Read the relevant module's `context/*/key_elements.md` for current function signatures and behaviors.
4. **Do not infer behavior from names alone** — read the actual source files for functions you will modify.

## Step 1 — Make the change

Follow the cross-layer consistency rules from `context/routing_global.md`. Common patterns:

### Adding a new backend endpoint

1. Find or create the right router in `app/backend/routers/`
2. Add Pydantic request/response models at the top of the router
3. Implement the endpoint (it should call src/ services or external APIs)
4. Mount the router in `app/backend/main.py` if it's a new file
5. Add the API client function to `app/frontend/src/api/platformClient.ts`
6. Add TypeScript interfaces for request/response types

### Adding a new DSL transformer

1. Add the `Transformer` or `Estimator` subclass in `src/dsl/transformers.py` or `estimators.py`
2. **Also add the equivalent stage in `src/dsl/numpy_executor.py`** — the online inference path mirrors every Spark transformer
3. Register the new type in `src/dsl/state_registry.py`
4. Test: DSL YAML with the new stage type should work in both Spark fit+transform and NumPy executor

### Adding a new SkyPilot YAML

1. Add file in `k3s/sky/` with naming convention `ray-{purpose}-{provider}.yaml`
2. Update routing table in `app/backend/services/job_builder.py`
3. Update `k3s/airflow/dags/sky_runner.py` if the DAG selects YAMLs dynamically
4. Mark old generic YAML as LEGACY if this is a provider-specific replacement

### Adding a new Airflow DAG

1. Create `k3s/airflow/dags/{dag_id}.py`
2. Follow the existing pattern: PythonOperator for Spark/Ray, KubernetesPodOperator for SkyPilot
3. Use `k3s/airflow/k8s_helpers.py` for idempotent K8s resource cleanup
4. Add backend endpoint to trigger the DAG (set `dag_run.conf` dict)
5. Update `platformClient.ts` with a client function for the new trigger

### Adding a new frontend page

1. Create `app/frontend/src/pages/{Name}Page.tsx`
2. Add the page type to `uiStore.ts` `page` union type
3. Add the tab/link in `app/frontend/src/App.tsx`
4. Add API functions in `platformClient.ts` for any new backend calls

### Adding a new GPU provider

1. Add `_query_{provider}()` method in `src/services/gpu_catalog.py`
2. Add provider name to `query_availability()` routing logic
3. Add mapping dict `_{PROVIDER}_TO_SKYPILOT` for name normalization
4. Add IB detection logic if applicable
5. Update `_sky_catalog_query.py` subprocess if SkyPilot supports the new provider
6. Test: `GET /api/v2/gpu-resources/catalog?providers={provider}` returns correct offers

## Step 2 — Update /context

Update the context ONLY when your change affects one of these:
- **Behavior** of a module (new input/output, changed side effects)
- **Architecture** (new layer, new service, new integration point)
- **Feature exposure** (feature exists in src/ but not in app/, or vice versa)
- **Routing** (new file that future LLMs need to navigate to)
- **Legacy status** (code becomes obsolete or is deleted)

Do NOT update /context for:
- Bug fixes that don't change the interface
- Internal refactors with no external behavior change
- Style/formatting changes

### What to update

| Change type | Files to update |
|------------|-----------------|
| New endpoint | `context/app/back/key_elements.md` (add to router table) |
| New frontend page | `context/app/front/key_elements.md` + `routing.md` |
| New API client function | `context/app/front/key_elements.md` |
| New DSL stage | `context/src/dsl/key_elements.md` |
| New GPU provider | `context/src/services/key_elements.md` |
| New Airflow DAG | `context/k3s/key_elements.md` |
| New SkyPilot YAML | `context/k3s/key_elements.md` |
| Feature now fully implemented | Remove from `context/*/missing_features.md` or `missing_features_global.md` |
| Mismatch resolved | Remove from `context/*/mismatches.md` or `mismatches_global.md` |
| Code becomes legacy | Add to `context/k3s/legacy.md` |
| New cross-layer dependency | Update `context/cross_relations.md` |

### Format for key_elements entries

```markdown
Function/Class: <name>
File: <path>

Does:
- <what it does>

Inputs:
- <input: type — description>

Outputs:
- <output: type — description>

Side effects:
- <side effect>

Depends on:
- <dependency>
```

## Step 3 — Registry of changes

Maintain a running log in `context/changelog.md` (create if missing) for significant architectural changes:

```markdown
## YYYY-MM-DD — Short title
- What changed
- Why (motivation)
- Which context files were updated
```

## Constraints to respect

- `/context/` is a **navigation map**, not a specification — keep entries brief
- Do not copy source code into context files — reference file paths instead
- Do not document transient state (in-progress work, current conversation)
- `routing_global.md` lines > 200 risk truncation — keep the index concise
- Always verify a file/function exists before documenting it in context
