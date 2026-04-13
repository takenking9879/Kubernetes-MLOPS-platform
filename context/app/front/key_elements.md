# app/front/ — Key Elements

## Pages

### DatasetPage
File: `app/frontend/src/pages/DatasetPage.tsx`

Does:
- List, create datasets (S3 prefixes)
- Upload `.parquet` files to S3
- Trigger Iceberg ingestion + poll status

Inputs: none (reads from datasetStore, listDatasets API)
Outputs: side effects (creates dataset, uploads files, triggers DAG)
API calls: `listDatasets`, `createDataset`, `uploadParquets`, `submitIngest`, `getIngestStatus`

---

### ProcessingPage
File: `app/frontend/src/pages/ProcessingPage.tsx`

Does:
- 3-step form: Configure (dataset, DSL version, splits) → Review → Status
- Uploads full schema YAML
- Triggers preprocessing_pipeline DAG + polls

API calls: `listDatasets`, `listDsls`, `submitProcessingRun`, `uploadFullSchema`, `getProcessingRunStatus`

---

### RunPage
File: `app/frontend/src/pages/RunPage.tsx`

Does:
- 4-step form: Preprocessing run → Model config → Tuning → Review
- Selects preprocess_run_id, auto-loads lineage
- Configures XGBoost / PyTorch hyperparams
- Triggers training DAG + polls

API calls: `listPreprocessRunIds`, `getPreprocessParams`, `getTrainingConfig`, `submitRun`, `getRunStatus`
Components used: `GPUResourceSelector`

---

### ServingPage
File: `app/frontend/src/pages/ServingPage.tsx`

Does:
- ray_only mode (3 steps) or kafka mode (4 steps)
- LLM serving via vLLM (SkyPilot-based)
- Triggers serving_pipeline or vllm_serving_pipeline DAG + polls

API calls: `listTrainingRunIds`, `submitServingConfig`, `triggerServingDeploy`, `getServingDeployStatus`, `getLLMCatalog`, `triggerVllmDeploy`, `getVllmEndpoint`

---

### LaunchWizardPage
File: `app/frontend/src/pages/LaunchWizardPage.tsx`

Does:
- 4-step wizard: Model Selection → Resources → Review → Launch
- Handles tabular, LLM, and custom model upload paths
- Dry-run capable (shows orchestration recommendation + YAML preview)

API calls: `listPreprocessRunIds`, `getPreprocessParams`, `submitRun`, `launchJob`, `getLLMCatalog`, `listArchitectures`, `uploadArchitecture`
Components used: `GPUResourceSelector`, `GPUPricingPanel`, `OrchestrationRecommendation`

---

## Components

### Form inputs
Files: `app/frontend/src/components/forms/NumericInput.tsx`, `app/frontend/src/components/forms/ChipInput.tsx`

Does:
- Provide shared string-backed numeric editing without spinner behavior
- Provide chip/tag editing with Enter-to-add, remove buttons, duplicate prevention, and optional suggestions

Used by:
- `NodePropertiesForm.tsx`
- `EdgeSelectorConfig.tsx`
- `RunPage.tsx` (hyperparameter editors and search-space overrides)

Depends on:
- `app/frontend/src/lib/formValues.ts`

### Node parameter schema
File: `app/frontend/src/registry/NodeRegistry.ts`

Does:
- Declares per-parameter array element metadata for structured form rendering
- Validates array contents by element type before node params are accepted

Depends on:
- `app/frontend/src/components/Properties/NodePropertiesForm.tsx`
- `app/frontend/src/registry/__tests__/NodeRegistry.test.ts`

### GPUResourceSelector
File: `app/frontend/src/components/GPUResourceSelector.tsx`

Does:
- Collapsible GPU configuration panel with two modes:
  - **Auto (catalog)**: provider checkboxes, VRAM filter, spot preference, regions, cost estimate — backend auto-generates `any_of`
  - **Manual (fallback list)**: user builds an ordered list of `{infra, accelerators, use_spot}` entries; first = highest priority; stored in `ResourceConstraints.gpu_fallbacks`
- Mode toggle at top of expanded section
- Manual mode supports add/remove/reorder (↑↓) of entries
- `gpu_fallbacks` propagated to DAG via `RESOURCE_CONSTRAINTS_JSON`; `_load_task()` injects them directly into SkyPilot `any_of`, bypassing the catalog selector

Inputs: `value: ResourceConstraints`, `onChange`, `disabled?`
API calls: `queryGPUCatalog` (auto mode only), `selectGPUResources` (auto mode only)

---

### GPUPricingPanel
File: `app/frontend/src/components/GPUPricingPanel.tsx`

Does:
- Live pricing table filtered to current constraints
- Default: shows only `skypilot_supported=true` GPUs; toggle to show all
- "no sky" badge on non-SkyPilot GPUs

Inputs: `constraints`, `runtimeHours?`
API calls: `queryGPUCatalog`

---

### OrchestrationRecommendation
File: `app/frontend/src/components/OrchestrationRecommendation.tsx`

Does:
- Display-only: orchestration badge, cost estimate, warnings, SkyPilot YAML preview

Inputs: `recommendation`, `yamlPreview?`, `runtimeHours?`
API calls: none

---

## API Client — platformClient.ts

File: `app/frontend/src/api/platformClient.ts`

All TypeScript interfaces and API functions live here. Key groups:

| Group | Functions |
|-------|-----------|
| Datasets | `listDatasets`, `createDataset`, `uploadParquets`, `submitIngest`, `getIngestStatus`, `getIcebergSample`, `getIcebergSchema` |
| DSLs | `listDsls`, `getDsl`, `getDslFeatures`, `saveDsl` |
| Processing runs | `listPreprocessRunIds`, `submitProcessingRun`, `listProcessingRuns`, `getPreprocessParams`, `getProcessingRunStatus` |
| Training runs | `submitRun`, `listTrainingRunIds`, `getRunStatus`, `checkArtifact`, `getTrainingConfig` |
| Schemas | `uploadSchemas`, `uploadFullSchema`, `uploadRawSchema` |
| Serving | `submitServingConfig`, `triggerServingDeploy`, `getServingDeployStatus`, `triggerVllmDeploy`, `getVllmEndpoint` |
| GPU resources | `queryGPUCatalog`, `selectGPUResources`, `getLLMCatalog`; `GPUFallbackEntry` interface exported |
| Jobs | `launchJob`, `getJobStatus`, `listJobs`, `cancelJob` |
| Architectures | `listArchitectures`, `uploadArchitecture` |

TypeScript types defined here: `GPUOffer` (includes `skypilot_supported: boolean`), `ResourceConstraints`, `GPUSelectResult` (uses `infra` field, not `cloud`).

---

## Stores

### uiStore
File: `app/frontend/src/store/uiStore.ts`

State: `page: 'datasets' | 'dsl-builder' | 'processing' | 'run-pipeline' | 'serving' | 'launch'`
Persisted: yes (localStorage)

---

### datasetStore
File: `app/frontend/src/store/datasetStore.ts`

State: `activeDataset: string | null`
Persisted: yes (shared across DSL Builder + Run Pipeline)

---

### pipelineStore
File: `app/frontend/src/store/pipelineStore.ts`

State: nodes, edges, dirtyNodeIds, datasetSchema, validationResult, dryRunResult, consoleMessages, pipelineName, pipelineVersion, selectedNodeId, selectedEdgeId
Persisted: partial (nodes/edges/metadata only; validation state is ephemeral)
Key actions: `addNode`, `updateNodeData`, `deleteNode`, `addEdge`, `updateEdgeSelector`, `validate`, `dryRun`, `exportYAML`, `importYAML`, `loadSchemaFromIceberg`
