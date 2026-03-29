# app/front/ — Purpose

### Purpose
React SPA (TypeScript, Tailwind CSS) that provides the entire user-facing UI for the MLOps platform. Single entry point: `src/App.tsx`. All backend communication through `src/api/platformClient.ts`.

### When to use
- Adding or modifying a UI page
- Changing component behavior, props, or display logic
- Adding a new API call from the browser
- Modifying global state (Zustand stores)
- Changing tab navigation or page routing

### When not to use
- Backend logic → `app/backend/`
- Domain logic (feature engineering, GPU selection) → `src/`

### Physical layout
```
app/frontend/src/
  App.tsx                     ← tab routing (uiStore.page switch)
  pages/
    DatasetPage.tsx            ← dataset CRUD + ingest
    ProcessingPage.tsx         ← preprocessing run submission
    RunPage.tsx                ← training configuration + submit
    ServingPage.tsx            ← serving config + deploy (Ray Serve + vLLM)
    LaunchWizardPage.tsx       ← unified GPU job wizard
    (DSL Builder — MainLayout.tsx + panels)
  components/
    Layout/MainLayout.tsx      ← DSL builder layout
    Canvas/CanvasPanel.tsx     ← ReactFlow pipeline editor
    Catalog/CatalogPanel.tsx   ← DSL node catalog sidebar
    Properties/PropertiesPanel.tsx ← node editor
    Console/ConsolePanel.tsx   ← validation/dry-run log
    GPUResourceSelector.tsx    ← GPU constraints + pricing estimate
    GPUPricingPanel.tsx        ← live pricing table (filters by skypilot_supported)
    OrchestrationRecommendation.tsx ← orchestration badge + YAML preview
    schema/
      FullYamlEditor.tsx       ← feature schema editor (full.yaml)
      RawYamlEditor.tsx        ← raw schema editor (Kafka mode)
      PreprocessedYamlEditor.tsx ← read-only DSL features display
  api/
    platformClient.ts          ← ALL API calls; all TypeScript types
  store/
    uiStore.ts                 ← active page (persisted)
    datasetStore.ts            ← active dataset (persisted)
    pipelineStore.ts           ← DSL pipeline state (nodes, edges, validation, dry-run)
```
