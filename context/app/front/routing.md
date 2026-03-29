# app/front/ — Routing

## If the task is...

- Change tab navigation → `App.tsx` (switch on `uiStore.page`) + `uiStore.ts` (page union type)
- Modify processing stepper → `ProcessingPage.tsx`
- Modify training configuration form → `RunPage.tsx`
- Modify serving deployment form → `ServingPage.tsx`
- Modify GPU job wizard → `LaunchWizardPage.tsx`
- Change DSL visual editor → `components/Canvas/CanvasPanel.tsx`
- Change DSL node catalog → `components/Catalog/CatalogPanel.tsx`
- Change node properties form → `components/Properties/PropertiesPanel.tsx`
- Change pipeline validation/dry-run log → `components/Console/ConsolePanel.tsx`
- Change GPU pricing display → `components/GPUPricingPanel.tsx`
- Change GPU constraints form → `components/GPUResourceSelector.tsx`
- Change orchestration recommendation display → `components/OrchestrationRecommendation.tsx`
- Change full schema editor → `components/schema/FullYamlEditor.tsx`
- Change raw schema editor (Kafka mode) → `components/schema/RawYamlEditor.tsx`
- Add a new API function or type → `api/platformClient.ts`
- Change persisted active dataset → `store/datasetStore.ts`
- Change pipeline canvas state → `store/pipelineStore.ts`
