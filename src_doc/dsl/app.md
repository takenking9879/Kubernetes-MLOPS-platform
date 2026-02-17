# Visual LAF Compiler – Quick Architecture Reference

Purpose:  
Quickly identify which file to modify when changing behavior, UI, compiler logic, or API.

---

# 1. Entry & Config

Modify these only if changing build, tooling, or global setup.

- package.json → dependencies (React Flow, Zustand, Tailwind, etc.)
- tsconfig.json → strict typing rules
- vite.config.ts → dev proxy (/api → backend)
- tailwind.config.js → theme / colors
- index.css → global styles
- index.html → root entry

---

# 2. Domain Model (Core Types)

Modify these if changing data model, schema behavior, or stage structure.

## schema.ts
Change here if:
- Adding new Spark data types
- Modifying schema structure
- Changing column metadata behavior
- Updating schema evolution logic types

## nodes.ts
Change here if:
- Adding a new stage type
- Modifying transformer/estimator definitions
- Updating discriminated unions

## edges.ts
Change here if:
- Modifying selector types
- Adding new selector mode
- Changing edge data structure

## validation.ts
Change here if:
- Adding new validation error types
- Modifying validation result structure

## dryrun.ts
Change here if:
- Modifying dry-run result model

---

# 3. Stage Definitions

## NodeRegistry.ts

Central stage configuration.

Modify here if:
- Adding a new stage
- Changing parameter schema (ParamSchema)
- Modifying input cardinality constraints
- Changing output type inference
- Updating stage-level validation

This is the authoritative stage definition file.

---

# 4. Compiler (Core Engine)

Modify here if changing DAG logic, selector resolution, or schema propagation.

## toposort.ts
Change if:
- Modifying cycle detection
- Changing graph traversal logic

## resolver.ts
Change if:
- Modifying selector resolution behavior
- Updating manual/group/rule selector logic
- Changing fail-fast behavior for empty resolution

## schemaSimulator.ts
Change if:
- Modifying schema propagation
- Changing how outputs are inferred
- Updating duplicate detection
- Adjusting symbolic evolution logic

## lafValidator.ts
Main compiler orchestration.

Change if:
- Modifying compilation flow
- Changing validation order
- Integrating resolver + simulator differently
- Adjusting pipeline compilation strategy

---

# 5. State Management

## pipelineStore.ts

Central Zustand store.

Modify here if:
- Changing how nodes/edges are stored
- Updating compile state persistence
- Adjusting validate/export/dryRun behavior
- Changing console logging logic

All global state flows through this file.

---

# 6. YAML & API Layer

## generator.ts
Modify if:
- Changing YAML structure
- Updating how inputs/outputs are serialized
- Adjusting stage ordering in export

## client.ts
Modify if:
- Changing API endpoints
- Updating request/response typing

---

# 7. UI Components

Modify here for visual or interaction changes.

## CanvasPanel.tsx
- React Flow integration
- Node/edge rendering behavior

## DatasetNode.tsx
- Root node UI

## StageNode.tsx
- Transformer/Estimator node UI
- Validation highlighting

## FeatureSelectorNode.tsx
- Final features node UI

## EdgeWithSelector.tsx
- Edge label display
- Selector summary rendering

## CatalogPanel.tsx
- Left sidebar layout

## CatalogItem.tsx
- Draggable stage entry

## PropertiesPanel.tsx
- Right sidebar routing

## NodePropertiesForm.tsx
- Dynamic parameter form rendering
- Modify if changing ParamSchema UI behavior

## EdgeSelectorConfig.tsx
- Selector UI (manual/group/rule)
- Modify if selector options need upstream schema awareness

## ConsolePanel.tsx
- Log rendering behavior

---

# 8. Layout

## Header.tsx
- Validate / Dry-Run / Export buttons
- Status badge behavior

## MainLayout.tsx
- Panel structure

## App.tsx / main.tsx
- Application bootstrap

---

# 9. Backend

## main.py

Modify if:
- Changing schema loading logic
- Updating dry-run execution
- Adjusting Spark error handling
- Adding new API endpoints

Endpoints:
- POST /api/schema/from-csv
- GET /api/schema/from-iceberg
- POST /api/dry-run

---

# 10. YAML Import

## importer.ts
Modify if:
- Changing YAML import logic
- Updating naming strategy inference
- Modifying edge creation from column provenance
- Adjusting legacy (v1) format handling

## autoLayout.ts
Modify if:
- Changing auto-layout algorithm for imported DAGs
- Adjusting node spacing or layering

---

# 11. Schema Hash

## schemaHash.ts (frontend)
- `computeSchemaHash()` — async SHA-256 hash
- `computeSchemaHashSync()` — sync FNV-1a hash for UI display

## main.py (backend)
- `compute_schema_hash()` — SHA-256 hash (must match frontend async version)

Used to detect when a YAML pipeline was generated against a different dataset schema.

---

# 12. Pipeline Settings

## PipelineSettingsPanel.tsx
- Editable pipeline name and version
- Read-only schema hash display
- Shown in PropertiesPanel when no node/edge is selected

---

# 13. Type Validation

## NodeRegistry.ts — `inputTypeConstraint`
- Optional per-stage constraint on input column Spark types
- Validated during schema simulation in `schemaSimulator.ts`
- Produces `type_mismatch` validation error

Stages with constraints: `temporal_extractor` (timestamp), `log_transformer` (numeric), `arithmetic_transformer` (numeric), `cyclic_transformer` (numeric), `ratio_transformer` (numeric), `string_indexer` (string), `frequency_encoder` (string), `concat_transformer` (string).

---

# 14. Incremental Compilation

## pipelineStore.ts — `dirtyNodeIds`
- Tracks which nodes need recompilation
- `updateNodeData`, `updateEdgeSelector`, `addEdge`, `deleteEdge` mark target + downstream dirty
- `addNode`, `deleteNode` mark all nodes dirty (structural change)
- Dirty set is cleared after each `validate()` call

## lafValidator.ts — `IncrementalOptions`
- Accepts optional `previousResult` and `dirtyNodeIds`
- Reuses compiled state for clean nodes
- Always does full topo sort and disconnected-node check

---

# 15. Testing

Run: `cd app/frontend && npm test`

Test files:
- `toposort.test.ts` — graph sorting and cycle detection
- `resolver.test.ts` — manual/group/rule selector resolution
- `schemaSimulator.test.ts` — stage simulation, naming strategies, type constraints
- `generator.test.ts` — YAML export structure
- `importer.test.ts` — YAML import, naming inference, edge creation
- `schemaHash.test.ts` — hash determinism and correctness
- `NodeRegistry.test.ts` — registry completeness, param validation

---

# Quick Modification Guide

If you want to…

Add a new stage →
    nodes.ts + NodeRegistry.ts
    (see docs/ADDING_TRANSFORMERS_AND_ESTIMATORS.md)

Change selector behavior →
    edges.ts + resolver.ts + EdgeSelectorConfig.tsx

Fix schema propagation →
    schemaSimulator.ts + lafValidator.ts

Change compilation flow →
    lafValidator.ts

Change YAML structure →
    generator.ts + importer.ts (keep in sync)

Import YAML pipeline →
    importer.ts + autoLayout.ts + pipelineStore.ts (importYAML)

Add type constraint to a stage →
    NodeRegistry.ts (inputTypeConstraint) + schemaSimulator.ts

Modify UI form behavior →
    NodePropertiesForm.tsx

Update global state behavior →
    pipelineStore.ts

Modify DAG logic →
    toposort.ts

Add backend feature →
    main.py + client.ts

Validate YAML on backend →
    POST /api/validate-yaml + client.ts (validateYAMLOnBackend)

---

End of Quick Reference
