# Feature Engineering DAG

## Status
- `working`: drag-and-drop graph authoring, local validation, backend dry-run hooks
- `partial`: comprehensive UX guardrails and enterprise governance features

## Design
- DSL builder uses typed node contracts and edge selectors.
- Store tracks dirty nodes and downstream impact to avoid full recomputation on every edit.
- Validation runs locally first, then backend validation/dry-run can be invoked before heavy execution.

## Why It Matters
This graph is executable metadata, not a static drawing. It reduces waste by catching graph/schema issues before Spark/Ray jobs.

## Real UI Snapshot
![GPU Options and DAG Screenshot](./images/real-gpu-dag-screenshot.png)

## Trade-Offs
- Rich client-side logic adds complexity to store/state management.
- Validation consistency must be maintained across frontend and backend layers.

## Evidence Pointers
- `app/frontend/src/components/Canvas/CanvasPanel.tsx`
- `app/frontend/src/store/pipelineStore.ts`
- `app/frontend/src/lib/validation/lafValidator.ts`
- `app/backend/main.py`
