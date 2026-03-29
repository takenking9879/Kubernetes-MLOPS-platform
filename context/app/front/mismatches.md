# app/front/ — Mismatches

- **`uploadSchemas()` batch endpoint**: Defined in platformClient.ts, corresponding backend endpoint exists (`POST /api/v2/datasets/{name}/schemas`), but no page calls it. Pages use `uploadFullSchema` and `uploadRawSchema` (individual endpoints) instead. The batch function is dead code.

- **`checkArtifact()` never called**: Backend endpoint exists (`GET /api/v2/runs/check`), function defined in client, but RunPage/LaunchWizardPage never invoke it. Preprocess run validation is skipped.

- **`listProcessingRuns()` never called**: `GET /api/v2/processing-runs` endpoint exists and function is defined, but no page uses it. Pages use `listPreprocessRunIds()` (the `/ids` endpoint) instead.

- **`GPUSelectResult` uses `infra` field**: TypeScript type correctly uses `infra: string` (not deprecated `cloud`). Ensure any new page consuming `any_of` entries reads `infra`, not `cloud`.

- **Polling terminal states are fragile string checks**: `DatasetPage` checks `['ResourceReleased', 'RESOURCE_RELEASED', 'Succeeded', 'SUCCEEDED', 'FAILED', 'Failed']`. If Airflow/Spark returns a new state string, polling will loop forever. Consider normalizing in backend.

- **LLM endpoint status polling**: `getVllmEndpoint()` returns status `'healthy'/'pending'/'not_found'` but the UI doesn't clearly communicate these states to the user.
