# app/front/ — Nice-to-Have

- **Job cancel button** in status polling pages (RunPage, ProcessingPage) — quick win, API already exists
- **Dataset preview tab** using `getIcebergSample()` — useful for verifying ingestion worked
- **Global error toast** — persistent across navigation; replaces silent catches
- **Unified status polling hook** — extract common polling logic (interval, terminal states) into a `usePolling` hook to reduce duplication across pages
- **Preprocess run validation** — call `checkArtifact()` after user selects preprocess_run_id; show error if not found
- **Training run params viewer** — collapsible section showing `params_training.yaml` content after submit
- **Step progress bar** in long-running operations (ingestion, preprocessing, training)
- **Retry on API error** — currently one failed call ends the workflow; add retry with exponential backoff
- **`preferred_regions` dropdown** in `GPUResourceSelector` — instead of free text, show known AWS regions + RunPod zones from catalog
