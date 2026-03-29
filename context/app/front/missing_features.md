# app/front/ — Missing Features

- **Job cancellation** — `cancelJob()` exists in platformClient.ts; no page or component exposes it
- **Custom architecture full upload workflow** — `listArchitectures()` + `uploadArchitecture()` are defined; LaunchWizardPage fetches list but doesn't wire an upload form with preview/select
- **Dataset preview** — `getIcebergSample()` is defined; no page renders a data preview table
- **Artifact check before training** — `checkArtifact()` is defined; no page calls it to validate preprocess_run_id
- **Training params viewer** — no page fetches or displays `params_training.yaml` for a completed run
- **Serving config history** — no list endpoint available from backend; no history page
- **Persistent error state** — errors disappear on tab navigation; no global error toast/boundary
- **Polling progress bars** — all status polling is text-only; no visual progress indicator
- **DAG / task log viewer** — no streaming log viewer for Airflow tasks
