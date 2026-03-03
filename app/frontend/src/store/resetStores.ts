/**
 * Utility to wipe all persisted state across stores.
 *
 * Usage (e.g. a "Clear session" button or during logout):
 *   import { resetAllStores } from './store/resetStores';
 *   resetAllStores();
 *
 * This clears both the in-memory Zustand state and the corresponding
 * localStorage keys so that the next page load starts fresh.
 */

import { useDatasetStore } from './datasetStore';
import { usePipelineStore } from './pipelineStore';
import { useUIStore } from './uiStore';

const STORAGE_KEYS = [
  'mlops-dataset-store',
  'mlops-pipeline-store',
  'mlops-ui-store',
] as const;

export function resetAllStores(): void {
  useDatasetStore.getState().reset();
  usePipelineStore.getState().reset();
  useUIStore.getState().reset();

  for (const key of STORAGE_KEYS) {
    localStorage.removeItem(key);
  }
}
