/**
 * Global active-dataset store.
 *
 * Keeps track of which dataset the user is currently working in.
 * This context is shared between the DSL Builder and the Run Pipeline tab.
 */
import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';

interface DatasetState {
  activeDataset: string | null;
  setActiveDataset: (name: string | null) => void;
  reset: () => void;
}

const INITIAL_STATE: Pick<DatasetState, 'activeDataset'> = { activeDataset: null };

export const useDatasetStore = create<DatasetState>()(
  devtools(
    persist(
      (set) => ({
        ...INITIAL_STATE,
        setActiveDataset: (name) => set({ activeDataset: name }),
        reset: () => set(INITIAL_STATE),
      }),
      {
        name: 'mlops-dataset-store',
        version: 1,
        partialize: (state) => ({ activeDataset: state.activeDataset }),
      },
    ),
    { name: 'DatasetStore' },
  ),
);
