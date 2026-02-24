/**
 * Global active-dataset store.
 *
 * Keeps track of which dataset the user is currently working in.
 * This context is shared between the DSL Builder and the Run Pipeline tab.
 */
import { create } from 'zustand';
import { devtools } from 'zustand/middleware';

interface DatasetState {
  activeDataset: string | null;
  setActiveDataset: (name: string | null) => void;
}

export const useDatasetStore = create<DatasetState>()(
  devtools(
    (set) => ({
      activeDataset: null,
      setActiveDataset: (name) => set({ activeDataset: name }),
    }),
    { name: 'DatasetStore' },
  ),
);
