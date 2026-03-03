/**
 * UI state store — persists the active tab across sessions.
 *
 * Kept intentionally minimal: only navigation state belongs here.
 * Loading flags, modals, and transient UI state should NOT be added.
 */

import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';

export type Page = 'datasets' | 'dsl-builder' | 'processing' | 'run-pipeline' | 'serving';

interface UIState {
  page: Page;
  setPage: (page: Page) => void;
  reset: () => void;
}

const INITIAL_STATE: Pick<UIState, 'page'> = { page: 'datasets' };

export const useUIStore = create<UIState>()(
  devtools(
    persist(
      (set) => ({
        ...INITIAL_STATE,
        setPage: (page) => set({ page }),
        reset: () => set(INITIAL_STATE),
      }),
      {
        name: 'mlops-ui-store',
        version: 1,
        partialize: (state) => ({ page: state.page }),
      },
    ),
    { name: 'UIStore' },
  ),
);
