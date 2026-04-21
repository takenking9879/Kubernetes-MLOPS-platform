/**
 * UI state store — persists the active tab across sessions.
 *
 * Kept intentionally minimal: only navigation state belongs here.
 * Loading flags, modals, and transient UI state should NOT be added.
 */

import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';

export type Page = 'canvas' | 'dsl-builder';

interface UIState {
  page: Page;
  setPage: (page: Page) => void;
  reset: () => void;
}

const INITIAL_STATE: Pick<UIState, 'page'> = { page: 'canvas' };

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
        version: 3,
        partialize: (state) => ({ page: state.page }),
        migrate: (state: unknown, _version: number) => {
          const s = state as { page?: string };
          // Migrate all old page values → canvas
          if (!s.page || !['canvas', 'dsl-builder'].includes(s.page)) {
            return { ...s, page: 'canvas' };
          }
          return state;
        },
      },
    ),
    { name: 'UIStore' },
  ),
);
