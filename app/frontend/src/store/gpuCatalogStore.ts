/**
 * Shared GPU catalog store — single source of truth for both Training and Serving inspectors.
 * Stale-while-revalidate: old data stays visible while a refresh is in flight.
 * Not persisted — catalog data is live pricing info, always re-fetched per session.
 */

import { create } from 'zustand';
import { queryGPUCatalog, type GPUOffer } from '../api/platformClient';

const STALE_MS = 90_000; // 1.5 min before auto-refresh on next open

interface GPUCatalogStore {
  catalog: GPUOffer[];
  lastFetched: number | null;
  loading: boolean;
  /** Fire-and-forget refresh. Never clears existing catalog — stale data stays until new arrives. */
  refresh: () => Promise<void>;
  /** Refresh only if catalog is empty or stale. Safe to call on every panel open. */
  refreshIfStale: () => void;
}

export const useGpuCatalogStore = create<GPUCatalogStore>((set, get) => ({
  catalog: [],
  lastFetched: null,
  loading: false,

  refresh: async () => {
    if (get().loading) return;
    set({ loading: true });
    try {
      const data = await queryGPUCatalog({ providers: 'runpod,vast,aws,gcp,azure' });
      set({ catalog: data, lastFetched: Date.now() });
    } catch {
      // Keep existing catalog — stale data > empty panel
    } finally {
      set({ loading: false });
    }
  },

  refreshIfStale: () => {
    const { catalog, lastFetched, loading, refresh } = get();
    if (loading) return;
    const stale = !lastFetched || Date.now() - lastFetched > STALE_MS;
    if (stale || catalog.length === 0) void refresh();
  },
}));
