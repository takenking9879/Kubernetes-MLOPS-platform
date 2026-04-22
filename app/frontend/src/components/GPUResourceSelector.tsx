/**
 * GPUResourceSelector — collapsible GPU resource configuration panel.
 *
 * Two modes (toggle at the top):
 *
 *  Auto (catalog)   — original behaviour: provider checkboxes, GPU type
 *                     dropdown, VRAM filter, spot preference, cost estimate.
 *                     The backend auto-generates a ranked any_of list.
 *
 *  Manual (fallback list) — user builds an ordered list of GPU options.
 *                     Each entry: infra (runpod/vast/aws/…) + accelerator
 *                     string (e.g. "A100-80GB:1") + spot toggle.
 *                     First entry = highest priority.  Entries are passed
 *                     as ResourceConstraints.gpu_fallbacks and injected
 *                     directly into the SkyPilot any_of list, bypassing
 *                     the catalog selector.
 *
 * Parent receives the final ResourceConstraints via onChange.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { ChevronDown, ChevronUp, Zap, Plus, Trash2, ArrowUp, ArrowDown } from 'lucide-react';
import {
  getRunpodSupportedRegions,
  queryRunpodRegionAvailability,
  selectGPUResources,
  type GPUSelectResult,
  type ResourceConstraints,
  type GPUFallbackEntry,
  type RunPodRegionOption,
  type RunPodRegionAvailability,
} from '../api/platformClient';
import { useGpuCatalogStore } from '../store/gpuCatalogStore';

// ── Constants ─────────────────────────────────────────────────────────────────

const PROVIDERS = [
  { id: 'runpod', label: 'RunPod' },
  { id: 'vast', label: 'Vast.ai' },
  { id: 'aws', label: 'AWS' },
  { id: 'gcp', label: 'GCP' },
  { id: 'azure', label: 'Azure' },
  { id: 'local', label: 'Local GPU' },
];

const INFRA_OPTIONS = [
  { id: 'runpod',        label: 'RunPod' },
  { id: 'vast',          label: 'Vast.ai' },
  { id: 'aws',           label: 'AWS' },
  { id: 'gcp',           label: 'GCP' },
  { id: 'azure',         label: 'Azure' },
  { id: 'ssh/local-gpu', label: 'Local GPU' },
];

const VRAM_OPTIONS = [0, 8, 16, 24, 40, 80];

const RUNTIME_PRESETS = [
  { label: '0.5 h', hours: 0.5 },
  { label: '2 h',   hours: 2   },
  { label: '8 h',   hours: 8   },
  { label: '24 h',  hours: 24  },
];

const DEFAULT_FALLBACK: GPUFallbackEntry = {
  infra: 'runpod',
  accelerators: 'A100-80GB:1',
  use_spot: true,
};

import { INPUT_CLS, SELECT_CLS } from '../lib/uiTokens';

// ── Types ─────────────────────────────────────────────────────────────────────

interface Props {
  value: ResourceConstraints;
  onChange: (c: ResourceConstraints) => void;
  disabled?: boolean;
}

// ── Component ─────────────────────────────────────────────────────────────────

export function GPUResourceSelector({ value, onChange, disabled }: Props) {
  const [open, setOpen] = useState(false);
  const catalog        = useGpuCatalogStore((s) => s.catalog);
  const catalogLoading = useGpuCatalogStore((s) => s.loading);
  const refreshCatalog = useGpuCatalogStore((s) => s.refreshIfStale);
  void catalogLoading; // available for future loading indicator if needed
  const [selectResult, setSelectResult] = useState<GPUSelectResult | null>(null);
  const [runpodRegions, setRunpodRegions] = useState<RunPodRegionOption[]>([]);
  const [runpodAvailability, setRunpodAvailability] = useState<RunPodRegionAvailability[]>([]);
  const [manualRunpodRegionOptions, setManualRunpodRegionOptions] = useState<Record<string, RunPodRegionOption[]>>({});
  const [manualRunpodRegionFetchedAt, setManualRunpodRegionFetchedAt] = useState<Record<string, number>>({});
  const [manualRunpodRegionLoading, setManualRunpodRegionLoading] = useState<Record<string, boolean>>({});
  const [checkingRegionAvailability, setCheckingRegionAvailability] = useState(false);
  const [loading, setLoading] = useState(false);
  const [runtimeHours, setRuntimeHours] = useState<number>(2);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const manualDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Mode derived from value — external writes (e.g. catalog panel adding a fallback) auto-switch UI
  const mode: 'auto' | 'manual' = value.gpu_fallbacks && value.gpu_fallbacks.length > 0 ? 'manual' : 'auto';
  const fallbacks = value.gpu_fallbacks ?? [];

  const set = useCallback(
    (patch: Partial<ResourceConstraints>) => onChange({ ...value, ...patch }),
    [value, onChange],
  );

  const switchMode = useCallback(
    (m: 'auto' | 'manual') => {
      if (m === 'auto') {
        set({ gpu_fallbacks: null });
      } else if (!value.gpu_fallbacks || value.gpu_fallbacks.length === 0) {
        set({ gpu_fallbacks: [{ ...DEFAULT_FALLBACK }] });
      }
    },
    [value.gpu_fallbacks, set],
  );

  // Trigger catalog refresh on open (stale-while-revalidate via shared store)
  useEffect(() => {
    if (open) refreshCatalog();
  }, [open, refreshCatalog]);

  useEffect(() => {
    if (!open) return;
    getRunpodSupportedRegions()
      .then(setRunpodRegions)
      .catch(() => setRunpodRegions([]));
  }, [open]);

  // Debounced select call whenever constraints change (auto mode only)
  useEffect(() => {
    if (!open || mode !== 'auto') return;
    // Local SSH node pool: no catalog needed; backend derives infra from YAML.
    const _providers = value.providers ?? ['runpod'];
    if (_providers.length === 1 && _providers[0] === 'local') {
      setSelectResult(null);
      return;
    }
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      setLoading(true);
      selectGPUResources(value)
        .then(setSelectResult)
        .catch(() => setSelectResult(null))
        .finally(() => setLoading(false));
    }, 400);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [value, open, mode]);

  // Region-specific RunPod availability for selected GPU + selected RunPod regions only.
  useEffect(() => {
    if (!open || mode !== 'auto') return;

    const providers = value.providers ?? ['runpod'];
    if (!providers.includes('runpod')) {
      setRunpodAvailability([]);
      return;
    }

    const selectedGpu = value.gpu_types?.[0];
    const runpodRegionSet = new Set(runpodRegions.map((r) => r.provider_region_id.toUpperCase()));
    const selectedRunpodRegions = (value.preferred_regions ?? [])
      .map((r) => r.toUpperCase())
      .filter((r) => runpodRegionSet.has(r));

    if (!selectedGpu || selectedRunpodRegions.length === 0) {
      setRunpodAvailability([]);
      return;
    }

    setCheckingRegionAvailability(true);
    queryRunpodRegionAvailability({
      gpu_types: [selectedGpu],
      regions: selectedRunpodRegions,
    })
      .then(setRunpodAvailability)
      .catch(() => setRunpodAvailability([]))
      .finally(() => setCheckingRegionAvailability(false));
  }, [value.gpu_types, value.preferred_regions, value.providers, open, mode, runpodRegions]);

  const toggleProvider = useCallback(
    (id: string) => {
      const cur = value.providers ?? ['runpod'];
      const next = cur.includes(id) ? cur.filter((p) => p !== id) : [...cur, id];
      if (next.length === 0) return;
      set({ providers: next });
    },
    [value.providers, set],
  );

  const toggleRunpodRegion = useCallback(
    (regionId: string) => {
      const id = regionId.toUpperCase();
      const cur = value.preferred_regions ?? [];
      const exists = cur.some((r) => r.toUpperCase() === id);
      const next = exists
        ? cur.filter((r) => r.toUpperCase() !== id)
        : [...cur, id];
      set({ preferred_regions: next });
    },
    [set, value.preferred_regions],
  );

  const parseAcceleratorGpuType = useCallback((accelerators: string) => {
    const head = String(accelerators || '').split(':')[0] || '';
    return head.trim();
  }, []);

  // Manual-mode region options: only show regions that are live-available for
  // the selected GPU type in each RunPod fallback row.
  useEffect(() => {
    if (!open || mode !== 'manual' || runpodRegions.length === 0) return;

    const regionIds = runpodRegions.map((r) => r.provider_region_id.toUpperCase());
    const now = Date.now();
    const staleAfterMs = 20_000;
    const staleAfterEmptyMs = 5_000;
    const gpuTypesToQuery = Array.from(
      new Set(
        fallbacks
          .filter((entry) => entry.infra.split('/')[0] === 'runpod')
          .map((entry) => parseAcceleratorGpuType(entry.accelerators))
          .filter((gpu) => gpu.length > 0),
      ),
    ).filter((gpuType) => {
      const cachedOptions = manualRunpodRegionOptions[gpuType];
      const hasCached = cachedOptions !== undefined;
      const lastFetchAt = manualRunpodRegionFetchedAt[gpuType] ?? 0;
      const ttlMs = hasCached && cachedOptions.length > 0
        ? staleAfterMs
        : staleAfterEmptyMs;
      const isFresh = hasCached && (now - lastFetchAt) < ttlMs;
      if (isFresh) return false;
      return !manualRunpodRegionLoading[gpuType];
    });

    if (gpuTypesToQuery.length === 0) return;

    if (manualDebounceRef.current) clearTimeout(manualDebounceRef.current);
    manualDebounceRef.current = setTimeout(() => {
      setManualRunpodRegionLoading((prev) => {
        const next = { ...prev };
        gpuTypesToQuery.forEach((gpuType) => {
          next[gpuType] = true;
        });
        return next;
      });

      queryRunpodRegionAvailability({
        gpu_types: gpuTypesToQuery,
        regions: regionIds,
      })
        .then((rows) => {
          const byGpuType: Record<string, Set<string>> = {};
          rows
            .filter((row) => row.available)
            .forEach((row) => {
              const gpuType = row.gpu_type;
              if (!byGpuType[gpuType]) byGpuType[gpuType] = new Set<string>();
              byGpuType[gpuType].add(row.provider_region_id.toUpperCase());
            });

          const fetchedAt = Date.now();
          setManualRunpodRegionOptions((prev) => {
            const next = { ...prev };
            gpuTypesToQuery.forEach((gpuType) => {
              const available = byGpuType[gpuType] ?? new Set<string>();
              next[gpuType] = runpodRegions.filter((region) =>
                available.has(region.provider_region_id.toUpperCase()),
              );
            });
            return next;
          });
          setManualRunpodRegionFetchedAt((prev) => {
            const next = { ...prev };
            gpuTypesToQuery.forEach((gpuType) => {
              next[gpuType] = fetchedAt;
            });
            return next;
          });
        })
        .catch(() => {
          const fetchedAt = Date.now();
          setManualRunpodRegionOptions((prev) => {
            const next = { ...prev };
            gpuTypesToQuery.forEach((gpuType) => {
              if (next[gpuType] === undefined) {
                next[gpuType] = [];
              }
            });
            return next;
          });
          setManualRunpodRegionFetchedAt((prev) => {
            const next = { ...prev };
            gpuTypesToQuery.forEach((gpuType) => {
              next[gpuType] = fetchedAt;
            });
            return next;
          });
        })
        .finally(() => {
          setManualRunpodRegionLoading((prev) => {
            const next = { ...prev };
            gpuTypesToQuery.forEach((gpuType) => {
              next[gpuType] = false;
            });
            return next;
          });
        });
    }, 350);

    return () => {
      if (manualDebounceRef.current) clearTimeout(manualDebounceRef.current);
    };
  }, [
    open,
    mode,
    runpodRegions,
    fallbacks,
    parseAcceleratorGpuType,
    manualRunpodRegionOptions,
    manualRunpodRegionFetchedAt,
    manualRunpodRegionLoading,
  ]);

  // ── Fallback list helpers ─────────────────────────────────────────────────

  const addFallback = useCallback(() => {
    const cur = value.gpu_fallbacks ?? [];
    // Clone last entry as a starting point (common pattern: same GPU, different spot)
    const last = cur[cur.length - 1];
    const next: GPUFallbackEntry = last
      ? { ...last, use_spot: last.use_spot ? false : true }
      : { ...DEFAULT_FALLBACK };
    set({ gpu_fallbacks: [...cur, next] });
  }, [value.gpu_fallbacks, set]);

  const removeFallback = useCallback(
    (idx: number) => {
      const cur = value.gpu_fallbacks ?? [];
      set({ gpu_fallbacks: cur.filter((_, i) => i !== idx) });
    },
    [value.gpu_fallbacks, set],
  );

  const updateFallback = useCallback(
    (idx: number, patch: Partial<GPUFallbackEntry>) => {
      const cur = [...(value.gpu_fallbacks ?? [])];
      if (!cur[idx]) return;
      cur[idx] = { ...cur[idx], ...patch };
      set({ gpu_fallbacks: cur });
    },
    [value.gpu_fallbacks, set],
  );

  const moveFallback = useCallback(
    (idx: number, dir: -1 | 1) => {
      const cur = [...(value.gpu_fallbacks ?? [])];
      const target = idx + dir;
      if (target < 0 || target >= cur.length) return;
      if (!cur[idx] || !cur[target]) return;
      [cur[idx], cur[target]] = [cur[target], cur[idx]];
      set({ gpu_fallbacks: cur });
    },
    [value.gpu_fallbacks, set],
  );

  // ── Derived (auto mode) ───────────────────────────────────────────────────

  const STATIC_GPU_TYPES = [
    'A100-80GB', 'A100-40GB', 'H100', 'H100-80GB',
    'RTX-4090', 'RTX-3090', 'RTX-3080',
    'A10G', 'A10', 'A30', 'V100', 'V100-32GB', 'L4', 'L40',
  ];
  const gpuTypes = catalog.length > 0
    ? Array.from(new Set(catalog.map((o) => o.gpu_type))).sort()
    : STATIC_GPU_TYPES;
  const runpodRegionSet = new Set(runpodRegions.map((r) => r.provider_region_id.toUpperCase()));
  const selectedRunpodRegions = (value.preferred_regions ?? [])
    .map((r) => r.toUpperCase())
    .filter((r) => runpodRegionSet.has(r));
  const customRegions = (value.preferred_regions ?? []).filter(
    (r) => !runpodRegionSet.has(r.toUpperCase()),
  );

  const isLocalOnly =
    (value.providers ?? ['runpod']).length === 1 &&
    (value.providers ?? ['runpod'])[0] === 'local';

  const savings =
    selectResult?.estimated_cost_spot != null &&
    selectResult?.estimated_cost_ondemand != null &&
    selectResult.estimated_cost_ondemand > 0
      ? Math.round(
          (1 - selectResult.estimated_cost_spot / selectResult.estimated_cost_ondemand) * 100,
        )
      : null;

  const spotTotal =
    selectResult?.estimated_cost_spot != null
      ? selectResult.estimated_cost_spot * runtimeHours
      : null;
  const ondemandTotal =
    selectResult?.estimated_cost_ondemand != null
      ? selectResult.estimated_cost_ondemand * runtimeHours
      : null;

  return (
    <div className="rounded border border-slate-700 bg-slate-900/50">
      {/* Header toggle */}
      <button
        type="button"
        disabled={disabled}
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between px-3 py-2 text-xs text-slate-300 hover:bg-slate-800/50 disabled:opacity-40"
      >
        <span className="flex items-center gap-1.5 font-medium">
          <Zap size={12} className="text-yellow-400" />
          GPU Resource Selection
          {mode === 'manual' && fallbacks.length > 0 && (
            <span className="rounded-full bg-purple-900/60 px-1.5 py-0.5 text-[10px] text-purple-300">
              {fallbacks.length} manual {fallbacks.length === 1 ? 'option' : 'options'}
            </span>
          )}
          {mode === 'auto' && value.prefer_spot !== false && (
            <span className="rounded-full bg-yellow-900/60 px-1.5 py-0.5 text-[10px] text-yellow-300">
              spot-first
            </span>
          )}
        </span>
        {open ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
      </button>

      {open && (
        <div className="flex flex-col gap-3 border-t border-slate-700 p-3">

          {/* ── Mode toggle ──────────────────────────────────────────────── */}
          <div className="flex flex-col gap-1">
            <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">
              Selection mode
            </span>
            <div className="flex gap-2">
              {(['auto', 'manual'] as const).map((m) => (
                <button
                  key={m}
                  type="button"
                  onClick={() => switchMode(m)}
                  className={`rounded px-3 py-0.5 text-xs font-medium transition-colors ${
                    mode === m
                      ? 'bg-blue-700 text-white'
                      : 'bg-slate-700 text-slate-400 hover:bg-slate-600'
                  }`}
                >
                  {m === 'auto' ? 'Auto (catalog)' : 'Manual (fallback list)'}
                </button>
              ))}
            </div>
            <span className="text-[9px] text-slate-600">
              {mode === 'auto'
                ? 'Catalog picks the best available GPU matching your filters.'
                : 'You define the priority order — SkyPilot tries each option in sequence.'}
            </span>
          </div>

          {/* ══════════════════════════════════════════════════════════════
              AUTO MODE — original catalog-based controls
          ══════════════════════════════════════════════════════════════ */}
          {mode === 'auto' && (
            <>
              {/* Provider checkboxes */}
              <div className="flex flex-col gap-1">
                <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">
                  Providers
                </span>
                <div className="flex flex-wrap gap-2">
                  {PROVIDERS.map(({ id, label }) => {
                    const active = (value.providers ?? ['runpod']).includes(id);
                    return (
                      <button
                        key={id}
                        type="button"
                        onClick={() => toggleProvider(id)}
                        className={`rounded px-2 py-0.5 text-xs font-medium transition-colors ${
                          active
                            ? 'bg-blue-700 text-white'
                            : 'bg-slate-700 text-slate-400 hover:bg-slate-600'
                        }`}
                      >
                        {label}
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Local GPU note (no catalog needed) */}
              {isLocalOnly && (
                <p className="rounded border border-blue-700/40 bg-blue-950/30 px-3 py-2 text-[11px] text-blue-300">
                  Routes to local Docker container via SSH Node Pool
                  (<span className="font-mono">host.docker.internal:2222</span>).
                  No GPU catalog or cost estimate — uses the pre-running container with{' '}
                  <span className="font-mono">--gpus all</span>.
                </p>
              )}

              {!isLocalOnly && (<>

              {/* GPU type + VRAM row */}
              <div className="grid grid-cols-2 gap-3">
                <div className="flex flex-col gap-1">
                  <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">
                    GPU Type
                  </span>
                  <select
                    className={SELECT_CLS}
                    value={value.gpu_types?.[0] ?? ''}
                    onChange={(e) =>
                      set({ gpu_types: e.target.value ? [e.target.value] : null })
                    }
                  >
                    <option value="">Auto (cheapest)</option>
                    {gpuTypes.map((g) => (
                      <option key={g} value={g}>
                        {g}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="flex flex-col gap-1">
                  <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">
                    Min VRAM (GB)
                  </span>
                  <select
                    className={SELECT_CLS}
                    value={value.min_vram_gb ?? 0}
                    onChange={(e) => set({ min_vram_gb: Number(e.target.value) })}
                  >
                    {VRAM_OPTIONS.map((v) => (
                      <option key={v} value={v}>
                        {v === 0 ? 'Any' : `≥ ${v} GB`}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Max price + spot toggle row */}
              <div className="grid grid-cols-2 gap-3">
                <div className="flex flex-col gap-1">
                  <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">
                    Max $/hr
                  </span>
                  <input
                    type="number"
                    className={INPUT_CLS}
                    value={value.max_price_per_hour === 9999 ? '' : (value.max_price_per_hour ?? '')}
                    placeholder="No limit"
                    min={0}
                    step={0.1}
                    onChange={(e) =>
                      set({
                        max_price_per_hour: e.target.value ? parseFloat(e.target.value) : 9999,
                      })
                    }
                  />
                </div>

                <div className="flex flex-col gap-1">
                  <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">
                    Spot instances
                  </span>
                  <div className="flex gap-2">
                    {[true, false].map((v) => (
                      <button
                        key={String(v)}
                        type="button"
                        onClick={() => set({ prefer_spot: v })}
                        className={`rounded-full px-3 py-1 text-xs font-medium transition-colors ${
                          (value.prefer_spot ?? true) === v
                            ? 'bg-blue-600 text-white'
                            : 'bg-slate-700 text-slate-400 hover:bg-slate-600'
                        }`}
                      >
                        {v ? 'Preferred' : 'Off'}
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              {/* Runtime duration presets */}
              <div className="flex flex-col gap-1">
                <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">
                  Est. Runtime
                </span>
                <div className="flex gap-1.5">
                  {RUNTIME_PRESETS.map(({ label, hours }) => (
                    <button
                      key={label}
                      type="button"
                      onClick={() => setRuntimeHours(hours)}
                      className={`rounded px-2 py-0.5 text-xs font-medium transition-colors ${
                        runtimeHours === hours
                          ? 'bg-blue-700 text-white'
                          : 'bg-slate-700 text-slate-400 hover:bg-slate-600'
                      }`}
                    >
                      {label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Preferred regions */}
              <div className="flex flex-col gap-1">
                <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">
                  RunPod Regions (SkyPilot-supported)
                </span>
                <div className="flex flex-wrap gap-1.5">
                  {runpodRegions.length === 0 && (
                    <span className="text-[10px] text-slate-600">No region data available</span>
                  )}
                  {runpodRegions.map((region) => {
                    const id = region.provider_region_id.toUpperCase();
                    const active = selectedRunpodRegions.includes(id);
                    return (
                      <button
                        key={id}
                        type="button"
                        onClick={() => toggleRunpodRegion(id)}
                        className={`rounded px-2 py-0.5 text-[10px] font-medium transition-colors ${
                          active
                            ? 'bg-blue-700 text-white'
                            : 'bg-slate-700 text-slate-400 hover:bg-slate-600'
                        }`}
                        title={`${region.name} (${region.skypilot_infra})`}
                      >
                        {id}
                      </button>
                    );
                  })}
                </div>
                <input
                  type="text"
                  className={INPUT_CLS}
                  value={customRegions.join(', ')}
                  placeholder="Extra regions (AWS/GCP/etc), comma-separated"
                  onChange={(e) => {
                    const manual = e.target.value
                      .split(',')
                      .map((r) => r.trim())
                      .filter(Boolean);
                    set({ preferred_regions: [...selectedRunpodRegions, ...manual] });
                  }}
                />
                <span className="text-[9px] text-slate-600">
                  Only RunPod regions supported by SkyPilot are listed here.
                </span>
              </div>

              {(value.gpu_types?.[0] && selectedRunpodRegions.length > 0) && (
                <div className="flex flex-col gap-1">
                  <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">
                    Region Availability
                    {checkingRegionAvailability && (
                      <span className="ml-2 font-normal normal-case text-slate-600">checking…</span>
                    )}
                  </span>
                  {!checkingRegionAvailability && runpodAvailability.length === 0 && (
                    <span className="text-[10px] text-slate-600">No data for selected GPU/regions.</span>
                  )}
                  {runpodAvailability.length > 0 && (
                    <div className="overflow-hidden rounded border border-slate-700">
                      <table className="w-full text-[10px]">
                        <thead>
                          <tr className="bg-slate-800 text-left text-slate-400">
                            <th className="px-2 py-1">Region</th>
                            <th className="px-2 py-1">Status</th>
                            <th className="px-2 py-1 text-right">Max</th>
                          </tr>
                        </thead>
                        <tbody>
                          {runpodAvailability.map((row) => (
                            <tr key={`${row.gpu_type}-${row.provider_region_id}`} className="border-t border-slate-700">
                              <td className="px-2 py-1 font-mono text-slate-300">{row.provider_region_id}</td>
                              <td className="px-2 py-1">
                                {row.available ? (
                                  <span className="rounded bg-green-900/50 px-1 py-0.5 text-green-300">available</span>
                                ) : (
                                  <span className="rounded bg-red-900/50 px-1 py-0.5 text-red-300">no-stock</span>
                                )}
                              </td>
                              <td className="px-2 py-1 text-right font-mono text-slate-400">{row.max_available}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              )}

              {/* Cost comparison table */}
              {(selectResult || loading) && (
                <div className="flex flex-col gap-1">
                  <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">
                    Cost Estimate
                    {loading && (
                      <span className="ml-2 font-normal normal-case text-slate-600">
                        refreshing…
                      </span>
                    )}
                  </span>
                  {selectResult && !loading && (
                    <div className="overflow-hidden rounded border border-slate-700">
                      <table className="w-full text-xs">
                        <thead>
                          <tr className="bg-slate-800 text-left text-[10px] text-slate-400">
                            <th className="px-2 py-1.5">Type</th>
                            <th className="px-2 py-1.5 text-right">$/hr</th>
                            <th className="px-2 py-1.5 text-right">
                              Total ({runtimeHours >= 1 ? `${runtimeHours}h` : `${runtimeHours * 60}m`})
                            </th>
                            <th className="px-2 py-1.5 text-right">Entries</th>
                          </tr>
                        </thead>
                        <tbody>
                          {selectResult.estimated_cost_spot != null && (
                            <tr className="border-t border-slate-700 bg-green-950/20">
                              <td className="px-2 py-1.5 text-green-400">
                                Spot
                                {savings != null && savings > 0 && (
                                  <span className="ml-1.5 rounded-full bg-green-900 px-1.5 text-[10px] text-green-300">
                                    ~{savings}% off
                                  </span>
                                )}
                              </td>
                              <td className="px-2 py-1.5 text-right font-mono text-green-300">
                                ${selectResult.estimated_cost_spot.toFixed(3)}
                              </td>
                              <td className="px-2 py-1.5 text-right font-mono font-semibold text-green-300">
                                {spotTotal != null ? `$${spotTotal.toFixed(2)}` : '—'}
                              </td>
                              <td className="px-2 py-1.5 text-right text-slate-400">
                                {selectResult.spot_entries}
                              </td>
                            </tr>
                          )}
                          {selectResult.estimated_cost_ondemand != null && (
                            <tr className="border-t border-slate-700">
                              <td className="px-2 py-1.5 text-slate-300">On-demand</td>
                              <td className="px-2 py-1.5 text-right font-mono text-slate-300">
                                ${selectResult.estimated_cost_ondemand.toFixed(3)}
                              </td>
                              <td className="px-2 py-1.5 text-right font-mono text-slate-400">
                                {ondemandTotal != null ? `$${ondemandTotal.toFixed(2)}` : '—'}
                              </td>
                              <td className="px-2 py-1.5 text-right text-slate-400">
                                {selectResult.ondemand_entries}
                              </td>
                            </tr>
                          )}
                        </tbody>
                      </table>
                    </div>
                  )}
                  {selectResult && !loading && spotTotal != null && ondemandTotal != null && savings != null && savings > 0 && (
                    <p className="text-[11px] text-green-400">
                      Est. savings with spot: <strong>${(ondemandTotal - spotTotal).toFixed(2)}</strong> over {runtimeHours >= 1 ? `${runtimeHours}h` : `${runtimeHours * 60}m`}
                    </p>
                  )}
                  {selectResult && !loading && selectResult.any_of.length === 0 && (
                    <p className="text-xs text-amber-400">
                      No matching GPUs found — relax filters or add more providers.
                    </p>
                  )}
                </div>
              )}
              </>)}
            </>
          )}

          {/* ══════════════════════════════════════════════════════════════
              MANUAL MODE — ranked fallback list
          ══════════════════════════════════════════════════════════════ */}
          {mode === 'manual' && (
            <div className="flex flex-col gap-2">
              <div className="flex items-center justify-between">
                <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">
                  GPU Fallback Options
                  <span className="ml-1 font-normal normal-case text-slate-600">
                    — priority: top → bottom
                  </span>
                </span>
              </div>

              {fallbacks.length === 0 && (
                <p className="text-[11px] text-amber-400">
                  Add at least one GPU option below.
                </p>
              )}

              {fallbacks.map((entry, idx) => {
                const provider = entry.infra.split('/')[0];
                const rowGpuType = parseAcceleratorGpuType(entry.accelerators);
                const isLoadingRegions = rowGpuType
                  ? (manualRunpodRegionLoading[rowGpuType] ?? false)
                  : false;
                const availableRegions = rowGpuType
                  ? (manualRunpodRegionOptions[rowGpuType] ?? [])
                  : [];
                const currentRegionInfra = entry.infra.startsWith('runpod/') ? entry.infra : '';
                const hasCurrent = currentRegionInfra
                  ? availableRegions.some((r) => r.skypilot_infra === currentRegionInfra)
                  : false;
                const currentRegion = currentRegionInfra
                  ? runpodRegions.find((r) => r.skypilot_infra === currentRegionInfra)
                  : undefined;
                const rowRunpodRegions = !hasCurrent && currentRegion
                  ? [currentRegion, ...availableRegions]
                  : availableRegions;

                return (
                  <div
                    key={idx}
                    className={`flex flex-col gap-1.5 rounded border px-2 py-2 ${
                      idx === 0
                        ? 'border-blue-700/60 bg-blue-950/20'
                        : 'border-slate-700 bg-slate-800/40'
                    }`}
                  >
                    {/* ── Row 1: priority + infra + region ── */}
                    <div className="flex items-center gap-1.5">
                      {/* Priority badge */}
                      <span
                        className={`w-5 shrink-0 text-center text-[10px] font-bold ${
                          idx === 0 ? 'text-blue-400' : 'text-slate-600'
                        }`}
                      >
                        {idx + 1}
                      </span>

                      {/* Cloud / infra */}
                      <select
                        className="w-24 shrink-0 rounded bg-slate-800 px-1.5 py-1 text-xs text-slate-100 outline-none focus:ring-1 focus:ring-blue-500"
                        value={provider}
                        onChange={(e) => updateFallback(idx, { infra: e.target.value })}
                      >
                        {INFRA_OPTIONS.map(({ id, label }) => (
                          <option key={id} value={id}>{label}</option>
                        ))}
                      </select>

                      {provider === 'runpod' && (
                        <select
                          className="min-w-0 flex-1 rounded bg-slate-800 px-1.5 py-1 text-xs text-slate-100 outline-none focus:ring-1 focus:ring-blue-500"
                          value={entry.infra.startsWith('runpod/') ? entry.infra : 'runpod'}
                          onChange={(e) => updateFallback(idx, { infra: e.target.value })}
                          title="RunPod region"
                        >
                          <option value="runpod">Any RunPod region</option>
                          {isLoadingRegions && (
                            <option value="" disabled>Checking live availability…</option>
                          )}
                          {!isLoadingRegions && rowGpuType && rowRunpodRegions.length === 0 && (
                            <option value="" disabled>No regions currently available</option>
                          )}
                          {rowRunpodRegions.map((r) => (
                            <option key={r.provider_region_id} value={r.skypilot_infra}>
                              {r.provider_region_id}
                            </option>
                          ))}
                        </select>
                      )}
                    </div>

                    {/* ── Row 2: GPU type + count + spot + reorder + delete ── */}
                    <div className="flex items-center gap-1.5">
                      {/* GPU type select */}
                      <select
                        className="min-w-0 flex-1 rounded bg-slate-800 px-1.5 py-1 text-xs text-slate-100 outline-none focus:ring-1 focus:ring-blue-500"
                        value={parseAcceleratorGpuType(entry.accelerators) || ''}
                        onChange={(e) => {
                          const count = entry.accelerators.split(':')[1] ?? '1';
                          updateFallback(idx, { accelerators: `${e.target.value}:${count}` });
                        }}
                      >
                        <option value="">— select GPU —</option>
                        {gpuTypes.map((g) => (
                          <option key={g} value={g}>{g}</option>
                        ))}
                      </select>

                      {/* GPU count */}
                      <select
                        className="w-12 shrink-0 rounded bg-slate-800 px-1 py-1 text-xs text-slate-100 outline-none focus:ring-1 focus:ring-blue-500"
                        value={entry.accelerators.split(':')[1] ?? '1'}
                        onChange={(e) => {
                          const gpuType = parseAcceleratorGpuType(entry.accelerators);
                          updateFallback(idx, { accelerators: `${gpuType}:${e.target.value}` });
                        }}
                      >
                        {['1','2','4','8'].map((n) => (
                          <option key={n} value={n}>×{n}</option>
                        ))}
                      </select>

                      {/* Spot toggle */}
                      <button
                        type="button"
                        onClick={() => updateFallback(idx, { use_spot: !entry.use_spot })}
                        className={`shrink-0 rounded px-2 py-0.5 text-[10px] font-medium transition-colors ${
                          entry.use_spot
                            ? 'bg-yellow-800/60 text-yellow-300 hover:bg-yellow-700/60'
                            : 'bg-slate-700 text-slate-400 hover:bg-slate-600'
                        }`}
                        title="Toggle spot / on-demand"
                      >
                        {entry.use_spot ? 'spot' : 'on-dem'}
                      </button>

                      {/* Reorder arrows */}
                      <div className="flex shrink-0 flex-col">
                        <button
                          type="button"
                          onClick={() => moveFallback(idx, -1)}
                          disabled={idx === 0}
                          className="text-slate-500 hover:text-slate-300 disabled:opacity-25"
                          title="Move up (higher priority)"
                        >
                          <ArrowUp size={10} />
                        </button>
                        <button
                          type="button"
                          onClick={() => moveFallback(idx, 1)}
                          disabled={idx === fallbacks.length - 1}
                          className="text-slate-500 hover:text-slate-300 disabled:opacity-25"
                          title="Move down (lower priority)"
                        >
                          <ArrowDown size={10} />
                        </button>
                      </div>

                      {/* Remove */}
                      <button
                        type="button"
                        onClick={() => removeFallback(idx)}
                        className="shrink-0 text-slate-600 hover:text-red-400 transition-colors"
                        title="Remove this option"
                      >
                        <Trash2 size={12} />
                      </button>
                    </div>
                  </div>
                );
              })}

              <button
                type="button"
                onClick={addFallback}
                className="flex items-center gap-1.5 self-start rounded border border-dashed border-slate-600 px-3 py-1 text-xs text-slate-400 hover:border-slate-400 hover:text-slate-200 transition-colors"
              >
                <Plus size={11} />
                Add option
              </button>

              <p className="text-[9px] text-slate-600">
                Tip: put spot first, then on-demand as fallback. SkyPilot tries
                each entry in order until one succeeds.
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
