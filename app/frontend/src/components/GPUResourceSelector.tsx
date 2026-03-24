/**
 * GPUResourceSelector — collapsible GPU resource configuration panel.
 *
 * Shows:
 *  - Provider checkboxes (RunPod / Vast.ai / AWS / GCP / Azure)
 *  - GPU type dropdown ("Auto" = cheapest, or specific type)
 *  - Min VRAM slider
 *  - Spot preference toggle (default ON)
 *  - Live cost comparison table (spot vs on-demand)
 *  - Estimated savings badge
 *
 * Calls POST /api/v2/gpu-resources/select on each change (debounced).
 * Parent receives the final ResourceConstraints via onChange.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { ChevronDown, ChevronUp, Zap } from 'lucide-react';
import {
  queryGPUCatalog,
  selectGPUResources,
  type GPUOffer,
  type GPUSelectResult,
  type ResourceConstraints,
} from '../api/platformClient';

// ── Constants ─────────────────────────────────────────────────────────────────

const PROVIDERS = [
  { id: 'runpod', label: 'RunPod' },
  { id: 'vast', label: 'Vast.ai' },
  { id: 'aws', label: 'AWS' },
  { id: 'gcp', label: 'GCP' },
  { id: 'azure', label: 'Azure' },
];

const VRAM_OPTIONS = [0, 8, 16, 24, 40, 80];

const RUNTIME_PRESETS = [
  { label: '0.5 h', hours: 0.5 },
  { label: '2 h',   hours: 2   },
  { label: '8 h',   hours: 8   },
  { label: '24 h',  hours: 24  },
];

const INPUT_CLS =
  'rounded bg-slate-800 px-2 py-1 text-xs text-slate-100 outline-none focus:ring-1 focus:ring-blue-500 w-full';
const SELECT_CLS =
  'rounded bg-slate-800 px-2 py-1 text-xs text-slate-100 outline-none focus:ring-1 focus:ring-blue-500 w-full';

// ── Types ─────────────────────────────────────────────────────────────────────

interface Props {
  value: ResourceConstraints;
  onChange: (c: ResourceConstraints) => void;
  disabled?: boolean;
}

// ── Component ─────────────────────────────────────────────────────────────────

export function GPUResourceSelector({ value, onChange, disabled }: Props) {
  const [open, setOpen] = useState(false);
  const [catalog, setCatalog] = useState<GPUOffer[]>([]);
  const [selectResult, setSelectResult] = useState<GPUSelectResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [runtimeHours, setRuntimeHours] = useState<number>(2);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Fetch catalog once on open
  useEffect(() => {
    if (!open || catalog.length > 0) return;
    queryGPUCatalog({ providers: (value.providers ?? ['runpod']).join(',') })
      .then(setCatalog)
      .catch(() => {/* silent — catalog unavailable */});
  }, [open]); // eslint-disable-line react-hooks/exhaustive-deps

  // Debounced select call whenever constraints change
  useEffect(() => {
    if (!open) return;
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
  }, [value, open]);

  const set = useCallback(
    (patch: Partial<ResourceConstraints>) => onChange({ ...value, ...patch }),
    [value, onChange],
  );

  const toggleProvider = useCallback(
    (id: string) => {
      const cur = value.providers ?? ['runpod'];
      const next = cur.includes(id) ? cur.filter((p) => p !== id) : [...cur, id];
      if (next.length === 0) return; // must keep at least one
      set({ providers: next });
    },
    [value.providers, set],
  );

  // Unique GPU types from catalog for the dropdown
  const gpuTypes = Array.from(new Set(catalog.map((o) => o.gpu_type))).sort();

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
          {value.prefer_spot !== false && (
            <span className="rounded-full bg-yellow-900/60 px-1.5 py-0.5 text-[10px] text-yellow-300">
              spot-first
            </span>
          )}
        </span>
        {open ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
      </button>

      {open && (
        <div className="flex flex-col gap-3 border-t border-slate-700 p-3">
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
        </div>
      )}
    </div>
  );
}
