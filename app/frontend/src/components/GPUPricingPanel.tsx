/**
 * GPUPricingPanel — live GPU price comparison table.
 *
 * Fetches GET /api/v2/gpu-resources/catalog and displays a filterable,
 * sortable table of GPU offers across RunPod, Vast.ai, and AWS.
 *
 * Shown on the left column of the Launch tab alongside the wizard.
 *
 * Props:
 *   onAddToFallback — when provided (manual-mode, step 2), rows become
 *     clickable and show a "+" button; clicking appends the GPU to the
 *     manual fallback list in GPUResourceSelector.
 */

import { useCallback, useEffect, useState } from 'react';
import { Plus, RefreshCw, Zap } from 'lucide-react';
import { queryGPUCatalog, type GPUFallbackEntry, type GPUOffer } from '../api/platformClient';

// ── Constants ─────────────────────────────────────────────────────────────────

const PROVIDERS = [
  { id: 'runpod', label: 'RunPod' },
  { id: 'vast',   label: 'Vast.ai' },
  { id: 'aws',    label: 'AWS' },
];

const VRAM_OPTIONS = [0, 8, 16, 24, 40, 80];

const LABEL_CLS = 'text-[10px] font-semibold uppercase tracking-wider text-slate-500';
const CHIP_ACTIVE = 'rounded px-2 py-0.5 text-[10px] font-medium bg-blue-700 text-white transition-colors';
const CHIP_INACTIVE = 'rounded px-2 py-0.5 text-[10px] font-medium bg-slate-700 text-slate-400 hover:bg-slate-600 transition-colors cursor-pointer';

// ── Helpers ───────────────────────────────────────────────────────────────────

function fmt(price: number | null): string {
  if (price == null) return '—';
  return `$${price.toFixed(3)}`;
}

function savings(spot: number | null, ondemand: number): number | null {
  if (spot == null || ondemand <= 0) return null;
  return Math.round((1 - spot / ondemand) * 100);
}

function secAgo(d: Date): string {
  const s = Math.round((Date.now() - d.getTime()) / 1000);
  if (s < 60) return `${s}s ago`;
  return `${Math.round(s / 60)}m ago`;
}

function buildEntry(o: GPUOffer): GPUFallbackEntry {
  const accel = o.skypilot_accelerator.includes(':')
    ? o.skypilot_accelerator
    : `${o.skypilot_accelerator}:${o.gpu_count}`;
  return {
    infra: o.provider,
    accelerators: accel,
    use_spot: o.spot_available && (o.price_spot ?? 0) > 0,
  };
}

// ── Props ─────────────────────────────────────────────────────────────────────

interface Props {
  /** When defined, rows become clickable and show a "+" button to add the GPU
   *  directly to the manual fallback list. Pass undefined in auto mode. */
  onAddToFallback?: (entry: GPUFallbackEntry) => void;
}

// ── Component ─────────────────────────────────────────────────────────────────

export function GPUPricingPanel({ onAddToFallback }: Props) {
  const [offers, setOffers] = useState<GPUOffer[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  // Filters
  const [selectedProviders, setSelectedProviders] = useState<string[]>(['runpod', 'vast', 'aws']);
  const [minVram, setMinVram] = useState(0);
  const [sortBy, setSortBy] = useState<'spot' | 'ondemand'>('spot');
  const [search, setSearch] = useState('');
  // Default: show only SkyPilot-launchable GPUs (reduces noise)
  const [showAllGPUs, setShowAllGPUs] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const data = await queryGPUCatalog();
      setOffers(data);
      setLastUpdated(new Date());
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  // Update "X seconds ago" label every 10s
  useEffect(() => {
    const id = setInterval(() => {
      setLastUpdated(prev => (prev ? new Date(prev) : prev));
    }, 10_000);
    return () => clearInterval(id);
  }, []);

  // ── Filtering + sorting ──────────────────────────────────────────────────

  const filtered = offers
    .filter(o => selectedProviders.includes(o.provider))
    .filter(o => showAllGPUs || o.skypilot_supported)
    .filter(o => o.vram_gb >= minVram)
    .filter(o => !search || o.gpu_type.toLowerCase().includes(search.toLowerCase()))
    .sort((a, b) => {
      if (sortBy === 'spot') {
        const sa = a.price_spot ?? Infinity;
        const sb = b.price_spot ?? Infinity;
        return sa - sb;
      }
      return a.price_on_demand - b.price_on_demand;
    });

  const toggleProvider = (id: string) => {
    setSelectedProviders(prev =>
      prev.includes(id)
        ? prev.length > 1 ? prev.filter(p => p !== id) : prev
        : [...prev, id]
    );
  };

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col gap-3 rounded border border-slate-700 bg-slate-900/50 p-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5">
          <Zap size={13} className="text-amber-400" />
          <span className="text-xs font-semibold text-slate-200">GPU Pricing</span>
          {onAddToFallback && (
            <span className="text-[9px] text-blue-400 font-medium">— click row to add</span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {lastUpdated && (
            <span className="text-[10px] text-slate-600">{secAgo(lastUpdated)}</span>
          )}
          <button
            type="button"
            onClick={() => setShowAllGPUs(v => !v)}
            className={`text-[10px] font-medium transition-colors ${
              showAllGPUs
                ? 'text-amber-400 hover:text-amber-300'
                : 'text-slate-500 hover:text-slate-300'
            }`}
            title={showAllGPUs ? 'Showing all GPUs — click to show SkyPilot-only' : 'Showing SkyPilot-only — click to show all'}
          >
            {showAllGPUs ? 'All GPUs' : 'SkyPilot only'}
          </button>
          <button
            type="button"
            onClick={load}
            disabled={loading}
            className="flex items-center gap-1 text-[10px] text-slate-500 hover:text-slate-300"
          >
            <RefreshCw size={10} className={loading ? 'animate-spin' : ''} />
            Refresh
          </button>
        </div>
      </div>

      {/* Provider chips */}
      <div className="flex flex-col gap-1">
        <span className={LABEL_CLS}>Provider</span>
        <div className="flex gap-1.5 flex-wrap">
          {PROVIDERS.map(({ id, label }) => (
            <button
              key={id}
              type="button"
              onClick={() => toggleProvider(id)}
              className={selectedProviders.includes(id) ? CHIP_ACTIVE : CHIP_INACTIVE}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* VRAM + search row */}
      <div className="grid grid-cols-2 gap-3">
        <div className="flex flex-col gap-1">
          <span className={LABEL_CLS}>Min VRAM</span>
          <div className="flex gap-1 flex-wrap">
            {VRAM_OPTIONS.map(v => (
              <button
                key={v}
                type="button"
                onClick={() => setMinVram(v)}
                className={minVram === v ? CHIP_ACTIVE : CHIP_INACTIVE}
              >
                {v === 0 ? 'Any' : `${v}G`}
              </button>
            ))}
          </div>
        </div>
        <div className="flex flex-col gap-1">
          <span className={LABEL_CLS}>Sort by</span>
          <div className="flex gap-1.5">
            {(['spot', 'ondemand'] as const).map(s => (
              <button
                key={s}
                type="button"
                onClick={() => setSortBy(s)}
                className={sortBy === s ? CHIP_ACTIVE : CHIP_INACTIVE}
              >
                {s === 'spot' ? 'Spot' : 'On-demand'}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* GPU search */}
      <input
        type="text"
        value={search}
        onChange={e => setSearch(e.target.value)}
        placeholder="Filter GPU type…"
        className="rounded bg-slate-800 px-2 py-1 text-xs text-slate-100 outline-none focus:ring-1 focus:ring-blue-500 w-full"
      />

      {/* Error */}
      {error && (
        <p className="text-[10px] text-red-400">{error}</p>
      )}

      {/* Table */}
      <div className="overflow-x-auto">
        {loading && offers.length === 0 ? (
          <div className="text-xs text-slate-500 py-4 text-center">Loading…</div>
        ) : filtered.length === 0 ? (
          <div className="text-xs text-slate-500 py-4 text-center">No GPUs match filters</div>
        ) : (
          <table className="w-full text-[10px] border-collapse">
            <thead>
              <tr className="border-b border-slate-700 text-slate-500">
                <th className="text-left py-1 pr-2 font-semibold">GPU</th>
                <th className="text-right py-1 pr-2 font-semibold">VRAM</th>
                <th className="text-left py-1 pr-2 font-semibold">Provider</th>
                <th className="text-right py-1 pr-2 font-semibold">Spot/hr</th>
                <th className="text-right py-1 pr-2 font-semibold">OD/hr</th>
                <th className="text-right py-1 font-semibold">Save</th>
                {onAddToFallback && <th className="w-5" />}
              </tr>
            </thead>
            <tbody>
              {filtered.map((o, i) => {
                const sv = savings(o.price_spot, o.price_on_demand);
                const noStock = o.available_count === 0;
                return (
                  <tr
                    key={`${o.provider}-${o.gpu_type}-${i}`}
                    className={`group border-b border-slate-800 transition-colors ${
                      onAddToFallback
                        ? 'cursor-pointer hover:bg-slate-700/60'
                        : 'hover:bg-slate-800/40'
                    }`}
                    onClick={onAddToFallback ? () => onAddToFallback(buildEntry(o)) : undefined}
                  >
                    <td className="py-1 pr-2 font-mono">
                      <span className={o.skypilot_supported ? 'text-slate-200' : 'text-slate-500'}>
                        {o.gpu_type}
                      </span>
                      {!o.skypilot_supported && (
                        <span className="ml-1 rounded bg-slate-800 px-1 py-0.5 text-[8px] text-slate-600">
                          no sky
                        </span>
                      )}
                      {noStock && (
                        <span className="ml-1 rounded bg-red-900/60 px-1 py-0.5 text-[8px] text-red-400 font-mono">
                          no stock
                        </span>
                      )}
                    </td>
                    <td className="py-1 pr-2 text-right text-slate-400">{o.vram_gb}G</td>
                    <td className="py-1 pr-2 text-slate-400 capitalize">{o.provider}</td>
                    <td className={`py-1 pr-2 text-right font-mono ${o.price_spot != null ? 'text-green-400' : 'text-slate-600'}`}>
                      {fmt(o.price_spot)}
                    </td>
                    <td className="py-1 pr-2 text-right font-mono text-slate-400">
                      {fmt(o.price_on_demand)}
                    </td>
                    <td className="py-1 text-right">
                      {sv != null && sv > 0 ? (
                        <span className="rounded bg-green-900/50 px-1 py-0.5 text-green-400 font-semibold">
                          {sv}%
                        </span>
                      ) : (
                        <span className="text-slate-600">—</span>
                      )}
                    </td>
                    {onAddToFallback && (
                      <td className="py-1 pl-1 text-center">
                        <button
                          type="button"
                          title={`Add ${o.gpu_type} to fallback list`}
                          onClick={e => { e.stopPropagation(); onAddToFallback(buildEntry(o)); }}
                          className="opacity-0 group-hover:opacity-100 rounded p-0.5 text-blue-400 hover:bg-blue-900/50 transition-opacity"
                        >
                          <Plus size={10} />
                        </button>
                      </td>
                    )}
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>

      <p className="text-[9px] text-slate-600">
        {filtered.length} offer{filtered.length !== 1 ? 's' : ''} · prices are live estimates
      </p>
    </div>
  );
}
