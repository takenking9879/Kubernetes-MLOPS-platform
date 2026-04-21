/**
 * GPUCatalogPanel — floating panel rendered to the left of the NodeInspector (440px wide).
 *
 * Mounts via React portal directly to document.body so fixed positioning is always
 * relative to the viewport regardless of ancestor overflow/backdrop-filter.
 *
 * Open/close is driven by parent (Resources & Launch section toggle).
 * Catalog stays cached in gpuCatalogStore between opens — no re-fetch delay.
 */

import { useEffect, useMemo, useState } from 'react';
import { createPortal } from 'react-dom';
import { RefreshCw, Loader2, Search } from 'lucide-react';
import { useGpuCatalogStore } from '../store/gpuCatalogStore';
import type { GPUOffer } from '../api/platformClient';

// ── Static fallback when backend unavailable ──────────────────────────────────

const STATIC_GPU_TYPES = [
  { gpuType: 'H100',        vramGb: 80, spotMin: null, ondemandMin: null, providers: new Set(['aws']),    count: 0 },
  { gpuType: 'H100-80GB',   vramGb: 80, spotMin: null, ondemandMin: null, providers: new Set(['gcp']),    count: 0 },
  { gpuType: 'A100-80GB',   vramGb: 80, spotMin: null, ondemandMin: null, providers: new Set(['runpod']), count: 0 },
  { gpuType: 'A100-40GB',   vramGb: 40, spotMin: null, ondemandMin: null, providers: new Set(['runpod']), count: 0 },
  { gpuType: 'L40',         vramGb: 48, spotMin: null, ondemandMin: null, providers: new Set(['vast']),   count: 0 },
  { gpuType: 'L4',          vramGb: 24, spotMin: null, ondemandMin: null, providers: new Set(['gcp']),    count: 0 },
  { gpuType: 'A10G',        vramGb: 24, spotMin: null, ondemandMin: null, providers: new Set(['aws']),    count: 0 },
  { gpuType: 'A10',         vramGb: 24, spotMin: null, ondemandMin: null, providers: new Set(['vast']),   count: 0 },
  { gpuType: 'A30',         vramGb: 24, spotMin: null, ondemandMin: null, providers: new Set(['vast']),   count: 0 },
  { gpuType: 'RTX-4090',    vramGb: 24, spotMin: null, ondemandMin: null, providers: new Set(['runpod']), count: 0 },
  { gpuType: 'RTX-3090',    vramGb: 24, spotMin: null, ondemandMin: null, providers: new Set(['runpod']), count: 0 },
  { gpuType: 'V100-32GB',   vramGb: 32, spotMin: null, ondemandMin: null, providers: new Set(['aws']),    count: 0 },
  { gpuType: 'V100',        vramGb: 16, spotMin: null, ondemandMin: null, providers: new Set(['aws']),    count: 0 },
  { gpuType: 'RTX-3080',    vramGb: 10, spotMin: null, ondemandMin: null, providers: new Set(['runpod']), count: 0 },
];

// ── Aggregation ───────────────────────────────────────────────────────────────

interface AggRow {
  gpuType: string;
  vramGb: number;
  spotMin: number | null;
  ondemandMin: number | null;
  providers: Set<string>;
  count: number;
}

function aggregateCatalog(catalog: GPUOffer[]): AggRow[] {
  const map = new Map<string, AggRow>();
  for (const o of catalog) {
    if (!o.skypilot_supported) continue;
    let row = map.get(o.gpu_type);
    if (!row) {
      row = { gpuType: o.gpu_type, vramGb: o.vram_gb, spotMin: null, ondemandMin: null, providers: new Set(), count: 0 };
      map.set(o.gpu_type, row);
    }
    row.count++;
    row.providers.add(o.provider);
    if (o.vram_gb > row.vramGb) row.vramGb = o.vram_gb;
    if (o.price_spot != null && (row.spotMin === null || o.price_spot < row.spotMin)) row.spotMin = o.price_spot;
    if (o.price_on_demand > 0 && (row.ondemandMin === null || o.price_on_demand < row.ondemandMin)) row.ondemandMin = o.price_on_demand;
  }
  return Array.from(map.values()).sort((a, b) => b.vramGb - a.vramGb || a.gpuType.localeCompare(b.gpuType));
}

function fmtPrice(p: number | null) {
  if (p === null) return <span className="text-slate-600">—</span>;
  return <span>${p.toFixed(3)}</span>;
}

function fmtTime(ts: number | null) {
  if (!ts) return 'never';
  const diff = Math.round((Date.now() - ts) / 1000);
  if (diff < 60) return `${diff}s ago`;
  return `${Math.round(diff / 60)}m ago`;
}

// ── Panel ─────────────────────────────────────────────────────────────────────

interface Props {
  open: boolean;
  selectedGpuType: string | null;
  onSelect: (gpuType: string) => void;
}

export function GPUCatalogPanel({ open, selectedGpuType, onSelect }: Props) {
  const catalog    = useGpuCatalogStore((s) => s.catalog);
  const loading    = useGpuCatalogStore((s) => s.loading);
  const lastFetched = useGpuCatalogStore((s) => s.lastFetched);
  const refresh    = useGpuCatalogStore((s) => s.refresh);
  const refreshIfStale = useGpuCatalogStore((s) => s.refreshIfStale);

  const [query, setQuery] = useState('');

  // Trigger refresh on open (stale-while-revalidate)
  useEffect(() => {
    if (open) refreshIfStale();
  }, [open, refreshIfStale]);

  const rows = useMemo(() => {
    const base = catalog.length > 0 ? aggregateCatalog(catalog) : STATIC_GPU_TYPES;
    if (!query.trim()) return base;
    const q = query.trim().toLowerCase();
    return base.filter((r) => r.gpuType.toLowerCase().includes(q));
  }, [catalog, query]);

  if (!open) return null;

  return createPortal(
    <div
      className="fixed inset-y-0 z-50 flex flex-col"
      style={{ right: 440, width: 380, background: '#080e14', borderLeft: '1px solid rgba(6,182,212,0.15)' }}
    >
      {/* Header */}
      <div
        className="flex shrink-0 items-center justify-between gap-2 px-3 py-2.5"
        style={{ borderBottom: '1px solid rgba(6,182,212,0.12)', background: 'rgba(6,182,212,0.04)' }}
      >
        <div className="flex items-center gap-2">
          <span className="text-[11px] font-bold uppercase tracking-widest text-cyan-400">
            GPU Catalog
          </span>
          {loading
            ? <Loader2 size={11} className="animate-spin text-cyan-500/60" />
            : <span className="text-[10px] text-slate-600">{fmtTime(lastFetched)}</span>
          }
        </div>
        <button
          type="button"
          onClick={() => void refresh()}
          disabled={loading}
          className="flex items-center gap-1 rounded px-2 py-0.5 text-[10px] text-slate-500 hover:text-cyan-300 disabled:opacity-40 transition-colors"
          title="Refresh catalog"
        >
          <RefreshCw size={11} className={loading ? 'animate-spin' : ''} />
          Refresh
        </button>
      </div>

      {/* Search */}
      <div className="shrink-0 px-3 py-2" style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
        <div className="flex items-center gap-1.5 rounded border border-slate-700/60 bg-slate-900/60 px-2 py-1">
          <Search size={10} className="text-slate-600" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Filter GPU types…"
            className="flex-1 bg-transparent text-xs text-slate-200 outline-none placeholder:text-slate-600"
          />
        </div>
      </div>

      {/* Table header */}
      <div
        className="grid shrink-0 px-3 py-1.5 text-[10px] font-semibold uppercase tracking-wider text-slate-500"
        style={{ gridTemplateColumns: '1fr 52px 72px 72px', borderBottom: '1px solid rgba(255,255,255,0.05)' }}
      >
        <span>GPU Type</span>
        <span className="text-right">VRAM</span>
        <span className="text-right">Spot/hr</span>
        <span className="text-right">OD/hr</span>
      </div>

      {/* Rows */}
      <div className="flex-1 overflow-y-auto">
        {rows.length === 0 && (
          <p className="px-3 py-4 text-[11px] text-slate-600">No GPUs match filter.</p>
        )}
        {rows.map((row) => {
          const isSelected = selectedGpuType === row.gpuType;
          return (
            <button
              key={row.gpuType}
              type="button"
              onClick={() => onSelect(row.gpuType)}
              className={`grid w-full px-3 py-2 text-left text-xs transition-colors ${
                isSelected
                  ? 'bg-cyan-500/10 text-cyan-200'
                  : 'text-slate-300 hover:bg-slate-800/60'
              }`}
              style={{
                gridTemplateColumns: '1fr 52px 72px 72px',
                borderBottom: '1px solid rgba(255,255,255,0.03)',
                ...(isSelected ? { boxShadow: 'inset 2px 0 0 rgba(6,182,212,0.7)' } : {}),
              }}
            >
              <span className="font-mono font-medium">{row.gpuType}</span>
              <span className="text-right text-slate-400">{row.vramGb}GB</span>
              <span className={`text-right font-mono ${row.spotMin != null ? 'text-emerald-400' : 'text-slate-600'}`}>
                {fmtPrice(row.spotMin)}
              </span>
              <span className="text-right font-mono text-slate-400">
                {fmtPrice(row.ondemandMin)}
              </span>
            </button>
          );
        })}
      </div>

      {/* Footer */}
      <div
        className="shrink-0 px-3 py-1.5"
        style={{ borderTop: '1px solid rgba(255,255,255,0.04)' }}
      >
        <span className="text-[9px] text-slate-700">
          {catalog.length > 0
            ? `${rows.length} GPU types · ${catalog.length} offers`
            : 'Static list — backend unavailable'
          }
          {loading && ' · refreshing…'}
        </span>
      </div>
    </div>,
    document.body,
  );
}
