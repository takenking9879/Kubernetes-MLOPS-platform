/**
 * GPUCatalogPanel — floating panel rendered left of the NodeInspector (440px).
 *
 * Portal to document.body so fixed positioning is always viewport-relative.
 * Provider-per-row aggregation: same GPU from different clouds shown separately
 * with neon colored left-border glow per provider.
 */

import { useEffect, useMemo, useState } from 'react';
import { createPortal } from 'react-dom';
import { RefreshCw, Loader2, Search, TrendingDown } from 'lucide-react';
import { useGpuCatalogStore } from '../store/gpuCatalogStore';
import type { GPUOffer } from '../api/platformClient';

// ── Provider config ───────────────────────────────────────────────────────────

const PROVIDER_CFG: Record<string, {
  label: string;
  leftGlow: string;       // CSS box-shadow color for left border
  borderColor: string;    // rgba for inset shadow
  dot: string;            // hex dot indicator
  pill: string;           // tailwind classes for provider pill
  filterActive: string;   // tailwind classes when filter pill is ON
}> = {
  runpod: {
    label: 'RunPod',
    leftGlow: 'rgba(168,85,247,0.65)',
    borderColor: '#a855f7',
    dot: '#a855f7',
    pill: 'border-violet-500/40 text-violet-300 bg-violet-500/10',
    filterActive: 'bg-violet-600/30 border-violet-400/70 text-violet-200 shadow-[0_0_8px_rgba(168,85,247,0.45)]',
  },
  vast: {
    label: 'Vast.ai',
    leftGlow: 'rgba(148,163,184,0.55)',
    borderColor: '#94a3b8',
    dot: '#94a3b8',
    pill: 'border-slate-500/40 text-slate-300 bg-slate-500/10',
    filterActive: 'bg-slate-600/40 border-slate-400/70 text-slate-200 shadow-[0_0_8px_rgba(148,163,184,0.4)]',
  },
  aws: {
    label: 'AWS',
    leftGlow: 'rgba(249,115,22,0.65)',
    borderColor: '#f97316',
    dot: '#f97316',
    pill: 'border-orange-500/40 text-orange-300 bg-orange-500/10',
    filterActive: 'bg-orange-600/25 border-orange-400/70 text-orange-200 shadow-[0_0_8px_rgba(249,115,22,0.45)]',
  },
  gcp: {
    label: 'GCP',
    leftGlow: 'rgba(59,130,246,0.65)',
    borderColor: '#3b82f6',
    dot: '#3b82f6',
    pill: 'border-blue-500/40 text-blue-300 bg-blue-500/10',
    filterActive: 'bg-blue-600/25 border-blue-400/70 text-blue-200 shadow-[0_0_8px_rgba(59,130,246,0.45)]',
  },
  azure: {
    label: 'Azure',
    leftGlow: 'rgba(14,165,233,0.65)',
    borderColor: '#0ea5e9',
    dot: '#0ea5e9',
    pill: 'border-sky-500/40 text-sky-300 bg-sky-500/10',
    filterActive: 'bg-sky-600/25 border-sky-400/70 text-sky-200 shadow-[0_0_8px_rgba(14,165,233,0.45)]',
  },
};

const ALL_PROVIDERS = ['runpod', 'vast', 'aws', 'gcp', 'azure'];

// ── Aggregation ───────────────────────────────────────────────────────────────

interface AggRow {
  gpuType: string;
  provider: string;
  vramGb: number;
  spotMin: number | null;
  ondemandMin: number | null;
  savePercent: number | null;
  anyAvailable: boolean;  // at least one offer not confirmed out-of-stock
  count: number;
}

function aggregateCatalog(catalog: GPUOffer[]): AggRow[] {
  const map = new Map<string, AggRow>();
  for (const o of catalog) {
    if (!o.skypilot_supported) continue;
    const key = `${o.gpu_type}::${o.provider}`;
    let row = map.get(key);
    if (!row) {
      row = {
        gpuType: o.gpu_type,
        provider: o.provider,
        vramGb: o.vram_gb,
        spotMin: null,
        ondemandMin: null,
        savePercent: null,
        anyAvailable: false,
        count: 0,
      };
      map.set(key, row);
    }
    row.count++;
    if (o.vram_gb > row.vramGb) row.vramGb = o.vram_gb;
    if (o.price_spot != null && (row.spotMin === null || o.price_spot < row.spotMin)) row.spotMin = o.price_spot;
    if (o.price_on_demand > 0 && (row.ondemandMin === null || o.price_on_demand < row.ondemandMin)) row.ondemandMin = o.price_on_demand;
    if (o.available_count !== 0) row.anyAvailable = true;
  }
  for (const row of map.values()) {
    if (row.spotMin != null && row.ondemandMin != null && row.ondemandMin > 0) {
      row.savePercent = Math.round((1 - row.spotMin / row.ondemandMin) * 100);
    }
  }
  return Array.from(map.values());
}

type SortMode = 'spot' | 'od' | 'save';

function sortRows(rows: AggRow[], mode: SortMode): AggRow[] {
  return [...rows].sort((a, b) => {
    if (mode === 'spot') {
      if (a.spotMin === null && b.spotMin === null) return a.gpuType.localeCompare(b.gpuType);
      if (a.spotMin === null) return 1;
      if (b.spotMin === null) return -1;
      return a.spotMin - b.spotMin;
    }
    if (mode === 'od') {
      if (a.ondemandMin === null && b.ondemandMin === null) return a.gpuType.localeCompare(b.gpuType);
      if (a.ondemandMin === null) return 1;
      if (b.ondemandMin === null) return -1;
      return a.ondemandMin - b.ondemandMin;
    }
    // save — highest % first
    if (a.savePercent === null && b.savePercent === null) return a.gpuType.localeCompare(b.gpuType);
    if (a.savePercent === null) return 1;
    if (b.savePercent === null) return -1;
    return b.savePercent - a.savePercent;
  });
}

// ── Static fallback ───────────────────────────────────────────────────────────

const STATIC_ROWS: AggRow[] = [
  { gpuType: 'H100',       provider: 'aws',    vramGb: 80, spotMin: null, ondemandMin: null, savePercent: null, anyAvailable: false, count: 0 },
  { gpuType: 'H100-80GB',  provider: 'gcp',    vramGb: 80, spotMin: null, ondemandMin: null, savePercent: null, anyAvailable: false, count: 0 },
  { gpuType: 'A100-80GB',  provider: 'runpod', vramGb: 80, spotMin: null, ondemandMin: null, savePercent: null, anyAvailable: false, count: 0 },
  { gpuType: 'A100-40GB',  provider: 'runpod', vramGb: 40, spotMin: null, ondemandMin: null, savePercent: null, anyAvailable: false, count: 0 },
  { gpuType: 'L40S',       provider: 'vast',   vramGb: 48, spotMin: null, ondemandMin: null, savePercent: null, anyAvailable: false, count: 0 },
  { gpuType: 'L40',        provider: 'vast',   vramGb: 48, spotMin: null, ondemandMin: null, savePercent: null, anyAvailable: false, count: 0 },
  { gpuType: 'L4',         provider: 'gcp',    vramGb: 24, spotMin: null, ondemandMin: null, savePercent: null, anyAvailable: false, count: 0 },
  { gpuType: 'A10G',       provider: 'aws',    vramGb: 24, spotMin: null, ondemandMin: null, savePercent: null, anyAvailable: false, count: 0 },
  { gpuType: 'RTX-4090',   provider: 'runpod', vramGb: 24, spotMin: null, ondemandMin: null, savePercent: null, anyAvailable: false, count: 0 },
  { gpuType: 'RTX-3090',   provider: 'runpod', vramGb: 24, spotMin: null, ondemandMin: null, savePercent: null, anyAvailable: false, count: 0 },
  { gpuType: 'V100-32GB',  provider: 'aws',    vramGb: 32, spotMin: null, ondemandMin: null, savePercent: null, anyAvailable: false, count: 0 },
  { gpuType: 'V100',       provider: 'aws',    vramGb: 16, spotMin: null, ondemandMin: null, savePercent: null, anyAvailable: false, count: 0 },
];

// ── Helpers ───────────────────────────────────────────────────────────────────

const FALLBACK_CFG = PROVIDER_CFG.runpod!;
function getProviderCfg(providerKey: string) {
  return PROVIDER_CFG[providerKey] ?? FALLBACK_CFG;
}

function fmtTime(ts: number | null) {
  if (!ts) return 'never';
  const diff = Math.round((Date.now() - ts) / 1000);
  if (diff < 60) return `${diff}s ago`;
  return `${Math.round(diff / 60)}m ago`;
}

function saveColor(pct: number | null): string {
  if (pct === null) return 'text-slate-600';
  if (pct >= 70) return 'text-emerald-400';
  if (pct >= 45) return 'text-amber-400';
  return 'text-yellow-500';
}

// ── Panel ─────────────────────────────────────────────────────────────────────

interface Props {
  open: boolean;
  selectedGpuType: string | null;
  onSelect: (gpuType: string) => void;
}

export function GPUCatalogPanel({ open, selectedGpuType, onSelect }: Props) {
  const catalog      = useGpuCatalogStore((s) => s.catalog);
  const loading      = useGpuCatalogStore((s) => s.loading);
  const lastFetched  = useGpuCatalogStore((s) => s.lastFetched);
  const refresh      = useGpuCatalogStore((s) => s.refresh);
  const refreshIfStale = useGpuCatalogStore((s) => s.refreshIfStale);

  const [query, setQuery]               = useState('');
  const [activeProviders, setActiveProviders] = useState<string[]>(ALL_PROVIDERS);
  const [sortMode, setSortMode]         = useState<SortMode>('spot');
  const [showSave, setShowSave]         = useState(false);

  useEffect(() => {
    if (open) refreshIfStale();
  }, [open, refreshIfStale]);

  const toggleProvider = (p: string) => {
    setActiveProviders((prev) =>
      prev.includes(p)
        ? prev.length > 1 ? prev.filter((x) => x !== p) : prev  // keep at least one
        : [...prev, p],
    );
  };

  const presentProviders = useMemo(() => {
    const base = catalog.length > 0 ? aggregateCatalog(catalog) : STATIC_ROWS;
    return new Set(base.map((r) => {
      const k = r.provider.toLowerCase();
      return k === 'vast.ai' ? 'vast' : k;
    }));
  }, [catalog]);

  const rows = useMemo(() => {
    const base = catalog.length > 0 ? aggregateCatalog(catalog) : STATIC_ROWS;
    const filtered = base.filter((r) => {
      const providerKey = r.provider.toLowerCase();
      const normalized = providerKey === 'vast.ai' ? 'vast' : providerKey;
      if (!activeProviders.some((p) => normalized.startsWith(p))) return false;
      if (query.trim()) return r.gpuType.toLowerCase().includes(query.trim().toLowerCase());
      return true;
    });
    return sortRows(filtered, sortMode);
  }, [catalog, activeProviders, query, sortMode]);

  if (!open) return null;

  return createPortal(
    <div
      className="fixed inset-y-0 z-50 flex flex-col"
      style={{
        right: 440,
        width: 400,
        background: '#060c12',
        borderLeft: '1px solid rgba(6,182,212,0.12)',
        borderRight: '1px solid rgba(6,182,212,0.06)',
      }}
    >
      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <div
        className="flex shrink-0 items-center justify-between gap-2 px-3 py-2.5"
        style={{ borderBottom: '1px solid rgba(6,182,212,0.10)', background: 'rgba(6,182,212,0.03)' }}
      >
        <div className="flex items-center gap-2">
          <span className="text-[11px] font-extrabold uppercase tracking-widest text-cyan-400"
            style={{ textShadow: '0 0 12px rgba(6,182,212,0.6)' }}>
            GPU Catalog
          </span>
          {loading
            ? <Loader2 size={11} className="animate-spin text-cyan-500/60" />
            : <span className="text-[9px] text-slate-600">{fmtTime(lastFetched)}</span>
          }
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setShowSave((v) => !v)}
            className={`flex items-center gap-1 rounded border px-2 py-0.5 text-[10px] transition-colors ${
              showSave
                ? 'border-amber-500/50 bg-amber-500/10 text-amber-300'
                : 'border-slate-700 text-slate-500 hover:text-amber-300'
            }`}
            title="Toggle savings column"
          >
            <TrendingDown size={10} />
            Save%
          </button>
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
      </div>

      {/* ── Provider filter pills ───────────────────────────────────────────── */}
      <div
        className="flex shrink-0 flex-wrap items-center gap-1.5 px-3 py-2"
        style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}
      >
        {ALL_PROVIDERS.filter((p) => presentProviders.has(p)).map((p) => {
          const cfg = getProviderCfg(p);
          const active = activeProviders.includes(p);
          return (
            <button
              key={p}
              type="button"
              onClick={() => toggleProvider(p)}
              className={`rounded border px-2 py-0.5 text-[10px] font-semibold transition-all ${
                active ? cfg.filterActive : 'border-slate-700/50 text-slate-600 hover:text-slate-400'
              }`}
            >
              {cfg.label}
            </button>
          );
        })}
      </div>

      {/* ── Sort mode ──────────────────────────────────────────────────────── */}
      <div
        className="flex shrink-0 items-center gap-1 px-3 py-1.5"
        style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}
      >
        <span className="mr-1 text-[9px] font-semibold uppercase tracking-wider text-slate-600">Sort</span>
        {(['spot', 'od', 'save'] as SortMode[]).map((m) => (
          <button
            key={m}
            type="button"
            onClick={() => setSortMode(m)}
            className={`rounded px-2 py-0.5 text-[10px] font-medium transition-colors ${
              sortMode === m
                ? 'bg-cyan-500/15 text-cyan-300 shadow-[0_0_6px_rgba(6,182,212,0.3)]'
                : 'text-slate-500 hover:text-slate-300'
            }`}
          >
            {m === 'spot' ? 'Spot ↑' : m === 'od' ? 'OD ↑' : 'Save ↓'}
          </button>
        ))}
      </div>

      {/* ── Search ─────────────────────────────────────────────────────────── */}
      <div className="shrink-0 px-3 py-2" style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
        <div className="flex items-center gap-1.5 rounded border border-slate-700/50 bg-slate-900/60 px-2 py-1">
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

      {/* ── Column headers ─────────────────────────────────────────────────── */}
      <div
        className="shrink-0 px-3 py-1.5 text-[9px] font-bold uppercase tracking-widest text-slate-600"
        style={{
          display: 'grid',
          gridTemplateColumns: showSave ? '1fr 44px 64px 64px 44px' : '1fr 44px 64px 64px',
          borderBottom: '1px solid rgba(255,255,255,0.05)',
        }}
      >
        <span>GPU / Provider</span>
        <span className="text-right">VRAM</span>
        <span className="text-right">Spot/hr</span>
        <span className="text-right">OD/hr</span>
        {showSave && <span className="text-right">Save</span>}
      </div>

      {/* ── Rows ───────────────────────────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto" style={{ scrollbarWidth: 'thin', scrollbarColor: 'rgba(6,182,212,0.2) transparent' }}>
        {rows.length === 0 && (
          <p className="px-3 py-4 text-[11px] text-slate-600">No GPUs match filter.</p>
        )}
        {rows.map((row) => {
          const providerKey = row.provider.toLowerCase() === 'vast.ai' ? 'vast' : row.provider.toLowerCase();
          const cfg = getProviderCfg(providerKey);
          const isSelected = selectedGpuType === row.gpuType;

          return (
            <button
              key={`${row.gpuType}::${row.provider}`}
              type="button"
              onClick={() => onSelect(row.gpuType)}
              className={`w-full px-3 py-2 text-left text-xs transition-colors ${
                isSelected ? 'bg-cyan-500/8' : 'hover:bg-slate-800/50'
              }`}
              style={{
                display: 'grid',
                gridTemplateColumns: showSave ? '1fr 44px 64px 64px 44px' : '1fr 44px 64px 64px',
                borderBottom: '1px solid rgba(255,255,255,0.025)',
                boxShadow: isSelected
                  ? 'inset 3px 0 0 rgba(6,182,212,0.7), inset 0 0 20px rgba(6,182,212,0.03)'
                  : `inset 3px 0 0 ${cfg.leftGlow}`,
              }}
            >
              {/* GPU name + provider badge */}
              <span className="flex min-w-0 flex-col gap-0.5">
                <span
                  className={`truncate font-mono text-[11px] font-semibold ${isSelected ? 'text-cyan-200' : 'text-slate-100'}`}
                  style={isSelected ? { textShadow: '0 0 8px rgba(6,182,212,0.5)' } : undefined}
                >
                  {row.gpuType}
                </span>
                <span
                  className={`inline-flex w-fit items-center gap-1 rounded border px-1 py-px text-[8px] font-bold uppercase tracking-wide ${cfg.pill}`}
                >
                  <span
                    className="inline-block h-1.5 w-1.5 rounded-full"
                    style={{ background: cfg.dot, boxShadow: `0 0 4px ${cfg.dot}` }}
                  />
                  {cfg.label}
                </span>
              </span>

              {/* VRAM */}
              <span className="self-start pt-0.5 text-right text-[11px] text-slate-500">
                {row.vramGb}G
              </span>

              {/* Spot */}
              <span
                className="self-start pt-0.5 text-right font-mono text-[11px]"
                style={row.spotMin != null ? { color: '#22d3ee', textShadow: '0 0 6px rgba(34,211,238,0.4)' } : undefined}
              >
                {row.spotMin != null ? `$${row.spotMin.toFixed(3)}` : <span className="text-slate-700">—</span>}
              </span>

              {/* OD */}
              <span className={`self-start pt-0.5 text-right font-mono text-[11px] ${row.ondemandMin != null ? 'text-slate-300' : 'text-slate-700'}`}>
                {row.ondemandMin != null ? `$${row.ondemandMin.toFixed(3)}` : '—'}
              </span>

              {/* Save % (optional) */}
              {showSave && (
                <span className={`self-start pt-0.5 text-right font-mono text-[11px] font-bold ${saveColor(row.savePercent)}`}>
                  {row.savePercent != null ? `${row.savePercent}%` : '—'}
                </span>
              )}
            </button>
          );
        })}
      </div>

      {/* ── Footer ─────────────────────────────────────────────────────────── */}
      <div
        className="shrink-0 px-3 py-1.5"
        style={{ borderTop: '1px solid rgba(255,255,255,0.04)', background: 'rgba(6,182,212,0.02)' }}
      >
        <span className="text-[9px] text-slate-700">
          {catalog.length > 0
            ? `${rows.length} entries · ${catalog.length} offers`
            : 'Static list — backend unavailable'}
          {loading && <span className="ml-1 text-cyan-700/60">· refreshing…</span>}
        </span>
      </div>
    </div>,
    document.body,
  );
}
