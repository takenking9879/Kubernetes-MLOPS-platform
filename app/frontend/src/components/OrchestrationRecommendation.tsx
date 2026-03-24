/**
 * OrchestrationRecommendation — shows the smart recommendation from Step 3
 * of the Launch Wizard.
 *
 * Displays:
 *  - Orchestration badge (color-coded by complexity)
 *  - Cost estimate (spot + on-demand per hour and per day)
 *  - Reason / explanation
 *  - Warnings list
 *  - Expandable generated SkyPilot YAML preview
 */

import { useState } from 'react';
import { ChevronDown, ChevronUp, AlertTriangle, CheckCircle, Zap, Layers } from 'lucide-react';
import type { OrchestratorRecommendation } from '../api/platformClient';

// ── Constants ──────────────────────────────────────────────────────────────────

const ORCH_META: Record<string, { label: string; color: string; icon: React.ReactNode }> = {
  ray_train: {
    label: 'Ray Train',
    color: 'bg-blue-900/60 text-blue-300 ring-blue-700/50',
    icon: <Zap size={12} className="inline mr-1 text-blue-400" />,
  },
  llm_finetune: {
    label: 'HF+TRL+DeepSpeed',
    color: 'bg-purple-900/60 text-purple-300 ring-purple-700/50',
    icon: <Layers size={12} className="inline mr-1 text-purple-400" />,
  },
  vllm_single_node: {
    label: 'vLLM (single-node)',
    color: 'bg-green-900/60 text-green-300 ring-green-700/50',
    icon: <CheckCircle size={12} className="inline mr-1 text-green-400" />,
  },
  ray_vllm_multinode: {
    label: 'Ray + vLLM (multi-node)',
    color: 'bg-amber-900/60 text-amber-300 ring-amber-700/50',
    icon: <Layers size={12} className="inline mr-1 text-amber-400" />,
  },
};

// ── Props ──────────────────────────────────────────────────────────────────────

interface Props {
  recommendation: OrchestratorRecommendation;
  yamlPreview?: string;
  runtimeHours?: number;               // for total cost estimate
}

// ── Component ──────────────────────────────────────────────────────────────────

export function OrchestrationRecommendation({
  recommendation: rec,
  yamlPreview,
  runtimeHours = 2,
}: Props) {
  const [yamlOpen, setYamlOpen] = useState(false);

  const meta = ORCH_META[rec.orchestration] ?? {
    label: rec.orchestration,
    color: 'bg-slate-700 text-slate-300 ring-slate-600',
    icon: null,
  };

  const spotHourly = rec.estimated_cost_spot;
  const odHourly = rec.estimated_cost_ondemand;
  const spotTotal = spotHourly != null ? spotHourly * runtimeHours : null;
  const odTotal = odHourly != null ? odHourly * runtimeHours : null;

  const savings =
    spotHourly != null && odHourly != null && odHourly > 0
      ? Math.round((1 - spotHourly / odHourly) * 100)
      : null;

  return (
    <div className="flex flex-col gap-3">
      {/* ── Orchestration badge + reason ─────────────────────────────────── */}
      <div className="flex flex-wrap items-start gap-3">
        <span
          className={`inline-flex items-center rounded-full px-3 py-1 text-xs font-semibold ring-1 ${meta.color}`}
        >
          {meta.icon}
          {meta.label}
        </span>
        <p className="flex-1 text-xs text-slate-400 leading-relaxed">{rec.reason}</p>
      </div>

      {/* ── Cost estimate table ───────────────────────────────────────────── */}
      {(spotHourly != null || odHourly != null) && (
        <div className="overflow-hidden rounded border border-slate-700">
          <table className="w-full text-xs">
            <thead>
              <tr className="bg-slate-800 text-left text-[10px] text-slate-400">
                <th className="px-2 py-1.5">Type</th>
                <th className="px-2 py-1.5 text-right">$/hr</th>
                <th className="px-2 py-1.5 text-right">
                  Total ({runtimeHours >= 1 ? `${runtimeHours}h` : `${runtimeHours * 60}m`})
                </th>
                <th className="px-2 py-1.5 text-right">$/day</th>
              </tr>
            </thead>
            <tbody>
              {spotHourly != null && (
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
                    ${spotHourly.toFixed(3)}
                  </td>
                  <td className="px-2 py-1.5 text-right font-mono font-semibold text-green-300">
                    {spotTotal != null ? `$${spotTotal.toFixed(2)}` : '—'}
                  </td>
                  <td className="px-2 py-1.5 text-right font-mono text-slate-400">
                    ${(spotHourly * 24).toFixed(2)}
                  </td>
                </tr>
              )}
              {odHourly != null && (
                <tr className="border-t border-slate-700">
                  <td className="px-2 py-1.5 text-slate-300">On-demand</td>
                  <td className="px-2 py-1.5 text-right font-mono text-slate-300">
                    ${odHourly.toFixed(3)}
                  </td>
                  <td className="px-2 py-1.5 text-right font-mono text-slate-400">
                    {odTotal != null ? `$${odTotal.toFixed(2)}` : '—'}
                  </td>
                  <td className="px-2 py-1.5 text-right font-mono text-slate-500">
                    ${(odHourly * 24).toFixed(2)}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}

      {/* ── Savings callout ───────────────────────────────────────────────── */}
      {spotTotal != null && odTotal != null && savings != null && savings > 0 && (
        <p className="text-[11px] text-green-400">
          Est. savings with spot:{' '}
          <strong>${(odTotal - spotTotal).toFixed(2)}</strong> over{' '}
          {runtimeHours >= 1 ? `${runtimeHours}h` : `${runtimeHours * 60}m`}
        </p>
      )}

      {/* ── Warnings ──────────────────────────────────────────────────────── */}
      {rec.warnings.length > 0 && (
        <div className="flex flex-col gap-1">
          {rec.warnings.map((w, i) => (
            <div key={i} className="flex items-start gap-1.5 text-xs text-amber-400">
              <AlertTriangle size={12} className="mt-0.5 shrink-0" />
              {w}
            </div>
          ))}
        </div>
      )}

      {/* ── YAML preview (expandable) ─────────────────────────────────────── */}
      {yamlPreview && (
        <div className="rounded border border-slate-700">
          <button
            type="button"
            onClick={() => setYamlOpen((o) => !o)}
            className="flex w-full items-center justify-between px-3 py-1.5 text-[11px] text-slate-400 hover:bg-slate-800/50"
          >
            <span className="font-medium">SkyPilot YAML preview</span>
            {yamlOpen ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
          </button>
          {yamlOpen && (
            <pre className="max-h-64 overflow-y-auto border-t border-slate-700 bg-slate-950 p-3 text-[10px] text-slate-300 leading-relaxed">
              {yamlPreview}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}
