import { useMemo, useState } from 'react';
import {
  getTransformers,
  getEstimators,
  getTerminalNodes,
  getNodeContract,
} from '../../registry/NodeRegistry';
import { CatalogItem } from './CatalogItem';
import { usePipelineStore } from '../../store/pipelineStore';

export function CatalogPanel() {
  const [query, setQuery] = useState('');
  const nodes = usePipelineStore((s) => s.nodes);

  const transformers = getTransformers();
  const estimators = getEstimators();
  const terminalNodes = getTerminalNodes();

  const q = query.trim().toLowerCase();
  const byFilter = useMemo(
    () => (label: string, description: string) => {
      if (!q) return true;
      return label.toLowerCase().includes(q) || description.toLowerCase().includes(q);
    },
    [q],
  );

  const visibleTransformers = transformers.filter((d) => byFilter(d.label, d.description));
  const visibleEstimators = estimators.filter((d) => byFilter(d.label, d.description));
  const visibleTerminal = terminalNodes.filter((d) => byFilter(d.label, d.description));

  const canCreate = (type: string): boolean => {
    const contract = getNodeContract(type);
    const instances = nodes.filter((n) => n.data.type === type).length;
    return instances < contract.maxInstances;
  };

  return (
    <div className="flex h-full w-60 flex-col border-r border-slate-800 bg-slate-900">
      <div className="border-b border-slate-800 px-4 py-3">
        <h2 className="text-sm font-semibold text-slate-200">Stage Catalog</h2>
        <p className="text-[10px] text-slate-500">Drag stages onto the canvas</p>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search..."
          className="mt-2 w-full rounded border border-slate-700 bg-slate-950 px-2 py-1 text-xs text-slate-200 outline-none focus:border-slate-500"
        />
      </div>

      <div className="flex-1 overflow-y-auto p-3">
        <details open className="mb-3 rounded border border-slate-800 bg-slate-950/50">
          <summary className="cursor-pointer px-2 py-2 text-xs font-semibold uppercase tracking-wider text-blue-400">
            Transformers ({visibleTransformers.length})
          </summary>
          <div className="space-y-1.5 p-2 pt-0">
            {visibleTransformers.map((def) => (
              <CatalogItem key={def.type} definition={def} disabled={!canCreate(def.type)} />
            ))}
          </div>
        </details>

        <details open className="mb-3 rounded border border-slate-800 bg-slate-950/50">
          <summary className="cursor-pointer px-2 py-2 text-xs font-semibold uppercase tracking-wider text-purple-400">
            Estimators ({visibleEstimators.length})
          </summary>
          <div className="space-y-1.5 p-2 pt-0">
            {visibleEstimators.map((def) => (
              <CatalogItem key={def.type} definition={def} disabled={!canCreate(def.type)} />
            ))}
          </div>
        </details>

        <details open className="rounded border border-slate-800 bg-slate-950/50">
          <summary className="cursor-pointer px-2 py-2 text-xs font-semibold uppercase tracking-wider text-red-400">
            Terminal ({visibleTerminal.length})
          </summary>
          <div className="space-y-1.5 p-2 pt-0">
            {visibleTerminal.map((def) => (
              <CatalogItem key={def.type} definition={def} disabled={!canCreate(def.type)} />
            ))}
          </div>
        </details>
      </div>
    </div>
  );
}
