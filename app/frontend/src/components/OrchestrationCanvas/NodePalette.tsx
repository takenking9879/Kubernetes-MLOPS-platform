import { useDagStore } from '../../store/dagStore';
import type { OrchestrationNodeData } from '../../types/dag';

interface PaletteItem {
  kind: OrchestrationNodeData['nodeKind'] | 'merge' | 'decision' | 'model_selector';
  label: string;
  color: string;
  borderClass: string;
  disabled?: boolean;
  tag?: string;
}

const ITEMS: PaletteItem[] = [
  { kind: 'dataset',    label: 'Dataset',     color: 'text-emerald-300', borderClass: 'border-emerald-500/30 hover:bg-emerald-500/5' },
  { kind: 'processing', label: 'Processing',  color: 'text-cyan-300',    borderClass: 'border-cyan-500/30 hover:bg-cyan-500/5'    },
  { kind: 'training',   label: 'Training',    color: 'text-orange-300',  borderClass: 'border-orange-500/30 hover:bg-orange-500/5' },
  { kind: 'serving',    label: 'Serving',     color: 'text-purple-300',  borderClass: 'border-purple-500/30 hover:bg-purple-500/5' },
  { kind: 'merge',      label: 'Merge',       color: 'text-slate-500',   borderClass: 'border-slate-700/30',  disabled: true, tag: 'Soon' },
  { kind: 'decision',   label: 'Decision',    color: 'text-slate-500',   borderClass: 'border-slate-700/30',  disabled: true, tag: 'Soon' },
  { kind: 'model_selector', label: 'Model Sel.', color: 'text-slate-500', borderClass: 'border-slate-700/30', disabled: true, tag: 'Soon' },
];

export function NodePalette() {
  const addNode = useDagStore((s) => s.addNode);

  return (
    <aside className="flex w-[160px] flex-shrink-0 flex-col h-full glass-panel"
      style={{ borderTop: 'none', borderLeft: 'none', borderBottom: 'none' }}>
      <div className="px-3 py-3 border-b border-cyan-500/15">
        <p className="text-[10px] font-bold uppercase tracking-widest text-cyan-500/60">Node Palette</p>
      </div>
      <div className="flex-1 overflow-y-auto p-2 space-y-1.5">
        {ITEMS.map((item) => (
          <button
            key={item.kind}
            disabled={item.disabled}
            title={item.disabled ? 'Coming soon' : `Add ${item.label} node`}
            onClick={() => {
              if (!item.disabled) {
                addNode(item.kind as OrchestrationNodeData['nodeKind']);
              }
            }}
            className={`
              w-full rounded border bg-brand-panel px-2 py-2 text-left text-xs font-semibold
              transition-colors relative
              ${item.borderClass}
              ${item.color}
              ${item.disabled ? 'opacity-40 cursor-not-allowed' : 'cursor-pointer'}
            `}
          >
            {item.label}
            {item.tag && (
              <span className="absolute right-1.5 top-1/2 -translate-y-1/2 text-[8px] text-slate-600 uppercase tracking-wide">
                {item.tag}
              </span>
            )}
          </button>
        ))}
      </div>
      <div className="px-3 py-2 border-t border-cyan-500/10">
        <p className="text-[9px] text-slate-700 uppercase tracking-wide">Click to add</p>
      </div>
    </aside>
  );
}
