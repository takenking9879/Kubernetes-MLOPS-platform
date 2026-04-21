import { useDagStore } from '../../store/dagStore';
import { StatusLED } from '../../design/components/StatusLED';

export function StatusBar() {
  const nodes = useDagStore((s) => s.nodes);

  if (nodes.length === 0) return null;

  return (
    <footer
      className="flex h-8 shrink-0 items-center gap-4 px-4 glass-panel overflow-x-auto"
      style={{ borderBottom: 'none', borderLeft: 'none', borderRight: 'none' }}
    >
      <span className="text-[9px] font-bold uppercase tracking-widest text-slate-700 mr-2 shrink-0">Pipeline</span>
      {nodes.map((n) => (
        <span key={n.id} className="flex items-center gap-1.5 shrink-0">
          <StatusLED status={n.data.status} size="sm" label={n.data.label || n.data.nodeKind} />
        </span>
      ))}
      <span className="ml-auto text-[9px] font-mono text-slate-700 shrink-0">
        {nodes.length} node{nodes.length !== 1 ? 's' : ''}
      </span>
    </footer>
  );
}
