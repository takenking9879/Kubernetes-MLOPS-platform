import { memo } from 'react';
import { Handle, Position, type NodeProps } from 'reactflow';
import type { ServingOrchNodeData } from '../../../types/dag';
import { StatusLED } from '../../../design/components/StatusLED';
import { useDagStore } from '../../../store/dagStore';

// Ray Serve / deploy icon — stylized server stack
function ServeIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect x="6" y="8"  width="28" height="7" rx="2" fill="currentColor" opacity="0.9"/>
      <rect x="6" y="18" width="28" height="7" rx="2" fill="currentColor" opacity="0.65"/>
      <rect x="6" y="28" width="28" height="7" rx="2" fill="currentColor" opacity="0.4"/>
      <circle cx="30" cy="11.5" r="1.5" fill="#03060a" opacity="0.8"/>
      <circle cx="30" cy="21.5" r="1.5" fill="#03060a" opacity="0.8"/>
      <circle cx="30" cy="31.5" r="1.5" fill="#03060a" opacity="0.8"/>
    </svg>
  );
}

function ServingOrchNodeInner({ id, data, selected }: NodeProps<ServingOrchNodeData>) {
  const selectNode = useDagStore((s) => s.selectNode);

  return (
    <div
      onClick={() => selectNode(id)}
      className={`min-w-[160px] rounded-xl transition-all cursor-pointer ${selected ? 'neon-selected-cyan' : 'neon-border-purple'}`}
      style={{ background: '#03060a' }}
    >
      <div className="flex items-center gap-2 rounded-t-xl px-3 py-2" style={{ background: 'rgba(167,139,250,0.08)' }}>
        <ServeIcon className="h-4 w-4 text-purple-400 shrink-0" />
        <span className="text-[10px] font-bold uppercase tracking-widest text-purple-300">Serving</span>
      </div>
      <div className="px-3 py-2.5">
        <p className="text-sm font-semibold text-slate-100">
          {data.servingMode || <span className="text-slate-600 italic">not configured</span>}
        </p>
        {data.trainRunId
          ? <p className="mt-0.5 text-[10px] font-mono text-purple-400/50 truncate max-w-[140px]">← {data.trainRunId}</p>
          : <p className="mt-0.5 text-[10px] text-slate-600">no training run</p>
        }
        <div className="mt-2">
          <StatusLED status={data.status} />
        </div>
      </div>
      <Handle type="target" position={Position.Left} className="!h-3 !w-3 !border-2 !border-purple-500 !bg-brand-surface" />
    </div>
  );
}

export const ServingOrchNode = memo(ServingOrchNodeInner);
