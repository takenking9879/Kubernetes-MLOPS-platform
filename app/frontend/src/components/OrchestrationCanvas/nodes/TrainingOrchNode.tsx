import { memo } from 'react';
import { Handle, Position, type NodeProps } from 'reactflow';
import type { TrainingOrchNodeData } from '../../../types/dag';
import { StatusLED } from '../../../design/components/StatusLED';
import { useDagStore } from '../../../store/dagStore';

// Ray logo — stylized sun/starburst
function RayIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 40 40" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
      <circle cx="20" cy="20" r="5.5" opacity="0.95"/>
      {[0, 45, 90, 135, 180, 225, 270, 315].map((deg) => {
        const r = Math.PI * deg / 180;
        const x1 = 20 + 8.5 * Math.cos(r);
        const y1 = 20 + 8.5 * Math.sin(r);
        const x2 = 20 + 17 * Math.cos(r);
        const y2 = 20 + 17 * Math.sin(r);
        return <line key={deg} x1={x1} y1={y1} x2={x2} y2={y2} stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" opacity="0.8"/>;
      })}
    </svg>
  );
}

function TrainingOrchNodeInner({ id, data, selected }: NodeProps<TrainingOrchNodeData>) {
  const selectNode = useDagStore((s) => s.selectNode);

  return (
    <div
      onClick={() => selectNode(id)}
      className={`min-w-[170px] rounded-xl transition-all cursor-pointer ${selected ? 'neon-selected-cyan' : 'neon-border-orange'}`}
      style={{ background: '#03060a' }}
    >
      <div className="flex items-center gap-2 rounded-t-xl px-3 py-2" style={{ background: 'rgba(249,115,22,0.08)' }}>
        <RayIcon className="h-4 w-4 text-orange-400 shrink-0" />
        <span className="text-[10px] font-bold uppercase tracking-widest text-orange-300">Training</span>
      </div>
      <div className="px-3 py-2.5">
        <p className="text-sm font-semibold text-slate-100">
          {data.modelType || <span className="text-slate-600 italic">not configured</span>}
          {data.framework && <span className="ml-1 text-orange-400/60 text-xs">· {data.framework}</span>}
        </p>
        {data.preprocessRunId
          ? <p className="mt-0.5 text-[10px] font-mono text-orange-400/50 truncate max-w-[150px]">← {data.preprocessRunId}</p>
          : <p className="mt-0.5 text-[10px] text-slate-600">no preprocess run</p>
        }
        {data.trainRunId && (
          <p className="mt-0.5 text-[10px] font-mono text-orange-300/70 truncate max-w-[150px]">
            {data.trainRunId}
          </p>
        )}
        <div className="mt-2">
          <StatusLED status={data.status} />
        </div>
      </div>
      <Handle type="target" position={Position.Left}  className="!h-3 !w-3 !border-2 !border-orange-500 !bg-brand-surface" />
      <Handle type="source" position={Position.Right} className="!h-3 !w-3 !border-2 !border-orange-500 !bg-brand-surface" />
    </div>
  );
}

export const TrainingOrchNode = memo(TrainingOrchNodeInner);
