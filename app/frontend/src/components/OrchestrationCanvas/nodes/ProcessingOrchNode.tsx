import { memo } from 'react';
import { Handle, Position, type NodeProps } from 'reactflow';
import type { ProcessingOrchNodeData } from '../../../types/dag';
import { StatusLED } from '../../../design/components/StatusLED';
import { useDagStore } from '../../../store/dagStore';

// Apache Spark flame/lightning icon
function SparkIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 40 40" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
      <path d="M22 4L10 22h10l-4 14 18-20H24l4-12z" opacity="0.9"/>
    </svg>
  );
}

function ProcessingOrchNodeInner({ id, data, selected }: NodeProps<ProcessingOrchNodeData>) {
  const selectNode = useDagStore((s) => s.selectNode);

  return (
    <div
      onClick={() => selectNode(id)}
      className={`min-w-[170px] rounded-xl transition-all cursor-pointer ${selected ? 'neon-selected-cyan' : 'neon-border-cyan'}`}
      style={{ background: '#03060a' }}
    >
      <div className="flex items-center gap-2 rounded-t-xl px-3 py-2" style={{ background: 'rgba(34,211,238,0.06)' }}>
        <SparkIcon className="h-4 w-4 text-cyan-400 shrink-0" />
        <span className="text-[10px] font-bold uppercase tracking-widest text-cyan-300">Processing</span>
      </div>
      <div className="px-3 py-2.5">
        <p className="text-sm font-semibold text-slate-100">
          {data.datasetName
            ? <>{data.datasetName} {data.dslVersion ? <span className="text-cyan-500/60 text-xs">v{data.dslVersion}</span> : null}</>
            : <span className="text-slate-600 italic">no dataset</span>
          }
        </p>
        {data.preprocessRunId && (
          <p className="mt-1 text-[10px] font-mono text-cyan-400/70 truncate max-w-[150px]">
            {data.preprocessRunId}
          </p>
        )}
        <div className="mt-2">
          <StatusLED status={data.status} />
        </div>
      </div>
      <Handle type="target" position={Position.Left}  className="!h-3 !w-3 !border-2 !border-cyan-500 !bg-brand-surface" />
      <Handle type="source" position={Position.Right} className="!h-3 !w-3 !border-2 !border-cyan-500 !bg-brand-surface" />
    </div>
  );
}

export const ProcessingOrchNode = memo(ProcessingOrchNodeInner);
