import { memo } from 'react';
import { Handle, Position, type NodeProps } from 'reactflow';
import type { DatasetOrchNodeData } from '../../../types/dag';
import { StatusLED } from '../../../design/components/StatusLED';
import { useDagStore } from '../../../store/dagStore';

// AWS S3 bucket icon
function S3Icon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
      <ellipse cx="20" cy="9" rx="12" ry="5" fill="currentColor" opacity="0.9"/>
      <path d="M8 9v22c0 2.76 5.37 5 12 5s12-2.24 12-5V9" stroke="currentColor" strokeWidth="1.5" fill="none" opacity="0.7"/>
      <path d="M8 20c0 2.76 5.37 5 12 5s12-2.24 12-5" stroke="currentColor" strokeWidth="1.5" fill="none" opacity="0.5"/>
      <path d="M8 14.5c0 2.76 5.37 5 12 5s12-2.24 12-5" stroke="currentColor" strokeWidth="1" fill="none" opacity="0.35"/>
    </svg>
  );
}

function DatasetOrchNodeInner({ id, data, selected }: NodeProps<DatasetOrchNodeData>) {
  const selectNode = useDagStore((s) => s.selectNode);

  return (
    <div
      onClick={() => selectNode(id)}
      className={`min-w-[160px] rounded-xl transition-all cursor-pointer ${selected ? 'neon-selected-cyan' : 'neon-border-emerald'}`}
      style={{ background: '#03060a' }}
    >
      <div className="flex items-center gap-2 rounded-t-xl px-3 py-2" style={{ background: 'rgba(16,185,129,0.08)' }}>
        <S3Icon className="h-4 w-4 text-emerald-400 shrink-0" />
        <span className="text-[10px] font-bold uppercase tracking-widest text-emerald-300">Dataset</span>
      </div>
      <div className="px-3 py-2.5">
        <p className="text-sm font-semibold text-slate-100">
          {data.datasetName || <span className="text-slate-600 italic">not set</span>}
        </p>
        <div className="mt-2">
          <StatusLED status={data.status} />
        </div>
      </div>
      <Handle
        type="source"
        position={Position.Right}
        className="!h-3 !w-3 !border-2 !border-emerald-500 !bg-brand-surface"
      />
    </div>
  );
}

export const DatasetOrchNode = memo(DatasetOrchNodeInner);
