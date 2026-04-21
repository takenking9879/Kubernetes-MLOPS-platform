import { memo } from 'react';
import { Handle, Position, type NodeProps } from 'reactflow';
import type { DatasetNodeData } from '../../../types/nodes';
import { usePipelineStore } from '../../../store/pipelineStore';

function DatasetNodeInner({ data, id, selected }: NodeProps<DatasetNodeData>) {
  const selectNode = usePipelineStore((s) => s.selectNode);

  return (
    <div
      onClick={() => selectNode(id)}
      className={`min-w-[180px] rounded-xl transition-all ${
        selected ? 'neon-selected-cyan' : 'neon-border-emerald'
      }`}
      style={{ background: '#03060a' }}
    >
      {/* Header */}
      <div className="flex items-center gap-2 rounded-t-xl px-3 py-2" style={{ background: 'rgba(16,185,129,0.08)' }}>
        <div className="h-2 w-2 rounded-full bg-emerald-400 led-pulse" />
        <span className="text-[10px] font-bold uppercase tracking-widest text-emerald-300">
          Dataset
        </span>
      </div>

      {/* Body */}
      <div className="px-3 py-2">
        <p className="text-sm font-medium text-slate-100">{data.label}</p>
        <p className="mt-1 text-[10px] text-slate-500 uppercase tracking-wide">
          {data.source === 'csv' ? 'CSV / Parquet' : 'Iceberg'} · {data.path || 'not set'}
        </p>
        {data.schema && (
          <p className="mt-1 text-[10px] text-emerald-400">
            {data.schema.columns.length} cols
          </p>
        )}
      </div>

      {/* Output handle */}
      <div className="relative">
        <Handle
          type="source"
          position={Position.Bottom}
          className="!h-3 !w-3 !border-2 !border-emerald-500 !bg-brand-surface"
        />
        <span className="absolute -bottom-4 left-1/2 -translate-x-1/2 text-[8px] text-slate-600 uppercase tracking-wide">
          out
        </span>
      </div>
    </div>
  );
}

export const DatasetNode = memo(DatasetNodeInner);
