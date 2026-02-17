import { memo } from 'react';
import { Handle, Position, type NodeProps } from 'reactflow';
import type { DatasetNodeData } from '../../../types/nodes';
import { usePipelineStore } from '../../../store/pipelineStore';

function DatasetNodeInner({ data, id, selected }: NodeProps<DatasetNodeData>) {
  const selectNode = usePipelineStore((s) => s.selectNode);

  return (
    <div
      onClick={() => selectNode(id)}
      className={`
        min-w-[180px] rounded-lg border-2 bg-slate-900 shadow-lg
        ${selected ? 'border-emerald-400 ring-2 ring-emerald-400/30' : 'border-emerald-600/50'}
      `}
    >
      {/* Header */}
      <div className="flex items-center gap-2 rounded-t-md bg-emerald-600/20 px-3 py-2">
        <div className="h-2.5 w-2.5 rounded-full bg-emerald-400" />
        <span className="text-xs font-semibold uppercase tracking-wider text-emerald-300">
          Dataset
        </span>
      </div>

      {/* Body */}
      <div className="px-3 py-2">
        <p className="text-sm font-medium text-slate-200">{data.label}</p>
        <p className="mt-1 text-xs text-slate-400">
          {data.source === 'csv' ? 'Local file (CSV / Parquet)' : 'Iceberg'}: {data.path || '(not set)'}
        </p>
        {data.schema && (
          <p className="mt-1 text-xs text-emerald-400">
            {data.schema.columns.length} columns
          </p>
        )}
      </div>

      {/* Output handle */}
      <div className="relative">
        <Handle
          type="source"
          position={Position.Bottom}
          className="!h-3 !w-3 !border-2 !border-emerald-500 !bg-slate-900"
        />
        <span className="absolute -bottom-4 left-1/2 -translate-x-1/2 text-[8px] text-slate-500">
          Output
        </span>
      </div>
    </div>
  );
}

export const DatasetNode = memo(DatasetNodeInner);
