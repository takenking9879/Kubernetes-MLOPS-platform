import { memo } from 'react';
import { Handle, Position, type NodeProps } from 'reactflow';
import type { FinalFeatureSelectorData } from '../../../types/nodes';
import { usePipelineStore } from '../../../store/pipelineStore';

function FeatureSelectorNodeInner({ data, id, selected }: NodeProps<FinalFeatureSelectorData>) {
  const selectNode = usePipelineStore((s) => s.selectNode);

  const total =
    data.features.length +
    data.target.length +
    data.metadata.length;

  return (
    <div
      onClick={() => selectNode(id)}
      className={`
        min-w-[180px] rounded-lg border-2 bg-slate-900 shadow-lg
        ${selected ? 'border-red-400 ring-2 ring-red-400/30' : 'border-red-600/40'}
      `}
    >
      {/* Input handle */}
      <div className="relative">
        <Handle
          type="target"
          position={Position.Top}
          className="!h-3 !w-3 !border-2 !border-red-500 !bg-slate-900"
        />
        <span className="absolute -top-4 left-1/2 -translate-x-1/2 text-[8px] text-slate-500">
          Input
        </span>
      </div>

      {/* Header */}
      <div className="flex items-center gap-2 rounded-t-md bg-red-600/15 px-3 py-2">
        <div className="h-2.5 w-2.5 rounded-full bg-red-400" />
        <span className="text-xs font-semibold uppercase tracking-wider text-red-300">
          Final Features
        </span>
      </div>

      {/* Body */}
      <div className="px-3 py-2 text-xs text-slate-400">
        <p className="font-medium text-slate-200">{data.label}</p>
        <div className="mt-1 space-y-0.5">
          <p>Features: {data.features.length}</p>
          <p>Target: {data.target.length}</p>
          <p>Metadata: {data.metadata.length}</p>
        </div>
        <p className="mt-1 font-medium text-red-300">Total: {total}</p>
      </div>
    </div>
  );
}

export const FeatureSelectorNode = memo(FeatureSelectorNodeInner);
