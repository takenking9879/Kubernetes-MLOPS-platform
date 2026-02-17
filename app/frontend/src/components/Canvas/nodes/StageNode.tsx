import { memo } from 'react';
import { Handle, Position, type NodeProps } from 'reactflow';
import type { StageNodeData } from '../../../types/nodes';
import { usePipelineStore } from '../../../store/pipelineStore';
import { getNodeDefinition } from '../../../registry/NodeRegistry';

function StageNodeInner({ data, id, selected }: NodeProps<StageNodeData>) {
  const selectNode = usePipelineStore((s) => s.selectNode);
  const validationResult = usePipelineStore((s) => s.validationResult);
  const incomingEdges = usePipelineStore((s) => s.edges.filter((edge) => edge.target === id));

  const def = getNodeDefinition(data.type);
  const isEstimator = data.stageType === 'estimator';
  const accentColor = isEstimator ? 'purple' : 'blue';

  // Check if this node has validation errors
  const hasError = validationResult?.errors.some((e) => e.nodeId === id) ?? false;
  const inputModes = incomingEdges.map((edge) => edge.selector.type);
  const uniqueInputModes = Array.from(new Set(inputModes));

  const borderClass = hasError
    ? 'border-red-500 ring-2 ring-red-500/30'
    : selected
      ? `border-${accentColor}-400 ring-2 ring-${accentColor}-400/30`
      : `border-${accentColor}-600/50`;

  return (
    <div
      onClick={() => selectNode(id)}
      className={`min-w-[170px] rounded-lg border-2 bg-slate-900 shadow-lg ${borderClass}`}
    >
      {/* Input handle */}
      <div className="relative">
        <Handle
          type="target"
          position={Position.Top}
          className={`!h-3 !w-3 !border-2 !bg-slate-900 ${
            isEstimator ? '!border-purple-500' : '!border-blue-500'
          }`}
        />
        <span className="absolute -top-4 left-1/2 -translate-x-1/2 text-[8px] text-slate-500">
          Input
        </span>
      </div>

      {/* Header */}
      <div
        className={`flex items-center gap-2 rounded-t-md px-3 py-2 ${
          isEstimator ? 'bg-purple-600/20' : 'bg-blue-600/20'
        }`}
      >
        <div
          className={`h-2.5 w-2.5 rounded-full ${
            isEstimator ? 'bg-purple-400' : 'bg-blue-400'
          }`}
        />
        <span
          className={`text-xs font-semibold uppercase tracking-wider ${
            isEstimator ? 'text-purple-300' : 'text-blue-300'
          }`}
        >
          {isEstimator ? 'Estimator' : 'Transformer'}
        </span>
      </div>

      {/* Body */}
      <div className="px-3 py-2">
        <p className="text-sm font-medium text-slate-200">{data.label}</p>
        <p className="mt-0.5 text-[10px] text-slate-500">{def.description}</p>

        {/* Inputs/Outputs summary */}
        <div className="mt-2 flex gap-3 text-[10px]">
          <span className="text-slate-400">
            in: {data.resolvedInputs.length}
          </span>
          <span className="text-slate-400">
            out: {data.resolvedOutputs.length}
          </span>
        </div>

        {/* Compilation state + Input Mode on same row */}
        <div className="mt-1 flex items-center justify-between gap-1 text-[10px]">
          <span className={`rounded px-1.5 py-0.5 ${
            data.compilation.status === 'compiled'
              ? 'bg-emerald-900/40 text-emerald-400'
              : data.compilation.status === 'error'
                ? 'bg-red-900/40 text-red-400'
                : 'bg-slate-800 text-slate-500'
          }`}>
            {data.compilation.status}
          </span>
          <div className="flex gap-1">
            {uniqueInputModes.map((mode) => (
              <span key={mode} className="rounded bg-slate-800 px-1.5 py-0.5 text-slate-300">
                {mode}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* Output handle */}
      <div className="relative">
        <Handle
          type="source"
          position={Position.Bottom}
          className={`!h-3 !w-3 !border-2 !bg-slate-900 ${
            isEstimator ? '!border-purple-500' : '!border-blue-500'
          }`}
        />
        <span className="absolute -bottom-4 left-1/2 -translate-x-1/2 text-[8px] text-slate-500">
          Output
        </span>
      </div>
    </div>
  );
}

export const StageNode = memo(StageNodeInner);
