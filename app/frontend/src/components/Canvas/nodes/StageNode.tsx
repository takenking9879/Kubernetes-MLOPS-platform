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

  // Check if this node has validation errors
  const hasError = validationResult?.errors.some((e) => e.nodeId === id) ?? false;
  const inputModes = incomingEdges.map((edge) => edge.selector.type);
  const uniqueInputModes = Array.from(new Set(inputModes));

  const glowClass = hasError
    ? 'neon-border-red'
    : selected
      ? 'neon-selected-cyan'
      : isEstimator ? 'neon-border-purple' : 'neon-border-cyan';

  const headerBg = isEstimator ? 'rgba(167,139,250,0.08)' : 'rgba(34,211,238,0.06)';
  const dotColor = isEstimator ? 'bg-purple-400' : 'bg-cyan-400';
  const labelColor = isEstimator ? 'text-purple-300' : 'text-cyan-300';
  const handleBorder = isEstimator ? '!border-purple-500' : '!border-cyan-500';

  return (
    <div
      onClick={() => selectNode(id)}
      className={`min-w-[170px] rounded-xl transition-all ${glowClass}`}
      style={{ background: '#03060a' }}
    >
      {/* Input handle */}
      <div className="relative">
        <Handle
          type="target"
          position={Position.Top}
          className={`!h-3 !w-3 !border-2 !bg-brand-surface ${handleBorder}`}
        />
        <span className="absolute -top-4 left-1/2 -translate-x-1/2 text-[8px] text-slate-600 uppercase tracking-wide">in</span>
      </div>

      {/* Header */}
      <div className="flex items-center gap-2 rounded-t-xl px-3 py-2" style={{ background: headerBg }}>
        <div className={`h-2 w-2 rounded-full ${dotColor}`} />
        <span className={`text-[10px] font-bold uppercase tracking-widest ${labelColor}`}>
          {isEstimator ? 'Estimator' : 'Transformer'}
        </span>
      </div>

      {/* Body */}
      <div className="px-3 py-2">
        <p className="text-sm font-medium text-slate-100">{data.label}</p>
        <p className="mt-0.5 text-[10px] text-slate-500">{def.description}</p>

        <div className="mt-2 flex gap-3 text-[10px]">
          <span className="text-slate-500">in: {data.resolvedInputs.length}</span>
          <span className="text-slate-500">out: {data.resolvedOutputs.length}</span>
        </div>

        <div className="mt-1 flex items-center justify-between gap-1 text-[10px]">
          <span className={`rounded border px-1.5 py-0.5 ${
            data.compilation.status === 'compiled'
              ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-400'
              : data.compilation.status === 'error'
                ? 'border-red-500/30 bg-red-500/10 text-red-400'
                : 'border-slate-700/40 bg-brand-panel text-slate-500'
          }`}>
            {data.compilation.status}
          </span>
          <div className="flex gap-1">
            {uniqueInputModes.map((mode) => (
              <span key={mode} className="rounded border border-cyan-500/20 bg-brand-panel px-1.5 py-0.5 text-cyan-400/70">
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
          className={`!h-3 !w-3 !border-2 !bg-brand-surface ${handleBorder}`}
        />
        <span className="absolute -bottom-4 left-1/2 -translate-x-1/2 text-[8px] text-slate-600 uppercase tracking-wide">out</span>
      </div>
    </div>
  );
}

export const StageNode = memo(StageNodeInner);
