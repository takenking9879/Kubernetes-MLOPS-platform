import { memo } from 'react';
import {
  BaseEdge,
  EdgeLabelRenderer,
  getBezierPath,
  type EdgeProps,
} from 'reactflow';
import { usePipelineStore } from '../../../store/pipelineStore';
import { summarizeSelector } from '../../../types/edges';
import type { FlowEdge } from '../../../types/edges';

function EdgeWithSelectorInner(props: EdgeProps) {
  const { id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition } = props;

  const edge = usePipelineStore((s) => s.edges.find((e) => e.id === id));
  const selectEdge = usePipelineStore((s) => s.selectEdge);
  const selectedEdgeId = usePipelineStore((s) => s.selectedEdgeId);

  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
  });

  const isSelected = selectedEdgeId === id;
  const label = edge ? summarizeSelector(edge as FlowEdge) : '';

  return (
    <>
      <BaseEdge
        id={id}
        path={edgePath}
        style={{
          stroke: isSelected ? '#60a5fa' : '#475569',
          strokeWidth: isSelected ? 2.5 : 1.5,
        }}
      />
      <EdgeLabelRenderer>
        <div
          onClick={(e) => {
            e.stopPropagation();
            selectEdge(id);
          }}
          style={{
            position: 'absolute',
            transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)`,
            pointerEvents: 'all',
          }}
          className={`
            cursor-pointer rounded-md px-2 py-0.5 text-[10px] font-medium
            ${isSelected
              ? 'bg-blue-600 text-white'
              : 'bg-slate-800 text-slate-300 hover:bg-slate-700'}
          `}
        >
          {label}
        </div>
      </EdgeLabelRenderer>
    </>
  );
}

export const EdgeWithSelector = memo(EdgeWithSelectorInner);
