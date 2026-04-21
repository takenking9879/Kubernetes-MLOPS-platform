import { memo } from 'react';
import {
  getBezierPath,
  type EdgeProps,
} from 'reactflow';
import type { OrchestrationEdgeData } from '../../../types/dag';

function FiberOpticEdgeInner({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  data,
}: EdgeProps<OrchestrationEdgeData>) {
  const [path] = getBezierPath({ sourceX, sourceY, sourcePosition, targetX, targetY, targetPosition });
  const hasPropagated = !!data?.propagatedValue;

  return (
    <g>
      {/* Glow halo */}
      <path
        d={path}
        fill="none"
        stroke="rgba(34,211,238,0.15)"
        strokeWidth={6}
        strokeLinecap="round"
      />
      {/* Main fiber line */}
      <path
        id={id}
        d={path}
        fill="none"
        stroke={hasPropagated ? 'rgba(34,211,238,0.8)' : 'rgba(34,211,238,0.35)'}
        strokeWidth={1.5}
        strokeLinecap="round"
        className={hasPropagated ? 'edge-flow' : ''}
        style={{ filter: 'drop-shadow(0 0 3px rgba(34,211,238,0.5))' }}
      />
      {/* Propagated value label */}
      {hasPropagated && data?.propagatedValue && (
        <text
          style={{
            fontSize: 9,
            fill: 'rgba(34,211,238,0.6)',
            fontFamily: 'monospace',
          }}
        >
          <textPath href={`#${id}`} startOffset="50%" textAnchor="middle">
            {data.propagatedValue.length > 18
              ? `…${data.propagatedValue.slice(-14)}`
              : data.propagatedValue}
          </textPath>
        </text>
      )}
    </g>
  );
}

export const FiberOpticEdge = memo(FiberOpticEdgeInner);
