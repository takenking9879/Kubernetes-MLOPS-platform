import { useCallback } from 'react';
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  type Connection,
  type NodeChange,
  type EdgeChange,
  type NodeTypes,
  type EdgeTypes,
  applyNodeChanges,
  applyEdgeChanges,
} from 'reactflow';
import 'reactflow/dist/style.css';

import { useDagStore } from '../../store/dagStore';
import { DatasetOrchNode }    from './nodes/DatasetOrchNode';
import { ProcessingOrchNode } from './nodes/ProcessingOrchNode';
import { TrainingOrchNode }   from './nodes/TrainingOrchNode';
import { ServingOrchNode }    from './nodes/ServingOrchNode';
import { FiberOpticEdge }     from './edges/FiberOpticEdge';
import type { OrchestrationNode, OrchestrationEdge } from '../../types/dag';

const nodeTypes: NodeTypes = {
  dataset:    DatasetOrchNode,
  processing: ProcessingOrchNode,
  training:   TrainingOrchNode,
  serving:    ServingOrchNode,
};

const edgeTypes: EdgeTypes = {
  fiber: FiberOpticEdge,
};

export function OrchestrationCanvas() {
  const nodes      = useDagStore((s) => s.nodes);
  const edges      = useDagStore((s) => s.edges);
  const setNodes   = useDagStore((s) => s.setNodes);
  const setEdges   = useDagStore((s) => s.setEdges);
  const addEdge    = useDagStore((s) => s.addEdge);
  const removeEdge = useDagStore((s) => s.removeEdge);
  const selectNode = useDagStore((s) => s.selectNode);

  const onNodesChange = useCallback(
    (changes: NodeChange[]) => {
      const updated = applyNodeChanges(changes, nodes) as OrchestrationNode[];
      setNodes(updated);
    },
    [nodes, setNodes],
  );

  const onEdgesChange = useCallback(
    (changes: EdgeChange[]) => {
      for (const c of changes) {
        if (c.type === 'remove') removeEdge(c.id);
      }
      const updated = applyEdgeChanges(changes, edges) as OrchestrationEdge[];
      setEdges(updated);
    },
    [edges, removeEdge, setEdges],
  );

  const onConnect = useCallback(
    (connection: Connection) => {
      if (connection.source && connection.target) {
        addEdge(connection.source, connection.target);
      }
    },
    [addEdge],
  );

  const onPaneClick = useCallback(() => selectNode(null), [selectNode]);

  return (
    <div className="h-full w-full" style={{ background: '#000000' }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onPaneClick={onPaneClick}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        defaultEdgeOptions={{ type: 'fiber' }}
        fitView
        fitViewOptions={{ padding: 0.3 }}
        proOptions={{ hideAttribution: true }}
      >
        <Background color="#0e1f2e" gap={28} size={1} />
        <Controls />
        <MiniMap
          nodeColor={(n) => {
            switch (n.type) {
              case 'dataset':    return '#10b981';
              case 'processing': return '#22d3ee';
              case 'training':   return '#f97316';
              case 'serving':    return '#a78bfa';
              default:           return '#475569';
            }
          }}
        />
      </ReactFlow>
    </div>
  );
}
