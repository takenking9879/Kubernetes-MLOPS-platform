import { useCallback, useRef, useMemo } from 'react';
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
  type Edge,
  useReactFlow,
} from 'reactflow';
import 'reactflow/dist/style.css';

import { usePipelineStore } from '../../store/pipelineStore';
import { DatasetNode } from './nodes/DatasetNode';
import { StageNode } from './nodes/StageNode';
import { FeatureSelectorNode } from './nodes/FeatureSelectorNode';
import { EdgeWithSelector } from './edges/EdgeWithSelector';
import {
  getNodeDefinition,
  getNodeContract,
  DATASET_NODE_CONTRACT,
  FINAL_FEATURES_NODE_CONTRACT,
  type GraphNodeKind,
} from '../../registry/NodeRegistry';
import type { NodeData, StageNodeData } from '../../types/nodes';
import type { Node } from 'reactflow';

const nodeTypes: NodeTypes = {
  dataset: DatasetNode,
  stage: StageNode,
  final_features: FeatureSelectorNode,
};

const edgeTypes: EdgeTypes = {
  selector: EdgeWithSelector,
};

let nodeIdCounter = 0;

export function CanvasPanel() {
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const { screenToFlowPosition } = useReactFlow();

  const nodes = usePipelineStore((s) => s.nodes);
  const edges = usePipelineStore((s) => s.edges);
  const setNodes = usePipelineStore((s) => s.setNodes);
  const setEdges = usePipelineStore((s) => s.setEdges);
  const addNode = usePipelineStore((s) => s.addNode);
  const addEdge = usePipelineStore((s) => s.addEdge);
  const validate = usePipelineStore((s) => s.validate);
  const selectNode = usePipelineStore((s) => s.selectNode);
  const log = usePipelineStore((s) => s.log);

  const getNodeKind = useCallback((nodeId: string): GraphNodeKind | null => {
    const node = nodes.find((n) => n.id === nodeId);
    if (!node) return null;
    if (node.data.type === 'dataset') return 'dataset';
    if (node.data.type === 'final_features') return 'terminal';
    if ('stageType' in node.data) return node.data.stageType;
    return null;
  }, [nodes]);

  // Map FlowEdge[] to React Flow Edge[] for rendering
  const rfEdges: Edge[] = useMemo(
    () =>
      edges.map((e) => ({
        id: e.id,
        source: e.source,
        target: e.target,
        type: 'selector',
        sourceHandle: e.sourceHandle,
        targetHandle: e.targetHandle,
      })),
    [edges],
  );

  const onNodesChange = useCallback(
    (changes: NodeChange[]) => {
      const updated = applyNodeChanges(changes, nodes);
      setNodes(updated as Node<NodeData>[]);
    },
    [nodes, setNodes],
  );

  const onEdgesChange = useCallback(
    (changes: EdgeChange[]) => {
      // We only handle removes via EdgeChange; selectors are managed separately
      const removedIds = new Set(
        changes.filter((c) => c.type === 'remove').map((c) => c.id),
      );
      if (removedIds.size > 0) {
        setEdges(edges.filter((e) => !removedIds.has(e.id)));
      }
    },
    [edges, setEdges],
  );

  const onConnect = useCallback(
    (connection: Connection) => {
      if (!connection.source || !connection.target) return;

      const sourceNode = nodes.find((n) => n.id === connection.source);
      const targetNode = nodes.find((n) => n.id === connection.target);
      if (!sourceNode || !targetNode) return;

      if (targetNode.data.type === 'dataset') {
        log('Dataset node cannot have incoming edges', 'error');
        return;
      }

      if (sourceNode.data.type === 'final_features') {
        log('Final Features node is terminal and cannot have outgoing edges', 'error');
        return;
      }

      const sourceKind = getNodeKind(sourceNode.id);
      const targetKind = getNodeKind(targetNode.id);
      if (!sourceKind || !targetKind) return;

      const sourceContract = getNodeContract(sourceNode.data.type);
      const targetContract = getNodeContract(targetNode.data.type);

      if (!sourceContract.allowedOutputs.includes(targetKind)) {
        log(`Connection blocked: ${sourceContract.label} cannot output to ${targetContract.label}`, 'error');
        return;
      }

      if (!targetContract.allowedInputs.includes(sourceKind)) {
        log(`Connection blocked: ${targetContract.label} cannot accept ${sourceContract.label} as input`, 'error');
        return;
      }

      addEdge(connection.source, connection.target);
      validate({ silent: true });
    },
    [addEdge, getNodeKind, log, nodes, validate],
  );

  const onNodeClick = useCallback(
    (_: React.MouseEvent, node: Node<NodeData>) => {
      selectNode(node.id);
    },
    [selectNode],
  );

  const onPaneClick = useCallback(() => {
    selectNode(null);
  }, [selectNode]);

  // Drag-and-drop from catalog
  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();

      const stageType = event.dataTransfer.getData('application/spark-stage-type');
      if (!stageType) return;

      if (!reactFlowWrapper.current) return;

      const position = screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });

      const contract = getNodeContract(stageType);
      const existingInstances = nodes.filter((n) => n.data.type === stageType).length;
      if (existingInstances >= contract.maxInstances) {
        log(`Cannot add '${contract.label}': maxInstances=${contract.maxInstances}`, 'warning');
        return;
      }

      if (stageType === 'final_features') {
        const finalId = FINAL_FEATURES_NODE_CONTRACT.id;
        const newNode: Node<NodeData> = {
          id: finalId,
          type: 'final_features',
          position,
          data: {
            id: finalId,
            label: 'Final Features',
            type: 'final_features',
            features: [],
            target: [],
            metadata: [],
            passthrough: [],
            manualOverrides: {},
            confirmedColumns: [],
          },
        };
        addNode(newNode);
        validate({ silent: true });
        return;
      }

      const def = getNodeDefinition(stageType);

      const id = `${stageType}-${++nodeIdCounter}`;

      const data: StageNodeData = {
        id,
        label: def.label,
        type: stageType as StageNodeData['type'],
        stageType: def.category,
        inputs: [],
        outputs: [],
        params: { ...def.defaultParams },
        namingStrategy: def.defaultNamingStrategy,
        resolvedInputs: [],
        resolvedOutputs: [],
        outputSchema: null,
        compilation: { status: 'draft' },
      } as StageNodeData;

      const newNode: Node<NodeData> = {
        id,
        type: 'stage',
        position,
        data,
      };

      addNode(newNode);
      validate({ silent: true });
    },
    [addNode, log, nodes, screenToFlowPosition, validate],
  );

  const datasetNodeCount = nodes.filter((n) => n.data.type === 'dataset').length;
  const finalNodeCount = nodes.filter((n) => n.data.type === 'final_features').length;
  const finalNode = nodes.find((n) => n.data.type === 'final_features');
  const finalIncoming = finalNode ? edges.filter((e) => e.target === finalNode.id).length : 0;
  const finalOutgoing = finalNode ? edges.filter((e) => e.source === finalNode.id).length : 0;

  const datasetOk = datasetNodeCount === DATASET_NODE_CONTRACT.maxInstances;
  const finalOk =
    finalNodeCount === FINAL_FEATURES_NODE_CONTRACT.maxInstances
    && finalIncoming > 0
    && finalOutgoing === 0;

  return (
    <div ref={reactFlowWrapper} className="relative h-full w-full">
      <div className="pointer-events-none absolute z-10 ml-2 mt-2 flex gap-2">
        <span className={`rounded px-2 py-1 text-[10px] ${datasetOk ? 'bg-emerald-900/70 text-emerald-300' : 'bg-slate-800 text-slate-300'}`}>
          Dataset {datasetOk ? '✓' : 'required'}
        </span>
        <span className={`rounded px-2 py-1 text-[10px] ${finalOk ? 'bg-red-900/70 text-red-300' : 'bg-slate-800 text-slate-300'}`}>
          Final Features {finalOk ? '✓' : 'required'}
        </span>
      </div>
      <ReactFlow
        nodes={nodes}
        edges={rfEdges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeClick={onNodeClick}
        onPaneClick={onPaneClick}
        onDragOver={onDragOver}
        onDrop={onDrop}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        defaultEdgeOptions={{ type: 'selector' }}
        fitView
        proOptions={{ hideAttribution: true }}
        className="bg-canvas-bg"
      >
        <Background color="#1e293b" gap={20} size={1} />
        <Controls className="!bg-slate-800 !border-slate-700 [&>button]:!bg-slate-800 [&>button]:!border-slate-700 [&>button]:!text-slate-300" />
        <MiniMap
          nodeColor={(node) => {
            const d = node.data as NodeData;
            if (d.type === 'dataset') return '#10b981';
            if ('stageType' in d && d.stageType === 'estimator') return '#8b5cf6';
            if ('stageType' in d) return '#3b82f6';
            return '#ef4444';
          }}
          className="!bg-slate-900 !border-slate-700"
        />
      </ReactFlow>
    </div>
  );
}
