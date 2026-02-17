import { usePipelineStore } from '../../store/pipelineStore';
import { NodePropertiesForm } from './NodePropertiesForm';
import { EdgeSelectorConfig } from './EdgeSelectorConfig';
import { PipelineSettingsPanel } from './PipelineSettingsPanel';

export function PropertiesPanel() {
  const selectedNodeId = usePipelineStore((s) => s.selectedNodeId);
  const selectedEdgeId = usePipelineStore((s) => s.selectedEdgeId);
  const nodes = usePipelineStore((s) => s.nodes);
  const edges = usePipelineStore((s) => s.edges);

  const selectedNode = selectedNodeId
    ? nodes.find((n) => n.id === selectedNodeId)
    : null;

  const selectedEdge = selectedEdgeId
    ? edges.find((e) => e.id === selectedEdgeId)
    : null;

  return (
    <div className="flex h-full w-72 flex-col border-l border-slate-800 bg-slate-900">
      <div className="border-b border-slate-800 px-4 py-3">
        <h2 className="text-sm font-semibold text-slate-200">Properties</h2>
      </div>

      <div className="flex-1 overflow-y-auto p-4">
        {selectedNode && <NodePropertiesForm node={selectedNode} />}
        {selectedEdge && <EdgeSelectorConfig edge={selectedEdge} />}
        {!selectedNode && !selectedEdge && <PipelineSettingsPanel />}
      </div>
    </div>
  );
}
