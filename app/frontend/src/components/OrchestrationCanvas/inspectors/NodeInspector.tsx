import { useDagStore } from '../../../store/dagStore';
import type {
  DatasetOrchNodeData,
  ProcessingOrchNodeData,
  TrainingOrchNodeData,
  ServingOrchNodeData,
} from '../../../types/dag';
import { DatasetInspector }    from './DatasetInspector';
import { ProcessingInspector } from './ProcessingInspector';
import { TrainingInspector }   from './TrainingInspector';
import { ServingInspector }    from './ServingInspector';
import { X } from 'lucide-react';

export function NodeInspector() {
  const selectedNodeId = useDagStore((s) => s.selectedNodeId);
  const nodes          = useDagStore((s) => s.nodes);
  const selectNode     = useDagStore((s) => s.selectNode);

  const node = nodes.find((n) => n.id === selectedNodeId);

  const isOpen = !!node;

  return (
    <aside
      className={`
        flex-shrink-0 flex flex-col h-full
        transition-all duration-300 ease-in-out overflow-hidden
        ${isOpen ? 'w-[440px]' : 'w-0'}
      `}
    >
      {isOpen && node && (
        <div
          className="flex flex-col h-full glass-panel"
          style={{ borderTop: 'none', borderRight: 'none', borderBottom: 'none', width: 440 }}
        >
          {/* Inspector header */}
          <div className="flex items-center justify-between px-4 py-3 border-b border-cyan-500/15">
            <span className="text-[10px] font-bold uppercase tracking-widest text-cyan-500/60">
              {node.data.nodeKind} · {node.id}
            </span>
            <button
              onClick={() => selectNode(null)}
              className="text-slate-500 hover:text-cyan-300 transition-colors"
            >
              <X size={14} />
            </button>
          </div>

          {/* Inspector body — scrollable */}
          <div className="flex-1 overflow-y-auto p-4">
            {node.data.nodeKind === 'dataset' && (
              <DatasetInspector nodeId={node.id} data={node.data as DatasetOrchNodeData} />
            )}
            {node.data.nodeKind === 'processing' && (
              <ProcessingInspector nodeId={node.id} data={node.data as ProcessingOrchNodeData} />
            )}
            {node.data.nodeKind === 'training' && (
              <TrainingInspector nodeId={node.id} data={node.data as TrainingOrchNodeData} />
            )}
            {node.data.nodeKind === 'serving' && (
              <ServingInspector nodeId={node.id} data={node.data as ServingOrchNodeData} />
            )}
          </div>
        </div>
      )}
    </aside>
  );
}
