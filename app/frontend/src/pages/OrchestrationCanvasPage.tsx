import { useEffect } from 'react';
import { ReactFlowProvider } from 'reactflow';
import { BrandHeader }           from '../components/ZenthrosShell/BrandHeader';
import { NodePalette }           from '../components/OrchestrationCanvas/NodePalette';
import { OrchestrationCanvas }   from '../components/OrchestrationCanvas/OrchestrationCanvas';
import { NodeInspector }         from '../components/OrchestrationCanvas/inspectors/NodeInspector';
import { StatusBar }             from '../components/OrchestrationCanvas/StatusBar';
import { useDagStore }           from '../store/dagStore';

function OrchestrationCanvasPageInner() {
  const initDefaultPipeline = useDagStore((s) => s.initDefaultPipeline);

  useEffect(() => {
    initDefaultPipeline();
  }, [initDefaultPipeline]);

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden" style={{ background: '#03060a' }}>
      <BrandHeader />

      <div className="flex flex-1 min-h-0 overflow-hidden">
        {/* Left: Node palette */}
        <NodePalette />

        {/* Center: DAG canvas */}
        <div className="flex-1 min-w-0">
          <OrchestrationCanvas />
        </div>

        {/* Right: Node inspector (slides in when node selected) */}
        <NodeInspector />
      </div>

      <StatusBar />
    </div>
  );
}

export function OrchestrationCanvasPage() {
  return (
    <ReactFlowProvider>
      <OrchestrationCanvasPageInner />
    </ReactFlowProvider>
  );
}
