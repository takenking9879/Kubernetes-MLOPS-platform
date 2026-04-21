import { useRef } from 'react';
import { ReactFlowProvider } from 'reactflow';
import { MainLayout }              from './components/Layout/MainLayout';
import { OrchestrationCanvasPage } from './pages/OrchestrationCanvasPage';
import { usePipelineStore }        from './store/pipelineStore';
import { useUIStore }              from './store/uiStore';

export default function App() {
  const { page, setPage } = useUIStore();

  const exportYAML  = usePipelineStore((s) => s.exportYAML);
  const importYAML  = usePipelineStore((s) => s.importYAML);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleExport = () => {
    const yaml = exportYAML();
    if (!yaml) return;
    const blob = new Blob([yaml], { type: 'text/yaml' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = 'pipeline.yaml';
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleImport = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target?.result;
      if (typeof text === 'string') importYAML(text);
    };
    reader.readAsText(file);
    event.target.value = '';
  };

  return (
    <div className="flex h-screen flex-col overflow-hidden text-slate-100" style={{ background: '#03060a' }}>
      {/* ── Orchestration canvas (default) ── */}
      {page === 'canvas' && (
        <div className="flex-1 min-h-0 overflow-hidden">
          <OrchestrationCanvasPage />
        </div>
      )}

      {/* ── DSL Builder ── */}
      {page === 'dsl-builder' && (
        <div className="flex-1 min-h-0 overflow-hidden flex flex-col">
          {/* Minimal DSL nav bar */}
          <nav
            className="flex shrink-0 items-center gap-2 px-4 py-1.5 glass-panel"
            style={{ borderTop: 'none', borderLeft: 'none', borderRight: 'none' }}
          >
            <button
              onClick={() => setPage('canvas')}
              className="rounded border border-cyan-500/30 bg-cyan-500/10 px-2 py-1 text-[10px] font-bold text-cyan-300 hover:bg-cyan-500/20 transition-colors"
            >
              ← Pipeline Canvas
            </button>
            <span className="text-[10px] font-bold text-cyan-400/60 uppercase tracking-widest ml-1">
              DSL Builder
            </span>
            <div className="ml-auto flex items-center gap-2">
              <button
                onClick={handleExport}
                className="rounded border border-emerald-500/30 bg-emerald-500/10 px-2 py-1 text-[10px] font-bold text-emerald-300 hover:bg-emerald-500/20 transition-colors"
              >
                Export YAML
              </button>
              <button
                onClick={() => fileInputRef.current?.click()}
                className="rounded border border-orange-500/30 bg-orange-500/10 px-2 py-1 text-[10px] font-bold text-orange-300 hover:bg-orange-500/20 transition-colors"
              >
                Import YAML
              </button>
              <input ref={fileInputRef} type="file" accept=".yaml,.yml" className="hidden" onChange={handleImport} />
            </div>
          </nav>

          <div className="flex-1 min-h-0">
            <ReactFlowProvider>
              <MainLayout />
            </ReactFlowProvider>
          </div>
        </div>
      )}
    </div>
  );
}
