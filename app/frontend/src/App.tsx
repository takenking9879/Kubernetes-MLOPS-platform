import { useRef } from 'react';
import { ReactFlowProvider } from 'reactflow';
import { MainLayout } from './components/Layout/MainLayout';
import { DatasetPage } from './pages/DatasetPage';
import { ProcessingPage } from './pages/ProcessingPage';
import { LaunchWizardPage } from './pages/LaunchWizardPage';
import { useDatasetStore } from './store/datasetStore';
import { usePipelineStore } from './store/pipelineStore';
import { useUIStore, type Page } from './store/uiStore';

const TAB_LABELS: Record<Page, string> = {
  'datasets':    'Datasets',
  'dsl-builder': 'DSL Builder',
  'processing':  'Processing',
  'launch':      'Launch',
};

export default function App() {
  const { page, setPage } = useUIStore();
  const { activeDataset } = useDatasetStore();
  
  // YAML Actions (lifting logic for visibility)
  const exportYAML = usePipelineStore((s) => s.exportYAML);
  const importYAML = usePipelineStore((s) => s.importYAML);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleExport = () => {
    const yaml = exportYAML();
    if (!yaml) return;
    const blob = new Blob([yaml], { type: 'text/yaml' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
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
    <div className="flex h-screen flex-col bg-slate-950 text-slate-100">
      {/* ── Top navigation ── */}
      <nav className="flex shrink-0 items-center border-b border-slate-700 bg-slate-900 px-4 py-1.5 shadow-md">
        <div className="flex items-center gap-1">
          {(Object.entries(TAB_LABELS) as [Page, string][]).map(([id, label]) => (
            <button
              key={id}
              onClick={() => setPage(id)}
              className={`rounded px-3 py-1.5 text-sm font-semibold transition-all ${
                page === id
                  ? 'bg-slate-700 text-white shadow-inner'
                  : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'
              }`}
            >
              {label}
            </button>
          ))}
        </div>

        {/* Global Pipeline Actions (Visible when in DSL Builder) */}
        {page === 'dsl-builder' && (
          <div className="ml-6 flex items-center gap-2 border-l border-slate-700 pl-6">
            <button
              onClick={handleExport}
              className="rounded bg-emerald-600 px-2.5 py-1 text-[11px] font-bold text-white hover:bg-emerald-500 transition-colors"
            >
              Export YAML
            </button>
            <button
              onClick={() => fileInputRef.current?.click()}
              className="rounded bg-amber-600 px-2.5 py-1 text-[11px] font-bold text-white hover:bg-amber-500 transition-colors"
            >
              Import YAML
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".yaml,.yml"
              className="hidden"
              onChange={handleImport}
            />
          </div>
        )}

        {activeDataset && (
          <div className="ml-auto flex items-center gap-2">
            <span className="rounded bg-blue-900/40 px-2 py-0.5 text-[10px] font-bold text-blue-300 ring-1 ring-blue-700/50">
              dataset: {activeDataset}
            </span>
          </div>
        )}
      </nav>

      {/* ── Page content ── */}
      <div
        className={`flex-1 min-h-0 ${page === 'dsl-builder' ? 'overflow-hidden' : 'overflow-y-auto'}`}
      >
        {page === 'datasets' && <DatasetPage />}

        {/* ReactFlowProvider only wraps the DSL Builder */}
        {page === 'dsl-builder' && (
          <div className="h-full min-h-0">
            <ReactFlowProvider>
              <MainLayout />
            </ReactFlowProvider>
          </div>
        )}

        {page === 'processing' && <ProcessingPage />}

        {page === 'launch' && <LaunchWizardPage />}
      </div>
    </div>
  );
}
