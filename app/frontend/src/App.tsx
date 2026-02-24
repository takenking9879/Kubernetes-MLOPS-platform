import { useState } from 'react';
import { ReactFlowProvider } from 'reactflow';
import { MainLayout } from './components/Layout/MainLayout';
import { DatasetPage } from './pages/DatasetPage';
import { RunPage } from './pages/RunPage';
import { useDatasetStore } from './store/datasetStore';

type Page = 'datasets' | 'dsl-builder' | 'run-pipeline';

const TAB_LABELS: Record<Page, string> = {
  'datasets':    'Datasets',
  'dsl-builder': 'DSL Builder',
  'run-pipeline':'Run Pipeline',
};

export default function App() {
  const [page, setPage] = useState<Page>('datasets');
  const { activeDataset } = useDatasetStore();

  return (
    <div className="flex h-screen flex-col bg-slate-950 text-slate-100">
      {/* ── Top navigation ── */}
      <nav className="flex shrink-0 items-center gap-1 border-b border-slate-700 bg-slate-900 px-4 py-1">
        {(Object.entries(TAB_LABELS) as [Page, string][]).map(([id, label]) => (
          <button
            key={id}
            onClick={() => setPage(id)}
            className={`rounded px-3 py-1.5 text-sm font-medium transition-colors ${
              page === id
                ? 'bg-slate-700 text-slate-100'
                : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'
            }`}
          >
            {label}
          </button>
        ))}
        {activeDataset && (
          <span className="ml-auto rounded bg-blue-900 px-2 py-0.5 text-xs text-blue-300">
            dataset: {activeDataset}
          </span>
        )}
      </nav>

      {/* ── Page content ── */}
      <div className="flex-1 overflow-hidden">
        {page === 'datasets' && <DatasetPage />}

        {/* ReactFlowProvider only wraps the DSL Builder */}
        {page === 'dsl-builder' && (
          <ReactFlowProvider>
            <MainLayout />
          </ReactFlowProvider>
        )}

        {page === 'run-pipeline' && <RunPage />}
      </div>
    </div>
  );
}
