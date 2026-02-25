import { useState } from 'react';
import { usePipelineStore } from '../../store/pipelineStore';
import { useDatasetStore } from '../../store/datasetStore';
import { saveDsl } from '../../api/platformClient';
import type { Node } from 'reactflow';
import type { NodeData } from '../../types/nodes';
import { DATASET_NODE_CONTRACT } from '../../registry/NodeRegistry';

export function Header() {
  const validate = usePipelineStore((s) => s.validate);
  const exportYAML = usePipelineStore((s) => s.exportYAML);
  const dryRun = usePipelineStore((s) => s.dryRun);
  const isValidating = usePipelineStore((s) => s.isValidating);
  const isDryRunning = usePipelineStore((s) => s.isDryRunning);
  const validationResult = usePipelineStore((s) => s.validationResult);
  const dryRunResult = usePipelineStore((s) => s.dryRunResult);

  const { activeDataset } = useDatasetStore();
  const [isSavingDSL, setIsSavingDSL] = useState(false);

  const handleSaveDSLToS3 = async () => {
    const log = usePipelineStore.getState().log;
    if (!activeDataset) {
      log('Select a dataset in the Datasets tab before saving the DSL to S3.', 'warning');
      return;
    }
    const yaml = exportYAML();
    if (!yaml) return; // exportYAML already logged the error

    const name = window.prompt('Enter a descriptive name for this DSL version:');
    if (!name?.trim()) return;

    setIsSavingDSL(true);
    try {
      const result = await saveDsl(activeDataset, name.trim(), yaml);
      log(`DSL saved to S3: dsl/dsl_${activeDataset}/v${result.version}__${result.slug}.yaml`, 'info');
    } catch (e) {
      log(`Failed to save DSL to S3: ${e instanceof Error ? e.message : String(e)}`, 'error');
    } finally {
      setIsSavingDSL(false);
    }
  };

  return (
    <header className="flex h-10 shrink-0 items-center justify-between border-b border-slate-800 bg-slate-900 px-4 shadow-sm">
      {/* Title */}
      <div className="flex items-center gap-3">
        <h1 className="text-xs font-bold text-slate-400 uppercase tracking-wider">Feature Designer</h1>
        {validationResult && (
          <span
            className={`rounded-full px-2 py-0.5 text-[10px] font-medium ${
              validationResult.valid
                ? 'bg-emerald-900/40 text-emerald-400'
                : 'bg-red-900/40 text-red-400'
            }`}
          >
            {validationResult.valid
              ? `Valid (${validationResult.topologicalOrder.length})`
              : `${validationResult.errors.length} ERR`}
          </span>
        )}
      </div>

      {/* Dataset badge + S3 save */}
      <div className="flex items-center gap-2">
        {activeDataset && (
          <span className="rounded bg-blue-900/50 px-2 py-0.5 text-[10px] font-bold text-blue-300 ring-1 ring-blue-700/30">
            {activeDataset}
          </span>
        )}
        <button
          onClick={() => { void handleSaveDSLToS3(); }}
          disabled={isSavingDSL}
          className="rounded border border-indigo-500/30 bg-indigo-600/20 px-2 py-0.5 text-[10px] font-bold text-indigo-300 hover:bg-indigo-600/40 disabled:opacity-40"
        >
          {isSavingDSL ? 'Saving…' : 'Save to S3'}
        </button>
      </div>

      {/* Actions */}
      <div className="flex items-center gap-2">
        <button
          onClick={() => {
            const nodes = usePipelineStore.getState().nodes as Node<NodeData>[];
            const addNode = usePipelineStore.getState().addNode;
            const selectNode = usePipelineStore.getState().selectNode;
            const log = usePipelineStore.getState().log;

            const existing = nodes.find((n) => n.data.type === 'dataset');
            if (existing) {
              selectNode(existing.id);
              log('Dataset node exists.', 'warning');
              return;
            }

            const id = DATASET_NODE_CONTRACT.id;
            const datasetNode: Node<NodeData> = {
              id,
              type: 'dataset',
              position: { x: 40, y: 80 },
              data: {
                id,
                label: 'Dataset',
                type: 'dataset',
                source: 'csv',
                path: '',
                schema: null,
              },
            } as Node<NodeData>;

            addNode(datasetNode);
            setTimeout(() => selectNode(id), 50);
          }}
          className="rounded-md bg-slate-800 px-2 py-1 text-[10px] font-bold text-slate-300 hover:bg-slate-700"
        >
          + DATASET
        </button>

        <button
          onClick={() => validate()}
          disabled={isValidating}
          className="rounded-md bg-slate-800 px-2 py-1 text-[10px] font-bold text-slate-300 hover:bg-slate-700 disabled:opacity-40"
        >
          {isValidating ? '...' : 'VALIDATE'}
        </button>

        {/* Dry-run result badge (right of VALIDATE) */}
        {dryRunResult && (
          <span
            className={`ml-2 rounded-full px-2 py-0.5 text-[10px] font-medium ${
              dryRunResult.success
                ? 'bg-emerald-900/40 text-emerald-400'
                : 'bg-red-900/40 text-red-400'
            }`}
          >
            {dryRunResult.success ? 'DRY-RUN OK' : 'DRY-RUN ERR'}
          </span>
        )}

        <button
          onClick={() => { dryRun().catch(() => {}); }}
          disabled={isDryRunning}
          className="rounded-md bg-blue-700 px-2 py-1 text-[10px] font-bold text-white hover:bg-blue-600 disabled:opacity-40"
        >
          {isDryRunning ? '...' : 'DRY-RUN'}
        </button>
      </div>
    </header>
  );
}
