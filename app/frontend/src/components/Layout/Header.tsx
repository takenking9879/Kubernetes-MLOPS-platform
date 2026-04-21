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
    <header className="flex h-10 shrink-0 items-center justify-between px-4 glass-panel" style={{ borderTop: 'none', borderLeft: 'none', borderRight: 'none' }}>
      {/* Title */}
      <div className="flex items-center gap-3">
        <h1 className="text-xs font-bold text-cyan-400/80 uppercase tracking-widest">Feature Designer</h1>
        {validationResult && (
          <div className="flex items-center gap-2">
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

            {/* Dry-run result badge next to Valid */}
            {dryRunResult && (
              <span
                className={`rounded-full px-2 py-0.5 text-[10px] font-medium ${
                  dryRunResult.success
                    ? 'bg-emerald-900/40 text-emerald-400'
                    : 'bg-red-900/40 text-red-400'
                }`}
              >
                {dryRunResult.success ? 'DRY-RUN OK' : 'DRY-RUN ERR'}
              </span>
            )}
          </div>
        )}
      </div>

      {/* Dataset badge + S3 save */}
      <div className="flex items-center gap-2">
        {activeDataset && (
          <span className="rounded border border-cyan-500/30 bg-cyan-500/10 px-2 py-0.5 text-[10px] font-bold text-cyan-300">
            {activeDataset}
          </span>
        )}
        <button
          onClick={() => { void handleSaveDSLToS3(); }}
          disabled={isSavingDSL}
          className="rounded border border-cyan-500/30 bg-cyan-500/10 px-2 py-0.5 text-[10px] font-bold text-cyan-300 hover:bg-cyan-500/20 transition-colors disabled:opacity-40"
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
          className="rounded border border-emerald-500/30 bg-emerald-500/10 px-2 py-1 text-[10px] font-bold text-emerald-300 hover:bg-emerald-500/20 transition-colors"
        >
          + DATASET
        </button>

        <button
          onClick={() => validate()}
          disabled={isValidating}
          className="rounded border border-cyan-500/30 bg-cyan-500/10 px-2 py-1 text-[10px] font-bold text-cyan-300 hover:bg-cyan-500/20 transition-colors disabled:opacity-40"
        >
          {isValidating ? '...' : 'VALIDATE'}
        </button>

        <button
          onClick={() => { dryRun().catch(() => {}); }}
          disabled={isDryRunning}
          className="rounded border border-orange-500/30 bg-orange-500/10 px-2 py-1 text-[10px] font-bold text-orange-300 hover:bg-orange-500/20 transition-colors disabled:opacity-40"
        >
          {isDryRunning ? '...' : 'DRY-RUN'}
        </button>
      </div>
    </header>
  );
}
