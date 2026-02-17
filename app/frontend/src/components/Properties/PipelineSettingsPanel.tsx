import { usePipelineStore } from '../../store/pipelineStore';
import { computeSchemaHashSync } from '../../lib/schemaHash';

const DSL_VERSION = 2;

export function PipelineSettingsPanel() {
  const pipelineName = usePipelineStore((s) => s.pipelineName);
  const pipelineVersion = usePipelineStore((s) => s.pipelineVersion);
  const setPipelineName = usePipelineStore((s) => s.setPipelineName);
  const setPipelineVersion = usePipelineStore((s) => s.setPipelineVersion);
  const datasetSchema = usePipelineStore((s) => s.datasetSchema);

  const schemaHash = datasetSchema ? computeSchemaHashSync(datasetSchema) : null;

  return (
    <div className="space-y-3">
      <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
        Pipeline Settings
      </h3>

      <div>
        <label className="mb-1 block text-[10px] font-medium text-slate-400">Name</label>
        <input
          type="text"
          value={pipelineName}
          onChange={(e) => setPipelineName(e.target.value)}
          placeholder="my_pipeline"
          className="input-field"
        />
      </div>

      <div>
        <label className="mb-1 block text-[10px] font-medium text-slate-400">Version</label>
        <input
          type="text"
          value={pipelineVersion}
          onChange={(e) => setPipelineVersion(e.target.value)}
          placeholder="1.0"
          className="input-field"
        />
      </div>

      <div className="space-y-1 rounded border border-slate-800 bg-slate-900/40 p-2">
        <div className="flex items-center justify-between">
          <span className="text-[10px] text-slate-500">DSL Version</span>
          <span className="text-[10px] text-slate-300">{DSL_VERSION}</span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-[10px] text-slate-500">Schema Hash</span>
          <span className="font-mono text-[10px] text-slate-300">
            {schemaHash ?? 'No schema loaded'}
          </span>
        </div>
      </div>
    </div>
  );
}
