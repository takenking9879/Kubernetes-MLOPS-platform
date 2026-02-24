/**
 * RunPage — configure and submit a full training pipeline run.
 *
 * Flow:
 *   1. Select dataset (auto-filled from activeDataset)
 *   2. Select DSL version
 *   3. Configure model (framework, splits, hyperparams)
 *   4. (Optional) check if execution_id artifact already exists
 *   5. Submit → params.yaml generated + uploaded + Airflow DAG triggered
 *   6. Poll run status
 */
import { useEffect, useRef, useState } from 'react';
import {
  checkArtifact,
  getRunStatus,
  listDsls,
  submitRun,
  type DslVersion,
  type RunRequest,
  type RunResult,
} from '../api/platformClient';
import { useDatasetStore } from '../store/datasetStore';

const DEFAULT_SPLITS = {
  train: { start: '2026-01-01 00:00:00', end: '2026-01-01 12:00:00' },
  val:   { start: '2026-01-01 12:00:00', end: '2026-01-02 00:00:00' },
  test:  { start: '2026-01-02 00:00:00', end: '2026-01-03 00:00:00' },
};

function nowId(): string {
  const d = new Date();
  return [
    d.getUTCFullYear(),
    String(d.getUTCMonth() + 1).padStart(2, '0'),
    String(d.getUTCDate()).padStart(2, '0'),
    '_',
    String(d.getUTCHours()).padStart(2, '0'),
    String(d.getUTCMinutes()).padStart(2, '0'),
    String(d.getUTCSeconds()).padStart(2, '0'),
  ].join('');
}

export function RunPage() {
  const { activeDataset } = useDatasetStore();

  const [dataset, setDataset] = useState(activeDataset ?? '');
  const [dsls, setDsls] = useState<DslVersion[]>([]);
  const [dslVersion, setDslVersion] = useState<number | ''>('');
  const [executionId, setExecutionId] = useState(nowId);
  const [framework, setFramework] = useState<'xgboost' | 'pytorch'>('xgboost');
  const [splits, setSplits] = useState(DEFAULT_SPLITS);
  const [hyperparams, setHyperparams] = useState('');  // raw JSON or empty

  const [artifactCheck, setArtifactCheck] = useState<{
    loading: boolean;
    exists?: boolean;
    processedTable?: string | null;
    error?: string;
  }>({ loading: false });

  const [submitResult, setSubmitResult] = useState<RunResult | null>(null);
  const [submitError, setSubmitError] = useState('');
  const [runStatus, setRunStatus] = useState('');
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Sync dataset field when store changes
  useEffect(() => {
    if (activeDataset) setDataset(activeDataset);
  }, [activeDataset]);

  // Load DSL list whenever dataset changes
  useEffect(() => {
    if (!dataset) { setDsls([]); return; }
    listDsls(dataset)
      .then((r) => setDsls(r.dsls))
      .catch(() => setDsls([]));
  }, [dataset]);

  // Poll run status after submission
  useEffect(() => {
    if (!submitResult?.dag_run_id) return;
    const TERMINAL = ['success', 'failed', 'upstream_failed'];
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const s = await getRunStatus(submitResult.dag_run_id);
        setRunStatus(s.state);
        if (TERMINAL.includes(s.state.toLowerCase())) {
          if (pollRef.current) clearInterval(pollRef.current);
        }
      } catch {
        // keep polling
      }
    }, 15_000);
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [submitResult]);

  const handleCheckArtifact = async () => {
    if (!dataset || !executionId) return;
    setArtifactCheck({ loading: true });
    try {
      const r = await checkArtifact(executionId, dataset);
      setArtifactCheck({ loading: false, exists: r.exists, processedTable: r.processed_table });
    } catch (e) {
      setArtifactCheck({ loading: false, error: e instanceof Error ? e.message : String(e) });
    }
  };

  const handleSubmit = async () => {
    setSubmitError('');
    setSubmitResult(null);
    setRunStatus('');
    if (!dataset || dslVersion === '') {
      setSubmitError('Select a dataset and DSL version first.');
      return;
    }

    let parsedHyperparams: Record<string, unknown> = {};
    if (hyperparams.trim()) {
      try {
        parsedHyperparams = JSON.parse(hyperparams) as Record<string, unknown>;
      } catch {
        setSubmitError('Hyperparams must be valid JSON (e.g. {"eta": 0.1}).');
        return;
      }
    }

    const req: RunRequest = {
      dataset,
      dsl_version: dslVersion as number,
      execution_id: executionId.trim(),
      framework,
      splits,
      hyperparams: parsedHyperparams,
    };

    try {
      const result = await submitRun(req);
      setSubmitResult(result);
      setRunStatus('queued');
    } catch (e) {
      setSubmitError(e instanceof Error ? e.message : String(e));
    }
  };

  const splitField = (
    label: string,
    split: keyof typeof splits,
    field: 'start' | 'end',
  ) => (
    <div className="flex flex-col gap-1">
      <label className="text-xs text-slate-400">{label}</label>
      <input
        type="text"
        value={splits[split][field]}
        onChange={(e) =>
          setSplits((s) => ({
            ...s,
            [split]: { ...s[split], [field]: e.target.value },
          }))
        }
        placeholder="YYYY-MM-DD HH:MM:SS"
        className="rounded bg-slate-800 px-2 py-1 text-xs text-slate-100 outline-none focus:ring-1 focus:ring-blue-500"
      />
    </div>
  );

  const stateColor = (state: string) =>
    state === 'success' ? 'text-green-400'
    : state === 'running' ? 'text-yellow-400'
    : state.includes('fail') ? 'text-red-400'
    : 'text-slate-400';

  return (
    <div className="h-full overflow-y-auto p-6">
      <div className="mx-auto max-w-2xl space-y-6">
        <h2 className="text-lg font-semibold text-slate-100">Run Pipeline</h2>

        {/* Dataset + DSL */}
        <section className="rounded-lg border border-slate-700 bg-slate-900 p-4 space-y-3">
          <h3 className="text-sm font-medium text-slate-300">Dataset &amp; DSL</h3>
          <div className="grid grid-cols-2 gap-3">
            <div className="flex flex-col gap-1">
              <label className="text-xs text-slate-400">Dataset</label>
              <input
                value={dataset}
                onChange={(e) => setDataset(e.target.value)}
                placeholder="network_traffic"
                className="rounded bg-slate-800 px-2 py-1 text-xs text-slate-100 outline-none focus:ring-1 focus:ring-blue-500"
              />
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-xs text-slate-400">DSL Version</label>
              <select
                value={dslVersion}
                onChange={(e) => setDslVersion(e.target.value === '' ? '' : Number(e.target.value))}
                className="rounded bg-slate-800 px-2 py-1 text-xs text-slate-100 outline-none focus:ring-1 focus:ring-blue-500"
              >
                <option value="">— select —</option>
                {dsls.map((d) => (
                  <option key={d.version} value={d.version}>
                    v{d.version} — {d.slug}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </section>

        {/* Execution identity */}
        <section className="rounded-lg border border-slate-700 bg-slate-900 p-4 space-y-3">
          <h3 className="text-sm font-medium text-slate-300">Execution</h3>
          <div className="grid grid-cols-2 gap-3">
            <div className="flex flex-col gap-1">
              <label className="text-xs text-slate-400">Execution ID</label>
              <div className="flex gap-1">
                <input
                  value={executionId}
                  onChange={(e) => setExecutionId(e.target.value)}
                  className="flex-1 rounded bg-slate-800 px-2 py-1 text-xs text-slate-100 outline-none focus:ring-1 focus:ring-blue-500"
                />
                <button
                  onClick={() => setExecutionId(nowId())}
                  className="rounded bg-slate-700 px-2 py-1 text-xs text-slate-300 hover:bg-slate-600"
                  title="Regenerate"
                >
                  ↺
                </button>
              </div>
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-xs text-slate-400">Framework</label>
              <select
                value={framework}
                onChange={(e) => setFramework(e.target.value as 'xgboost' | 'pytorch')}
                className="rounded bg-slate-800 px-2 py-1 text-xs text-slate-100 outline-none focus:ring-1 focus:ring-blue-500"
              >
                <option value="xgboost">xgboost</option>
                <option value="pytorch">pytorch</option>
              </select>
            </div>
          </div>

          {/* Artifact check */}
          <div className="flex items-center gap-3">
            <button
              onClick={() => void handleCheckArtifact()}
              disabled={artifactCheck.loading}
              className="rounded bg-slate-700 px-3 py-1 text-xs text-slate-300 hover:bg-slate-600 disabled:opacity-40"
            >
              {artifactCheck.loading ? 'Checking…' : 'Check artifact'}
            </button>
            {artifactCheck.exists !== undefined && (
              <span className={`text-xs font-medium ${artifactCheck.exists ? 'text-yellow-400' : 'text-green-400'}`}>
                {artifactCheck.exists
                  ? `⚠ Artifact exists — transformation will be skipped (${artifactCheck.processedTable ?? ''})`
                  : '✓ New execution — full pipeline will run'}
              </span>
            )}
            {artifactCheck.error && (
              <span className="text-xs text-red-400">{artifactCheck.error}</span>
            )}
          </div>
        </section>

        {/* Date splits */}
        <section className="rounded-lg border border-slate-700 bg-slate-900 p-4 space-y-3">
          <h3 className="text-sm font-medium text-slate-300">Date Splits</h3>
          <div className="grid grid-cols-2 gap-3">
            {splitField('Train start', 'train', 'start')}
            {splitField('Train end', 'train', 'end')}
            {splitField('Val start', 'val', 'start')}
            {splitField('Val end', 'val', 'end')}
            {splitField('Test start', 'test', 'start')}
            {splitField('Test end', 'test', 'end')}
          </div>
        </section>

        {/* Hyperparameters */}
        <section className="rounded-lg border border-slate-700 bg-slate-900 p-4 space-y-2">
          <h3 className="text-sm font-medium text-slate-300">
            Hyperparameter Overrides{' '}
            <span className="text-xs font-normal text-slate-500">(optional, JSON)</span>
          </h3>
          <textarea
            value={hyperparams}
            onChange={(e) => setHyperparams(e.target.value)}
            placeholder={`{"eta": 0.1, "max_depth": 8}`}
            rows={3}
            className="w-full rounded bg-slate-800 px-2 py-1 font-mono text-xs text-slate-100 outline-none focus:ring-1 focus:ring-blue-500"
          />
          <p className="text-xs text-slate-500">
            Keys must match the allowed set defined in{' '}
            <code>src/schemas/model/{framework}_params.py</code>.
            Leave empty to use all defaults.
          </p>
        </section>

        {/* Submit */}
        <div className="flex items-center gap-4">
          <button
            onClick={() => void handleSubmit()}
            disabled={!!submitResult}
            className="rounded bg-blue-600 px-5 py-2 text-sm font-medium text-white hover:bg-blue-500 disabled:opacity-40"
          >
            Run Pipeline
          </button>
          {submitError && (
            <p className="text-sm text-red-400">{submitError}</p>
          )}
        </div>

        {/* Run status */}
        {submitResult && (
          <section className="rounded-lg border border-slate-700 bg-slate-900 p-4 space-y-2">
            <h3 className="text-sm font-medium text-slate-300">Run Status</h3>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <span className="text-slate-400">DAG run ID</span>
              <span className="break-all text-slate-200">{submitResult.dag_run_id}</span>
              <span className="text-slate-400">Execution ID</span>
              <span className="text-slate-200">{submitResult.execution_id}</span>
              <span className="text-slate-400">params.yaml</span>
              <span className="break-all text-slate-200">{submitResult.params_s3_path}</span>
              <span className="text-slate-400">DSL</span>
              <span className="break-all text-slate-200">{submitResult.dsl_s3_path}</span>
              <span className="text-slate-400">State</span>
              <span className={`font-medium ${stateColor(runStatus.toLowerCase())}`}>
                {runStatus || 'queued'}
              </span>
            </div>
          </section>
        )}
      </div>
    </div>
  );
}
