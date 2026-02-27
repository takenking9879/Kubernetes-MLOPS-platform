/**
 * ProcessingPage — Submit and monitor Spark preprocessing runs.
 *
 * 3-step workflow:
 *   1. Configure — dataset, DSL version, execution ID
 *   2. Review    — read-only summary
 *   3. Status    — post-submit DAG polling
 */

import { useEffect, useRef, useState } from 'react';
import { RefreshCw } from 'lucide-react';
import {
  getProcessingRunStatus,
  listDatasets,
  listDsls,
  submitProcessingRun,
  type DatasetInfo,
  type DslVersion,
  type RunResult,
} from '../api/platformClient';
import { useDatasetStore } from '../store/datasetStore';

// ─── Style tokens (mirror RunPage) ───────────────────────────────────────────

const INPUT_CLS =
  'rounded bg-slate-800 px-2 py-1 text-xs text-slate-100 outline-none focus:ring-1 focus:ring-blue-500 w-full';
const SELECT_CLS =
  'rounded bg-slate-800 px-2 py-1 text-xs text-slate-100 outline-none focus:ring-1 focus:ring-blue-500 w-full';
const BTN_NEUTRAL =
  'rounded bg-slate-700 px-2 py-1 text-xs text-slate-300 hover:bg-slate-600 disabled:opacity-40';
const BTN_PRIMARY =
  'rounded bg-blue-600 px-4 py-1.5 text-xs font-medium text-white hover:bg-blue-500 disabled:opacity-40';
const SUB_HEADING =
  'text-xs font-semibold text-slate-400 uppercase tracking-wider';

// ─── Types / constants ───────────────────────────────────────────────────────

const DATE_RE = /^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$/;

interface SplitRange { start: string; end: string }
interface Splits { train: SplitRange; val: SplitRange; test: SplitRange }

const DEFAULT_SPLITS: Splits = {
  train: { start: '2026-01-01 00:30:00', end: '2026-01-01 13:00:00' },
  val:   { start: '2026-01-01 13:00:00', end: '2026-01-02 05:40:00' },
  test:  { start: '2026-01-02 07:00:00', end: '2026-01-03 03:50:00' },
};

// ─── Helpers ─────────────────────────────────────────────────────────────────

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

// ─── Sub-components ───────────────────────────────────────────────────────────

function Field({
  label,
  tooltip,
  children,
}: {
  label: string;
  tooltip?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-xs text-slate-400" title={tooltip}>
        {label}
      </label>
      {children}
    </div>
  );
}

const STEP_LABELS = ['Configure', 'Review', 'Status'];

function StepperHeader({ current }: { current: number }) {
  return (
    <div className="flex items-center gap-1">
      {STEP_LABELS.map((label, idx) => {
        const n = idx + 1;
        const active = n === current;
        const done = n < current;
        return (
          <div key={n} className="flex items-center gap-1">
            <div
              className={`flex h-6 w-6 items-center justify-center rounded-full text-[10px] font-bold ${
                active
                  ? 'bg-blue-600 text-white'
                  : done
                    ? 'bg-green-700 text-white'
                    : 'bg-slate-700 text-slate-400'
              }`}
            >
              {n}
            </div>
            <span
              className={`text-xs ${active ? 'font-semibold text-slate-100' : 'text-slate-500'}`}
            >
              {label}
            </span>
            {idx < STEP_LABELS.length - 1 && (
              <span className="mx-1 text-slate-700">›</span>
            )}
          </div>
        );
      })}
    </div>
  );
}

// ─── ProcessingPage ───────────────────────────────────────────────────────────

export function ProcessingPage() {
  const { activeDataset } = useDatasetStore();

  const [currentStep, setCurrentStep] = useState(1);

  // ── Form state ────────────────────────────────────────────────────────────
  const [dataset, setDataset] = useState(activeDataset ?? '');
  const [availableDatasets, setAvailableDatasets] = useState<DatasetInfo[]>([]);
  const [dsls, setDsls] = useState<DslVersion[]>([]);
  const [dslVersion, setDslVersion] = useState<number | ''>('');
  const [executionId, setExecutionId] = useState(nowId);

  // ── DSL search state ──────────────────────────────────────────────────────
  const [dslSearchStatus, setDslSearchStatus] = useState<
    'idle' | 'loading' | 'success' | 'error'
  >('idle');
  const [dslSearchMessage, setDslSearchMessage] = useState('');

  // ── Splits ────────────────────────────────────────────────────────────────
  const [splits, setSplits] = useState<Splits>(DEFAULT_SPLITS);

  // ── Submission state ──────────────────────────────────────────────────────
  const [submitResult, setSubmitResult] = useState<RunResult | null>(null);
  const [submitError, setSubmitError] = useState('');
  const [runStatus, setRunStatus] = useState('');
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Effects ───────────────────────────────────────────────────────────────

  useEffect(() => {
    listDatasets()
      .then((ds) => setAvailableDatasets(ds))
      .catch(() => setAvailableDatasets([]));
  }, []);

  useEffect(() => {
    if (activeDataset) setDataset(activeDataset);
  }, [activeDataset]);

  // Reset DSL search when dataset changes
  useEffect(() => {
    setDsls([]);
    setDslVersion('');
    setDslSearchStatus('idle');
    setDslSearchMessage('');
  }, [dataset]);

  // Poll run status after submission
  useEffect(() => {
    if (!submitResult?.dag_run_id) return;
    const TERMINAL = ['success', 'failed', 'upstream_failed'];
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const s = await getProcessingRunStatus(submitResult.dag_run_id);
        setRunStatus(s.state);
        if (TERMINAL.includes(s.state.toLowerCase())) {
          if (pollRef.current) clearInterval(pollRef.current);
        }
      } catch {
        /* keep polling */
      }
    }, 15_000);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [submitResult]);

  // ── Handlers ─────────────────────────────────────────────────────────────

  const handleDslSearch = async () => {
    if (!dataset) return;
    setDslSearchStatus('loading');
    setDslSearchMessage('');
    setDsls([]);
    setDslVersion('');
    try {
      const r = await listDsls(dataset);
      setDsls(r.dsls);
      setDslSearchStatus('success');
      setDslSearchMessage(
        `✔ ${r.dsls.length} DSL version${r.dsls.length !== 1 ? 's' : ''} available for ${dataset}`,
      );
    } catch (e) {
      setDslSearchStatus('error');
      setDslSearchMessage(
        `⚠ Could not fetch DSLs: ${e instanceof Error ? e.message : String(e)}`,
      );
    }
  };

  const stepErrors = (): string[] => {
    const e: string[] = [];
    if (!dataset.trim()) e.push('Dataset is required');
    if (dslVersion === '') e.push('Select a DSL version');
    if (!executionId.trim()) e.push('Execution ID is required');

    // Splits validation
    const splitKeys: Array<keyof Splits> = ['train', 'val', 'test'];
    for (const k of splitKeys) {
      if (!DATE_RE.test(splits[k].start)) e.push(`${k} start: must be YYYY-MM-DD HH:MM:SS`);
      if (!DATE_RE.test(splits[k].end))   e.push(`${k} end: must be YYYY-MM-DD HH:MM:SS`);
    }
    if (e.length === 0) {
      const ts = (s: string) => Date.parse(s.replace(' ', 'T') + 'Z');
      for (const k of splitKeys) {
        if (ts(splits[k].start) >= ts(splits[k].end))
          e.push(`${k} start must be before ${k} end`);
      }
      const pairs: Array<[keyof Splits, keyof Splits]> = [
        ['train', 'val'], ['train', 'test'], ['val', 'test'],
      ];
      for (const [a, b] of pairs) {
        if (ts(splits[a].start) < ts(splits[b].end) && ts(splits[b].start) < ts(splits[a].end))
          e.push(`${a} and ${b} ranges overlap`);
      }
    }
    return e;
  };

  const goForward = () => {
    const errs = stepErrors();
    if (currentStep === 1 && errs.length > 0) return;
    setCurrentStep((s) => Math.min(s + 1, 3));
  };

  const handleSubmit = async () => {
    setSubmitError('');
    setSubmitResult(null);
    setRunStatus('');
    try {
      const result = await submitProcessingRun({
        dataset,
        dsl_version: dslVersion as number,
        execution_id: executionId.trim(),
        splits,
      });
      setSubmitResult(result);
      setRunStatus('queued');
      setCurrentStep(3);
    } catch (e) {
      setSubmitError(e instanceof Error ? e.message : String(e));
    }
  };

  const handleReset = () => {
    setCurrentStep(1);
    setSubmitResult(null);
    setSubmitError('');
    setRunStatus('');
    setExecutionId(nowId());
    setDslVersion('');
    setDsls([]);
    setDslSearchStatus('idle');
    setDslSearchMessage('');
    setSplits(DEFAULT_SPLITS);
  };

  const stateColor = (state: string) =>
    state === 'success'
      ? 'text-green-400'
      : state === 'running'
        ? 'text-yellow-400'
        : state.includes('fail')
          ? 'text-red-400'
          : 'text-slate-400';

  const errors = stepErrors();

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="mx-auto max-w-2xl space-y-4 p-4">
      <StepperHeader current={currentStep} />

      <div className="rounded-lg border border-slate-700 bg-slate-900 px-4 py-4">

        {/* ── STEP 1: Configure ── */}
        {currentStep === 1 && (
          <div className="space-y-4">
            <p className={SUB_HEADING}>Processing Configuration</p>

            <div className="grid grid-cols-2 gap-3">
              <Field label="Dataset">
                {availableDatasets.length > 0 ? (
                  <select
                    value={dataset}
                    onChange={(e) => setDataset(e.target.value)}
                    className={SELECT_CLS}
                  >
                    <option value="">— select dataset —</option>
                    {availableDatasets.map((d) => (
                      <option key={d.name} value={d.name}>
                        {d.name}
                      </option>
                    ))}
                  </select>
                ) : (
                  <input
                    value={dataset}
                    onChange={(e) => setDataset(e.target.value)}
                    placeholder="network_traffic_raw"
                    className={INPUT_CLS}
                  />
                )}
              </Field>

              <Field
                label="DSL Version"
                tooltip="Saved DSL pipeline version to fit and apply."
              >
                <div className="flex gap-1">
                  <select
                    value={dslVersion}
                    onChange={(e) =>
                      setDslVersion(
                        e.target.value === '' ? '' : Number(e.target.value),
                      )
                    }
                    className={SELECT_CLS}
                    disabled={dsls.length === 0}
                  >
                    <option value="">
                      {dsls.length === 0 ? '— search first —' : '— select —'}
                    </option>
                    {dsls.map((d) => (
                      <option key={d.version} value={d.version}>
                        v{d.version} — {d.slug}
                      </option>
                    ))}
                  </select>
                  <button
                    onClick={() => void handleDslSearch()}
                    disabled={!dataset || dslSearchStatus === 'loading'}
                    className={BTN_NEUTRAL}
                    title="Find DSL versions for this dataset"
                  >
                    {dslSearchStatus === 'loading' ? '…' : '🔍'}
                  </button>
                </div>
              </Field>

              <Field
                label="Execution ID"
                tooltip="Unique processing run identifier."
              >
                <div className="flex gap-1">
                  <input
                    value={executionId}
                    onChange={(e) => setExecutionId(e.target.value)}
                    className={INPUT_CLS}
                  />
                  <button
                    onClick={() => setExecutionId(nowId())}
                    className={BTN_NEUTRAL}
                    title="Regenerate"
                  >
                    <RefreshCw size={12} />
                  </button>
                </div>
              </Field>
            </div>

            {/* ── Splits ── */}
            <div className="space-y-2 pt-1">
              <p className={SUB_HEADING}>Temporal Splits</p>
              <p className="text-xs text-slate-500">
                Spark fits the DSL pipeline on the <span className="font-mono text-slate-300">train</span> split and
                transforms all splits. Ranges must not overlap.{' '}
                <span className="font-mono text-slate-400">YYYY-MM-DD HH:MM:SS</span>
              </p>
              {(['train', 'val', 'test'] as const).map((k) => (
                <div key={k} className="grid grid-cols-[60px_1fr_1fr] items-center gap-2">
                  <span className="text-xs capitalize text-slate-400">{k}</span>
                  <input
                    value={splits[k].start}
                    onChange={(e) =>
                      setSplits((prev) => ({ ...prev, [k]: { ...prev[k], start: e.target.value } }))
                    }
                    placeholder="start"
                    className={INPUT_CLS}
                  />
                  <input
                    value={splits[k].end}
                    onChange={(e) =>
                      setSplits((prev) => ({ ...prev, [k]: { ...prev[k], end: e.target.value } }))
                    }
                    placeholder="end"
                    className={INPUT_CLS}
                  />
                </div>
              ))}
            </div>

            {/* DSL search result banner */}
            {dslSearchStatus !== 'idle' && (
              <div
                className={`rounded border px-3 py-2 text-xs font-medium ${
                  dslSearchStatus === 'success'
                    ? 'border-green-800/30 bg-green-900/10 text-green-400'
                    : dslSearchStatus === 'error'
                      ? 'border-red-800/30 bg-red-900/10 text-red-400'
                      : 'border-slate-700 bg-slate-800/50 text-slate-400'
                }`}
              >
                {dslSearchStatus === 'loading' ? 'Searching…' : dslSearchMessage}
              </div>
            )}

            {/* Validation errors */}
            {errors.length > 0 && (
              <div className="space-y-1 rounded border border-red-800/40 bg-red-900/20 px-3 py-2">
                {errors.map((e, i) => (
                  <p key={i} className="text-xs text-red-400">
                    • {e}
                  </p>
                ))}
              </div>
            )}

            <button
              onClick={goForward}
              disabled={errors.length > 0}
              className={`${BTN_PRIMARY} w-full`}
            >
              Review →
            </button>
          </div>
        )}

        {/* ── STEP 2: Review ── */}
        {currentStep === 2 && (
          <div className="space-y-4">
            <p className={SUB_HEADING}>Review & Launch</p>

            <div className="rounded border border-slate-700 bg-slate-800/40 px-3 py-3 text-xs space-y-2">
              <div className="flex justify-between">
                <span className="text-slate-400">Dataset</span>
                <span className="font-mono text-slate-100">{dataset}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">DSL Version</span>
                <span className="font-mono text-slate-100">
                  v{dslVersion}
                  {dsls.find((d) => d.version === dslVersion)
                    ? ` — ${dsls.find((d) => d.version === dslVersion)!.slug}`
                    : ''}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Execution ID</span>
                <span className="font-mono text-slate-100">{executionId}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Output table</span>
                <span className="font-mono text-slate-500">
                  iceberg.processed.{dataset}_{executionId}
                </span>
              </div>
              <div className="border-t border-slate-700 pt-2 space-y-1">
                {(['train', 'val', 'test'] as const).map((k) => (
                  <div key={k} className="flex justify-between">
                    <span className="capitalize text-slate-400">{k} split</span>
                    <span className="font-mono text-slate-100 text-[11px]">
                      {splits[k].start} → {splits[k].end}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {submitError && (
              <p className="text-xs text-red-400">{submitError}</p>
            )}

            <div className="flex gap-2">
              <button
                onClick={() => setCurrentStep(1)}
                className={BTN_NEUTRAL}
              >
                ← Back
              </button>
              <button
                onClick={() => void handleSubmit()}
                className={`${BTN_PRIMARY} flex-1`}
              >
                Launch Processing
              </button>
            </div>
          </div>
        )}

        {/* ── STEP 3: Status ── */}
        {currentStep === 3 && submitResult && (
          <div className="space-y-4">
            <p className={SUB_HEADING}>Processing Run Status</p>

            <div className="rounded border border-slate-700 bg-slate-800/40 px-3 py-3 text-xs space-y-2">
              <div className="flex justify-between">
                <span className="text-slate-400">DAG Run ID</span>
                <span className="font-mono text-slate-100">
                  {submitResult.dag_run_id}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Execution ID</span>
                <span className="font-mono text-slate-100">
                  {submitResult.execution_id}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">params.yaml</span>
                <span className="font-mono text-slate-500 text-[10px]">
                  {submitResult.params_s3_path}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">State</span>
                <span
                  className={`font-semibold ${stateColor(runStatus)}`}
                >
                  {runStatus || 'queued'}
                </span>
              </div>
            </div>

            {runStatus === 'success' && (
              <div className="rounded border border-green-800/30 bg-green-900/10 px-3 py-2 text-xs text-green-400">
                ✔ Preprocessing complete. The processed table is now available
                for Training runs.
              </div>
            )}

            <button onClick={handleReset} className={`${BTN_NEUTRAL} w-full`}>
              New Processing Run
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
