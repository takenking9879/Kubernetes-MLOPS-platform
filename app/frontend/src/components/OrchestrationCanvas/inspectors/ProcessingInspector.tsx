import { useEffect, useRef, useState } from 'react';
import { ChevronDown, ChevronUp, RefreshCw } from 'lucide-react';
import { useDagStore } from '../../../store/dagStore';
import type { ProcessingOrchNodeData, SplitConfig } from '../../../types/dag';
import {
  listDsls, listPreprocessRunIds, uploadFullSchema,
  type DslVersion, type PreprocessRunId, type SchemaUploadSingleResult,
} from '../../../api/platformClient';
import {
  INPUT_CLS, SELECT_CLS, BTN_PRIMARY, BTN_NEUTRAL, BTN_SUCCESS, SUB_HEADING,
  INSPECTOR_HELP_CLS, INSPECTOR_LABEL_CLS, INSPECTOR_META_CLS,
} from '../../../lib/uiTokens';
import { StatusLED } from '../../../design/components/StatusLED';
import { SectionTitle } from '../../../design/components/SectionTitle';
import { useUIStore } from '../../../store/uiStore';
import { FullYamlEditor } from '../../schema/FullYamlEditor';
import type { FullFieldEntry } from '../../../lib/schemaYaml';

// ─── Constants ─────────────────────────────────────────────────────────────────

const DATE_RE = /^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$/;

// ─── DateTime segmented input ─────────────────────────────────────────────────

const SEG_WIDTHS = ['w-11', 'w-7', 'w-7', 'w-7', 'w-7', 'w-7'] as const;
const SEG_MAX    = [4, 2, 2, 2, 2, 2] as const;
const SEG_HINTS  = ['YYYY', 'MM', 'DD', 'HH', 'MM', 'SS'] as const;

function parseDtParts(value: string): string[] {
  const m = value.match(/^(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})$/);
  return m ? [m[1]!, m[2]!, m[3]!, m[4]!, m[5]!, m[6]!] : ['2026', '01', '01', '00', '00', '00'];
}

function buildDtString(parts: string[]): string {
  return `${parts[0]}-${parts[1]}-${parts[2]} ${parts[3]}:${parts[4]}:${parts[5]}`;
}

function DateTimeSegInput({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  const parts = parseDtParts(value);
  const inputRefs = useRef<Array<HTMLInputElement | null>>(Array(6).fill(null));

  const commit = (idx: number, raw: string) => {
    const cleaned = raw.replace(/\D/g, '').padStart(SEG_MAX[idx]!, '0').slice(-(SEG_MAX[idx]!));
    const next = [...parts];
    next[idx] = cleaned;
    onChange(buildDtString(next));
  };

  return (
    <div className="flex items-center gap-0.5 font-mono text-xs">
      {parts.map((p, idx) => (
        <span key={idx} className="flex items-center gap-0.5">
          {idx === 3 && <span className="mx-1 select-none text-slate-700">·</span>}
          {idx > 0 && idx !== 3 && (
            <span className="select-none text-slate-700">{idx < 3 ? '-' : ':'}</span>
          )}
          <input
            ref={(el) => { inputRefs.current[idx] = el; }}
            type="text"
            inputMode="numeric"
            maxLength={SEG_MAX[idx]}
            value={p}
            placeholder={SEG_HINTS[idx]}
            title={SEG_HINTS[idx]}
            onFocus={(e) => e.target.select()}
            onChange={(e) => {
              const raw = e.target.value.replace(/\D/g, '').slice(0, SEG_MAX[idx]);
              const next = [...parts];
              next[idx] = raw;
              onChange(buildDtString(next));
              if (raw.length === SEG_MAX[idx] && idx < 5) {
                inputRefs.current[idx + 1]?.focus();
                inputRefs.current[idx + 1]?.select();
              }
            }}
            onBlur={(e) => commit(idx, e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
                e.preventDefault();
                const curr = parseInt(p, 10) || 0;
                const next = [...parts];
                next[idx] = String(Math.max(0, curr + (e.key === 'ArrowUp' ? 1 : -1))).padStart(SEG_MAX[idx]!, '0');
                onChange(buildDtString(next));
              }
            }}
            className={`${SEG_WIDTHS[idx]} rounded border border-slate-700/50 bg-slate-800 px-1 py-0.5 text-center text-cyan-200 transition-colors focus:border-cyan-500/60 focus:bg-slate-700 focus:outline-none`}
          />
        </span>
      ))}
    </div>
  );
}

// ─── Split timeline visualization ─────────────────────────────────────────────

function parseDt(s: string): number | null {
  if (!DATE_RE.test(s)) return null;
  return Date.parse(s.replace(' ', 'T') + 'Z');
}

function fmtDuration(ms: number): string {
  const abs = Math.abs(ms);
  const h   = Math.floor(abs / 3_600_000);
  const m   = Math.floor((abs % 3_600_000) / 60_000);
  return h > 0 ? `${h}h ${m}m` : `${m}m`;
}

const SPLIT_META = [
  { key: 'train' as const, label: 'Train', bar: 'bg-blue-500',    text: 'text-blue-400'    },
  { key: 'val'   as const, label: 'Val',   bar: 'bg-amber-500',   text: 'text-amber-400'   },
  { key: 'test'  as const, label: 'Test',  bar: 'bg-emerald-500', text: 'text-emerald-400' },
];

function SplitTimeline({ splits }: { splits: SplitConfig }) {
  const data = SPLIT_META.map((m) => ({
    ...m,
    start: parseDt(splits[m.key].start),
    end:   parseDt(splits[m.key].end),
  })).map((d) => ({ ...d, dur: d.start !== null && d.end !== null ? d.end - d.start : null }));

  if (data.every((d) => d.dur === null)) return null;

  const validStarts = data.map((d) => d.start).filter((v): v is number => v !== null);
  const validEnds   = data.map((d) => d.end).filter((v): v is number => v !== null);
  const tMin = validStarts.length ? Math.min(...validStarts) : null;
  const tMax = validEnds.length   ? Math.max(...validEnds)   : null;
  const span = tMin !== null && tMax !== null ? tMax - tMin : null;

  return (
    <div className="rounded border border-slate-800/60 bg-[#03060a]/60 p-2.5 space-y-2">
      <p className="text-xs uppercase tracking-wider text-slate-400">Split coverage</p>

      {/* Proportional timeline bar */}
      {span && span > 0 && tMin !== null && (
        <div className="relative h-2.5 overflow-hidden rounded bg-slate-800/80">
          {data.map((d) => {
            if (d.start === null || d.dur === null || d.dur <= 0) return null;
            const left  = ((d.start - tMin) / span) * 100;
            const width = (d.dur / span) * 100;
            return (
              <div
                key={d.key}
                className={`absolute inset-y-0 ${d.bar} opacity-70`}
                style={{ left: `${left}%`, width: `${width}%` }}
                title={`${d.label}: ${fmtDuration(d.dur)}`}
              />
            );
          })}
        </div>
      )}

      {/* Duration rows + gap indicators */}
      <div className="space-y-0.5">
        {data.map((d, i) => {
          const next = data[i + 1];
          const gap  = next && d.end !== null && next.start !== null ? next.start - d.end : null;
          return (
            <div key={d.key}>
              <div className="flex items-center gap-2">
                <div className={`h-2 w-2 flex-shrink-0 rounded-sm ${d.bar} opacity-80`} />
                <span className={`w-10 ${LBL}`}>{d.label}</span>
                <span className={`text-[10px] font-mono ${d.dur !== null && d.dur > 0 ? d.text : 'text-slate-700'}`}>
                  {d.dur !== null && d.dur > 0 ? fmtDuration(d.dur) : '—'}
                </span>
              </div>
              {gap !== null && (
                <div className={`ml-4 mt-0.5 text-xs ${gap < 0 ? 'text-red-400' : 'text-slate-300'}`}>
                  {gap < 0
                    ? `⚠ overlaps ${fmtDuration(gap)}`
                    : `↕ gap ${fmtDuration(gap)}`}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ─── Schema YAML helper ───────────────────────────────────────────────────────

function generateFullYaml(fields: FullFieldEntry[], dataset: string): string {
  const lines = [`# Schema: ${dataset} — full (features + target label)`, 'fields:'];
  for (const f of fields) {
    const name = f.editedName ?? f.name;
    const type = f.editedType ?? f.sparkType;
    lines.push(`  - {name: ${name}, type: ${type}, nullable: ${f.nullable}}`);
  }
  return lines.join('\n') + '\n';
}

// ─── Props ────────────────────────────────────────────────────────────────────

interface Props {
  nodeId: string;
  data: ProcessingOrchNodeData;
}

type SplitKey  = 'train' | 'val' | 'test';
type BoundKey  = 'start' | 'end';

const LBL = INSPECTOR_LABEL_CLS;
const HELP = INSPECTOR_HELP_CLS;
const META = INSPECTOR_META_CLS;

// ─── Main component ───────────────────────────────────────────────────────────

export function ProcessingInspector({ nodeId, data }: Props) {
  const updateNodeData = useDagStore((s) => s.updateNodeData);
  const runNode        = useDagStore((s) => s.runNode);
  const runUpTo        = useDagStore((s) => s.runUpTo);
  const assignRunId    = useDagStore((s) => s.assignRunId);
  const setPage        = useUIStore((s) => s.setPage);

  const [dsls, setDsls]             = useState<DslVersion[]>([]);
  const [runIds, setRunIds]         = useState<PreprocessRunId[]>([]);
  const [loading, setLoading]       = useState(false);

  // Assign existing run
  const [selectedRunId, setSelectedRunId] = useState('');

  // Schema section
  const [schemaOpen, setSchemaOpen]     = useState(false);
  const [fullFields, setFullFields]     = useState<FullFieldEntry[]>([]);
  const [isSavingSchema, setIsSavingSchema] = useState(false);
  const [schemaResult, setSchemaResult] = useState<SchemaUploadSingleResult | null>(null);
  const [schemaError, setSchemaError]   = useState('');

  // Load DSLs and existing run IDs when dataset changes
  useEffect(() => {
    setDsls([]);
    setRunIds([]);
    setSelectedRunId('');
    if (data.datasetName) {
      listDsls(data.datasetName).then((r) => setDsls(r.dsls)).catch(() => {});
      listPreprocessRunIds(data.datasetName).then((r) => setRunIds(r.runs)).catch(() => {});
    }
  }, [data.datasetName]);

  const set = (patch: Partial<ProcessingOrchNodeData>) =>
    updateNodeData(nodeId, (prev) => ({ ...prev, ...patch } as ProcessingOrchNodeData));

  const setSplit = (split: SplitKey, bound: BoundKey, val: string) =>
    set({ splits: { ...data.splits, [split]: { ...data.splits[split], [bound]: val } } });

  const handleSaveSchema = async () => {
    if (!data.datasetName || fullFields.length === 0) return;
    setIsSavingSchema(true);
    setSchemaError('');
    setSchemaResult(null);
    try {
      const yaml = generateFullYaml(fullFields, data.datasetName);
      const result = await uploadFullSchema(data.datasetName, yaml);
      setSchemaResult(result);
    } catch (e) {
      setSchemaError(e instanceof Error ? e.message : String(e));
    } finally {
      setIsSavingSchema(false);
    }
  };

  const SPLIT_LABEL: Record<SplitKey, string> = { train: 'Train', val: 'Val', test: 'Test' };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <SectionTitle>Processing Node</SectionTitle>
        <StatusLED status={data.status} />
      </div>

      <hr className="filament-divider" />

      {/* Dataset propagated */}
      <div className="space-y-1">
        <p className={SUB_HEADING}>Dataset (propagated)</p>
        <p className="text-xs font-mono text-cyan-300">
          {data.datasetName || <span className={META}>—</span>}
        </p>
      </div>

      {/* DSL version */}
      <div className="space-y-1">
        <p className={SUB_HEADING}>DSL Version</p>
        <select
          className={SELECT_CLS}
          value={data.dslVersion}
          onChange={(e) => set({ dslVersion: e.target.value ? Number(e.target.value) : '', status: 'configured' })}
          disabled={!data.datasetName}
        >
          <option value="">{dsls.length === 0 ? '— select dataset first —' : '— select —'}</option>
          {dsls.map((d) => (
            <option key={d.version} value={d.version}>v{d.version} · {d.slug}</option>
          ))}
        </select>
      </div>

      {/* Execution ID */}
      <div className="space-y-1">
        <p className={SUB_HEADING}>Execution ID (optional)</p>
        <div className="flex gap-1">
          <input
            className={INPUT_CLS}
            placeholder="e.g. 20260420_140000"
            value={data.executionId}
            onChange={(e) => set({ executionId: e.target.value })}
          />
          <button
            className={BTN_NEUTRAL}
            title="Regenerate"
            onClick={() => {
              const d = new Date();
              const id = [
                d.getUTCFullYear(),
                String(d.getUTCMonth() + 1).padStart(2, '0'),
                String(d.getUTCDate()).padStart(2, '0'),
                '_',
                String(d.getUTCHours()).padStart(2, '0'),
                String(d.getUTCMinutes()).padStart(2, '0'),
                String(d.getUTCSeconds()).padStart(2, '0'),
              ].join('');
              set({ executionId: id });
            }}
          >
            <RefreshCw size={11} />
          </button>
        </div>
      </div>

      {/* ── Temporal Splits ── */}
      <div className="space-y-2">
        <p className={SUB_HEADING}>Temporal Splits</p>
        <p className={HELP}>
          Click each segment to edit. <span className={META}>↑↓ arrows</span> nudge values.
          Ranges must not overlap.
        </p>

        {(['train', 'val', 'test'] as SplitKey[]).map((split) => (
          <div key={split} className="rounded border border-slate-800/60 p-2 space-y-1.5">
            <p className={`${LBL} uppercase tracking-wide`}>
              {SPLIT_LABEL[split]}
            </p>
            <div className="flex flex-col gap-1">
              <div className="flex items-center gap-1.5">
                <span className={`w-10 ${LBL}`}>Start</span>
                <DateTimeSegInput
                  value={data.splits[split].start}
                  onChange={(v) => setSplit(split, 'start', v)}
                />
              </div>
              <div className="flex items-center gap-1.5">
                <span className={`w-10 ${LBL}`}>End</span>
                <DateTimeSegInput
                  value={data.splits[split].end}
                  onChange={(v) => setSplit(split, 'end', v)}
                />
              </div>
            </div>
          </div>
        ))}

        <SplitTimeline splits={data.splits} />
      </div>

      {/* ── Preprocessed run ID (read-only, propagated) ── */}
      {data.preprocessRunId && (
        <div className="space-y-1">
          <p className={SUB_HEADING}>Preprocess Run ID</p>
          <p className="text-xs font-mono text-emerald-400 break-all">{data.preprocessRunId}</p>
        </div>
      )}

      {/* ── Assign Existing Run ── */}
      <div className="space-y-1">
        <p className={SUB_HEADING}>Assign Existing Run ID</p>
        {runIds.length === 0 ? (
          <p className={`${HELP} italic`}>
            {data.datasetName ? 'No previous runs found in S3 for this dataset.' : 'Set a dataset to load runs.'}
          </p>
        ) : (
          <div className="flex gap-1">
            <select
              className={SELECT_CLS}
              value={selectedRunId}
              onChange={(e) => setSelectedRunId(e.target.value)}
            >
              <option value="">— select run from S3 —</option>
              {runIds.map((r) => (
                <option key={r.preprocess_run_id} value={r.preprocess_run_id}>
                  {r.preprocess_run_id}
                </option>
              ))}
            </select>
            <button
              className={BTN_SUCCESS}
              disabled={!selectedRunId}
              onClick={() => { assignRunId(nodeId, selectedRunId); setSelectedRunId(''); }}
            >
              Assign
            </button>
          </div>
        )}
        <p className={HELP}>Marks node as success and propagates ID downstream.</p>
      </div>

      {/* ── Input Schema (full.yaml) ── */}
      <div className="rounded-lg border border-slate-700/60 bg-slate-900/60">
        <button
          onClick={() => setSchemaOpen((o) => !o)}
          className="flex w-full items-center justify-between px-3 py-2.5 text-xs font-medium text-slate-300 hover:text-slate-100"
        >
          <span className="flex items-center gap-2">
            Input Schema (full.yaml)
            {schemaResult && (
              <span className="rounded bg-emerald-900/40 px-1.5 py-0.5 text-[9px] font-bold text-emerald-400 ring-1 ring-emerald-700/40">
                Saved v{schemaResult.version}
              </span>
            )}
          </span>
          {schemaOpen ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
        </button>

        {schemaOpen && (
          <div className="space-y-3 border-t border-slate-700/60 px-3 pb-3 pt-2.5">
            <p className={HELP}>
              Defines columns for{' '}
              <code className="rounded bg-slate-800 px-1">iceberg.raw.{data.datasetName || '<dataset>'}</code>.
              Save before running — backend auto-detects latest version.
            </p>
            <FullYamlEditor
              fields={fullFields}
              onChange={setFullFields}
              dataset={data.datasetName}
            />
            <div className="flex items-center gap-2 border-t border-slate-800 pt-2">
              <button
                onClick={() => void handleSaveSchema()}
                disabled={isSavingSchema || !data.datasetName || fullFields.length === 0}
                className={BTN_PRIMARY}
              >
                {isSavingSchema ? 'Saving…' : 'Save full.yaml to S3'}
              </button>
              {schemaResult && (
                <span className="text-[10px] text-emerald-400">✓ v{schemaResult.version}</span>
              )}
              {schemaError && (
                <span className="text-[10px] text-red-400">{schemaError}</span>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Errors */}
      {data.errors.length > 0 && (
        <div className="rounded border border-red-500/20 bg-red-500/5 p-2">
          {data.errors.map((e, i) => (
            <p key={i} className="text-[10px] text-red-400">{e}</p>
          ))}
        </div>
      )}

      {/* Actions */}
      <div className="flex flex-wrap gap-2 pt-1">
        <button
          disabled={!data.datasetName || !data.dslVersion || loading}
          onClick={() => { setLoading(true); runNode(nodeId).finally(() => setLoading(false)); }}
          className={BTN_PRIMARY}
        >
          {loading ? 'Running…' : 'Run'}
        </button>
        <button
          disabled={!data.datasetName || !data.dslVersion || loading}
          onClick={() => { setLoading(true); runUpTo(nodeId).finally(() => setLoading(false)); }}
          className={BTN_NEUTRAL}
        >
          Run up to here
        </button>
        <button
          className={BTN_NEUTRAL}
          onClick={() => setPage('dsl-builder')}
          title="Open DSL Builder"
        >
          DSL Builder ↗
        </button>
      </div>
    </div>
  );
}
