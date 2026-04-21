import { useEffect, useState } from 'react';
import { useDagStore } from '../../../store/dagStore';
import type { ProcessingOrchNodeData } from '../../../types/dag';
import { listDsls, type DslVersion } from '../../../api/platformClient';
import { INPUT_CLS, SELECT_CLS, BTN_PRIMARY, BTN_NEUTRAL, BTN_SUCCESS, SUB_HEADING } from '../../../lib/uiTokens';
import { StatusLED } from '../../../design/components/StatusLED';
import { SectionTitle } from '../../../design/components/SectionTitle';
import { useUIStore } from '../../../store/uiStore';

interface Props {
  nodeId: string;
  data: ProcessingOrchNodeData;
}

type SplitKey = 'train' | 'val' | 'test';
type BoundKey = 'start' | 'end';

export function ProcessingInspector({ nodeId, data }: Props) {
  const updateNodeData = useDagStore((s) => s.updateNodeData);
  const runNode        = useDagStore((s) => s.runNode);
  const runUpTo        = useDagStore((s) => s.runUpTo);
  const assignRunId    = useDagStore((s) => s.assignRunId);
  const setPage        = useUIStore((s) => s.setPage);

  const [dsls, setDsls]         = useState<DslVersion[]>([]);
  const [loading, setLoading]   = useState(false);
  const [assignInput, setAssignInput] = useState('');

  useEffect(() => {
    if (data.datasetName) {
      listDsls(data.datasetName).then((r) => setDsls(r.dsls)).catch(() => {});
    }
  }, [data.datasetName]);

  const set = (patch: Partial<ProcessingOrchNodeData>) =>
    updateNodeData(nodeId, (prev) => ({ ...prev, ...patch } as ProcessingOrchNodeData));

  const setSplit = (split: SplitKey, bound: BoundKey, val: string) => {
    set({ splits: { ...data.splits, [split]: { ...data.splits[split], [bound]: val } } });
  };

  const SPLIT_LABEL: Record<SplitKey, string> = { train: 'Train', val: 'Validation', test: 'Test' };

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
        <p className="text-xs font-mono text-cyan-300">{data.datasetName || <span className="text-slate-600">—</span>}</p>
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
          <option value="">— select —</option>
          {dsls.map((d) => (
            <option key={d.version} value={d.version}>v{d.version} · {d.slug}</option>
          ))}
        </select>
      </div>

      {/* Execution ID */}
      <div className="space-y-1">
        <p className={SUB_HEADING}>Execution ID (optional)</p>
        <input
          className={INPUT_CLS}
          placeholder="e.g. 20260420_140000"
          value={data.executionId}
          onChange={(e) => set({ executionId: e.target.value })}
        />
      </div>

      {/* Splits */}
      <div className="space-y-2">
        <p className={SUB_HEADING}>Splits</p>
        {(['train', 'val', 'test'] as SplitKey[]).map((split) => (
          <div key={split} className="rounded border border-slate-800/60 p-2 space-y-1">
            <p className="text-[10px] text-slate-500 uppercase tracking-wide">{SPLIT_LABEL[split]}</p>
            <div className="grid grid-cols-2 gap-1">
              <input className={INPUT_CLS} placeholder="start" value={data.splits[split].start}
                onChange={(e) => setSplit(split, 'start', e.target.value)} />
              <input className={INPUT_CLS} placeholder="end" value={data.splits[split].end}
                onChange={(e) => setSplit(split, 'end', e.target.value)} />
            </div>
          </div>
        ))}
      </div>

      {/* Preprocessed run ID (read) */}
      {data.preprocessRunId && (
        <div className="space-y-1">
          <p className={SUB_HEADING}>Preprocess Run ID</p>
          <p className="text-xs font-mono text-emerald-400 break-all">{data.preprocessRunId}</p>
        </div>
      )}

      {/* Assign existing run */}
      <div className="space-y-1">
        <p className={SUB_HEADING}>Assign Existing Run ID</p>
        <div className="flex gap-1">
          <input
            className={INPUT_CLS}
            placeholder="paste preprocessRunId"
            value={assignInput}
            onChange={(e) => setAssignInput(e.target.value)}
          />
          <button
            className={BTN_SUCCESS}
            disabled={!assignInput.trim()}
            onClick={() => { assignRunId(nodeId, assignInput.trim()); setAssignInput(''); }}
          >
            Assign
          </button>
        </div>
        <p className="text-[10px] text-slate-600">Marks node as success and propagates ID downstream.</p>
      </div>

      {data.errors.length > 0 && (
        <div className="rounded border border-red-500/20 bg-red-500/5 p-2">
          {data.errors.map((e, i) => <p key={i} className="text-[10px] text-red-400">{e}</p>)}
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
