import { useEffect, useState } from 'react';
import { useDagStore } from '../../../store/dagStore';
import type { DatasetOrchNodeData } from '../../../types/dag';
import { listDatasets, type DatasetInfo } from '../../../api/platformClient';
import { SELECT_CLS, BTN_PRIMARY, BTN_NEUTRAL, SUB_HEADING } from '../../../lib/uiTokens';
import { StatusLED } from '../../../design/components/StatusLED';
import { SectionTitle } from '../../../design/components/SectionTitle';

interface Props {
  nodeId: string;
  data: DatasetOrchNodeData;
}

export function DatasetInspector({ nodeId, data }: Props) {
  const updateNodeData = useDagStore((s) => s.updateNodeData);
  const runNode        = useDagStore((s) => s.runNode);
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [loading, setLoading]   = useState(false);

  useEffect(() => {
    listDatasets().then(setDatasets).catch(() => {});
  }, []);

  const set = (patch: Partial<DatasetOrchNodeData>) =>
    updateNodeData(nodeId, (prev) => ({ ...prev, ...patch } as DatasetOrchNodeData));

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <SectionTitle>Dataset Node</SectionTitle>
        <StatusLED status={data.status} />
      </div>

      <hr className="filament-divider" />

      <div className="space-y-2">
        <p className={SUB_HEADING}>Dataset</p>
        <select
          className={SELECT_CLS}
          value={data.datasetName}
          onChange={(e) => set({ datasetName: e.target.value, status: 'configured' })}
        >
          <option value="">— select —</option>
          {datasets.map((d) => (
            <option key={d.name} value={d.name}>{d.name}</option>
          ))}
        </select>
      </div>

      {data.errors.length > 0 && (
        <div className="rounded border border-red-500/20 bg-red-500/5 p-2">
          {data.errors.map((e, i) => (
            <p key={i} className="text-[10px] text-red-400">{e}</p>
          ))}
        </div>
      )}

      <div className="flex gap-2 pt-1">
        <button
          disabled={!data.datasetName || loading}
          onClick={() => { setLoading(true); runNode(nodeId).finally(() => setLoading(false)); }}
          className={BTN_PRIMARY}
        >
          {loading ? 'Running…' : 'Ingest'}
        </button>
        <button className={BTN_NEUTRAL} onClick={() => set({ status: 'configured', errors: [] })}>
          Reset status
        </button>
      </div>
    </div>
  );
}
