import { useState } from 'react';
import { useDagStore } from '../../../store/dagStore';
import type { ServingOrchNodeData } from '../../../types/dag';
import {
  SELECT_CLS, INPUT_CLS, BTN_PRIMARY, BTN_NEUTRAL, BTN_SUCCESS, SUB_HEADING,
} from '../../../lib/uiTokens';
import { StatusLED } from '../../../design/components/StatusLED';
import { SectionTitle } from '../../../design/components/SectionTitle';
import { GPUResourceSelector } from '../../GPUResourceSelector';
import { GPUCatalogPanel } from '../../GPUCatalogPanel';
import { submitServingConfig, type ResourceConstraints } from '../../../api/platformClient';

interface Props {
  nodeId: string;
  data: ServingOrchNodeData;
}

const BTN_TOGGLE_ON  = 'rounded border border-cyan-500/40 bg-cyan-500/15 px-3 py-0.5 text-xs font-medium text-cyan-300 transition-colors';
const BTN_TOGGLE_OFF = 'rounded border border-slate-700/40 bg-brand-panel px-3 py-0.5 text-xs font-medium text-slate-500 hover:text-slate-300 transition-colors';

export function ServingInspector({ nodeId, data }: Props) {
  const updateNodeData = useDagStore((s) => s.updateNodeData);
  const runNode        = useDagStore((s) => s.runNode);
  const assignRunId    = useDagStore((s) => s.assignRunId);

  const [loading, setLoading]         = useState(false);
  const [assignInput, setAssignInput] = useState('');
  const [creating, setCreating]       = useState(false);
  const [showController, setShowController] = useState(false);
  const [showScaling, setShowScaling]       = useState(false);
  const [showServingGpu, setShowServingGpu] = useState(false);

  const set = (patch: Partial<ServingOrchNodeData>) =>
    updateNodeData(nodeId, (prev) => ({ ...prev, ...patch } as ServingOrchNodeData));

  const handleCreateConfig = async () => {
    if (!data.trainRunId) return;
    setCreating(true);
    try {
      const servingMode = data.servingMode === 'vllm' ? 'ray_only' : (data.servingMode || 'ray_only');
      const resp = await submitServingConfig({
        train_run_id: data.trainRunId,
        serving_mode: servingMode as 'ray_only' | 'kafka',
        ...(data.servingMode === 'kafka' && data.rawSchemaS3Path && { raw_schema_s3_path: data.rawSchemaS3Path }),
        alias: data.alias || undefined,
        canary: data.canary || undefined,
        ...(data.canary && {
          canary_alias: data.canaryAlias || undefined,
          canary_probability: data.canaryProbability,
        }),
        initial_replicas: data.initialReplicas || undefined,
        min_replicas: data.minReplicas,
        max_replicas: data.maxReplicas,
        target_qps_per_replica: data.targetQpsPerReplica,
        deployment_target: 'skypilot',
        ...(data.webhookPublicBaseUrl && { webhook_public_base_url: data.webhookPublicBaseUrl }),
        ...(data.webhookPath          && { webhook_path: data.webhookPath }),
        webhook_max_timestamp_age_seconds: data.webhookMaxTimestampAge || undefined,
        resource_constraints: { ...data.servingResourceConstraints, job_type: 'tabular' },
        ...(data.serveControllerInfra || data.serveControllerHighAvailability ? {
          serve_controller: {
            high_availability: data.serveControllerHighAvailability,
            resources: {
              infra:     data.serveControllerInfra   || undefined,
              cpus:      data.serveControllerCpus    || undefined,
              disk_size: data.serveControllerDiskSize || undefined,
            },
          },
        } : {}),
      });
      set({ servingRunId: resp.serve_run_id, status: 'configured' });
    } catch (e) {
      set({ errors: [e instanceof Error ? e.message : String(e)] });
    } finally {
      setCreating(false);
    }
  };

  const isVllm = data.servingMode === 'vllm';

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <SectionTitle>Serving Node</SectionTitle>
        <StatusLED status={data.status} />
      </div>

      <hr className="filament-divider" />

      <div className="space-y-1">
        <p className={SUB_HEADING}>Train Run ID (propagated)</p>
        <p className={`text-xs font-mono ${data.trainRunId ? 'text-purple-300' : 'text-slate-600'}`}>
          {data.trainRunId || 'not set — run Training node first'}
        </p>
      </div>

      {/* Serving mode */}
      <div className="space-y-1">
        <p className={SUB_HEADING}>Serving Mode</p>
        <select className={SELECT_CLS} value={data.servingMode}
          onChange={(e) => set({ servingMode: e.target.value as ServingOrchNodeData['servingMode'] })}>
          <option value="ray_only">Tabular — Ray Serve</option>
          <option value="kafka">Kafka Inference</option>
          <option value="vllm">vLLM (SkyPilot)</option>
        </select>
      </div>

      {/* ── vLLM config ─────────────────────────────────────── */}
      {isVllm && (
        <div className="space-y-2 rounded border border-purple-500/20 bg-purple-500/5 p-3">
          <p className="text-[10px] font-semibold uppercase tracking-wider text-purple-400">vLLM Config</p>
          {[
            { key: 'hfToken',       label: 'HF Token',          placeholder: 'hf_...',       type: 'text'   },
            { key: 'llmAdapterS3',  label: 'LoRA Adapter S3',   placeholder: 's3://...',     type: 'text'   },
          ].map(({ key, label, placeholder, type }) => (
            <div key={key} className="space-y-0.5">
              <p className="text-[10px] text-slate-500">{label}</p>
              <input className={INPUT_CLS} type={type} placeholder={placeholder}
                value={(data as unknown as Record<string, unknown>)[key] as string}
                onChange={(e) => set({ [key]: e.target.value } as Partial<ServingOrchNodeData>)} />
            </div>
          ))}
          <div className="grid grid-cols-2 gap-2">
            {[
              { key: 'vllmPort',             label: 'Port',              min: 1024,  max: 65535, step: 1    },
              { key: 'maxModelLen',           label: 'Max Model Len',     min: 512,   step: 512  },
              { key: 'tensorParallelSize',    label: 'Tensor Parallel',   min: 1,     step: 1    },
              { key: 'pipelineParallelSize',  label: 'Pipeline Parallel', min: 1,     step: 1    },
              { key: 'vllmReplicas',          label: 'Replicas',          min: 1,     step: 1    },
            ].map(({ key, label, min, max, step }) => (
              <div key={key} className="space-y-0.5">
                <p className="text-[10px] text-slate-500">{label}</p>
                <input className={INPUT_CLS} type="number" min={min} max={max} step={step}
                  value={(data as unknown as Record<string, unknown>)[key] as number}
                  onChange={(e) => set({ [key]: parseInt(e.target.value) || min } as Partial<ServingOrchNodeData>)} />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Tabular / Kafka config ───────────────────────────── */}
      {!isVllm && (
        <>
          {/* Alias */}
          <div className="space-y-1">
            <p className={SUB_HEADING}>Alias</p>
            <input className={INPUT_CLS} placeholder="champion"
              value={data.alias} onChange={(e) => set({ alias: e.target.value })} />
          </div>

          {/* Canary */}
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <p className={SUB_HEADING}>Canary</p>
              <div className="flex gap-1">
                {[true, false].map((v) => (
                  <button key={String(v)} type="button"
                    className={data.canary === v ? BTN_TOGGLE_ON : BTN_TOGGLE_OFF}
                    onClick={() => set({ canary: v })}>
                    {v ? 'ON' : 'OFF'}
                  </button>
                ))}
              </div>
            </div>
            {data.canary && (
              <div className="grid grid-cols-2 gap-2 pl-2 border-l border-orange-500/20">
                <div className="space-y-0.5">
                  <p className="text-[10px] text-slate-500">Canary Alias</p>
                  <input className={INPUT_CLS} placeholder="challenger"
                    value={data.canaryAlias} onChange={(e) => set({ canaryAlias: e.target.value })} />
                </div>
                <div className="space-y-0.5">
                  <p className="text-[10px] text-slate-500">Traffic Prob (0–1)</p>
                  <input className={INPUT_CLS} type="number" min={0} max={1} step={0.05}
                    value={data.canaryProbability}
                    onChange={(e) => set({ canaryProbability: parseFloat(e.target.value) || 0.1 })} />
                </div>
              </div>
            )}
          </div>

          {/* Replicas (collapsible) */}
          <div className="rounded border border-slate-700/60">
            <button type="button"
              className="flex w-full items-center justify-between px-3 py-2 text-[10px] font-semibold uppercase tracking-wider text-slate-400 hover:bg-slate-800/40"
              onClick={() => setShowScaling((v) => !v)}>
              <span>Autoscaling &amp; Replicas</span>
              <span>{showScaling ? '▲' : '▼'}</span>
            </button>
            {showScaling && (
              <div className="grid grid-cols-2 gap-2 border-t border-slate-700/60 p-3">
                {[
                  { key: 'initialReplicas',       label: 'Initial',     min: 0, step: 1 },
                  { key: 'minReplicas',            label: 'Min',         min: 0, step: 1 },
                  { key: 'maxReplicas',            label: 'Max',         min: 1, step: 1 },
                  { key: 'targetQpsPerReplica',    label: 'Target QPS',  min: 1, step: 1 },
                ].map(({ key, label, min, step }) => (
                  <div key={key} className="space-y-0.5">
                    <p className="text-[10px] text-slate-500">{label}</p>
                    <input className={INPUT_CLS} type="number" min={min} step={step}
                      value={(data as unknown as Record<string, unknown>)[key] as number}
                      onChange={(e) => set({ [key]: parseInt(e.target.value) || min } as Partial<ServingOrchNodeData>)} />
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Kafka-specific */}
          {data.servingMode === 'kafka' && (
            <div className="space-y-1">
              <p className={SUB_HEADING}>Raw Schema S3 Path <span className="text-red-400">*</span></p>
              <input className={INPUT_CLS} placeholder="s3://bucket/schemas/..."
                value={data.rawSchemaS3Path}
                onChange={(e) => set({ rawSchemaS3Path: e.target.value })} />
            </div>
          )}

          {/* Webhook */}
          <div className="space-y-2">
            <p className={SUB_HEADING}>Webhook (optional)</p>
            {[
              { key: 'webhookPublicBaseUrl', label: 'Base URL',    placeholder: 'https://example.com', type: 'text'   },
              { key: 'webhookPath',          label: 'Path',        placeholder: '/infer/webhook',       type: 'text'   },
              { key: 'webhookMaxTimestampAge', label: 'Max Age (s)', placeholder: '300',                type: 'number' },
            ].map(({ key, label, placeholder, type }) => (
              <div key={key} className="space-y-0.5">
                <p className="text-[10px] text-slate-500">{label}</p>
                <input className={INPUT_CLS} type={type} placeholder={placeholder}
                  value={(data as unknown as Record<string, unknown>)[key] as string | number}
                  onChange={(e) => set({ [key]: type === 'number' ? parseInt(e.target.value) || 300 : e.target.value } as Partial<ServingOrchNodeData>)} />
              </div>
            ))}
          </div>

          {/* GPU Catalog panel — floats left of inspector when GPU section open */}
          <GPUCatalogPanel
            open={showServingGpu}
            selectedGpuType={data.servingResourceConstraints.gpu_types?.[0] ?? null}
            onSelect={(gpuType) => set({ servingResourceConstraints: { ...data.servingResourceConstraints, gpu_types: [gpuType] } })}
          />

          {/* Serving GPU resources (collapsible) */}
          <div className="rounded border border-slate-700/60">
            <button type="button"
              className="flex w-full items-center justify-between px-3 py-2 text-[10px] font-semibold uppercase tracking-wider text-slate-400 hover:bg-slate-800/40"
              onClick={() => setShowServingGpu((v) => !v)}>
              <span>Serving GPU Resources</span>
              <span>{showServingGpu ? '▲' : '▼'}</span>
            </button>
            {showServingGpu && (
              <div className="border-t border-slate-700/60 p-3">
                <GPUResourceSelector
                  value={data.servingResourceConstraints as ResourceConstraints}
                  onChange={(rc) => set({ servingResourceConstraints: rc })}
                />
              </div>
            )}
          </div>

          {/* SkyServe Controller (collapsible) */}
          <div className="rounded border border-slate-700/60">
            <button type="button"
              className="flex w-full items-center justify-between px-3 py-2 text-[10px] font-semibold uppercase tracking-wider text-slate-400 hover:bg-slate-800/40"
              onClick={() => setShowController((v) => !v)}>
              <span>SkyServe Controller</span>
              <span>{showController ? '▲' : '▼'}</span>
            </button>
            {showController && (
              <div className="flex flex-col gap-2 border-t border-slate-700/60 p-3">
                <div className="flex items-center gap-2">
                  <input type="checkbox" id={`${nodeId}-ha`}
                    checked={data.serveControllerHighAvailability}
                    onChange={(e) => set({ serveControllerHighAvailability: e.target.checked })}
                    className="accent-cyan-400" />
                  <label htmlFor={`${nodeId}-ha`} className="text-[10px] text-slate-400">High Availability</label>
                </div>
                {[
                  { key: 'serveControllerInfra',    label: 'Infra',     placeholder: 'runpod/...',  type: 'text'   },
                  { key: 'serveControllerCpus',     label: 'CPUs',      placeholder: '4+',          type: 'text'   },
                  { key: 'serveControllerDiskSize', label: 'Disk (GB)', placeholder: '100',         type: 'number' },
                ].map(({ key, label, placeholder, type }) => (
                  <div key={key} className="space-y-0.5">
                    <p className="text-[10px] text-slate-500">{label}</p>
                    <input className={INPUT_CLS} type={type} placeholder={placeholder}
                      value={(data as unknown as Record<string, unknown>)[key] as string | number}
                      onChange={(e) => set({ [key]: type === 'number' ? parseInt(e.target.value) || 100 : e.target.value } as Partial<ServingOrchNodeData>)} />
                  </div>
                ))}
              </div>
            )}
          </div>
        </>
      )}

      {/* Serving run ID output */}
      {data.servingRunId && (
        <div className="space-y-1">
          <p className={SUB_HEADING}>Serving Run ID</p>
          <p className="text-xs font-mono text-emerald-400 break-all">{data.servingRunId}</p>
        </div>
      )}

      {/* Assign existing serving run */}
      <div className="space-y-1">
        <p className={SUB_HEADING}>Assign Existing Serving Run ID</p>
        <div className="flex gap-1">
          <input className={INPUT_CLS} placeholder="paste servingRunId"
            value={assignInput} onChange={(e) => setAssignInput(e.target.value)} />
          <button className={BTN_SUCCESS} disabled={!assignInput.trim()}
            onClick={() => { assignRunId(nodeId, assignInput.trim()); setAssignInput(''); }}>
            Assign
          </button>
        </div>
      </div>

      {data.errors.length > 0 && (
        <div className="rounded border border-red-500/20 bg-red-500/5 p-2">
          {data.errors.map((e, i) => <p key={i} className="text-[10px] text-red-400">{e}</p>)}
        </div>
      )}

      <div className="flex flex-wrap gap-2 pt-1">
        {!isVllm && !data.servingRunId && (
          <button disabled={!data.trainRunId || creating} className={BTN_NEUTRAL}
            onClick={() => { void handleCreateConfig(); }}>
            {creating ? 'Creating…' : 'Create Serving Config'}
          </button>
        )}
        <button
          disabled={(!isVllm && !data.servingRunId) || loading || !data.trainRunId}
          className={BTN_PRIMARY}
          onClick={() => { setLoading(true); runNode(nodeId).finally(() => setLoading(false)); }}
        >
          {loading ? 'Deploying…' : isVllm ? 'Launch vLLM' : 'Deploy'}
        </button>
      </div>
    </div>
  );
}
