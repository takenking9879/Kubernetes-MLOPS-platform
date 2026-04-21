/**
 * TrainingInspector — full parity with LaunchWizardPage tabular + LLM paths,
 * embedded in the DAG node inspector panel.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { UploadCloud, CheckCircle2, XCircle, Loader2 } from 'lucide-react';
import { useDagStore } from '../../../store/dagStore';
import type { TrainingOrchNodeData } from '../../../types/dag';
import {
  INPUT_CLS, SELECT_CLS, BTN_PRIMARY, BTN_NEUTRAL, BTN_SUCCESS, SUB_HEADING,
} from '../../../lib/uiTokens';
import { StatusLED } from '../../../design/components/StatusLED';
import { SectionTitle } from '../../../design/components/SectionTitle';
import { GPUResourceSelector } from '../../GPUResourceSelector';
import { ChipInput } from '../../forms/ChipInput';
import { NumericInput } from '../../forms/NumericInput';
import {
  getLLMCatalog, listArchitectures, uploadArchitecture,
  type LLMModelInfo, type ArchitectureInfo, type ResourceConstraints,
} from '../../../api/platformClient';
import {
  getDefaultsForTask, getTuneSettingsDefaults, getParamMetaForTask,
  getTuneSettingsMeta, getSearchSpace, getNonTunableKeys, getFinalTrainMeta,
} from '../../../types/hyperparams';
import { parseFiniteNumber, parseStringChip, stringifyNumberValue } from '../../../lib/formValues';
import { GPUCatalogPanel } from '../../GPUCatalogPanel';

interface Props {
  nodeId: string;
  data: TrainingOrchNodeData;
}

const PYTORCH_MODEL_TYPES = ['mlp', 'ssm', 'bae'] as const;
const BTN_ON  = 'rounded border border-cyan-500/40 bg-cyan-500/15 px-2.5 py-0.5 text-xs font-medium text-cyan-300 transition-colors';
const BTN_OFF = 'rounded border border-slate-700/40 bg-brand-panel px-2.5 py-0.5 text-xs font-medium text-slate-500 hover:text-slate-300 transition-colors';
const SECTION = 'rounded border border-slate-700/60';
const SEC_HDR = 'flex w-full items-center justify-between px-3 py-2 text-[10px] font-semibold uppercase tracking-wider text-slate-400 hover:bg-slate-800/30';
const SEC_BODY = 'flex flex-col gap-3 border-t border-slate-700/60 p-3';
const LBL = 'text-[10px] text-slate-500';

export function TrainingInspector({ nodeId, data }: Props) {
  const updateNodeData = useDagStore((s) => s.updateNodeData);
  const runNode        = useDagStore((s) => s.runNode);
  const runUpTo        = useDagStore((s) => s.runUpTo);
  const assignRunId    = useDagStore((s) => s.assignRunId);

  const [loading, setLoading]         = useState(false);
  const [assignInput, setAssignInput] = useState('');

  // Catalog state
  const [llmCatalog, setLlmCatalog]       = useState<LLMModelInfo[]>([]);
  const [architectures, setArchitectures] = useState<ArchitectureInfo[]>([]);

  // Architecture upload
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [uploadFile, setUploadFile]   = useState<File | null>(null);
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'uploading' | 'ok' | 'error'>('idle');
  const [uploadError, setUploadError] = useState('');

  // Collapsible sections
  const [showModelCfg, setShowModelCfg]     = useState(false);
  const [showHyperparams, setShowHyperparams] = useState(false);
  const [showResources, setShowResources]   = useState(false);

  const set = useCallback(
    (patch: Partial<TrainingOrchNodeData>) =>
      updateNodeData(nodeId, (prev) => ({ ...prev, ...patch } as TrainingOrchNodeData)),
    [nodeId, updateNodeData],
  );

  // Load catalogs once
  useEffect(() => {
    getLLMCatalog().then(setLlmCatalog).catch(() => {});
    listArchitectures().then(setArchitectures).catch(() => {});
  }, []);

  // Auto-select first LLM
  useEffect(() => {
    if (llmCatalog.length > 0 && !data.llmModelId) {
      set({ llmModelId: llmCatalog[0]!.model_id });
    }
  }, [llmCatalog]); // eslint-disable-line react-hooks/exhaustive-deps

  // Reset hyperparams + tune settings when framework/task changes
  useEffect(() => {
    set({
      hyperparams: getDefaultsForTask(data.framework, data.taskType),
      tuneSettings: getTuneSettingsDefaults(data.framework),
      searchSpaceOverrides: {},
      finalTrainOverrides: {},
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data.framework, data.taskType]);

  const paramMeta       = getParamMetaForTask(data.framework, data.taskType);
  const tuneSettingsMeta = getTuneSettingsMeta(data.framework);
  const nonTunableKeys  = getNonTunableKeys(data.framework);
  const finalTrainMeta  = getFinalTrainMeta(data.framework);
  const searchSpace     = getSearchSpace(data.framework);

  const hyperparams = Object.keys(data.hyperparams).length > 0
    ? data.hyperparams
    : getDefaultsForTask(data.framework, data.taskType);
  const tuneSettings = Object.keys(data.tuneSettings).length > 0
    ? data.tuneSettings
    : getTuneSettingsDefaults(data.framework);

  const selectedLLM = llmCatalog.find((m) => m.model_id === data.llmModelId);
  const canRun = !!data.preprocessRunId || data.modelCategory === 'llm';

  const handleUpload = async () => {
    if (!uploadFile || !data.uploadedArchId && !uploadFile) return;
    setUploadStatus('uploading');
    setUploadError('');
    try {
      const result = await uploadArchitecture(uploadFile, uploadFile.name.replace('.py', ''), '');
      set({ uploadedArchS3: result.s3_path, uploadedArchId: result.id });
      setUploadStatus('ok');
      listArchitectures().then(setArchitectures).catch(() => {});
    } catch (err) {
      setUploadStatus('error');
      setUploadError(String(err));
    }
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <SectionTitle>Training Node</SectionTitle>
        <StatusLED status={data.status} />
      </div>

      <hr className="filament-divider" />

      {/* Upstream ID */}
      <div className="space-y-0.5">
        <p className={SUB_HEADING}>Preprocess Run ID</p>
        <p className={`text-xs font-mono ${data.preprocessRunId ? 'text-orange-300' : 'text-slate-600'}`}>
          {data.preprocessRunId || 'not set — run Processing node first'}
        </p>
      </div>

      {/* ── Model Category ───────────────────────────────── */}
      <div className="space-y-1">
        <p className={SUB_HEADING}>Model Category</p>
        <div className="flex gap-1.5 flex-wrap">
          {([
            ['tabular', 'Tabular'],
            ['llm',     'LLM (fine-tune)'],
            ['upload',  'Upload .py'],
          ] as const).map(([id, label]) => (
            <button key={id} type="button"
              className={data.modelCategory === id ? BTN_ON : BTN_OFF}
              onClick={() => set({ modelCategory: id, modelType: id === 'llm' ? 'llm' : 'tabular' })}>
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* ══════════════════════════════════════════════════
          TABULAR path
      ══════════════════════════════════════════════════ */}
      {data.modelCategory === 'tabular' && (
        <>
          {/* Framework & Task */}
          <div className="grid grid-cols-2 gap-2">
            <div className="space-y-1">
              <p className={LBL}>Framework</p>
              <div className="flex gap-1">
                {(['xgboost', 'pytorch'] as const).map((f) => (
                  <button key={f} type="button" className={data.framework === f ? BTN_ON : BTN_OFF}
                    onClick={() => set({ framework: f })}>
                    {f === 'xgboost' ? 'XGBoost' : 'PyTorch'}
                  </button>
                ))}
              </div>
            </div>
            <div className="space-y-1">
              <p className={LBL}>Task</p>
              <div className="flex gap-1">
                {(['classification', 'regression'] as const).map((t) => (
                  <button key={t} type="button" className={data.taskType === t ? BTN_ON : BTN_OFF}
                    onClick={() => set({ taskType: t })}>
                    {t.charAt(0).toUpperCase() + t.slice(1)}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* PyTorch model architecture */}
          {data.framework === 'pytorch' && (
            <div className="space-y-1">
              <p className={LBL}>PyTorch Architecture</p>
              <div className="flex gap-1">
                {PYTORCH_MODEL_TYPES.map((m) => (
                  <button key={m} type="button"
                    className={data.pytorchModelType === m ? BTN_ON : BTN_OFF}
                    onClick={() => set({ pytorchModelType: m })}>
                    {m.toUpperCase()}
                  </button>
                ))}
                {/* Custom uploaded architectures */}
                {architectures.filter((a) => !a.builtin).map((a) => (
                  <button key={a.id} type="button"
                    className={data.uploadedArchId === a.id ? BTN_ON : BTN_OFF}
                    onClick={() => set({ uploadedArchId: a.id, uploadedArchS3: a.s3_path ?? '' })}>
                    {a.name}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Model Config (collapsible) */}
          <div className={SECTION}>
            <button type="button" className={SEC_HDR} onClick={() => setShowModelCfg((v) => !v)}>
              <span>Model Config</span><span>{showModelCfg ? '▲' : '▼'}</span>
            </button>
            {showModelCfg && (
              <div className={SEC_BODY}>
                <div className="grid grid-cols-2 gap-2">
                  {[
                    { key: 'experimentName',       label: 'Experiment Name',      placeholder: `dag_${data.framework}` },
                    { key: 'registryModelName',    label: 'Registry Name',        placeholder: data.framework },
                    { key: 'target',               label: 'Target Column',        placeholder: 'e.g. attack' },
                    { key: 'mlflowTrackingUri',    label: 'MLflow Tracking URI',  placeholder: 'Default: from Airflow env' },
                    { key: 'mlflowArtifactLocation', label: 'MLflow Artifact S3', placeholder: 's3://...' },
                  ].map(({ key, label, placeholder }) => (
                    <div key={key} className="space-y-0.5">
                      <p className={LBL}>{label}</p>
                      <input className={INPUT_CLS} placeholder={placeholder}
                        value={(data as unknown as Record<string, unknown>)[key] as string}
                        onChange={(e) => set({ [key]: e.target.value } as Partial<TrainingOrchNodeData>)} />
                    </div>
                  ))}
                  {data.taskType === 'classification' && (
                    <div className="space-y-0.5">
                      <p className={LBL}>Num Classes</p>
                      <input className={INPUT_CLS} type="number" min={2}
                        value={data.numClasses}
                        onChange={(e) => set({ numClasses: parseInt(e.target.value) || 2 })} />
                    </div>
                  )}
                  <div className="space-y-0.5">
                    <p className={LBL}>Seed</p>
                    <input className={INPUT_CLS} type="number"
                      value={data.seed}
                      onChange={(e) => set({ seed: parseInt(e.target.value) || 42 })} />
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Hyperparams & Tuning (collapsible) */}
          <div className={SECTION}>
            <button type="button" className={SEC_HDR} onClick={() => setShowHyperparams((v) => !v)}>
              <span>Hyperparameters &amp; Tuning{data.tuneEnabled ? <span className="ml-1 text-cyan-400 normal-case"> · Ray Tune ON</span> : ''}</span>
              <span>{showHyperparams ? '▲' : '▼'}</span>
            </button>
            {showHyperparams && (
              <div className={SEC_BODY}>
                {/* Ray Tune toggle */}
                <div className="flex items-center gap-2">
                  <p className={LBL}>Ray Tune</p>
                  <div className="flex gap-1">
                    {[true, false].map((v) => (
                      <button key={String(v)} type="button"
                        className={data.tuneEnabled === v ? BTN_ON : BTN_OFF}
                        onClick={() => set({ tuneEnabled: v })}>
                        {v ? 'Enabled' : 'Disabled'}
                      </button>
                    ))}
                  </div>
                </div>

                {data.tuneEnabled ? (
                  <>
                    {/* Search Space */}
                    <div className="space-y-2">
                      <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">Search Space</p>
                      {Object.entries(searchSpace).map(([k, entry]) => {
                        if (entry.type === 'fixed') {
                          return (
                            <div key={k} className="flex items-center justify-between">
                              <span className={LBL}>{k}</span>
                              <span className="rounded bg-slate-800 px-2 py-0.5 text-[10px] text-slate-400">
                                {Array.isArray(entry.value) ? entry.value.join(', ') : String(entry.value)}
                              </span>
                            </div>
                          );
                        }
                        if (entry.type === 'choice') {
                          const opts = (data.searchSpaceOverrides[k]?.options ?? entry.options) as number[];
                          return (
                            <div key={k} className="space-y-0.5">
                              <p className={LBL}>{k} <span className="text-slate-600">choice</span></p>
                              <ChipInput
                                value={opts}
                                onChange={(next) => set({ searchSpaceOverrides: { ...data.searchSpaceOverrides, [k]: { ...data.searchSpaceOverrides[k], options: next as number[] } } })}
                                parseItem={(raw) => { const n = parseFiniteNumber(raw); return n === null ? null : n; }}
                                formatItem={(item) => String(item)}
                                placeholder="value + Enter"
                                className="space-y-0.5"
                              />
                            </div>
                          );
                        }
                        const ov  = data.searchSpaceOverrides[k];
                        const re  = entry as { type: string; min: number; max: number };
                        return (
                          <div key={k} className="space-y-0.5">
                            <p className={LBL}>{k} <span className="text-slate-600">{entry.type}</span></p>
                            <div className="grid grid-cols-2 gap-1">
                              <div className="space-y-0.5">
                                <p className="text-[9px] text-slate-600">Min</p>
                                <NumericInput className={INPUT_CLS}
                                  value={stringifyNumberValue(ov?.min ?? re.min)} step="any"
                                  onChange={(v) => set({ searchSpaceOverrides: { ...data.searchSpaceOverrides, [k]: { ...ov, min: parseFiniteNumber(v) ?? undefined } } })} />
                              </div>
                              <div className="space-y-0.5">
                                <p className="text-[9px] text-slate-600">Max</p>
                                <NumericInput className={INPUT_CLS}
                                  value={stringifyNumberValue(ov?.max ?? re.max)} step="any"
                                  onChange={(v) => set({ searchSpaceOverrides: { ...data.searchSpaceOverrides, [k]: { ...ov, max: parseFiniteNumber(v) ?? undefined } } })} />
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>

                    {/* Scheduler */}
                    <div className="space-y-2">
                      <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">Scheduler</p>
                      <div className="grid grid-cols-2 gap-2">
                        <div className="space-y-0.5">
                          <p className={LBL}>Trials</p>
                          <NumericInput className={INPUT_CLS} value={String(data.numTrials)} min={1}
                            onChange={(v) => set({ numTrials: parseFiniteNumber(v) ?? 3 })} />
                        </div>
                        <div className="space-y-0.5">
                          <p className={LBL}>Sample Fraction</p>
                          <NumericInput className={INPUT_CLS} value={String(data.sampleFraction)} min={0.01} max={1} step={0.01}
                            onChange={(v) => set({ sampleFraction: parseFiniteNumber(v) ?? 0.2 })} />
                        </div>
                        {Object.entries(tuneSettingsMeta).map(([k, meta]) => (
                          <div key={k} className="space-y-0.5">
                            <p className={LBL}>{k}</p>
                            <NumericInput className={INPUT_CLS}
                              value={stringifyNumberValue(tuneSettings[k] ?? (meta.defaultValue as number))}
                              min={meta.min} step={meta.step ?? 1}
                              onChange={(v) => set({ tuneSettings: { ...tuneSettings, [k]: parseFiniteNumber(v) ?? (meta.defaultValue as number) } })} />
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Non-tunable constants */}
                    {nonTunableKeys.length > 0 && (
                      <div className="space-y-2">
                        <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">Non-Tunable Constants</p>
                        <p className={LBL}>Fixed in every trial.</p>
                        <div className="grid grid-cols-2 gap-2">
                          {nonTunableKeys.map((k) => {
                            const meta = paramMeta[k];
                            if (!meta) return null;
                            const val = hyperparams[k] ?? meta.defaultValue;
                            return (
                              <div key={k} className="space-y-0.5">
                                <p className={LBL}>{k}</p>
                                {meta.options ? (
                                  <select className={SELECT_CLS} value={String(val)}
                                    onChange={(e) => set({ hyperparams: { ...hyperparams, [k]: e.target.value } })}>
                                    {meta.options.map((o) => <option key={o} value={o}>{o}</option>)}
                                  </select>
                                ) : meta.type === 'array' ? (
                                  <ChipInput value={Array.isArray(val) ? (val as string[]) : []}
                                    onChange={(next) => set({ hyperparams: { ...hyperparams, [k]: next } })}
                                    parseItem={parseStringChip} formatItem={(i) => i} className="space-y-0.5" />
                                ) : (
                                  <NumericInput className={INPUT_CLS}
                                    value={stringifyNumberValue(val as number | string)}
                                    min={meta.min} max={meta.max} step={meta.step}
                                    onChange={(v) => { const n = parseFiniteNumber(v); if (n !== null) set({ hyperparams: { ...hyperparams, [k]: n } }); }} />
                                )}
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}

                    {/* Final Train Overrides */}
                    <div className="space-y-2">
                      <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">Final Train Overrides <span className="font-normal normal-case text-slate-600">— optional</span></p>
                      <div className="grid grid-cols-2 gap-2">
                        {Object.entries(finalTrainMeta).map(([k, meta]) => (
                          <div key={k} className="space-y-0.5">
                            <p className={LBL}>{k}</p>
                            <NumericInput className={INPUT_CLS} placeholder={String(meta.defaultValue)}
                              value={stringifyNumberValue(data.finalTrainOverrides[k] ?? '')}
                              min={meta.min} step={meta.step ?? 1}
                              onChange={(v) => {
                                if (v === '') { const { [k]: _r, ...rest } = data.finalTrainOverrides; set({ finalTrainOverrides: rest }); return; }
                                const n = parseFiniteNumber(v); if (n !== null) set({ finalTrainOverrides: { ...data.finalTrainOverrides, [k]: n } });
                              }} />
                          </div>
                        ))}
                      </div>
                    </div>
                  </>
                ) : (
                  /* Fixed hyperparams */
                  <div className="grid grid-cols-2 gap-2">
                    {Object.entries(paramMeta).map(([k, meta]) => {
                      const val = hyperparams[k] ?? meta.defaultValue;
                      return (
                        <div key={k} className="space-y-0.5">
                          <p className={LBL}>{k}</p>
                          {meta.options ? (
                            <select className={SELECT_CLS} value={String(val)}
                              onChange={(e) => set({ hyperparams: { ...hyperparams, [k]: e.target.value } })}>
                              {meta.options.map((o) => <option key={o} value={o}>{o}</option>)}
                            </select>
                          ) : meta.type === 'array' ? (
                            <ChipInput value={Array.isArray(val) ? (val as string[]) : []}
                              onChange={(next) => set({ hyperparams: { ...hyperparams, [k]: next } })}
                              parseItem={parseStringChip} formatItem={(i) => i} className="space-y-0.5" />
                          ) : (
                            <NumericInput className={INPUT_CLS}
                              value={stringifyNumberValue(val as number | string)}
                              min={meta.min} max={meta.max} step={meta.step}
                              onChange={(v) => { const n = parseFiniteNumber(v); if (n !== null) set({ hyperparams: { ...hyperparams, [k]: n } }); }} />
                          )}
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            )}
          </div>
        </>
      )}

      {/* ══════════════════════════════════════════════════
          LLM path
      ══════════════════════════════════════════════════ */}
      {data.modelCategory === 'llm' && (
        <div className="space-y-3">
          <div className="space-y-1">
            <p className={LBL}>LLM Model</p>
            <select className={SELECT_CLS} value={data.llmModelId}
              onChange={(e) => set({ llmModelId: e.target.value })}>
              {llmCatalog.length === 0
                ? <option value="">Loading catalog…</option>
                : llmCatalog.map((m) => <option key={m.model_id} value={m.model_id}>{m.model_id}</option>)
              }
            </select>
            {selectedLLM && (
              <div className="flex gap-3 text-[10px] text-slate-400">
                <span>VRAM: <strong className="text-slate-200">{selectedLLM.vram_gb} GB</strong></span>
                <span>Min GPUs: <strong className="text-slate-200">{selectedLLM.min_gpus}</strong></span>
                <span>Rec: <strong className="text-slate-200">{selectedLLM.recommended_gpu}</strong></span>
              </div>
            )}
          </div>
          <div className="space-y-1">
            <p className={LBL}>HuggingFace Token</p>
            <input className={INPUT_CLS} type="password" placeholder="hf_... (optional for public)"
              value={data.hfToken} onChange={(e) => set({ hfToken: e.target.value })} />
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div className="space-y-1">
              <p className={LBL}>Dataset S3 (JSONL)</p>
              <input className={INPUT_CLS} placeholder="s3://bucket/train.jsonl"
                value={data.datasetS3} onChange={(e) => set({ datasetS3: e.target.value })} />
            </div>
            <div className="space-y-1">
              <p className={LBL}>Max Steps</p>
              <input className={INPUT_CLS} type="number" min={1}
                value={data.maxSteps} onChange={(e) => set({ maxSteps: parseInt(e.target.value) || 500 })} />
            </div>
          </div>
          <div className="flex items-center gap-2">
            <p className={LBL}>DeepSpeed ZeRO</p>
            <div className="flex gap-1">
              {[true, false].map((v) => (
                <button key={String(v)} type="button"
                  className={data.useDeepspeed === v ? BTN_ON : BTN_OFF}
                  onClick={() => set({ useDeepspeed: v })}>
                  {v ? 'Enabled' : 'Disabled'}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ══════════════════════════════════════════════════
          Upload custom .py architecture
      ══════════════════════════════════════════════════ */}
      {data.modelCategory === 'upload' && (
        <div className="space-y-2 rounded border border-slate-700/60 p-3">
          <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">Custom Architecture</p>

          {/* Existing architectures */}
          {architectures.length > 0 && (
            <div className="space-y-1">
              <p className={LBL}>Available</p>
              <div className="flex flex-wrap gap-1">
                {architectures.map((a) => (
                  <button key={a.id} type="button"
                    className={data.uploadedArchId === a.id ? BTN_ON : BTN_OFF}
                    onClick={() => set({ uploadedArchId: a.id, uploadedArchS3: a.s3_path ?? '' })}>
                    {a.name}{a.builtin ? ' (built-in)' : ''}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Upload new */}
          <div className="flex items-center gap-2 pt-1">
            <button type="button" onClick={() => fileInputRef.current?.click()}
              className="flex items-center gap-1 rounded bg-slate-700 px-2.5 py-1 text-xs text-slate-300 hover:bg-slate-600">
              <UploadCloud size={11} />
              {uploadFile ? uploadFile.name : 'Select .py'}
            </button>
            <input ref={fileInputRef} type="file" accept=".py" className="hidden"
              onChange={(e) => setUploadFile(e.target.files?.[0] ?? null)} />
            <button type="button" disabled={!uploadFile || uploadStatus === 'uploading'}
              className={BTN_PRIMARY} onClick={() => void handleUpload()}>
              {uploadStatus === 'uploading' ? <><Loader2 size={10} className="animate-spin inline mr-1" />Uploading…</> : 'Validate & Upload'}
            </button>
          </div>
          {uploadStatus === 'ok' && (
            <p className="flex items-center gap-1 text-[10px] text-emerald-400">
              <CheckCircle2 size={10} /> Uploaded — ID: {data.uploadedArchId}
            </p>
          )}
          {uploadStatus === 'error' && (
            <p className="flex items-center gap-1 text-[10px] text-red-400">
              <XCircle size={10} /> {uploadError}
            </p>
          )}
        </div>
      )}

      {/* GPU Catalog panel — floats left of inspector when Resources open */}
      <GPUCatalogPanel
        open={showResources}
        selectedGpuType={data.resourceConstraints.gpu_types?.[0] ?? null}
        onSelect={(gpuType) => set({ resourceConstraints: { ...data.resourceConstraints, gpu_types: [gpuType] } })}
      />

      {/* ══════════════════════════════════════════════════
          Resources (collapsible)
      ══════════════════════════════════════════════════ */}
      <div className={SECTION}>
        <button type="button" className={SEC_HDR} onClick={() => setShowResources((v) => !v)}>
          <span>Resources &amp; Launch</span><span>{showResources ? '▲' : '▼'}</span>
        </button>
        {showResources && (
          <div className={SEC_BODY}>
            <GPUResourceSelector
              value={data.resourceConstraints as ResourceConstraints}
              onChange={(rc) => set({ resourceConstraints: rc })}
            />

            {/* Nodes */}
            <div className="space-y-1">
              <p className={LBL}>Nodes</p>
              <div className="flex gap-1">
                {[1, 2, 4, 8].map((n) => (
                  <button key={n} type="button"
                    className={data.resourceConstraints.num_nodes === n ? BTN_ON : BTN_OFF}
                    onClick={() => set({ resourceConstraints: { ...data.resourceConstraints, num_nodes: n } })}>
                    {n === 1 ? '1 (single)' : `${n}×`}
                  </button>
                ))}
              </div>
            </div>

            {/* InfiniBand */}
            <div className="flex items-center gap-2">
              <input type="checkbox" id={`${nodeId}-ib`} checked={data.useInfiniband}
                onChange={(e) => set({ useInfiniband: e.target.checked, resourceConstraints: { ...data.resourceConstraints, require_infiniband: e.target.checked } })}
                className="accent-cyan-400" />
              <label htmlFor={`${nodeId}-ib`} className={LBL}>Require InfiniBand</label>
            </div>

            {/* SkyPilot launch mode — only for tabular/upload */}
            {data.modelCategory !== 'llm' && (
              <div className="space-y-1">
                <p className={LBL}>SkyPilot Launch Mode</p>
                <div className="flex flex-wrap gap-1">
                  <button type="button"
                    className={!data.useManagedJobs ? BTN_ON : BTN_OFF}
                    onClick={() => set({ useManagedJobs: false })}>
                    sky launch --retry-until-up
                  </button>
                  <button type="button"
                    className={data.useManagedJobs ? BTN_ON : BTN_OFF}
                    onClick={() => set({ useManagedJobs: true })}>
                    sky jobs launch (managed)
                  </button>
                </div>
                <p className={LBL}>Direct launch keeps retry-until-up. Managed jobs skip that flag.</p>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Train run ID output */}
      {data.trainRunId && (
        <div className="space-y-0.5">
          <p className={SUB_HEADING}>Train Run ID</p>
          <p className="text-xs font-mono text-emerald-400 break-all">{data.trainRunId}</p>
        </div>
      )}

      {/* Assign existing */}
      <div className="space-y-1">
        <p className={SUB_HEADING}>Assign Existing Train Run ID</p>
        <div className="flex gap-1">
          <input className={INPUT_CLS} placeholder="paste trainRunId"
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
        <button disabled={!canRun || loading} className={BTN_PRIMARY}
          onClick={() => { setLoading(true); runNode(nodeId).finally(() => setLoading(false)); }}>
          {loading ? 'Running…' : 'Launch Training'}
        </button>
        <button disabled={!canRun || loading} className={BTN_NEUTRAL}
          onClick={() => { setLoading(true); runUpTo(nodeId).finally(() => setLoading(false)); }}>
          Run up to here
        </button>
      </div>
    </div>
  );
}
