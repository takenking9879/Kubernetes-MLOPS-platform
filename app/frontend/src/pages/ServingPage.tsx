/**
 * ServingPage — Kafka inference serving configuration.
 *
 * 4-step workflow:
 *   1. Training Run — select train_run_id (auto-resolves dataset)
 *   2. Kafka Schema (raw.yaml) — build & save raw.yaml to S3
 *   3. Serving Config — alias, canary, webhook settings
 *   4. Review & Submit — generates params_serving.yaml via backend
 */

import { useCallback, useEffect, useState } from 'react';
import { Check, Copy, RefreshCw } from 'lucide-react';
import {
  listTrainingRunIds,
  uploadRawSchema,
  submitServingConfig,
  type TrainingRunId,
  type SchemaUploadSingleResult,
  type ServingConfigResult,
} from '../api/platformClient';
import {
  generateRawYamlV2,
  type RawFieldEntry,
} from '../lib/schemaYaml';
import { RawYamlEditor } from '../components/schema/RawYamlEditor';

// ─── Style tokens ─────────────────────────────────────────────────────────────

const INPUT_CLS =
  'rounded bg-slate-800 px-2 py-1 text-xs text-slate-100 outline-none focus:ring-1 focus:ring-blue-500 w-full';
const SELECT_CLS =
  'rounded bg-slate-800 px-2 py-1 text-xs text-slate-100 outline-none focus:ring-1 focus:ring-blue-500 w-full';
const BTN_NEUTRAL =
  'rounded bg-slate-700 px-2 py-1 text-xs text-slate-300 hover:bg-slate-600 disabled:opacity-40';
const BTN_PRIMARY =
  'rounded bg-blue-600 px-4 py-1.5 text-xs font-medium text-white hover:bg-blue-500 disabled:opacity-40';
const SUB_HEADING = 'text-xs font-semibold text-slate-400 uppercase tracking-wider';

// ─── Constants ────────────────────────────────────────────────────────────────

const STEP_LABELS = ['Training Run', 'Kafka Schema', 'Serving Config', 'Review'];

// ─── Primitive UI components ──────────────────────────────────────────────────

function Label({ children, title }: { children: React.ReactNode; title?: string }) {
  return (
    <label className="text-xs text-slate-400" title={title}>
      {children}
    </label>
  );
}

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
      <Label title={tooltip}>
        {label}
        {tooltip && <span className="ml-1 cursor-help text-slate-600">ⓘ</span>}
      </Label>
      {children}
    </div>
  );
}

function ToggleButton({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={`rounded-full px-3 py-1 text-xs font-medium transition-colors ${
        active
          ? 'bg-blue-600 text-white'
          : 'bg-slate-700 text-slate-400 hover:bg-slate-600'
      }`}
    >
      {children}
    </button>
  );
}

// ─── StepperHeader ────────────────────────────────────────────────────────────

function StepperHeader({
  currentStep,
  stepStatus,
  onStepClick,
}: {
  currentStep: number;
  stepStatus: (n: number) => 'active' | 'completed' | 'error' | 'pending';
  onStepClick: (n: number) => void;
}) {
  return (
    <div className="flex w-full select-none items-start gap-0">
      {STEP_LABELS.map((label, i) => {
        const n = i + 1;
        const st = stepStatus(n);
        const clickable = n <= currentStep;

        const nodeStyle =
          st === 'active'
            ? 'bg-blue-600 text-white ring-2 ring-blue-400/30'
            : st === 'completed'
              ? 'bg-green-700 text-white'
              : st === 'error'
                ? 'bg-red-700 text-white'
                : 'bg-slate-700 text-slate-500';

        const labelStyle =
          st === 'active'
            ? 'text-slate-100 font-semibold'
            : st === 'completed'
              ? 'text-slate-400'
              : 'text-slate-600';

        return (
          <div key={n} className="flex flex-1 items-start">
            <div
              className={`flex flex-col items-center ${clickable ? 'cursor-pointer' : 'cursor-default'}`}
              onClick={() => clickable && onStepClick(n)}
            >
              <div
                className={`flex h-7 w-7 items-center justify-center rounded-full text-xs font-bold transition-colors ${nodeStyle}`}
              >
                {st === 'completed' ? (
                  <Check size={13} strokeWidth={3} />
                ) : st === 'error' ? (
                  '!'
                ) : (
                  n
                )}
              </div>
              <span className={`mt-1 text-[10px] transition-colors ${labelStyle}`}>
                {label}
              </span>
            </div>
            {i < STEP_LABELS.length - 1 && (
              <div
                className={`mx-2 mt-3.5 h-px flex-1 transition-colors ${
                  st === 'completed' ? 'bg-green-700' : 'bg-slate-700'
                }`}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}

// ─── SummaryCard ──────────────────────────────────────────────────────────────

function SummaryCard({
  title,
  items,
  onEdit,
}: {
  title: string;
  items: [string, string][];
  onEdit?: () => void;
}) {
  return (
    <div className="rounded-lg border border-slate-700 bg-slate-800/40 px-4 py-3">
      <div className="mb-2 flex items-center justify-between">
        <span className={SUB_HEADING}>{title}</span>
        {onEdit && (
          <button onClick={onEdit} className="text-[10px] text-blue-400 hover:text-blue-300">
            Edit
          </button>
        )}
      </div>
      <div className="grid grid-cols-[140px_1fr] gap-x-3 gap-y-1">
        {items.map(([k, v]) => (
          <div key={k} className="contents">
            <span className="text-[11px] text-slate-500">{k}</span>
            <span className="break-all font-mono text-[11px] text-slate-200">{v || '—'}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── YamlPreviewPanel ─────────────────────────────────────────────────────────

function YamlPreviewPanel({ content, label }: { content: string; label: string }) {
  const [copied, setCopied] = useState(false);
  const handleCopy = () => {
    void navigator.clipboard.writeText(content).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };
  return (
    <div className="flex h-full flex-col overflow-hidden rounded-lg border border-slate-700 bg-slate-900">
      <div className="flex shrink-0 items-center justify-between border-b border-slate-700 px-3 py-2">
        <span className="text-[10px] font-medium text-slate-400">{label}</span>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1 rounded px-2 py-0.5 text-[10px] text-slate-400 hover:bg-slate-800 hover:text-slate-200"
        >
          <Copy size={10} />
          {copied ? 'Copied!' : 'Copy'}
        </button>
      </div>
      <pre className="flex-1 overflow-auto whitespace-pre bg-slate-950 p-3 font-mono text-[11px] leading-relaxed text-green-300">
        {content || '# (empty)'}
      </pre>
    </div>
  );
}

// ─── ServingPage ──────────────────────────────────────────────────────────────

export function ServingPage() {
  // ── Step navigation ───────────────────────────────────────────────────────
  const [currentStep, setCurrentStep] = useState(1);
  const [navErrors, setNavErrors] = useState<string[]>([]);

  // ── Step 1: Training Run ──────────────────────────────────────────────────
  const [trainingRunIds, setTrainingRunIds] = useState<TrainingRunId[]>([]);
  const [trainingIdsLoading, setTrainingIdsLoading] = useState(false);
  const [selectedTrainRunId, setSelectedTrainRunId] = useState('');
  // dataset resolved from the selected TrainingRunId entry (parsed in /ids endpoint)
  const [resolvedDataset, setResolvedDataset] = useState('');

  // ── Step 2: Kafka Schema (raw.yaml) ───────────────────────────────────────
  const [rawFields, setRawFields] = useState<RawFieldEntry[]>([]);
  const [rawGroups, setRawGroups] = useState<Record<string, string>>({
    properties: 'properties',
  });
  const [idField, setIdField] = useState('');
  const [isSavingSchema, setIsSavingSchema] = useState(false);
  const [schemaSaveResult, setSchemaSaveResult] = useState<SchemaUploadSingleResult | null>(null);
  const [schemaSaveError, setSchemaSaveError] = useState('');
  // raw_schema_s3_path required before submitting
  const [rawSchemaS3Path, setRawSchemaS3Path] = useState('');

  // ── Step 3: Serving Config ────────────────────────────────────────────────
  const [alias, setAlias] = useState('champion');
  const [canary, setCanary] = useState(false);
  const [canaryAlias, setCanaryAlias] = useState('challenger');
  const [canaryProbability, setCanaryProbability] = useState(0.1);
  const [initialReplicas, setInitialReplicas] = useState(0);
  const [webhookPublicBaseUrl, setWebhookPublicBaseUrl] = useState('');
  const [webhookPath, setWebhookPath] = useState('/infer/webhook');
  const [webhookMaxTimestampAgeSeconds, setWebhookMaxTimestampAgeSeconds] = useState(300);

  // ── Step 4 / Submission ───────────────────────────────────────────────────
  const [submitResult, setSubmitResult] = useState<ServingConfigResult | null>(null);
  const [submitError, setSubmitError] = useState('');

  // ── Effects ───────────────────────────────────────────────────────────────

  const loadTrainingIds = useCallback(() => {
    setTrainingIdsLoading(true);
    listTrainingRunIds()
      .then((r) => setTrainingRunIds(r.runs))
      .catch(() => setTrainingRunIds([]))
      .finally(() => setTrainingIdsLoading(false));
  }, []);

  useEffect(() => {
    loadTrainingIds();
  }, [loadTrainingIds]);

  // Resolve dataset whenever selected train_run_id changes
  useEffect(() => {
    if (!selectedTrainRunId) {
      setResolvedDataset('');
      return;
    }
    const entry = trainingRunIds.find((r) => r.train_run_id === selectedTrainRunId);
    if (entry?.dataset) {
      setResolvedDataset(entry.dataset);
      return;
    }
    // Fallback: parse from run_id format train-{dataset}-{6hex}-{ts}Z-{6hex}
    const m = selectedTrainRunId.match(/^train-(.+)-[0-9a-f]{6}-\d{8}T\d{6}Z-[0-9a-f]{6}$/);
    setResolvedDataset(m ? m[1] : '');
  }, [selectedTrainRunId, trainingRunIds]);

  // Reset schema save state when dataset or raw fields change
  useEffect(() => {
    setSchemaSaveResult(null);
    setSchemaSaveError('');
    setRawSchemaS3Path('');
  }, [resolvedDataset, rawFields, rawGroups, idField]);

  // ── Derived: raw.yaml content ─────────────────────────────────────────────

  const rawYamlContent =
    rawFields.length > 0
      ? generateRawYamlV2({
          dataset: resolvedDataset,
          dslName: '',
          dslVersion: '',
          processedTableName: '',
          generatedFrom: '',
          allColumns: [],
          rawFields,
          rawGroups,
          idField,
          fullFields: [],
          preprocessedFields: [],
          // V1 compat (unused)
          rawTopLevel: [],
          propertiesFields: [],
          fullColumns: [],
          preprocessedColumns: [],
          targetColumn: '',
          idColumn: '',
          typeOverrides: {},
        })
      : '# Add fields to generate raw.yaml';

  // ── Step validation ────────────────────────────────────────────────────────

  function getStepErrors(step: number): string[] {
    switch (step) {
      case 1: {
        const e: string[] = [];
        if (!selectedTrainRunId) e.push('Select a training run');
        return e;
      }
      case 2: {
        const e: string[] = [];
        if (!rawSchemaS3Path)
          e.push('Save raw.yaml to S3 before continuing');
        if (rawFields.length === 0) e.push('Add at least one field to the schema');
        if (!idField) e.push('Select an id_field');
        return e;
      }
      case 3: {
        const e: string[] = [];
        if (!alias.trim()) e.push('Alias is required');
        if (canary && (canaryProbability < 0 || canaryProbability > 1))
          e.push('Canary probability must be between 0 and 1');
        return e;
      }
      default:
        return [];
    }
  }

  function validateForm(): string[] {
    const errors: string[] = [];
    for (let s = 1; s <= 3; s++) errors.push(...getStepErrors(s));
    return errors;
  }

  // ── Step navigation ────────────────────────────────────────────────────────

  const stepStatus = (n: number): 'active' | 'completed' | 'error' | 'pending' => {
    if (n === currentStep) return 'active';
    if (n > currentStep) return 'pending';
    return getStepErrors(n).length === 0 ? 'completed' : 'error';
  };

  const goForward = () => {
    const errs = getStepErrors(currentStep);
    if (errs.length > 0) { setNavErrors(errs); return; }
    setNavErrors([]);
    setCurrentStep((s) => Math.min(s + 1, 4));
  };

  const goBack = () => {
    setNavErrors([]);
    setCurrentStep((s) => Math.max(s - 1, 1));
  };

  const goToStep = (n: number) => {
    if (n <= currentStep) { setNavErrors([]); setCurrentStep(n); }
  };

  // ── Handlers ──────────────────────────────────────────────────────────────

  const handleSaveSchema = async () => {
    if (!resolvedDataset) return;
    setIsSavingSchema(true);
    setSchemaSaveError('');
    setSchemaSaveResult(null);
    setRawSchemaS3Path('');
    try {
      const result = await uploadRawSchema(resolvedDataset, rawYamlContent);
      setSchemaSaveResult(result);
      setRawSchemaS3Path(result.s3_path);
    } catch (e) {
      setSchemaSaveError(e instanceof Error ? e.message : String(e));
    } finally {
      setIsSavingSchema(false);
    }
  };

  const handleRawEditorChange = (
    fields: RawFieldEntry[],
    groups: Record<string, string>,
    newIdField: string,
  ) => {
    setRawFields(fields);
    setRawGroups(groups);
    setIdField(newIdField);
  };

  const handleSubmit = async () => {
    setSubmitError('');
    setSubmitResult(null);
    const errors = validateForm();
    if (errors.length > 0) return;
    try {
      const result = await submitServingConfig({
        train_run_id: selectedTrainRunId,
        raw_schema_s3_path: rawSchemaS3Path,
        alias,
        canary,
        canary_alias: canaryAlias,
        canary_probability: canaryProbability,
        initial_replicas: initialReplicas,
        webhook_public_base_url: webhookPublicBaseUrl,
        webhook_path: webhookPath,
        webhook_max_timestamp_age_seconds: webhookMaxTimestampAgeSeconds,
      });
      setSubmitResult(result);
    } catch (e) {
      setSubmitError(e instanceof Error ? e.message : String(e));
    }
  };

  const fullValidationErrors = validateForm();

  // ─────────────────────────────────────────────────────────────────────────────
  // RENDER
  // ─────────────────────────────────────────────────────────────────────────────

  return (
    <div className="flex h-full min-h-0 overflow-hidden">
      {/* ── LEFT: scrollable form ── */}
      <div className="min-w-0 flex-1 space-y-3 overflow-y-auto p-4">

        <StepperHeader
          currentStep={currentStep}
          stepStatus={stepStatus}
          onStepClick={goToStep}
        />

        <div className="rounded-lg border border-slate-700 bg-slate-900 px-4 py-4">

          {/* ── STEP 1: Training Run ── */}
          {currentStep === 1 && (
            <div className="space-y-4">
              <p className={SUB_HEADING}>Select Training Run</p>
              <p className="text-xs text-slate-500">
                Select the training run to serve. The dataset is inferred from the run ID.
              </p>

              <Field
                label="Training Run"
                tooltip="The train_run_id whose MLflow model will be deployed."
              >
                <div className="flex gap-1">
                  <select
                    value={selectedTrainRunId}
                    onChange={(e) => setSelectedTrainRunId(e.target.value)}
                    className={SELECT_CLS}
                    disabled={trainingIdsLoading}
                  >
                    <option value="">
                      {trainingIdsLoading ? '— loading… —' : '— select run —'}
                    </option>
                    {trainingRunIds.map((r) => (
                      <option key={r.train_run_id} value={r.train_run_id}>
                        {r.train_run_id}
                        {r.dataset ? ` · ${r.dataset}` : ''}
                      </option>
                    ))}
                  </select>
                  <button
                    onClick={loadTrainingIds}
                    disabled={trainingIdsLoading}
                    className={BTN_NEUTRAL}
                    title="Refresh list"
                  >
                    <RefreshCw size={12} />
                  </button>
                </div>
              </Field>

              {selectedTrainRunId && (
                <div className="rounded border border-slate-700 bg-slate-800/40 px-3 py-2.5">
                  <div className="grid grid-cols-[100px_1fr] gap-x-3 gap-y-1">
                    <span className="text-[11px] text-slate-500">Train run ID</span>
                    <span className="break-all font-mono text-[11px] text-slate-200">
                      {selectedTrainRunId}
                    </span>
                    <span className="text-[11px] text-slate-500">Dataset</span>
                    <span className="font-mono text-[11px] text-slate-200">
                      {resolvedDataset || '—'}
                    </span>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ── STEP 2: Kafka Schema ── */}
          {currentStep === 2 && (
            <div className="space-y-4">
              <p className={SUB_HEADING}>Kafka Input Schema (raw.yaml)</p>
              <p className="text-xs text-slate-500">
                Define the Kafka message schema for{' '}
                <code className="rounded bg-slate-800 px-1 text-slate-300">
                  {resolvedDataset || '(dataset)'}
                </code>
                . The <code className="rounded bg-slate-800 px-1">id_field</code> is read by{' '}
                <code className="rounded bg-slate-800 px-1">kafka_main.py</code> as the message key.
                Load columns from Iceberg or add them manually.
              </p>

              <RawYamlEditor
                fields={rawFields}
                groups={rawGroups}
                idField={idField}
                onChange={handleRawEditorChange}
                dataset={resolvedDataset}
              />

              {/* Save to S3 */}
              <div className="flex items-center gap-3 pt-1">
                <button
                  onClick={() => void handleSaveSchema()}
                  disabled={
                    isSavingSchema ||
                    rawFields.length === 0 ||
                    !idField ||
                    !resolvedDataset
                  }
                  className={BTN_PRIMARY}
                >
                  {isSavingSchema ? 'Saving…' : 'Save raw.yaml to S3'}
                </button>

                {schemaSaveResult && (
                  <span className="rounded-full bg-green-900/50 px-2 py-0.5 text-[10px] text-green-400">
                    ✓ Saved as v{schemaSaveResult.version}
                  </span>
                )}
                {rawSchemaS3Path && (
                  <span
                    className="max-w-[300px] truncate font-mono text-[10px] text-slate-500"
                    title={rawSchemaS3Path}
                  >
                    {rawSchemaS3Path}
                  </span>
                )}
              </div>

              {schemaSaveError && (
                <p className="text-xs text-red-400">{schemaSaveError}</p>
              )}
              {!rawSchemaS3Path && rawFields.length > 0 && idField && (
                <p className="text-xs text-amber-400">
                  Save raw.yaml to S3 before proceeding to the next step.
                </p>
              )}
            </div>
          )}

          {/* ── STEP 3: Serving Config ── */}
          {currentStep === 3 && (
            <div className="space-y-4">
              {/* Alias */}
              <div className="space-y-2">
                <p className={SUB_HEADING}>Model Alias</p>
                <div className="grid grid-cols-2 gap-3">
                  <Field label="Alias" tooltip="MLflow model alias for the champion model.">
                    <input
                      value={alias}
                      onChange={(e) => setAlias(e.target.value)}
                      placeholder="champion"
                      className={INPUT_CLS}
                    />
                  </Field>
                </div>
              </div>

              {/* Canary */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <p className={SUB_HEADING}>Canary Deployment</p>
                  <div className="flex gap-1">
                    <ToggleButton active={canary} onClick={() => setCanary(true)}>
                      Enabled
                    </ToggleButton>
                    <ToggleButton active={!canary} onClick={() => setCanary(false)}>
                      Disabled
                    </ToggleButton>
                  </div>
                </div>

                {canary && (
                  <div className="grid grid-cols-3 gap-3">
                    <Field label="Canary Alias">
                      <input
                        value={canaryAlias}
                        onChange={(e) => setCanaryAlias(e.target.value)}
                        placeholder="challenger"
                        className={INPUT_CLS}
                      />
                    </Field>
                    <Field
                      label="Canary Probability"
                      tooltip="Fraction of traffic routed to the canary model (0–1)."
                    >
                      <input
                        type="number"
                        min={0}
                        max={1}
                        step={0.01}
                        value={canaryProbability}
                        onChange={(e) =>
                          setCanaryProbability(parseFloat(e.target.value) || 0)
                        }
                        className={INPUT_CLS}
                      />
                    </Field>
                    <Field
                      label="Initial Replicas"
                      tooltip="Number of replicas for the canary service. 0 = not deployed yet."
                    >
                      <input
                        type="number"
                        min={0}
                        value={initialReplicas}
                        onChange={(e) =>
                          setInitialReplicas(parseInt(e.target.value) || 0)
                        }
                        className={INPUT_CLS}
                      />
                    </Field>
                  </div>
                )}
              </div>

              {/* Webhook */}
              <div className="space-y-2">
                <p className={SUB_HEADING}>Webhook</p>
                <p className="text-xs text-slate-500">
                  External HTTP endpoint that Kafka inference will POST predictions to.
                </p>
                <div className="grid grid-cols-2 gap-3">
                  <Field label="Public Base URL" tooltip="External base URL of the inference service.">
                    <input
                      value={webhookPublicBaseUrl}
                      onChange={(e) => setWebhookPublicBaseUrl(e.target.value)}
                      placeholder="http://model-serving-serve-svc.ray.svc.cluster.local:8000"
                      className={INPUT_CLS}
                    />
                  </Field>
                  <Field label="Webhook Path">
                    <input
                      value={webhookPath}
                      onChange={(e) => setWebhookPath(e.target.value)}
                      placeholder="/infer/webhook"
                      className={INPUT_CLS}
                    />
                  </Field>
                  <Field
                    label="Max Timestamp Age (s)"
                    tooltip="Reject Kafka messages older than this many seconds."
                  >
                    <input
                      type="number"
                      min={1}
                      value={webhookMaxTimestampAgeSeconds}
                      onChange={(e) =>
                        setWebhookMaxTimestampAgeSeconds(parseInt(e.target.value) || 300)
                      }
                      className={INPUT_CLS}
                    />
                  </Field>
                </div>
              </div>
            </div>
          )}

          {/* ── STEP 4: Review & Submit ── */}
          {currentStep === 4 && (
            <div className="space-y-3">
              <p className={SUB_HEADING}>Review Configuration</p>

              <SummaryCard
                title="Training Run"
                onEdit={() => goToStep(1)}
                items={[
                  ['Train run ID', selectedTrainRunId],
                  ['Dataset', resolvedDataset],
                ]}
              />

              <SummaryCard
                title="Kafka Schema"
                onEdit={() => goToStep(2)}
                items={[
                  ['Fields', String(rawFields.length)],
                  ['id_field', idField],
                  [
                    'Schema version',
                    schemaSaveResult ? `v${schemaSaveResult.version}` : '—',
                  ],
                  ['S3 path', rawSchemaS3Path],
                ]}
              />

              <SummaryCard
                title="Serving Config"
                onEdit={() => goToStep(3)}
                items={[
                  ['Alias', alias],
                  ['Canary', canary ? `Yes (${canaryAlias}, p=${canaryProbability})` : 'No'],
                  ['Webhook path', webhookPath || '—'],
                  [
                    'Max timestamp age',
                    `${webhookMaxTimestampAgeSeconds}s`,
                  ],
                ]}
              />

              {/* Validation */}
              {fullValidationErrors.length > 0 ? (
                <div className="space-y-1 rounded border border-red-800/40 bg-red-900/20 px-3 py-2">
                  <p className="text-xs font-semibold text-red-400">
                    Issues to resolve before submitting:
                  </p>
                  {fullValidationErrors.map((e, i) => (
                    <p key={i} className="text-xs text-red-400">
                      • {e}
                    </p>
                  ))}
                </div>
              ) : (
                <div className="rounded border border-green-800/30 bg-green-900/10 px-3 py-2">
                  <p className="text-xs text-green-400">✓ All configuration valid</p>
                </div>
              )}

              {/* Submit / Result */}
              {submitResult ? (
                <div className="space-y-2 rounded-lg border border-slate-700 bg-slate-800/40 p-4">
                  <p className="text-sm font-medium text-green-400">
                    ✓ Serving config saved
                  </p>
                  <div className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-xs">
                    <span className="text-slate-400">Serve run ID</span>
                    <span className="break-all text-slate-200">
                      {submitResult.serve_run_id}
                    </span>
                    <span className="text-slate-400">Dataset</span>
                    <span className="text-slate-200">{submitResult.dataset}</span>
                    <span className="text-slate-400">Train run ID</span>
                    <span className="break-all text-slate-200">
                      {submitResult.train_run_id}
                    </span>
                    <span className="text-slate-400">params S3 path</span>
                    <span className="break-all text-slate-200">
                      {submitResult.params_s3_path}
                    </span>
                  </div>
                  <button
                    onClick={() => {
                      setSubmitResult(null);
                      setCurrentStep(1);
                      setNavErrors([]);
                    }}
                    className={BTN_NEUTRAL}
                  >
                    New Serving Config
                  </button>
                </div>
              ) : (
                <>
                  {submitError && (
                    <div className="rounded border border-red-800/40 bg-red-900/20 px-3 py-2">
                      <p className="text-xs text-red-400">{submitError}</p>
                    </div>
                  )}
                  <button
                    onClick={() => void handleSubmit()}
                    disabled={fullValidationErrors.length > 0}
                    className="w-full rounded bg-blue-600 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-blue-500 disabled:opacity-40"
                  >
                    Submit Serving Config
                  </button>
                </>
              )}
            </div>
          )}
        </div>

        {/* Navigation errors */}
        {navErrors.length > 0 && (
          <div className="space-y-0.5 rounded border border-red-800/40 bg-red-900/20 px-3 py-2">
            {navErrors.map((e, i) => (
              <p key={i} className="text-xs text-red-400">
                • {e}
              </p>
            ))}
          </div>
        )}

        {/* Navigation buttons */}
        {currentStep < 4 ? (
          <div className="flex items-center justify-between">
            {currentStep > 1 ? (
              <button onClick={goBack} className={BTN_NEUTRAL}>
                ← Back
              </button>
            ) : (
              <div />
            )}
            <button onClick={goForward} className={BTN_PRIMARY}>
              Continue →
            </button>
          </div>
        ) : (
          <div>
            <button onClick={goBack} className={BTN_NEUTRAL}>
              ← Edit Configuration
            </button>
          </div>
        )}
      </div>

      {/* ── RIGHT: raw.yaml preview ── */}
      <div className="w-[420px] shrink-0 min-h-0 p-4">
        <YamlPreviewPanel
          content={rawYamlContent}
          label="raw.yaml preview"
        />
      </div>
    </div>
  );
}
