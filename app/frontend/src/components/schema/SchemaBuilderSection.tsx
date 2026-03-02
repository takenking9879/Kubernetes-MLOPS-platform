import { useEffect, useState } from 'react';
import type {
  FullFieldEntry,
  PreprocessedFieldEntry,
  RawFieldEntry,
  SchemaBuilderState,
} from '../../lib/schemaYaml';
import { syncLegacyFields, validateSchemaBuilder } from '../../lib/schemaYaml';
import type {
  DslVersion,
  ProcessedTableEntry,
  SchemaUploadResult,
} from '../../api/platformClient';
import { getDslFeatures, listDsls } from '../../api/platformClient';
import { RawYamlEditor } from './RawYamlEditor';
import { FullYamlEditor } from './FullYamlEditor';
import { PreprocessedYamlEditor } from './PreprocessedYamlEditor';
import { BTN_NEUTRAL, BTN_PRIMARY, INPUT_CLS, SELECT_CLS } from '../../lib/uiTokens';

// ─── Types ────────────────────────────────────────────────────────────────────

type ActiveEditor = 'raw' | 'full' | 'preprocessed';

interface SchemaBuilderSectionProps {
  state: SchemaBuilderState;
  onChange: (next: SchemaBuilderState) => void;

  /** Raw dataset name from RunPage rawDatasetFilter */
  inferredDataset: string;
  /** Processing runs loaded in RunPage (for DSL → processed_table mapping) */
  processingRuns: ProcessedTableEntry[];

  onSave: () => Promise<void>;
  onDownload: (content: string, filename: string) => void;
  isSaving: boolean;
  saveResult: SchemaUploadResult | null;

  rawYaml: string;
  fullYaml: string;
  preprocessedYaml: string;
}

// ─── Component ────────────────────────────────────────────────────────────────

export function SchemaBuilderSection({
  state,
  onChange,
  inferredDataset,
  processingRuns,
  onSave,
  onDownload,
  isSaving,
  saveResult,
  rawYaml,
  fullYaml,
  preprocessedYaml,
}: SchemaBuilderSectionProps) {
  const [dslList, setDslList] = useState<DslVersion[]>([]);
  const [dslLoading, setDslLoading] = useState(false);
  const [dslError, setDslError] = useState('');
  const [activeEditor, setActiveEditor] = useState<ActiveEditor>('raw');

  // Load DSL list when dataset changes
  useEffect(() => {
    if (!inferredDataset) {
      setDslList([]);
      return;
    }
    setDslLoading(true);
    setDslError('');
    listDsls(inferredDataset)
      .then((r) => setDslList(r.dsls))
      .catch((e: unknown) =>
        setDslError(e instanceof Error ? e.message : String(e)),
      )
      .finally(() => setDslLoading(false));
  }, [inferredDataset]);

  // ── DSL selection ──────────────────────────────────────────────────────────

  const handleDslSelect = async (version: number) => {
    if (!inferredDataset || !version) return;
    setDslError('');
    try {
      const result = await getDslFeatures(inferredDataset, version);

      // Build preprocessed fields from DSL final_features
      const preprocessedFields: PreprocessedFieldEntry[] =
        result.finalFeatures.features.map((name) => ({
          name,
          source: 'dsl_derived',
          dslLocked: true,
        }));

      // Infer processed_table_name from processing runs matching this dsl_version
      const matchingRun = processingRuns.find(
        (r) => r.dsl_version === version,
      );
      const processedTableName =
        matchingRun?.processed_table_name ?? state.processedTableName;

      const generatedFrom = `${result.slug} / iceberg.raw.${inferredDataset}`;

      const next: SchemaBuilderState = syncLegacyFields({
        ...state,
        dataset: inferredDataset,
        dslName: result.slug,
        dslVersion: version,
        processedTableName,
        generatedFrom,
        preprocessedFields,
      });
      onChange(next);
    } catch (e) {
      setDslError(e instanceof Error ? e.message : String(e));
    }
  };

  // ── Field update helpers ───────────────────────────────────────────────────

  const handleRawChange = (
    fields: RawFieldEntry[],
    groups: Record<string, string>,
    idField: string,
  ) => {
    onChange(
      syncLegacyFields({ ...state, rawFields: fields, rawGroups: groups, idField }),
    );
  };

  const handleFullChange = (fields: FullFieldEntry[]) => {
    onChange(syncLegacyFields({ ...state, fullFields: fields }));
  };

  const handlePreprocessedChange = (fields: PreprocessedFieldEntry[]) => {
    onChange(syncLegacyFields({ ...state, preprocessedFields: fields }));
  };

  // ── Validation ─────────────────────────────────────────────────────────────

  const validationErrors = validateSchemaBuilder(state);

  // ── Tab labels with counts ─────────────────────────────────────────────────

  const tabLabel = (
    key: ActiveEditor,
    label: string,
    count: number,
  ) => (
    <button
      key={key}
      onClick={() => setActiveEditor(key)}
      className={`rounded px-3 py-1 text-xs font-medium transition-colors ${
        activeEditor === key
          ? 'bg-slate-700 text-slate-100'
          : 'text-slate-500 hover:text-slate-300'
      }`}
    >
      {label}
      {count > 0 && (
        <span className="ml-1.5 rounded bg-slate-600/60 px-1 py-0.5 text-[9px]">
          {count}
        </span>
      )}
    </button>
  );

  return (
    <div className="space-y-4">
      <p className="text-xs text-slate-500">
        Generate versioned schema YAMLs. Select a DSL to populate features;
        then configure each schema layer (raw / full / preprocessed) independently.
        Saved to{' '}
        <code className="rounded bg-slate-800 px-1">
          s3://.../schemas/datasets/{'{name}'}/v{'{N}'}/
        </code>
      </p>

      {/* ── DSL Name selector ── */}
      <div className="flex flex-wrap items-start gap-3">
        <div className="flex-1 min-w-[200px] space-y-1">
          <label className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">
            DSL Name
          </label>
          {!inferredDataset ? (
            <p className="text-xs text-slate-600">
              Select a Raw Dataset in Step 1 first.
            </p>
          ) : dslLoading ? (
            <p className="text-xs text-slate-500">Loading DSLs…</p>
          ) : dslList.length === 0 ? (
            <p className="text-xs text-slate-600">
              No DSLs found for <code className="text-slate-400">{inferredDataset}</code>.
            </p>
          ) : (
            <select
              value={state.dslVersion === '' ? '' : String(state.dslVersion)}
              onChange={(e) => {
                const v = Number(e.target.value);
                if (v) void handleDslSelect(v);
              }}
              className={SELECT_CLS}
            >
              <option value="">— select DSL —</option>
              {dslList.map((d) => (
                <option key={d.version} value={String(d.version)}>
                  v{d.version} · {d.slug}
                </option>
              ))}
            </select>
          )}
          {dslError && (
            <p className="text-xs text-red-400">{dslError}</p>
          )}
        </div>

        {/* Inferred processed_table_name */}
        <div className="flex-1 min-w-[200px] space-y-1">
          <label className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">
            Processed Table{' '}
            <span className="font-normal text-slate-600">(inferred, editable)</span>
          </label>
          <input
            value={state.processedTableName}
            onChange={(e) =>
              onChange({ ...state, processedTableName: e.target.value })
            }
            placeholder="iceberg.processed.<dataset>_<exec_id>"
            className={INPUT_CLS}
          />
        </div>
      </div>

      {/* Provenance indicator */}
      {state.generatedFrom && (
        <p className="text-[10px] text-slate-600">
          Schema provenance:{' '}
          <code className="rounded bg-slate-800/60 px-1 text-slate-400">
            {state.generatedFrom}
          </code>
        </p>
      )}

      {/* ── Editor tabs ── */}
      <div className="flex items-center gap-1 border-b border-slate-800 pb-2">
        {tabLabel('raw', 'raw.yaml', state.rawFields.length)}
        {tabLabel('full', 'full.yaml', state.fullFields.length)}
        {tabLabel(
          'preprocessed',
          'preprocessed.yaml',
          state.preprocessedFields.length,
        )}
      </div>

      {/* ── Active editor ── */}
      <div>
        {activeEditor === 'raw' && (
          <RawYamlEditor
            fields={state.rawFields}
            groups={state.rawGroups}
            idField={state.idField}
            onChange={handleRawChange}
            dataset={state.dataset || inferredDataset}
            yamlPreview={rawYaml}
          />
        )}

        {activeEditor === 'full' && (
          <FullYamlEditor
            fields={state.fullFields}
            onChange={handleFullChange}
            dataset={state.dataset || inferredDataset}
          />
        )}

        {activeEditor === 'preprocessed' && (
          <PreprocessedYamlEditor
            fields={state.preprocessedFields}
            onChange={handlePreprocessedChange}
            dataset={state.dataset || inferredDataset}
            dslVersion={state.dslVersion}
            dslName={state.dslName}
          />
        )}
      </div>

      {/* ── Validation errors ── */}
      {validationErrors.length > 0 && (
        <div className="space-y-0.5 rounded border border-red-800/40 bg-red-900/20 px-3 py-2">
          {validationErrors.map((e, i) => (
            <p key={i} className="text-xs text-red-400">
              {e}
            </p>
          ))}
        </div>
      )}

      {/* ── Actions ── */}
      <div className="flex flex-wrap items-center gap-2 pt-1 border-t border-slate-800">
        <button
          onClick={() => onDownload(rawYaml, 'raw.yaml')}
          disabled={state.rawFields.length === 0 && state.rawTopLevel.length === 0}
          className={BTN_NEUTRAL}
        >
          ↓ raw.yaml
        </button>
        <button
          onClick={() => onDownload(fullYaml, 'full.yaml')}
          disabled={state.fullFields.length === 0 && state.fullColumns.length === 0}
          className={BTN_NEUTRAL}
        >
          ↓ full.yaml
        </button>
        <button
          onClick={() => onDownload(preprocessedYaml, 'preprocessed.yaml')}
          disabled={
            state.preprocessedFields.length === 0 &&
            state.preprocessedColumns.length === 0
          }
          className={BTN_NEUTRAL}
        >
          ↓ preprocessed.yaml
        </button>
        <button
          onClick={() => void onSave()}
          disabled={
            isSaving ||
            !inferredDataset ||
            (state.rawFields.length === 0 &&
              state.fullFields.length === 0 &&
              state.preprocessedFields.length === 0)
          }
          className={BTN_PRIMARY}
        >
          {isSaving ? 'Saving…' : 'Save schemas to S3'}
        </button>

        {saveResult && (
          <span className="text-xs text-green-400">
            ✓ v{saveResult.version} saved
          </span>
        )}
      </div>

      {/* ── Save result details ── */}
      {saveResult && (
        <div className="space-y-1 rounded border border-green-800/40 bg-green-900/10 px-3 py-2 text-xs text-green-400">
          <p className="font-medium">Schemas saved — version {saveResult.version}</p>
          {Object.entries(saveResult.uploaded).map(([k, v]) => (
            <p key={k} className="text-slate-400">
              <span className="text-green-500">{k}:</span> {v}
            </p>
          ))}
        </div>
      )}
    </div>
  );
}
