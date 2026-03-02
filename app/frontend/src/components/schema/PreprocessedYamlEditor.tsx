import { useState } from 'react';
import { Lock, Plus, RefreshCw } from 'lucide-react';
import type { PreprocessedFieldEntry } from '../../lib/schemaYaml';
import type { SparkDataType } from '../../types/schema';
import { getDslFeatures } from '../../api/platformClient';
import { SourceBadge } from './SourceBadge';
import { BTN_NEUTRAL, BTN_PRIMARY, INPUT_CLS, SELECT_CLS, SUB_HEADING } from '../../lib/uiTokens';

const TYPE_OPTIONS: SparkDataType[] = ['double', 'integer', 'long', 'string'];

interface PreprocessedYamlEditorProps {
  fields: PreprocessedFieldEntry[];
  onChange: (next: PreprocessedFieldEntry[]) => void;
  dataset: string;
  dslVersion: number | '';
  dslName?: string;
}

export function PreprocessedYamlEditor({
  fields,
  onChange,
  dataset,
  dslVersion,
  dslName,
}: PreprocessedYamlEditorProps) {
  const [syncing, setSyncing] = useState(false);
  const [syncError, setSyncError] = useState('');
  const [newFeatureName, setNewFeatureName] = useState('');
  const [newFeatureType, setNewFeatureType] = useState<SparkDataType>('double');

  const handleSyncFromDsl = async () => {
    if (!dataset || dslVersion === '') return;
    setSyncing(true);
    setSyncError('');
    try {
      const result = await getDslFeatures(dataset, dslVersion as number);
      const dslEntries: PreprocessedFieldEntry[] = result.finalFeatures.features.map(
        (name) => ({
          name,
          source: 'dsl_derived' as const,
          dslLocked: true,
        }),
      );
      // Keep existing custom entries that aren't in the DSL list
      const dslNames = new Set(dslEntries.map((e) => e.name));
      const customEntries = fields.filter(
        (f) => f.source === 'custom' && !dslNames.has(f.name),
      );
      onChange([...dslEntries, ...customEntries]);
    } catch (e) {
      setSyncError(e instanceof Error ? e.message : String(e));
    } finally {
      setSyncing(false);
    }
  };

  const handleTypeChange = (name: string, type: SparkDataType) => {
    onChange(
      fields.map((f) =>
        f.name === name
          ? { ...f, typeOverride: type, source: f.dslLocked ? 'dsl_derived' : f.source }
          : f,
      ),
    );
  };

  const handleRemove = (name: string) => {
    onChange(fields.filter((f) => f.name !== name));
  };

  const handleAddCustom = () => {
    const trimmed = newFeatureName.trim();
    if (!trimmed) return;
    if (fields.some((f) => f.name === trimmed)) return;
    onChange([
      ...fields,
      {
        name: trimmed,
        typeOverride: newFeatureType,
        source: 'custom',
        dslLocked: false,
      },
    ]);
    setNewFeatureName('');
    setNewFeatureType('double');
  };

  const isTypeOverridden = (f: PreprocessedFieldEntry) =>
    f.typeOverride !== undefined && f.typeOverride !== 'double';

  return (
    <div className="space-y-3">
      {/* Header actions */}
      <div className="flex items-center gap-2">
        <button
          onClick={() => void handleSyncFromDsl()}
          disabled={syncing || !dataset || dslVersion === ''}
          className={BTN_NEUTRAL}
          title={
            dslVersion === ''
              ? 'Select a DSL first'
              : `Sync features from DSL v${dslVersion}`
          }
        >
          <RefreshCw size={11} className={syncing ? 'animate-spin' : ''} />
          <span className="ml-1">
            {syncing ? 'Syncing…' : 'Sync from DSL'}
          </span>
        </button>

        {fields.length > 0 && (
          <span className="text-xs text-slate-500">
            {fields.filter((f) => f.source === 'dsl_derived').length} DSL ·{' '}
            {fields.filter((f) => f.source === 'custom').length} custom
          </span>
        )}

        {syncError && (
          <span className="text-xs text-red-400">{syncError}</span>
        )}
      </div>

      {/* Feature table */}
      {fields.length > 0 && (
        <div className="overflow-hidden rounded border border-slate-800">
          <div className="grid grid-cols-[1fr_100px_80px_28px] bg-slate-800/50 px-3 py-1.5 text-[10px] font-bold uppercase tracking-wider text-slate-400">
            <span>Feature</span>
            <span>Type</span>
            <span>Source</span>
            <span></span>
          </div>

          <div className="divide-y divide-slate-800/60">
            {fields.map((f) => (
              <div
                key={f.name}
                className="grid grid-cols-[1fr_100px_80px_28px] items-center px-3 py-1.5 hover:bg-slate-800/20"
              >
                {/* Name */}
                <div className="flex items-center gap-1.5">
                  <span className="font-mono text-xs text-slate-200">
                    {f.name}
                  </span>
                  {isTypeOverridden(f) && (
                    <span className="rounded bg-orange-900/30 px-1 py-0.5 text-[9px] text-orange-400">
                      overridden
                    </span>
                  )}
                </div>

                {/* Type selector */}
                <select
                  value={f.typeOverride ?? 'double'}
                  onChange={(e) =>
                    handleTypeChange(f.name, e.target.value as SparkDataType)
                  }
                  className="rounded bg-slate-800 px-1 py-0.5 text-[11px] text-slate-200 outline-none focus:ring-1 focus:ring-blue-500"
                >
                  {TYPE_OPTIONS.map((t) => (
                    <option key={t} value={t}>
                      {t}
                    </option>
                  ))}
                </select>

                {/* Source badge */}
                <div>
                  <SourceBadge source={f.source} dslName={dslName} />
                </div>

                {/* Remove / lock */}
                <div className="flex justify-center">
                  {f.dslLocked ? (
                    <span title="DSL-derived — cannot remove">
                      <Lock size={11} className="text-slate-600" />
                    </span>
                  ) : (
                    <button
                      type="button"
                      onClick={() => handleRemove(f.name)}
                      className="text-slate-600 hover:text-red-400"
                      title="Remove feature"
                    >
                      ×
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {fields.length === 0 && (
        <p className="text-xs text-slate-600">
          No features loaded. Click &quot;Sync from DSL&quot; to populate from the
          DSL pipeline&apos;s final_features block.
        </p>
      )}

      {/* Add custom feature */}
      <div className="space-y-1">
        <p className={SUB_HEADING}>Add Custom Feature</p>
        <div className="flex items-center gap-2">
          <input
            value={newFeatureName}
            onChange={(e) => setNewFeatureName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') handleAddCustom();
            }}
            placeholder="feature_name"
            className={`${INPUT_CLS} max-w-[200px]`}
          />
          <select
            value={newFeatureType}
            onChange={(e) => setNewFeatureType(e.target.value as SparkDataType)}
            className={`${SELECT_CLS} max-w-[100px]`}
          >
            {TYPE_OPTIONS.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
          <button
            type="button"
            onClick={handleAddCustom}
            disabled={!newFeatureName.trim()}
            className={BTN_PRIMARY}
          >
            <Plus size={11} className="mr-1 inline" />
            Add
          </button>
        </div>
      </div>
    </div>
  );
}
