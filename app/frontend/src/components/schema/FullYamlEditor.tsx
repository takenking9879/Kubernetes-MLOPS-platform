import { useState } from 'react';
import { RefreshCw, Pencil, Check, X } from 'lucide-react';
import type { FullFieldEntry } from '../../lib/schemaYaml';
import type { SparkDataType } from '../../types/schema';
import { getIcebergSchema } from '../../api/platformClient';
import { SourceBadge } from './SourceBadge';
import { BTN_NEUTRAL, BTN_PRIMARY, INPUT_CLS, SELECT_CLS } from '../../lib/uiTokens';

const SPARK_TYPES: SparkDataType[] = [
  'string',
  'integer',
  'long',
  'double',
  'boolean',
  'timestamp',
];

interface FullYamlEditorProps {
  fields: FullFieldEntry[];
  onChange: (next: FullFieldEntry[]) => void;
  dataset: string;
}

interface EditState {
  editedName: string;
  editedType: SparkDataType;
  note: string;
}

export function FullYamlEditor({ fields, onChange, dataset }: FullYamlEditorProps) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [editingRow, setEditingRow] = useState<string | null>(null);
  const [editState, setEditState] = useState<EditState>({
    editedName: '',
    editedType: 'string',
    note: '',
  });
  // Names of fields that would change on re-sync (for diff warning)
  const [resyncDiff, setResyncDiff] = useState<string[]>([]);
  const [showResyncConfirm, setShowResyncConfirm] = useState(false);
  // Pending Iceberg columns for re-sync
  const [pendingIcebergCols, setPendingIcebergCols] = useState<FullFieldEntry[]>([]);

  const mapToFullFields = (cols: { name: string; sparkType: string; nullable: boolean }[]): FullFieldEntry[] =>
    cols.map((c) => ({
      name: c.name,
      sparkType: c.sparkType as SparkDataType,
      nullable: c.nullable,
      source: 'iceberg' as const,
    }));

  const handleLoadFromIceberg = async () => {
    if (!dataset) return;
    setLoading(true);
    setError('');
    try {
      const result = await getIcebergSchema(dataset);
      const incoming = mapToFullFields(result.columns);

      if (fields.length === 0) {
        onChange(incoming);
      } else {
        // Compute diff: incoming names that conflict with user_override rows
        const overrideNames = new Set(
          fields
            .filter((f) => f.source === 'user_override')
            .map((f) => f.editedName ?? f.name),
        );
        const icebergNames = new Set(incoming.map((c) => c.name));
        const conflicts = [...overrideNames].filter((n) => icebergNames.has(n));

        if (conflicts.length > 0) {
          setPendingIcebergCols(incoming);
          setResyncDiff(conflicts);
          setShowResyncConfirm(true);
        } else {
          // No conflicts — just replace
          onChange(incoming);
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const handleResyncConfirm = () => {
    // Merge: keep user_override rows, replace iceberg rows
    const byName = new Map(pendingIcebergCols.map((c) => [c.name, c]));
    const merged: FullFieldEntry[] = [];

    for (const f of fields) {
      const effectiveName = f.editedName ?? f.name;
      if (f.source === 'user_override') {
        // Keep override unchanged
        merged.push(f);
        byName.delete(effectiveName);
      } else if (byName.has(f.name)) {
        // Replace with fresh Iceberg version
        merged.push(byName.get(f.name)!);
        byName.delete(f.name);
      }
    }
    // Add any new Iceberg columns not in the current list
    for (const c of byName.values()) {
      merged.push(c);
    }

    onChange(merged);
    setShowResyncConfirm(false);
    setPendingIcebergCols([]);
    setResyncDiff([]);
  };

  const handleResyncCancel = () => {
    setShowResyncConfirm(false);
    setPendingIcebergCols([]);
    setResyncDiff([]);
  };

  const startEdit = (f: FullFieldEntry) => {
    setEditingRow(f.name);
    setEditState({
      editedName: f.editedName ?? f.name,
      editedType: (f.editedType ?? f.sparkType) as SparkDataType,
      note: f.note ?? '',
    });
  };

  const commitEdit = (originalName: string) => {
    onChange(
      fields.map((f) => {
        if (f.name !== originalName) return f;
        const changed =
          editState.editedName !== f.name ||
          editState.editedType !== f.sparkType;
        return {
          ...f,
          editedName:
            editState.editedName !== f.name ? editState.editedName : undefined,
          editedType:
            editState.editedType !== f.sparkType
              ? editState.editedType
              : undefined,
          note: editState.note || undefined,
          source: changed ? ('user_override' as const) : f.source,
        };
      }),
    );
    setEditingRow(null);
  };

  const cancelEdit = () => setEditingRow(null);

  return (
    <div className="space-y-3">
      {/* Header actions */}
      <div className="flex flex-wrap items-center gap-2">
        <button
          onClick={() => void handleLoadFromIceberg()}
          disabled={loading || !dataset}
          className={BTN_NEUTRAL}
        >
          <RefreshCw
            size={11}
            className={`mr-1 inline ${loading ? 'animate-spin' : ''}`}
          />
          {fields.length === 0 ? 'Load from Iceberg' : 'Re-sync from Iceberg'}
        </button>

        {fields.length > 0 && (
          <span className="text-xs text-slate-500">
            {fields.length} columns ·{' '}
            {fields.filter((f) => f.source === 'user_override').length} overridden
          </span>
        )}

        {error && <span className="text-xs text-red-400">{error}</span>}
      </div>

      {/* Re-sync diff warning */}
      {showResyncConfirm && (
        <div className="rounded border border-amber-700/40 bg-amber-900/20 px-3 py-2 text-xs">
          <p className="font-medium text-amber-300">
            Re-sync warning: {resyncDiff.length} user-overridden field
            {resyncDiff.length !== 1 ? 's' : ''} will revert to Iceberg types:
          </p>
          <p className="mt-1 font-mono text-amber-400/80">
            {resyncDiff.join(', ')}
          </p>
          <div className="mt-2 flex gap-2">
            <button onClick={handleResyncConfirm} className={BTN_PRIMARY}>
              Confirm re-sync
            </button>
            <button onClick={handleResyncCancel} className={BTN_NEUTRAL}>
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Column table */}
      {fields.length > 0 && (
        <div className="overflow-hidden rounded border border-slate-800">
          <div className="grid grid-cols-[1fr_100px_60px_80px_44px] bg-slate-800/50 px-3 py-1.5 text-[10px] font-bold uppercase tracking-wider text-slate-400">
            <span>Column</span>
            <span>Type</span>
            <span>Nullable</span>
            <span>Source</span>
            <span></span>
          </div>

          <div className="divide-y divide-slate-800/60">
            {fields.map((f) => {
              const isEditing = editingRow === f.name;
              const displayName = f.editedName ?? f.name;
              const displayType = f.editedType ?? f.sparkType;

              return (
                <div
                  key={f.name}
                  className="divide-y divide-transparent"
                >
                  {isEditing ? (
                    /* Edit row */
                    <div className="bg-slate-800/40 px-3 py-2 space-y-2">
                      <div className="grid grid-cols-[1fr_100px_60px_80px_44px] items-center gap-2">
                        <input
                          value={editState.editedName}
                          onChange={(e) =>
                            setEditState((s) => ({
                              ...s,
                              editedName: e.target.value,
                            }))
                          }
                          className={INPUT_CLS}
                          autoFocus
                        />
                        <select
                          value={editState.editedType}
                          onChange={(e) =>
                            setEditState((s) => ({
                              ...s,
                              editedType: e.target.value as SparkDataType,
                            }))
                          }
                          className={SELECT_CLS}
                        >
                          {SPARK_TYPES.map((t) => (
                            <option key={t} value={t}>
                              {t}
                            </option>
                          ))}
                        </select>
                        <span className="text-xs text-slate-500">
                          {f.nullable ? 'yes' : 'no'}
                        </span>
                        <SourceBadge source="user_override" />
                        <div className="flex gap-1">
                          <button
                            type="button"
                            onClick={() => commitEdit(f.name)}
                            className="text-green-400 hover:text-green-300"
                            title="Save"
                          >
                            <Check size={13} />
                          </button>
                          <button
                            type="button"
                            onClick={cancelEdit}
                            className="text-slate-500 hover:text-slate-300"
                            title="Cancel"
                          >
                            <X size={13} />
                          </button>
                        </div>
                      </div>
                      <input
                        value={editState.note}
                        onChange={(e) =>
                          setEditState((s) => ({ ...s, note: e.target.value }))
                        }
                        placeholder="Note (optional, not written to YAML)"
                        className={`${INPUT_CLS} text-[11px] text-slate-400`}
                      />
                    </div>
                  ) : (
                    /* View row */
                    <div className="grid grid-cols-[1fr_100px_60px_80px_44px] items-center px-3 py-1.5 hover:bg-slate-800/20">
                      <div className="flex items-center gap-1.5">
                        <span className="font-mono text-xs text-slate-200">
                          {displayName}
                        </span>
                        {displayName !== f.name && (
                          <span className="text-[9px] text-slate-500 line-through">
                            {f.name}
                          </span>
                        )}
                        {f.note && (
                          <span
                            className="cursor-help text-[9px] text-slate-600"
                            title={f.note}
                          >
                            ℹ
                          </span>
                        )}
                      </div>
                      <span className="font-mono text-xs text-slate-400">
                        {displayType}
                      </span>
                      <span className="text-xs text-slate-600">
                        {f.nullable ? 'yes' : 'no'}
                      </span>
                      <SourceBadge source={f.source} />
                      <div className="flex justify-center">
                        <button
                          type="button"
                          onClick={() => startEdit(f)}
                          className="text-slate-600 hover:text-blue-400"
                          title="Edit column"
                          aria-label={`Edit column ${f.name}`}
                        >
                          <Pencil size={11} />
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {fields.length === 0 && (
        <p className="text-xs text-slate-600">
          No columns loaded. Click &quot;Load from Iceberg&quot; to read the schema
          from <code className="rounded bg-slate-800 px-1">iceberg.raw.{dataset || '<dataset>'}</code>.
        </p>
      )}
    </div>
  );
}
