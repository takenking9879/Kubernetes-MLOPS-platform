import { useRef, useState } from 'react';
import { ChevronDown, ChevronRight, GripVertical, Plus, RefreshCw } from 'lucide-react';
import type { RawFieldEntry, RawFieldRole } from '../../lib/schemaYaml';
import type { SparkDataType } from '../../types/schema';
import { getIcebergSchema } from '../../api/platformClient';
import { SourceBadge } from './SourceBadge';
import {
  BTN_NEUTRAL,
  BTN_PRIMARY,
  INPUT_CLS,
  SELECT_CLS,
  SUB_HEADING,
} from '../../lib/uiTokens';

// ─── Constants ────────────────────────────────────────────────────────────────

const SPARK_TYPES: SparkDataType[] = [
  'string', 'integer', 'long', 'double', 'boolean', 'timestamp',
];

const NUMERIC_TYPES = new Set<SparkDataType>(['integer', 'long', 'double']);
const CATEGORICAL_TYPES = new Set<SparkDataType>(['string', 'boolean']);

const ROLES: RawFieldRole[] = [
  'unassigned', 'feature', 'target', 'id', 'metadata',
];

// ─── Types ────────────────────────────────────────────────────────────────────

interface RawYamlEditorProps {
  fields: RawFieldEntry[];
  groups: Record<string, string>;
  idField: string;
  onChange: (
    fields: RawFieldEntry[],
    groups: Record<string, string>,
    idField: string,
  ) => void;
  dataset: string;
  /** Live YAML preview string — computed by RunPage useMemo, passed in. */
  yamlPreview: string;
}

// ─── Component ────────────────────────────────────────────────────────────────

export function RawYamlEditor({
  fields,
  groups,
  idField,
  onChange,
  dataset,
  yamlPreview,
}: RawYamlEditorProps) {
  // Drag-and-drop refs (avoid re-renders during dragOver at 60fps)
  const dragSrcRef = useRef<number | null>(null);
  const dragTargetRef = useRef<number | null>(null);
  const dragGroupRef = useRef<string | null>(null);

  // Multi-select
  const [selectedNames, setSelectedNames] = useState<Set<string>>(new Set());
  const [patternInput, setPatternInput] = useState('');
  const [bulkTargetGroup, setBulkTargetGroup] = useState('');
  const [newGroupName, setNewGroupName] = useState('');

  // Custom column add
  const [newColName, setNewColName] = useState('');
  const [newColType, setNewColType] = useState<SparkDataType>('string');
  const [newColGroup, setNewColGroup] = useState('');

  // Group collapse state
  const [collapsedGroups, setCollapsedGroups] = useState<Set<string>>(
    new Set(),
  );

  // Iceberg load
  const [loadingIceberg, setLoadingIceberg] = useState(false);
  const [loadError, setLoadError] = useState('');

  // ── Handlers ────────────────────────────────────────────────────────────────

  const handleLoadFromIceberg = async () => {
    if (!dataset) return;
    setLoadingIceberg(true);
    setLoadError('');
    try {
      const result = await getIcebergSchema(dataset);
      const incoming: RawFieldEntry[] = result.columns.map((c) => ({
        name: c.name,
        sparkType: c.sparkType as SparkDataType,
        nullable: c.nullable,
        role: 'unassigned',
        source: 'iceberg',
      }));
      // Only add columns not already in fields
      const existingNames = new Set(fields.map((f) => f.name));
      const toAdd = incoming.filter((f) => !existingNames.has(f.name));
      onChange([...fields, ...toAdd], groups, idField);
    } catch (e) {
      setLoadError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoadingIceberg(false);
    }
  };

  const updateField = (name: string, patch: Partial<RawFieldEntry>) => {
    onChange(
      fields.map((f) => (f.name === name ? { ...f, ...patch } : f)),
      groups,
      idField,
    );
  };

  const removeField = (name: string) => {
    onChange(
      fields.filter((f) => f.name !== name),
      groups,
      idField,
    );
  };

  const addCustomColumn = () => {
    const trimmed = newColName.trim();
    if (!trimmed || fields.some((f) => f.name === trimmed)) return;
    const newField: RawFieldEntry = {
      name: trimmed,
      sparkType: newColType,
      nullable: true,
      role: 'unassigned',
      source: 'custom',
      group: newColGroup || undefined,
    };
    onChange([...fields, newField], groups, idField);
    setNewColName('');
    setNewColType('string');
    setNewColGroup('');
  };

  const addGroup = () => {
    const trimmed = newGroupName.trim().toLowerCase().replace(/\s+/g, '_');
    if (!trimmed || groups[trimmed]) return;
    onChange(fields, { ...groups, [trimmed]: trimmed }, idField);
    setNewGroupName('');
  };

  const removeGroup = (key: string) => {
    // Move group members back to top-level
    const updated = fields.map((f) =>
      f.group === key ? { ...f, group: undefined } : f,
    );
    const { [key]: _removed, ...rest } = groups;
    onChange(updated, rest, idField);
  };

  const toggleCollapse = (key: string) => {
    setCollapsedGroups((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  // ── Selection helpers ────────────────────────────────────────────────────────

  const toggleSelect = (name: string) => {
    setSelectedNames((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  };

  const selectAll = () => setSelectedNames(new Set(fields.map((f) => f.name)));
  const selectNone = () => setSelectedNames(new Set());

  const selectByType = (types: Set<SparkDataType>) => {
    setSelectedNames(
      new Set(
        fields.filter((f) => types.has(f.sparkType)).map((f) => f.name),
      ),
    );
  };

  const selectByPattern = () => {
    if (!patternInput) return;
    try {
      const re = new RegExp(patternInput, 'i');
      setSelectedNames(
        new Set(fields.filter((f) => re.test(f.name)).map((f) => f.name)),
      );
    } catch {
      // invalid regex — ignore
    }
  };

  const deselectByPattern = () => {
    if (!patternInput) return;
    try {
      const re = new RegExp(patternInput, 'i');
      setSelectedNames((prev) => {
        const next = new Set(prev);
        for (const f of fields) {
          if (re.test(f.name)) next.delete(f.name);
        }
        return next;
      });
    } catch {
      // invalid regex — ignore
    }
  };

  const moveSelectedToGroup = (groupKey: string) => {
    if (!groupKey) return;
    onChange(
      fields.map((f) =>
        selectedNames.has(f.name) ? { ...f, group: groupKey } : f,
      ),
      groups,
      idField,
    );
    setSelectedNames(new Set());
    setBulkTargetGroup('');
  };

  const createGroupAndMoveSelected = () => {
    const trimmed = newGroupName.trim().toLowerCase().replace(/\s+/g, '_');
    if (!trimmed) return;
    const newGroups = { ...groups, [trimmed]: trimmed };
    onChange(
      fields.map((f) =>
        selectedNames.has(f.name) ? { ...f, group: trimmed } : f,
      ),
      newGroups,
      idField,
    );
    setSelectedNames(new Set());
    setNewGroupName('');
  };

  // ── Drag-and-drop ───────────────────────────────────────────────────────────

  const handleDragStart = (e: React.DragEvent, index: number) => {
    dragSrcRef.current = index;
    dragGroupRef.current = null;
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/plain', fields[index]?.name ?? '');
  };

  const handleDragOver = (e: React.DragEvent, index: number) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    dragTargetRef.current = index;
    dragGroupRef.current = null;
  };

  const handleDragOverGroup = (e: React.DragEvent, groupKey: string) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    dragGroupRef.current = groupKey;
    dragTargetRef.current = null;
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const src = dragSrcRef.current;

    if (dragGroupRef.current !== null) {
      // Drop onto group header → move field into that group
      const groupKey = dragGroupRef.current;
      if (src !== null) {
        const updated = fields.map((f, i) =>
          i === src ? { ...f, group: groupKey } : f,
        );
        onChange(updated, groups, idField);
      }
    } else if (dragTargetRef.current !== null) {
      const tgt = dragTargetRef.current;
      if (src !== null && src !== tgt) {
        const next = [...fields];
        const moved = next.splice(src, 1)[0];
        if (moved !== undefined) {
          next.splice(tgt, 0, moved);
          onChange(next, groups, idField);
        }
      }
    }

    dragSrcRef.current = null;
    dragTargetRef.current = null;
    dragGroupRef.current = null;
  };

  const handleDragEnd = () => {
    dragSrcRef.current = null;
    dragTargetRef.current = null;
    dragGroupRef.current = null;
  };

  // ── Render helpers ───────────────────────────────────────────────────────────

  const renderFieldRow = (f: RawFieldEntry, index: number) => {
    const isSelected = selectedNames.has(f.name);

    return (
      <div
        key={f.name}
        role="row"
        tabIndex={0}
        aria-label={`Field ${f.name}, type ${f.sparkType}, role ${f.role}`}
        draggable
        onDragStart={(e) => handleDragStart(e, index)}
        onDragOver={(e) => handleDragOver(e, index)}
        onDrop={handleDrop}
        onDragEnd={handleDragEnd}
        onKeyDown={(e) => {
          if (e.key === ' ') { e.preventDefault(); toggleSelect(f.name); }
        }}
        className={`grid grid-cols-[20px_20px_1fr_90px_110px_80px_24px] items-center gap-1 px-2 py-1.5 hover:bg-slate-800/20 focus:outline-none focus:bg-slate-800/30 ${
          isSelected ? 'bg-blue-900/10' : ''
        } ${f.group ? 'pl-6' : ''}`}
      >
        {/* Drag handle */}
        <GripVertical
          size={12}
          className="cursor-grab text-slate-700 active:cursor-grabbing"
        />

        {/* Checkbox */}
        <input
          type="checkbox"
          checked={isSelected}
          onChange={() => toggleSelect(f.name)}
          className="accent-blue-500"
          aria-label={`Select ${f.name}`}
        />

        {/* Name */}
        <span className="truncate font-mono text-xs text-slate-200">
          {f.name}
        </span>

        {/* Type */}
        <select
          value={f.sparkType}
          onChange={(e) =>
            updateField(f.name, {
              sparkType: e.target.value as SparkDataType,
              source: 'user_override',
            })
          }
          className="rounded bg-slate-800 px-1 py-0.5 text-[11px] text-slate-300 outline-none focus:ring-1 focus:ring-blue-500"
          onClick={(e) => e.stopPropagation()}
        >
          {SPARK_TYPES.map((t) => (
            <option key={t} value={t}>
              {t}
            </option>
          ))}
        </select>

        {/* Role */}
        <select
          value={f.role}
          onChange={(e) =>
            updateField(f.name, { role: e.target.value as RawFieldRole })
          }
          className="rounded bg-slate-800 px-1 py-0.5 text-[11px] text-slate-300 outline-none focus:ring-1 focus:ring-blue-500"
          onClick={(e) => e.stopPropagation()}
        >
          {ROLES.map((r) => (
            <option key={r} value={r}>
              {r}
            </option>
          ))}
        </select>

        {/* Source badge */}
        <SourceBadge source={f.source} />

        {/* Remove */}
        <button
          type="button"
          onClick={() => removeField(f.name)}
          className="text-slate-700 hover:text-red-400"
          title={`Remove ${f.name}`}
          aria-label={`Remove ${f.name}`}
        >
          ×
        </button>
      </div>
    );
  };

  // Build ordered render list: top-level fields, then each group block
  const topLevelFields = fields.filter((f) => !f.group);
  const topLevelIndices = topLevelFields.map((f) => fields.indexOf(f));

  return (
    <div className="space-y-3">
      {/* Header bar */}
      <div className="flex flex-wrap items-center gap-2">
        <button
          onClick={() => void handleLoadFromIceberg()}
          disabled={loadingIceberg || !dataset}
          className={BTN_NEUTRAL}
        >
          <RefreshCw
            size={11}
            className={`mr-1 inline ${loadingIceberg ? 'animate-spin' : ''}`}
          />
          {fields.length === 0 ? 'Load from Iceberg' : 'Merge from Iceberg'}
        </button>

        {fields.length > 0 && (
          <span className="text-xs text-slate-500">
            {fields.filter((f) => !f.group).length} top-level ·{' '}
            {Object.keys(groups).map(
              (g) => `${fields.filter((f) => f.group === g).length} in ${g}`,
            ).join(' · ')}
          </span>
        )}

        {loadError && (
          <span className="text-xs text-red-400">{loadError}</span>
        )}
      </div>

      {/* id_field selector */}
      {fields.length > 0 && (
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-500">id_field:</span>
          <select
            value={idField}
            onChange={(e) => onChange(fields, groups, e.target.value)}
            className={`${SELECT_CLS} max-w-[200px]`}
          >
            <option value="">— none —</option>
            {fields
              .filter((f) => !f.group)
              .map((f) => (
                <option key={f.name} value={f.name}>
                  {f.name}
                </option>
              ))}
          </select>
          <span className="text-[10px] text-slate-600">
            (read by kafka_main.py as message key)
          </span>
        </div>
      )}

      {/* Main grid: tree editor + YAML preview */}
      <div
        className={
          fields.length > 0
            ? 'grid grid-cols-[1fr_280px] gap-3'
            : 'space-y-2'
        }
      >
        {/* ── Left: tree editor ── */}
        <div className="space-y-2">
          {/* Bulk selection toolbar */}
          {fields.length > 0 && (
            <div className="flex flex-wrap items-center gap-1.5 rounded border border-slate-800 bg-slate-800/30 px-2 py-1.5">
              <span className="text-[10px] text-slate-500 font-medium uppercase tracking-wider mr-1">
                Select:
              </span>
              <button onClick={selectAll} className={`${BTN_NEUTRAL} py-0.5 text-[10px]`}>
                All
              </button>
              <button onClick={selectNone} className={`${BTN_NEUTRAL} py-0.5 text-[10px]`}>
                None
              </button>
              <button
                onClick={() => selectByType(NUMERIC_TYPES)}
                className={`${BTN_NEUTRAL} py-0.5 text-[10px]`}
              >
                Numeric
              </button>
              <button
                onClick={() => selectByType(CATEGORICAL_TYPES)}
                className={`${BTN_NEUTRAL} py-0.5 text-[10px]`}
              >
                Categorical
              </button>
              <div className="flex items-center gap-1">
                <input
                  value={patternInput}
                  onChange={(e) => setPatternInput(e.target.value)}
                  placeholder="regex…"
                  className="rounded bg-slate-800 px-1.5 py-0.5 text-[11px] text-slate-300 outline-none w-[80px] focus:ring-1 focus:ring-blue-500"
                />
                <button
                  onClick={selectByPattern}
                  className={`${BTN_NEUTRAL} py-0.5 text-[10px]`}
                  title="Select by pattern"
                >
                  + match
                </button>
                <button
                  onClick={deselectByPattern}
                  className={`${BTN_NEUTRAL} py-0.5 text-[10px]`}
                  title="Deselect by pattern"
                >
                  − match
                </button>
              </div>

              {selectedNames.size > 0 && (
                <>
                  <span className="text-[10px] text-blue-400 ml-1">
                    {selectedNames.size} selected →
                  </span>
                  <div className="flex items-center gap-1">
                    <select
                      value={bulkTargetGroup}
                      onChange={(e) => setBulkTargetGroup(e.target.value)}
                      className="rounded bg-slate-800 px-1 py-0.5 text-[11px] text-slate-300 outline-none focus:ring-1 focus:ring-blue-500"
                    >
                      <option value="">Move to group…</option>
                      {Object.keys(groups).map((g) => (
                        <option key={g} value={g}>
                          {g}
                        </option>
                      ))}
                    </select>
                    <button
                      onClick={() => moveSelectedToGroup(bulkTargetGroup)}
                      disabled={!bulkTargetGroup}
                      className={`${BTN_PRIMARY} py-0.5 text-[10px]`}
                    >
                      Move
                    </button>
                  </div>
                </>
              )}
            </div>
          )}

          {/* Column table header */}
          {fields.length > 0 && (
            <div className="overflow-hidden rounded border border-slate-800">
              <div className="grid grid-cols-[20px_20px_1fr_90px_110px_80px_24px] items-center gap-1 bg-slate-800/50 px-2 py-1.5 text-[10px] font-bold uppercase tracking-wider text-slate-400">
                <span></span>
                <span></span>
                <span>Field</span>
                <span>Type</span>
                <span>Role</span>
                <span>Source</span>
                <span></span>
              </div>

              {/* Top-level fields */}
              <div
                className="divide-y divide-slate-800/60"
                role="rowgroup"
                aria-label="Top-level fields"
              >
                {topLevelFields.map((f, i) =>
                  renderFieldRow(f, topLevelIndices[i] ?? i),
                )}
              </div>

              {/* Groups */}
              {Object.entries(groups).map(([groupKey, groupLabel]) => {
                const members = fields.filter((f) => f.group === groupKey);
                const isCollapsed = collapsedGroups.has(groupKey);

                return (
                  <div key={groupKey}>
                    {/* Group header — also a drop target */}
                    <div
                      role="row"
                      onDragOver={(e) => handleDragOverGroup(e, groupKey)}
                      onDrop={handleDrop}
                      className="flex items-center gap-2 border-t border-slate-700 bg-slate-800/40 px-2 py-1.5 cursor-pointer hover:bg-slate-700/40"
                      data-group-key={groupKey}
                      onClick={() => toggleCollapse(groupKey)}
                    >
                      <button
                        type="button"
                        className="text-slate-400"
                        aria-label={
                          isCollapsed
                            ? `Expand group ${groupLabel}`
                            : `Collapse group ${groupLabel}`
                        }
                      >
                        {isCollapsed ? (
                          <ChevronRight size={12} />
                        ) : (
                          <ChevronDown size={12} />
                        )}
                      </button>
                      <span className="font-mono text-xs font-medium text-slate-300">
                        {groupLabel}
                      </span>
                      <span className="text-[10px] text-slate-600">
                        struct · {members.length} fields
                      </span>
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          removeGroup(groupKey);
                        }}
                        className="ml-auto text-[10px] text-slate-700 hover:text-red-400"
                        title={`Remove group ${groupLabel}`}
                      >
                        remove group
                      </button>
                    </div>

                    {/* Group members */}
                    {!isCollapsed && (
                      <div
                        className="divide-y divide-slate-800/40 bg-slate-900/30"
                        role="rowgroup"
                        aria-label={`Fields in group ${groupLabel}`}
                      >
                        {members.map((f) => renderFieldRow(f, fields.indexOf(f)))}
                        {members.length === 0 && (
                          <p className="pl-8 py-1.5 text-[11px] text-slate-700 italic">
                            Drop fields here to add to group
                          </p>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}

          {fields.length === 0 && (
            <p className="text-xs text-slate-600">
              No fields. Load from Iceberg or add custom columns below.
            </p>
          )}

          {/* Add group */}
          <div className="space-y-1 pt-1">
            <p className={SUB_HEADING}>Add Group</p>
            <div className="flex items-center gap-2">
              <input
                value={newGroupName}
                onChange={(e) => setNewGroupName(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    if (selectedNames.size > 0) createGroupAndMoveSelected();
                    else addGroup();
                  }
                }}
                placeholder="group_name"
                className={`${INPUT_CLS} max-w-[160px]`}
              />
              <button
                type="button"
                onClick={selectedNames.size > 0 ? createGroupAndMoveSelected : addGroup}
                disabled={!newGroupName.trim()}
                className={BTN_NEUTRAL}
              >
                <Plus size={11} className="mr-1 inline" />
                {selectedNames.size > 0
                  ? `Create & move ${selectedNames.size} selected`
                  : 'Create group'}
              </button>
            </div>
          </div>

          {/* Add custom column */}
          <div className="space-y-1">
            <p className={SUB_HEADING}>Add Custom Column</p>
            <div className="flex flex-wrap items-center gap-2">
              <input
                value={newColName}
                onChange={(e) => setNewColName(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') addCustomColumn();
                }}
                placeholder="column_name"
                className={`${INPUT_CLS} max-w-[160px]`}
              />
              <select
                value={newColType}
                onChange={(e) => setNewColType(e.target.value as SparkDataType)}
                className={`${SELECT_CLS} max-w-[100px]`}
              >
                {SPARK_TYPES.map((t) => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </select>
              {Object.keys(groups).length > 0 && (
                <select
                  value={newColGroup}
                  onChange={(e) => setNewColGroup(e.target.value)}
                  className={`${SELECT_CLS} max-w-[120px]`}
                >
                  <option value="">top-level</option>
                  {Object.keys(groups).map((g) => (
                    <option key={g} value={g}>
                      {g}
                    </option>
                  ))}
                </select>
              )}
              <button
                type="button"
                onClick={addCustomColumn}
                disabled={!newColName.trim()}
                className={BTN_PRIMARY}
              >
                <Plus size={11} className="mr-1 inline" />
                Add
              </button>
            </div>
          </div>
        </div>

        {/* ── Right: live YAML preview ── */}
        {fields.length > 0 && (
          <div className="sticky top-0">
            <p className="mb-1 text-[10px] font-medium uppercase tracking-wider text-slate-500">
              YAML Preview
            </p>
            <pre className="max-h-[500px] overflow-auto rounded border border-slate-700 bg-slate-950 p-2 font-mono text-[10px] leading-relaxed text-green-300 whitespace-pre">
              {yamlPreview}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
}
