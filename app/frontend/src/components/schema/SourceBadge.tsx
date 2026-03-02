import type { ColumnSource } from '../../lib/schemaYaml';

const SOURCE_STYLES: Record<ColumnSource, string> = {
  iceberg: 'bg-blue-900/50 text-blue-300',
  dsl_derived: 'bg-purple-900/50 text-purple-300',
  custom: 'bg-amber-900/50 text-amber-300',
  user_override: 'bg-orange-900/50 text-orange-300',
};

const SOURCE_LABELS: Record<ColumnSource, string> = {
  iceberg: 'Iceberg',
  dsl_derived: 'DSL',
  custom: 'Custom',
  user_override: 'Override',
};

interface SourceBadgeProps {
  source: ColumnSource;
  /** Optional DSL name shown as a tooltip for dsl_derived columns. */
  dslName?: string;
}

export function SourceBadge({ source, dslName }: SourceBadgeProps) {
  const title =
    source === 'dsl_derived' && dslName
      ? `Derived from DSL: ${dslName}`
      : undefined;

  return (
    <span
      className={`inline-block rounded px-1.5 py-0.5 text-[9px] font-medium ${SOURCE_STYLES[source]}`}
      title={title}
    >
      {SOURCE_LABELS[source]}
    </span>
  );
}
