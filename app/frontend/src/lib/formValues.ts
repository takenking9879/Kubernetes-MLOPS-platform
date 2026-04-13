export function stringifyNumberValue(value: number | string | null | undefined): string {
  if (value === null || value === undefined) return '';
  return String(value);
}

export function parseFiniteNumber(raw: string): number | null {
  const trimmed = raw.trim();
  if (trimmed === '') return null;
  const parsed = Number(trimmed);
  return Number.isFinite(parsed) ? parsed : null;
}

export function parseInteger(raw: string): number | null {
  const parsed = parseFiniteNumber(raw);
  if (parsed === null || !Number.isInteger(parsed)) return null;
  return parsed;
}

export function parseStringChip(raw: string): string | null {
  const trimmed = raw.trim();
  return trimmed === '' ? null : trimmed;
}

export function dedupePrimitiveValues<T extends string | number>(values: readonly T[]): T[] {
  const seen = new Set<string>();
  const next: T[] = [];
  for (const value of values) {
    const key = `${typeof value}:${String(value)}`;
    if (seen.has(key)) continue;
    seen.add(key);
    next.push(value);
  }
  return next;
}
