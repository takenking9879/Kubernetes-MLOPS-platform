import { useEffect, useMemo, useState } from 'react';
import { X } from 'lucide-react';
import { dedupePrimitiveValues } from '../../lib/formValues';

interface ChipInputProps<T extends string | number> {
  readonly value: readonly T[];
  readonly onChange: (value: T[]) => void;
  readonly parseItem: (raw: string) => T | null;
  readonly formatItem?: (value: T) => string;
  readonly placeholder?: string;
  readonly suggestions?: readonly T[];
  readonly allowDuplicates?: boolean;
  readonly disabled?: boolean;
  readonly className?: string;
  readonly chipClassName?: string;
  readonly inputClassName?: string;
}

export function ChipInput<T extends string | number>({
  value,
  onChange,
  parseItem,
  formatItem = String,
  placeholder = 'Type a value and press Enter',
  suggestions,
  allowDuplicates = false,
  disabled = false,
  className = '',
  chipClassName = 'rounded bg-slate-700 px-2 py-0.5 text-[10px] text-slate-200',
  inputClassName = 'min-w-[120px] flex-1 bg-transparent text-xs text-slate-100 outline-none',
}: ChipInputProps<T>) {
  const [draft, setDraft] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    setError('');
  }, [value]);

  const visibleSuggestions = useMemo(() => {
    if (!suggestions || suggestions.length === 0) return [] as readonly T[];
    return allowDuplicates
      ? suggestions
      : suggestions.filter((suggestion) => !value.some((item) => Object.is(item, suggestion)));
  }, [allowDuplicates, suggestions, value]);

  const commitDraft = (raw: string) => {
    const parsed = parseItem(raw);
    if (parsed === null) {
      setError('Enter a valid value');
      return;
    }
    if (!allowDuplicates && value.some((item) => Object.is(item, parsed))) {
      setError('Duplicate values are not allowed');
      return;
    }

    onChange(dedupePrimitiveValues([...value, parsed]));
    setDraft('');
    setError('');
  };

  const removeItem = (target: T) => {
    onChange(value.filter((item) => !Object.is(item, target)));
  };

  return (
    <div className={className}>
      <div className="flex flex-wrap gap-1.5 rounded border border-slate-700 bg-slate-800 px-2 py-1.5">
        {value.map((item, index) => (
          <span key={`${String(item)}-${index}`} className={chipClassName}>
            <span className="mr-1">{formatItem(item)}</span>
            <button
              type="button"
              onClick={() => removeItem(item)}
              className="text-slate-400 hover:text-slate-100"
              aria-label={`Remove ${formatItem(item)}`}
            >
              <X size={10} />
            </button>
          </span>
        ))}

        <input
          type="text"
          inputMode="decimal"
          value={draft}
          onChange={(e) => {
            setDraft(e.target.value);
            setError('');
          }}
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              e.preventDefault();
              commitDraft(draft);
            }
          }}
          onBlur={() => {
            if (draft.trim()) {
              commitDraft(draft);
            }
          }}
          placeholder={value.length === 0 ? placeholder : undefined}
          disabled={disabled}
          className={inputClassName}
        />
      </div>

      {visibleSuggestions.length > 0 && (
        <div className="mt-1 flex flex-wrap gap-1.5">
          {visibleSuggestions.map((suggestion) => (
            <button
              key={`${String(suggestion)}`}
              type="button"
              onClick={() => onChange(dedupePrimitiveValues([...value, suggestion]))}
              className="rounded bg-slate-700 px-2 py-0.5 text-[10px] text-slate-300 hover:bg-slate-600"
            >
              {formatItem(suggestion)}
            </button>
          ))}
        </div>
      )}

      {error && <p className="mt-1 text-[10px] text-red-400">{error}</p>}
    </div>
  );
}
