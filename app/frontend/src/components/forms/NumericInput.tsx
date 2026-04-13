import { useEffect, useState } from 'react';

interface NumericInputProps {
  readonly value: string;
  readonly onChange: (value: string) => void;
  readonly placeholder?: string;
  readonly min?: number;
  readonly max?: number;
  readonly step?: number | string;
  readonly disabled?: boolean;
  readonly className?: string;
}

export function NumericInput({
  value,
  onChange,
  placeholder,
  min,
  max,
  step,
  disabled,
  className = 'input-field',
}: NumericInputProps) {
  const [draft, setDraft] = useState(value);

  useEffect(() => {
    setDraft(value);
  }, [value]);

  const hint = [
    min !== undefined ? `min ${min}` : null,
    max !== undefined ? `max ${max}` : null,
    step !== undefined ? `step ${step}` : null,
  ]
    .filter(Boolean)
    .join(' · ');

  return (
    <input
      type="text"
      inputMode="decimal"
      value={draft}
      onChange={(e) => {
        const next = e.target.value;
        setDraft(next);
        onChange(next);
      }}
      placeholder={placeholder}
      disabled={disabled}
      title={hint || undefined}
      aria-label={hint || undefined}
      className={className}
    />
  );
}
