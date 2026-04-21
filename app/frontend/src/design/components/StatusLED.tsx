type LEDStatus = 'idle' | 'configured' | 'running' | 'success' | 'error';

interface StatusLEDProps {
  status: LEDStatus;
  size?: 'sm' | 'md';
  label?: string;
}

const STATUS_CONFIG: Record<LEDStatus, { color: string; glow: string; pulse: boolean }> = {
  idle:       { color: 'bg-slate-600',   glow: 'shadow-[0_0_6px_rgba(100,116,139,0.5)]',    pulse: false },
  configured: { color: 'bg-cyan-400',    glow: 'shadow-[0_0_8px_rgba(34,211,238,0.6)]',     pulse: false },
  running:    { color: 'bg-orange-400',  glow: 'shadow-[0_0_8px_rgba(249,115,22,0.7)]',     pulse: true  },
  success:    { color: 'bg-emerald-400', glow: 'shadow-[0_0_8px_rgba(16,185,129,0.6)]',     pulse: false },
  error:      { color: 'bg-red-400',     glow: 'shadow-[0_0_8px_rgba(248,113,113,0.7)]',    pulse: true  },
};

const STATUS_LABEL: Record<LEDStatus, string> = {
  idle:       'IDLE',
  configured: 'CONFIGURED',
  running:    'RUNNING',
  success:    'SUCCESS',
  error:      'ERROR',
};

const STATUS_TEXT_COLOR: Record<LEDStatus, string> = {
  idle:       'text-slate-500',
  configured: 'text-cyan-400',
  running:    'text-orange-400',
  success:    'text-emerald-400',
  error:      'text-red-400',
};

export function StatusLED({ status, size = 'sm', label }: StatusLEDProps) {
  const cfg = STATUS_CONFIG[status];
  const dim = size === 'md' ? 'h-3 w-3' : 'h-2 w-2';

  return (
    <span className="inline-flex items-center gap-1.5">
      <span
        className={`inline-block rounded-full ${dim} ${cfg.color} ${cfg.glow} ${cfg.pulse ? 'led-pulse' : ''}`}
      />
      {label !== undefined ? (
        <span className={`text-[10px] font-semibold uppercase tracking-wider ${STATUS_TEXT_COLOR[status]}`}>
          {label}
        </span>
      ) : (
        <span className={`text-[10px] font-semibold uppercase tracking-wider ${STATUS_TEXT_COLOR[status]}`}>
          {STATUS_LABEL[status]}
        </span>
      )}
    </span>
  );
}
