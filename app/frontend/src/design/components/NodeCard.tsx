import type { ReactNode } from 'react';

type NodeColor = 'cyan' | 'orange' | 'emerald' | 'purple' | 'red' | 'slate';

interface NodeCardProps {
  children: ReactNode;
  color?: NodeColor;
  selected?: boolean;
  className?: string;
  onClick?: () => void;
}

const COLOR_MAP: Record<NodeColor, { border: string; headerBg: string; dot: string; label: string }> = {
  cyan:    { border: 'neon-border-cyan',    headerBg: 'rgba(34,211,238,0.06)',  dot: 'bg-cyan-400',    label: 'text-cyan-300'    },
  orange:  { border: 'neon-border-orange',  headerBg: 'rgba(249,115,22,0.08)',  dot: 'bg-orange-400',  label: 'text-orange-300'  },
  emerald: { border: 'neon-border-emerald', headerBg: 'rgba(16,185,129,0.08)',  dot: 'bg-emerald-400', label: 'text-emerald-300' },
  purple:  { border: 'neon-border-purple',  headerBg: 'rgba(167,139,250,0.08)', dot: 'bg-purple-400',  label: 'text-purple-300'  },
  red:     { border: 'neon-border-red',     headerBg: 'rgba(248,113,113,0.08)', dot: 'bg-red-400',     label: 'text-red-300'     },
  slate:   { border: 'border border-slate-700/40', headerBg: 'rgba(51,65,85,0.3)', dot: 'bg-slate-500', label: 'text-slate-400' },
};

export function NodeCard({ children, color = 'cyan', selected = false, className = '', onClick }: NodeCardProps) {
  const cfg = COLOR_MAP[color];
  const borderClass = selected ? 'neon-selected-cyan' : cfg.border;

  return (
    <div
      onClick={onClick}
      className={`rounded-xl transition-all ${borderClass} ${onClick ? 'cursor-pointer' : ''} ${className}`}
      style={{ background: '#03060a' }}
    >
      {children}
    </div>
  );
}

NodeCard.Header = function NodeCardHeader({
  children,
  color = 'cyan',
}: {
  children: ReactNode;
  color?: NodeColor;
}) {
  const cfg = COLOR_MAP[color];
  return (
    <div className="flex items-center gap-2 rounded-t-xl px-3 py-2" style={{ background: cfg.headerBg }}>
      <div className={`h-2 w-2 rounded-full ${cfg.dot}`} />
      <span className={`text-[10px] font-bold uppercase tracking-widest ${cfg.label}`}>
        {children}
      </span>
    </div>
  );
};

NodeCard.Body = function NodeCardBody({ children }: { children: ReactNode }) {
  return <div className="px-3 py-2">{children}</div>;
};
