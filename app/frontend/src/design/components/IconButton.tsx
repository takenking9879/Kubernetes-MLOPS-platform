import type { ReactNode } from 'react';

type BtnVariant = 'cyan' | 'orange' | 'emerald' | 'ghost';

interface IconButtonProps {
  children: ReactNode;
  onClick?: () => void;
  variant?: BtnVariant;
  disabled?: boolean;
  title?: string;
  className?: string;
  size?: 'sm' | 'md';
}

const VARIANT_MAP: Record<BtnVariant, string> = {
  cyan:    'border border-cyan-500/30 bg-cyan-500/10 text-cyan-300 hover:bg-cyan-500/20 hover:shadow-glow-cyan',
  orange:  'border border-orange-500/30 bg-orange-500/10 text-orange-300 hover:bg-orange-500/20 hover:shadow-glow-orange',
  emerald: 'border border-emerald-500/30 bg-emerald-500/10 text-emerald-300 hover:bg-emerald-500/20 hover:shadow-glow-emerald',
  ghost:   'border border-slate-700/40 bg-transparent text-slate-400 hover:border-cyan-500/30 hover:text-cyan-300',
};

export function IconButton({
  children,
  onClick,
  variant = 'ghost',
  disabled,
  title,
  className = '',
  size = 'sm',
}: IconButtonProps) {
  const padding = size === 'md' ? 'px-3 py-1.5' : 'px-2 py-1';
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={`inline-flex items-center gap-1.5 rounded text-xs font-medium transition-all disabled:opacity-40 ${padding} ${VARIANT_MAP[variant]} ${className}`}
    >
      {children}
    </button>
  );
}
