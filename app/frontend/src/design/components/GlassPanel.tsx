import type { ReactNode, CSSProperties } from 'react';

interface GlassPanelProps {
  children: ReactNode;
  className?: string;
  style?: CSSProperties;
  glow?: 'cyan' | 'orange' | 'emerald' | 'none';
}

export function GlassPanel({ children, className = '', style, glow = 'none' }: GlassPanelProps) {
  const glowClass = {
    cyan:    'shadow-glow-cyan',
    orange:  'shadow-glow-orange',
    emerald: 'shadow-glow-emerald',
    none:    '',
  }[glow];

  return (
    <div
      className={`glass-panel rounded-xl ${glowClass} ${className}`}
      style={style}
    >
      {children}
    </div>
  );
}
