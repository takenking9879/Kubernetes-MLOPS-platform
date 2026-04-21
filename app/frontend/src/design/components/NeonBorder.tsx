import type { ReactNode } from 'react';

type NeonColor = 'cyan' | 'orange' | 'emerald' | 'purple' | 'red';

interface NeonBorderProps {
  children: ReactNode;
  color?: NeonColor;
  active?: boolean;
  className?: string;
}

export function NeonBorder({ children, color = 'cyan', active = false, className = '' }: NeonBorderProps) {
  const borderClass = `neon-border-${color}`;
  const activeClass = active && color === 'cyan' ? 'neon-selected-cyan' : '';

  return (
    <div className={`rounded-xl transition-all ${active ? activeClass : borderClass} ${className}`}>
      {children}
    </div>
  );
}
