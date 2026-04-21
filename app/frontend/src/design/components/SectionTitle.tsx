import type { ReactNode } from 'react';

interface SectionTitleProps {
  children: ReactNode;
  className?: string;
  dim?: boolean;
}

export function SectionTitle({ children, className = '', dim = false }: SectionTitleProps) {
  return (
    <h2
      className={`text-xs font-extrabold italic uppercase tracking-widest ${
        dim ? 'text-slate-500' : 'text-cyan-400'
      } drop-shadow-[0_0_8px_rgba(34,211,238,0.5)] ${className}`}
    >
      {children}
    </h2>
  );
}
