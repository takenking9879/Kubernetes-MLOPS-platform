import type { ReactNode } from 'react';

interface CodePanelProps {
  children: ReactNode;
  title?: string;
  className?: string;
  maxHeight?: string;
}

export function CodePanel({ children, title, className = '', maxHeight = '200px' }: CodePanelProps) {
  return (
    <div className={`rounded-xl neon-border-cyan overflow-hidden ${className}`} style={{ background: '#000000' }}>
      {title && (
        <div className="flex items-center gap-2 px-3 py-1.5 border-b border-cyan-500/15" style={{ background: '#03060a' }}>
          <span className="text-[10px] font-bold uppercase tracking-widest text-cyan-500/60">{title}</span>
        </div>
      )}
      <div
        className="overflow-auto p-3 font-mono text-xs text-slate-300 leading-relaxed"
        style={{ maxHeight }}
      >
        {children}
      </div>
    </div>
  );
}
