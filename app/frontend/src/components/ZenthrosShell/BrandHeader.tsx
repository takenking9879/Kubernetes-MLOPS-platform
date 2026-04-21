import { useRef } from 'react';
import { Download, Upload, Play, Cpu } from 'lucide-react';
import { useDagStore } from '../../store/dagStore';
import { useUIStore } from '../../store/uiStore';
import { IconButton } from '../../design/components/IconButton';
import type { PipelineManifest } from '../../types/dag';

export function BrandHeader() {
  const pipelineName    = useDagStore((s) => s.pipelineName);
  const exportManifest  = useDagStore((s) => s.exportManifest);
  const importManifest  = useDagStore((s) => s.importManifest);
  const selectedNodeId  = useDagStore((s) => s.selectedNodeId);
  const runUpTo         = useDagStore((s) => s.runUpTo);
  const setPage         = useUIStore((s) => s.setPage);
  const fileInputRef    = useRef<HTMLInputElement>(null);

  const handleExport = () => {
    const manifest = exportManifest();
    const blob = new Blob([JSON.stringify(manifest, null, 2)], { type: 'application/json' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = `${pipelineName}_manifest.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleImport = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      try {
        const manifest = JSON.parse(ev.target?.result as string) as PipelineManifest;
        importManifest(manifest);
      } catch {
        alert('Invalid manifest JSON');
      }
    };
    reader.readAsText(file);
    e.target.value = '';
  };

  return (
    <header
      className="flex h-11 shrink-0 items-center justify-between px-4 glass-panel shadow-glow-cyan"
      style={{ borderTop: 'none', borderLeft: 'none', borderRight: 'none' }}
    >
      {/* Brand */}
      <div className="flex items-center gap-3">
        <h1
          className="text-sm font-extrabold italic tracking-[0.2em] text-cyan-400 select-none"
          style={{ textShadow: '0 0 16px rgba(34,211,238,0.6), 0 0 32px rgba(34,211,238,0.2)' }}
        >
          ZENTHROS
          <span className="ml-1 text-orange-400">ML</span>
        </h1>
        <span className="h-4 w-px bg-cyan-500/20" />
        <span className="text-[10px] font-mono text-slate-500 uppercase tracking-wide">{pipelineName}</span>
      </div>

      {/* Actions */}
      <div className="flex items-center gap-2">
        {selectedNodeId && (
          <IconButton
            variant="orange"
            onClick={() => { void runUpTo(selectedNodeId); }}
            title="Run up to selected node"
          >
            <Play size={12} />
            <span>Run to here</span>
          </IconButton>
        )}

        <IconButton variant="ghost" onClick={handleExport} title="Export pipeline manifest">
          <Download size={12} />
          <span>Export</span>
        </IconButton>

        <IconButton variant="ghost" onClick={() => fileInputRef.current?.click()} title="Import pipeline manifest">
          <Upload size={12} />
          <span>Import</span>
        </IconButton>

        <input ref={fileInputRef} type="file" accept=".json" className="hidden" onChange={handleImport} />

        <span className="h-4 w-px bg-cyan-500/20" />

        <IconButton variant="ghost" onClick={() => setPage('dsl-builder')} title="Open DSL Builder">
          <Cpu size={12} />
          <span>DSL Builder</span>
        </IconButton>
      </div>
    </header>
  );
}
