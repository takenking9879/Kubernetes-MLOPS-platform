import { useRef, useEffect } from 'react';
import { usePipelineStore } from '../../store/pipelineStore';

export function ConsolePanel() {
  const messages = usePipelineStore((s) => s.consoleMessages);
  const clearConsole = usePipelineStore((s) => s.clearConsole);
  const validationResult = usePipelineStore((s) => s.validationResult);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages.length]);

  const schemaEvolutions = validationResult?.schemaEvolutions;

  return (
    <div className="flex h-full flex-col border-t border-slate-800 bg-slate-950">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-800 px-4 py-1.5">
        <div className="flex items-center gap-3">
          <h3 className="text-xs font-semibold text-slate-400">Console</h3>
          {schemaEvolutions && schemaEvolutions.size > 0 && (
            <span className="text-[10px] text-slate-500">
              Schema: {Array.from(schemaEvolutions.values()).pop()?.outputSchema.columns.length ?? 0} cols
            </span>
          )}
        </div>
        <button
          onClick={clearConsole}
          className="text-[10px] text-slate-500 hover:text-slate-300"
        >
          Clear
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-2 font-mono text-[11px]">
        {messages.length === 0 && (
          <p className="text-slate-600">No messages yet. Load a dataset to begin.</p>
        )}
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`py-0.5 ${
              msg.type === 'error'
                ? 'text-red-400'
                : msg.type === 'warning'
                  ? 'text-amber-400'
                  : 'text-slate-400'
            }`}
          >
            <span className="text-slate-600">
              {new Date(msg.timestamp).toLocaleTimeString()}
            </span>{' '}
            {msg.message}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
