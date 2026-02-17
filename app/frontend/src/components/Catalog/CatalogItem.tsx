import type { NodeDefinition, TerminalNodeDefinition } from '../../registry/NodeRegistry';

interface CatalogItemProps {
  readonly definition: NodeDefinition | TerminalNodeDefinition;
  readonly disabled?: boolean;
}

export function CatalogItem({ definition, disabled = false }: CatalogItemProps) {
  const onDragStart = (event: React.DragEvent) => {
    if (disabled) {
      event.preventDefault();
      return;
    }
    event.dataTransfer.setData('application/spark-stage-type', definition.type);
    event.dataTransfer.effectAllowed = 'move';
  };

  const isEstimator = definition.category === 'estimator';
  const isTerminal = definition.category === 'terminal';

  return (
    <div
      draggable={!disabled}
      onDragStart={onDragStart}
      className={`
        flex items-center gap-2 rounded-md border px-3 py-2 transition-colors
        ${disabled
          ? 'cursor-not-allowed opacity-50'
          : 'cursor-grab active:cursor-grabbing'}
        ${isTerminal
          ? 'border-red-700/50 bg-red-900/20 hover:bg-red-900/35'
          : isEstimator
          ? 'border-purple-700/40 bg-purple-900/20 hover:bg-purple-900/40'
          : 'border-blue-700/40 bg-blue-900/20 hover:bg-blue-900/40'}
      `}
    >
      <div
        className={`h-2 w-2 rounded-full ${
          isTerminal ? 'bg-red-400' : isEstimator ? 'bg-purple-400' : 'bg-blue-400'
        }`}
      />
      <div className="min-w-0 flex-1">
        <p className="text-xs font-medium text-slate-200">{definition.label}</p>
        <p className="truncate text-[10px] text-slate-500">
          {disabled ? 'Maximum instances reached' : definition.description}
        </p>
      </div>
    </div>
  );
}
