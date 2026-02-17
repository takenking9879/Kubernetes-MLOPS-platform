export type OutputNamingStrategy =
  | { mode: 'auto' }
  | { mode: 'prefix'; value: string }
  | { mode: 'suffix'; value: string }
  | { mode: 'passthrough' }
  | { mode: 'custom'; columns: string[] };

export function isNamingStrategyComplete(strategy: OutputNamingStrategy): boolean {
  switch (strategy.mode) {
    case 'auto':
    case 'passthrough':
      return true;
    case 'prefix':
    case 'suffix':
      return strategy.value.trim().length > 0;
    case 'custom':
      return strategy.columns.length > 0;
  }
}
