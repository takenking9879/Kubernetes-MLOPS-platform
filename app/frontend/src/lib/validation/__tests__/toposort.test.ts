import { describe, it, expect } from 'vitest';
import { topologicalSort } from '../toposort';

describe('topologicalSort', () => {
  it('sorts a linear chain', () => {
    const result = topologicalSort(
      ['a', 'b', 'c'],
      [
        { source: 'a', target: 'b' },
        { source: 'b', target: 'c' },
      ],
    );

    expect(result.hasCycle).toBe(false);
    expect(result.order).toEqual(['a', 'b', 'c']);
  });

  it('sorts a diamond DAG', () => {
    const result = topologicalSort(
      ['a', 'b', 'c', 'd'],
      [
        { source: 'a', target: 'b' },
        { source: 'a', target: 'c' },
        { source: 'b', target: 'd' },
        { source: 'c', target: 'd' },
      ],
    );

    expect(result.hasCycle).toBe(false);
    expect(result.order[0]).toBe('a');
    expect(result.order[result.order.length - 1]).toBe('d');
    expect(result.order).toHaveLength(4);
  });

  it('detects a simple cycle', () => {
    const result = topologicalSort(
      ['a', 'b', 'c'],
      [
        { source: 'a', target: 'b' },
        { source: 'b', target: 'c' },
        { source: 'c', target: 'a' },
      ],
    );

    expect(result.hasCycle).toBe(true);
    expect(result.cyclePath.length).toBeGreaterThan(0);
  });

  it('handles an empty graph', () => {
    const result = topologicalSort([], []);

    expect(result.hasCycle).toBe(false);
    expect(result.order).toEqual([]);
  });

  it('handles isolated nodes', () => {
    const result = topologicalSort(['a', 'b', 'c'], []);

    expect(result.hasCycle).toBe(false);
    expect(result.order).toHaveLength(3);
  });

  it('handles a partial cycle with non-cycling nodes', () => {
    const result = topologicalSort(
      ['root', 'a', 'b'],
      [
        { source: 'root', target: 'a' },
        { source: 'a', target: 'b' },
        { source: 'b', target: 'a' },
      ],
    );

    expect(result.hasCycle).toBe(true);
  });
});
