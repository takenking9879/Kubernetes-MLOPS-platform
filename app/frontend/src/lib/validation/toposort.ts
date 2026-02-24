/**
 * Topological sort using Kahn's algorithm with cycle detection.
 *
 * Operates on node IDs and edge (source→target) pairs.
 * If a cycle exists, returns hasCycle: true and the cycle path for debugging.
 */

export interface TopoSortResult {
  readonly order: readonly string[];
  readonly hasCycle: boolean;
  readonly cyclePath: readonly string[];
}

export function topologicalSort(
  nodeIds: readonly string[],
  edges: readonly { readonly source: string; readonly target: string }[],
): TopoSortResult {
  // Build adjacency list and in-degree map
  const adj = new Map<string, string[]>();
  const inDeg = new Map<string, number>();

  for (const id of nodeIds) {
    adj.set(id, []);
    inDeg.set(id, 0);
  }

  for (const e of edges) {
    const neighbors = adj.get(e.source);
    if (!neighbors) throw new Error(`Edge source '${e.source}' not found in node list`);
    neighbors.push(e.target);

    const current = inDeg.get(e.target);
    if (current === undefined) throw new Error(`Edge target '${e.target}' not found in node list`);
    inDeg.set(e.target, current + 1);
  }

  // Kahn's: seed queue with zero in-degree nodes
  const queue: string[] = [];
  for (const [id, deg] of inDeg) {
    if (deg === 0) queue.push(id);
  }

  const order: string[] = [];

  while (queue.length > 0) {
    const current = queue.shift()!;
    order.push(current);

    for (const neighbor of adj.get(current) ?? []) {
      const newDeg = (inDeg.get(neighbor) ?? 1) - 1;
      inDeg.set(neighbor, newDeg);
      if (newDeg === 0) queue.push(neighbor);
    }
  }

  if (order.length === nodeIds.length) {
    return { order, hasCycle: false, cyclePath: [] };
  }

  // Cycle detected — find it via DFS on remaining nodes
  const inOrder = new Set(order);
  const cyclePath = findCycle(adj, nodeIds.filter((id) => !inOrder.has(id)));

  return { order: [], hasCycle: true, cyclePath };
}

// ─── Internal: DFS cycle finder ─────────────────────────────────────

function findCycle(
  adj: ReadonlyMap<string, readonly string[]>,
  candidates: readonly string[],
): string[] {
  const WHITE = 0;
  const GRAY = 1;
  const BLACK = 2;

  const color = new Map<string, number>();
  for (const id of candidates) color.set(id, WHITE);

  const parent = new Map<string, string | null>();
  let cycleEnd: string | null = null;
  let cycleStart: string | null = null;

  function dfs(u: string): boolean {
    color.set(u, GRAY);
    for (const v of adj.get(u) ?? []) {
      if (!color.has(v)) continue; // not in candidate set
      if (color.get(v) === GRAY) {
        cycleEnd = u;
        cycleStart = v;
        return true;
      }
      if (color.get(v) === WHITE) {
        parent.set(v, u);
        if (dfs(v)) return true;
      }
    }
    color.set(u, BLACK);
    return false;
  }

  for (const id of candidates) {
    if (color.get(id) === WHITE) {
      parent.set(id, null);
      if (dfs(id)) break;
    }
  }

  if (cycleStart === null || cycleEnd === null) return candidates.slice(0, 1);

  // Reconstruct cycle path
  const path: string[] = [cycleStart];
  let cur: string = cycleEnd;
  while (cur !== cycleStart) {
    path.push(cur);
    cur = parent.get(cur) ?? cycleStart;
  }
  path.push(cycleStart);
  path.reverse();

  return path;
}
