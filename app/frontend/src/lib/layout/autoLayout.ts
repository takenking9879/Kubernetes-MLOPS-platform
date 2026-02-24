/**
 * Auto-layout for imported DAGs.
 *
 * Uses a simple layered layout: assigns nodes to layers based on
 * longest-path from root, then positions them in a grid.
 */

import type { Node } from 'reactflow';
import type { NodeData } from '../../types/nodes';
import type { FlowEdge } from '../../types/edges';

const NODE_WIDTH = 200;
const NODE_HEIGHT = 140;
const HORIZONTAL_GAP = 60;
const VERTICAL_GAP = 80;

/**
 * Assign positions to nodes using layered layout.
 * Returns a new array with updated positions (does not mutate).
 */
export function autoLayoutNodes(
  nodes: Node<NodeData>[],
  edges: FlowEdge[],
): Node<NodeData>[] {
  if (nodes.length === 0) return [];

  // Build adjacency (source -> targets) and in-degree
  const children = new Map<string, string[]>();
  const inDegree = new Map<string, number>();

  for (const node of nodes) {
    children.set(node.id, []);
    inDegree.set(node.id, 0);
  }

  for (const edge of edges) {
    children.get(edge.source)?.push(edge.target);
    inDegree.set(edge.target, (inDegree.get(edge.target) ?? 0) + 1);
  }

  // Find roots (in-degree 0)
  const roots = nodes.filter((n) => (inDegree.get(n.id) ?? 0) === 0).map((n) => n.id);

  // Assign layers via BFS (longest path from any root)
  const layer = new Map<string, number>();
  const queue: string[] = [...roots];

  for (const r of roots) {
    layer.set(r, 0);
  }

  while (queue.length > 0) {
    const current = queue.shift()!;
    const currentLayer = layer.get(current) ?? 0;

    for (const child of children.get(current) ?? []) {
      const prevLayer = layer.get(child) ?? -1;
      if (currentLayer + 1 > prevLayer) {
        layer.set(child, currentLayer + 1);
        queue.push(child);
      }
    }
  }

  // Handle any disconnected nodes (no edges)
  for (const node of nodes) {
    if (!layer.has(node.id)) {
      layer.set(node.id, 0);
    }
  }

  // Group nodes by layer
  const layers = new Map<number, string[]>();
  for (const [nodeId, layerIdx] of layer) {
    const group = layers.get(layerIdx);
    if (group) {
      group.push(nodeId);
    } else {
      layers.set(layerIdx, [nodeId]);
    }
  }

  // Position nodes
  const nodeMap = new Map(nodes.map((n) => [n.id, n]));
  const positioned: Node<NodeData>[] = [];

  for (const [layerIdx, nodeIds] of layers) {
    const totalWidth = nodeIds.length * NODE_WIDTH + (nodeIds.length - 1) * HORIZONTAL_GAP;
    const startX = -totalWidth / 2;

    for (let i = 0; i < nodeIds.length; i++) {
      const node = nodeMap.get(nodeIds[i]!)!;
      positioned.push({
        ...node,
        position: {
          x: startX + i * (NODE_WIDTH + HORIZONTAL_GAP),
          y: layerIdx * (NODE_HEIGHT + VERTICAL_GAP),
        },
      });
    }
  }

  return positioned;
}
