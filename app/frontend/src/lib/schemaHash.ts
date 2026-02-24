/**
 * Schema Hash Utility.
 *
 * Computes a deterministic hash of a SparkSchema for YAML metadata
 * and import validation. The same schema always produces the same hash,
 * regardless of column order.
 *
 * Algorithm: sort columns by name, format as "name:sparkType:nullable",
 * join with "|", SHA-256, take first 16 hex chars.
 */

import type { SparkSchema } from '../types/schema';

/**
 * Compute a canonical hash of a SparkSchema.
 * Uses Web Crypto API (SHA-256) for consistency with backend.
 */
export async function computeSchemaHash(schema: SparkSchema): Promise<string> {
  const sorted = [...schema.columns].sort((a, b) => a.name.localeCompare(b.name));
  const content = sorted.map((c) => `${c.name}:${c.sparkType}:${c.nullable}`).join('|');
  const encoded = new TextEncoder().encode(content);
  const hashBuffer = await crypto.subtle.digest('SHA-256', encoded);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  const hexString = hashArray.map((b) => b.toString(16).padStart(2, '0')).join('');
  return hexString.slice(0, 16);
}

/**
 * Synchronous schema hash using simple FNV-1a for use in non-async contexts.
 * Produces a shorter, less collision-resistant hash but is deterministic.
 */
export function computeSchemaHashSync(schema: SparkSchema): string {
  const sorted = [...schema.columns].sort((a, b) => a.name.localeCompare(b.name));
  const content = sorted.map((c) => `${c.name}:${c.sparkType}:${c.nullable}`).join('|');

  // FNV-1a 32-bit
  let hash = 0x811c9dc5;
  for (let i = 0; i < content.length; i++) {
    hash ^= content.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }

  // Convert to unsigned hex
  return (hash >>> 0).toString(16).padStart(8, '0');
}
