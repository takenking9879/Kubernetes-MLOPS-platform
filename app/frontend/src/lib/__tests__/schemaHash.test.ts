import { describe, it, expect } from 'vitest';
import { computeSchemaHashSync } from '../schemaHash';
import type { SparkSchema } from '../../types/schema';

describe('computeSchemaHashSync', () => {
  it('produces a deterministic hash', () => {
    const schema: SparkSchema = {
      columns: [
        { name: 'a', sparkType: 'double', nullable: false },
        { name: 'b', sparkType: 'string', nullable: true },
      ],
    };

    const hash1 = computeSchemaHashSync(schema);
    const hash2 = computeSchemaHashSync(schema);
    expect(hash1).toBe(hash2);
  });

  it('is order-independent (columns sorted by name)', () => {
    const schema1: SparkSchema = {
      columns: [
        { name: 'z', sparkType: 'double', nullable: false },
        { name: 'a', sparkType: 'string', nullable: true },
      ],
    };

    const schema2: SparkSchema = {
      columns: [
        { name: 'a', sparkType: 'string', nullable: true },
        { name: 'z', sparkType: 'double', nullable: false },
      ],
    };

    expect(computeSchemaHashSync(schema1)).toBe(computeSchemaHashSync(schema2));
  });

  it('different schemas produce different hashes', () => {
    const schema1: SparkSchema = {
      columns: [{ name: 'a', sparkType: 'double', nullable: false }],
    };

    const schema2: SparkSchema = {
      columns: [{ name: 'a', sparkType: 'string', nullable: false }],
    };

    expect(computeSchemaHashSync(schema1)).not.toBe(computeSchemaHashSync(schema2));
  });

  it('nullable flag affects the hash', () => {
    const schema1: SparkSchema = {
      columns: [{ name: 'a', sparkType: 'double', nullable: false }],
    };

    const schema2: SparkSchema = {
      columns: [{ name: 'a', sparkType: 'double', nullable: true }],
    };

    expect(computeSchemaHashSync(schema1)).not.toBe(computeSchemaHashSync(schema2));
  });

  it('returns an 8-character hex string', () => {
    const schema: SparkSchema = {
      columns: [{ name: 'test', sparkType: 'integer', nullable: false }],
    };

    const hash = computeSchemaHashSync(schema);
    expect(hash).toMatch(/^[0-9a-f]{8}$/);
  });
});
