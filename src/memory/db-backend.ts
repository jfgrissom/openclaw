/**
 * Database backend abstraction for memory storage.
 *
 * Strategy Pattern: SqliteBackend (default) and PostgresBackend implement
 * the same MemoryDbBackend interface. The manager selects which to use
 * based on config. Existing SQLite behavior is preserved exactly.
 *
 * SOLID:
 * - Single Responsibility: Each backend handles only its own DB dialect
 * - Open/Closed: New backends added without modifying existing ones
 * - Liskov Substitution: Either backend is interchangeable
 * - Interface Segregation: MemoryDbBackend exposes only what manager needs
 * - Dependency Inversion: Manager depends on interface, not concrete DB
 */

// ─── Types ───────────────────────────────────────────────────────────────────

export type DbFileRow = {
  path: string;
  source: string;
  hash: string;
  mtime: number;
  size: number;
};

export type DbChunkRow = {
  id: string;
  path: string;
  source: string;
  start_line: number;
  end_line: number;
  hash: string;
  model: string;
  text: string;
  embedding: string;
  updated_at: number;
};

export type DbCacheRow = {
  provider: string;
  model: string;
  provider_key: string;
  hash: string;
  embedding: string;
  dims: number | null;
  updated_at: number;
};

export type VectorSearchRow = {
  id: string;
  path: string;
  start_line: number;
  end_line: number;
  text: string;
  source: string;
  dist: number;
};

export type FtsSearchRow = {
  id: string;
  path: string;
  source: string;
  start_line: number;
  end_line: number;
  text: string;
  rank: number;
};

export type SourceFilter = {
  sources: string[];
};

// ─── Interface ───────────────────────────────────────────────────────────────

export interface MemoryDbBackend {
  readonly type: "sqlite" | "postgres";

  // Schema
  ensureSchema(params: { embeddingCacheTable: string; ftsTable: string; ftsEnabled: boolean }): {
    ftsAvailable: boolean;
    ftsError?: string;
  };

  // Meta
  getMeta(key: string): string | undefined;
  setMeta(key: string, value: string): void;

  // Files
  getFile(path: string): DbFileRow | undefined;
  upsertFile(row: DbFileRow): void;
  deleteFile(path: string): void;
  listFiles(source?: string): DbFileRow[];
  countFiles(sourceFilter?: SourceFilter): number;

  // Chunks
  getChunk(id: string): DbChunkRow | undefined;
  getChunksByPath(path: string): DbChunkRow[];
  upsertChunk(row: DbChunkRow): void;
  deleteChunksByPath(path: string): void;
  deleteChunk(id: string): void;
  listChunks(params: { model: string; sourceFilter?: SourceFilter }): DbChunkRow[];
  countChunks(sourceFilter?: SourceFilter): number;
  countChunksBySource(): Array<{ source: string; files: number; chunks: number }>;

  // Embedding cache
  getCachedEmbedding(params: {
    provider: string;
    model: string;
    providerKey: string;
    hash: string;
  }): DbCacheRow | undefined;
  upsertCachedEmbedding(row: DbCacheRow): void;
  countCacheEntries(): number;
  trimCache(params: { maxEntries: number; table: string }): void;
  seedCacheFrom(sourceBackend: MemoryDbBackend, table: string): void;

  // FTS
  insertFts(params: {
    table: string;
    id: string;
    path: string;
    source: string;
    model: string;
    startLine: number;
    endLine: number;
    text: string;
  }): void;
  deleteFtsById(table: string, id: string): void;
  deleteFtsByPath(table: string, path: string): void;
  clearFts(table: string): void;
  searchFts(params: {
    table: string;
    model: string;
    query: string;
    sourceFilter?: SourceFilter;
    limit: number;
  }): FtsSearchRow[];

  // Vector
  ensureVectorIndex(params: {
    table: string;
    dimensions: number;
    extensionPath?: string;
  }): Promise<{ ok: boolean; extensionPath?: string; error?: string }>;
  insertVector(params: { table: string; id: string; embedding: number[] }): void;
  deleteVector(table: string, id: string): void;
  searchVector(params: {
    table: string;
    model: string;
    queryVec: number[];
    sourceFilter?: SourceFilter;
    limit: number;
  }): VectorSearchRow[];

  // Lifecycle
  close(): void;

  // Transaction support
  transaction<T>(fn: () => T): T;

  // Raw access (for operations not yet abstracted)
  /** @deprecated Use typed methods instead. Escape hatch for migration period. */
  rawExec(sql: string): void;
}
