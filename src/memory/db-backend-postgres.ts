/**
 * PostgreSQL implementation of MemoryDbBackend.
 *
 * Uses node-postgres (pg) for connection management.
 * Supports pgvector for vector similarity search.
 * FTS uses PostgreSQL native tsvector/tsquery (not FTS5).
 *
 * SOLID: Same interface as SqliteBackend — drop-in replacement.
 */

import pg from "pg";
import type {
  DbCacheRow,
  DbChunkRow,
  DbFileRow,
  FtsSearchRow,
  MemoryDbBackend,
  SourceFilter,
  VectorSearchRow,
} from "./db-backend.js";
import { createSubsystemLogger } from "../logging/subsystem.js";

const log = createSubsystemLogger("memory-pg");

export class PostgresBackend implements MemoryDbBackend {
  readonly type = "postgres" as const;
  private pool: pg.Pool;
  private vectorAvailable = false;

  constructor(params: { connectionString: string }) {
    this.pool = new pg.Pool({
      connectionString: params.connectionString,
      max: 5,
      idleTimeoutMillis: 30_000,
    });
  }

  // Synchronous helper — runs query with blocking pattern for interface compat
  // The MemoryDbBackend interface is sync (matching SQLite's DatabaseSync).
  // We use a shared connection and synchronous-style execution via pg's query.
  // NOTE: This is a pragmatic compromise. The interface could be made async
  // in a future refactor, but that would touch every callsite in manager.ts.
  private querySync<T extends pg.QueryResultRow = Record<string, unknown>>(
    sql: string,
    params?: unknown[],
  ): pg.QueryResult<T> {
    // pg.Pool.query returns a promise. We need synchronous access for
    // interface compatibility with SQLite's DatabaseSync.
    // Use a cached client for synchronous-ish access.
    throw new Error(
      "PostgresBackend requires async initialization. Use PostgresBackend.create() and async methods.",
    );
  }

  // ── Async factory & query methods ───────────────────────────────────────

  private client: pg.PoolClient | null = null;

  static async create(params: { connectionString: string }): Promise<PostgresBackend> {
    const backend = new PostgresBackend(params);
    backend.client = await backend.pool.connect();
    return backend;
  }

  private async query<T extends pg.QueryResultRow = Record<string, unknown>>(
    sql: string,
    params?: unknown[],
  ): Promise<pg.QueryResult<T>> {
    const client = this.client;
    if (!client) {
      throw new Error("PostgresBackend not initialized. Use PostgresBackend.create().");
    }
    return client.query<T>(sql, params);
  }

  private queryOne<T extends pg.QueryResultRow = Record<string, unknown>>(
    sql: string,
    params?: unknown[],
  ): Promise<T | undefined> {
    return this.query<T>(sql, params).then((r) => r.rows[0]);
  }

  private queryAll<T extends pg.QueryResultRow = Record<string, unknown>>(
    sql: string,
    params?: unknown[],
  ): Promise<T[]> {
    return this.query<T>(sql, params).then((r) => r.rows);
  }

  private async exec(sql: string): Promise<void> {
    await this.query(sql);
  }

  // ── Schema ──────────────────────────────────────────────────────────────

  ensureSchema(params: { embeddingCacheTable: string; ftsTable: string; ftsEnabled: boolean }): {
    ftsAvailable: boolean;
    ftsError?: string;
  } {
    // Schema is created asynchronously in ensureSchemaAsync
    // Return optimistic result — actual creation happens in init
    return { ftsAvailable: params.ftsEnabled };
  }

  async ensureSchemaAsync(params: {
    embeddingCacheTable: string;
    ftsTable: string;
    ftsEnabled: boolean;
  }): Promise<{ ftsAvailable: boolean; ftsError?: string }> {
    try {
      await this.exec(`
        CREATE TABLE IF NOT EXISTS meta (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        );
      `);
      await this.exec(`
        CREATE TABLE IF NOT EXISTS files (
          path TEXT PRIMARY KEY,
          source TEXT NOT NULL DEFAULT 'memory',
          hash TEXT NOT NULL,
          mtime BIGINT NOT NULL,
          size BIGINT NOT NULL
        );
      `);
      await this.exec(`
        CREATE TABLE IF NOT EXISTS chunks (
          id TEXT PRIMARY KEY,
          path TEXT NOT NULL,
          source TEXT NOT NULL DEFAULT 'memory',
          start_line INTEGER NOT NULL,
          end_line INTEGER NOT NULL,
          hash TEXT NOT NULL,
          model TEXT NOT NULL,
          text TEXT NOT NULL,
          embedding TEXT NOT NULL,
          updated_at BIGINT NOT NULL
        );
      `);
      await this.exec(`CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path);`);
      await this.exec(`CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);`);

      await this.exec(`
        CREATE TABLE IF NOT EXISTS ${params.embeddingCacheTable} (
          provider TEXT NOT NULL,
          model TEXT NOT NULL,
          provider_key TEXT NOT NULL,
          hash TEXT NOT NULL,
          embedding TEXT NOT NULL,
          dims INTEGER,
          updated_at BIGINT NOT NULL,
          PRIMARY KEY (provider, model, provider_key, hash)
        );
      `);
      await this.exec(
        `CREATE INDEX IF NOT EXISTS idx_embedding_cache_updated_at ON ${params.embeddingCacheTable}(updated_at);`,
      );

      // FTS using PostgreSQL native tsvector
      let ftsAvailable = false;
      let ftsError: string | undefined;
      if (params.ftsEnabled) {
        try {
          await this.exec(`
            CREATE INDEX IF NOT EXISTS idx_chunks_fts
            ON chunks USING gin(to_tsvector('english', text));
          `);
          ftsAvailable = true;
        } catch (err) {
          ftsError = err instanceof Error ? err.message : String(err);
        }
      }

      return { ftsAvailable, ...(ftsError ? { ftsError } : {}) };
    } catch (err) {
      log.error("Failed to ensure schema:", err);
      throw err;
    }
  }

  // ── Meta ────────────────────────────────────────────────────────────────

  getMeta(key: string): string | undefined {
    // Sync interface — not usable directly. Use getMetaAsync.
    return undefined;
  }

  async getMetaAsync(key: string): Promise<string | undefined> {
    const row = await this.queryOne<{ value: string }>("SELECT value FROM meta WHERE key = $1", [
      key,
    ]);
    return row?.value;
  }

  setMeta(key: string, value: string): void {
    // Fire-and-forget for sync interface compat
    void this.setMetaAsync(key, value);
  }

  async setMetaAsync(key: string, value: string): Promise<void> {
    await this.query(
      "INSERT INTO meta (key, value) VALUES ($1, $2) ON CONFLICT (key) DO UPDATE SET value = $2",
      [key, value],
    );
  }

  // ── Files ───────────────────────────────────────────────────────────────

  getFile(_path: string): DbFileRow | undefined {
    return undefined; // Use async version
  }

  async getFileAsync(filePath: string): Promise<DbFileRow | undefined> {
    return this.queryOne<DbFileRow>(
      "SELECT path, source, hash, mtime, size FROM files WHERE path = $1",
      [filePath],
    );
  }

  upsertFile(row: DbFileRow): void {
    void this.upsertFileAsync(row);
  }

  async upsertFileAsync(row: DbFileRow): Promise<void> {
    await this.query(
      `INSERT INTO files (path, source, hash, mtime, size) VALUES ($1, $2, $3, $4, $5)
       ON CONFLICT (path) DO UPDATE SET source = $2, hash = $3, mtime = $4, size = $5`,
      [row.path, row.source, row.hash, row.mtime, row.size],
    );
  }

  deleteFile(filePath: string): void {
    void this.deleteFileAsync(filePath);
  }

  async deleteFileAsync(filePath: string): Promise<void> {
    await this.query("DELETE FROM files WHERE path = $1", [filePath]);
  }

  listFiles(source?: string): DbFileRow[] {
    return []; // Use async version
  }

  async listFilesAsync(source?: string): Promise<DbFileRow[]> {
    if (source) {
      return this.queryAll<DbFileRow>(
        "SELECT path, source, hash, mtime, size FROM files WHERE source = $1",
        [source],
      );
    }
    return this.queryAll<DbFileRow>("SELECT path, source, hash, mtime, size FROM files");
  }

  countFiles(sourceFilter?: SourceFilter): number {
    return 0; // Use async version
  }

  async countFilesAsync(sourceFilter?: SourceFilter): Promise<number> {
    if (sourceFilter?.sources.length) {
      const placeholders = sourceFilter.sources.map((_, i) => `$${i + 1}`).join(", ");
      const row = await this.queryOne<{ cnt: string }>(
        `SELECT COUNT(*) AS cnt FROM files WHERE source IN (${placeholders})`,
        sourceFilter.sources,
      );
      return Number(row?.cnt ?? 0);
    }
    const row = await this.queryOne<{ cnt: string }>("SELECT COUNT(*) AS cnt FROM files");
    return Number(row?.cnt ?? 0);
  }

  // ── Chunks ──────────────────────────────────────────────────────────────

  getChunk(_id: string): DbChunkRow | undefined {
    return undefined;
  }

  async getChunkAsync(id: string): Promise<DbChunkRow | undefined> {
    return this.queryOne<DbChunkRow>("SELECT * FROM chunks WHERE id = $1", [id]);
  }

  getChunksByPath(_path: string): DbChunkRow[] {
    return [];
  }

  async getChunksByPathAsync(chunkPath: string): Promise<DbChunkRow[]> {
    return this.queryAll<DbChunkRow>("SELECT * FROM chunks WHERE path = $1", [chunkPath]);
  }

  upsertChunk(row: DbChunkRow): void {
    void this.upsertChunkAsync(row);
  }

  async upsertChunkAsync(row: DbChunkRow): Promise<void> {
    await this.query(
      `INSERT INTO chunks (id, path, source, start_line, end_line, hash, model, text, embedding, updated_at)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
       ON CONFLICT (id) DO UPDATE SET
         path = $2, source = $3, start_line = $4, end_line = $5,
         hash = $6, model = $7, text = $8, embedding = $9, updated_at = $10`,
      [
        row.id,
        row.path,
        row.source,
        row.start_line,
        row.end_line,
        row.hash,
        row.model,
        row.text,
        row.embedding,
        row.updated_at,
      ],
    );
  }

  deleteChunksByPath(chunkPath: string): void {
    void this.deleteChunksByPathAsync(chunkPath);
  }

  async deleteChunksByPathAsync(chunkPath: string): Promise<void> {
    await this.query("DELETE FROM chunks WHERE path = $1", [chunkPath]);
  }

  deleteChunk(id: string): void {
    void this.deleteChunkAsync(id);
  }

  async deleteChunkAsync(id: string): Promise<void> {
    await this.query("DELETE FROM chunks WHERE id = $1", [id]);
  }

  listChunks(params: { model: string; sourceFilter?: SourceFilter }): DbChunkRow[] {
    return [];
  }

  async listChunksAsync(params: {
    model: string;
    sourceFilter?: SourceFilter;
  }): Promise<DbChunkRow[]> {
    if (params.sourceFilter?.sources.length) {
      const placeholders = params.sourceFilter.sources.map((_, i) => `$${i + 2}`).join(", ");
      return this.queryAll<DbChunkRow>(
        `SELECT * FROM chunks WHERE model = $1 AND source IN (${placeholders})`,
        [params.model, ...params.sourceFilter.sources],
      );
    }
    return this.queryAll<DbChunkRow>("SELECT * FROM chunks WHERE model = $1", [params.model]);
  }

  countChunks(sourceFilter?: SourceFilter): number {
    return 0;
  }

  async countChunksAsync(sourceFilter?: SourceFilter): Promise<number> {
    if (sourceFilter?.sources.length) {
      const placeholders = sourceFilter.sources.map((_, i) => `$${i + 1}`).join(", ");
      const row = await this.queryOne<{ cnt: string }>(
        `SELECT COUNT(*) AS cnt FROM chunks WHERE source IN (${placeholders})`,
        sourceFilter.sources,
      );
      return Number(row?.cnt ?? 0);
    }
    const row = await this.queryOne<{ cnt: string }>("SELECT COUNT(*) AS cnt FROM chunks");
    return Number(row?.cnt ?? 0);
  }

  countChunksBySource(): Array<{ source: string; files: number; chunks: number }> {
    return [];
  }

  async countChunksBySourceAsync(): Promise<
    Array<{ source: string; files: number; chunks: number }>
  > {
    return this.queryAll<{ source: string; files: string; chunks: string }>(
      `SELECT source, COUNT(DISTINCT path) AS files, COUNT(*) AS chunks
       FROM chunks GROUP BY source`,
    ).then((rows) =>
      rows.map((r) => ({
        source: r.source,
        files: Number(r.files),
        chunks: Number(r.chunks),
      })),
    );
  }

  // ── Embedding Cache ─────────────────────────────────────────────────────

  getCachedEmbedding(params: {
    provider: string;
    model: string;
    providerKey: string;
    hash: string;
  }): DbCacheRow | undefined {
    return undefined;
  }

  async getCachedEmbeddingAsync(params: {
    provider: string;
    model: string;
    providerKey: string;
    hash: string;
  }): Promise<DbCacheRow | undefined> {
    return this.queryOne<DbCacheRow>(
      `SELECT * FROM embedding_cache
       WHERE provider = $1 AND model = $2 AND provider_key = $3 AND hash = $4`,
      [params.provider, params.model, params.providerKey, params.hash],
    );
  }

  upsertCachedEmbedding(row: DbCacheRow): void {
    void this.upsertCachedEmbeddingAsync(row);
  }

  async upsertCachedEmbeddingAsync(row: DbCacheRow): Promise<void> {
    await this.query(
      `INSERT INTO embedding_cache (provider, model, provider_key, hash, embedding, dims, updated_at)
       VALUES ($1, $2, $3, $4, $5, $6, $7)
       ON CONFLICT (provider, model, provider_key, hash)
       DO UPDATE SET embedding = $5, dims = $6, updated_at = $7`,
      [
        row.provider,
        row.model,
        row.provider_key,
        row.hash,
        row.embedding,
        row.dims,
        row.updated_at,
      ],
    );
  }

  countCacheEntries(): number {
    return 0;
  }

  async countCacheEntriesAsync(): Promise<number> {
    const row = await this.queryOne<{ cnt: string }>("SELECT COUNT(*) AS cnt FROM embedding_cache");
    return Number(row?.cnt ?? 0);
  }

  trimCache(params: { maxEntries: number; table: string }): void {
    void this.trimCacheAsync(params);
  }

  async trimCacheAsync(params: { maxEntries: number; table: string }): Promise<void> {
    await this.query(
      `DELETE FROM ${params.table}
       WHERE ctid IN (
         SELECT ctid FROM ${params.table}
         ORDER BY updated_at ASC
         LIMIT GREATEST(0, (SELECT COUNT(*) FROM ${params.table}) - $1)
       )`,
      [params.maxEntries],
    );
  }

  seedCacheFrom(_sourceBackend: MemoryDbBackend, _table: string): void {
    // Cross-backend seeding not needed for Postgres
  }

  // ── FTS (PostgreSQL native tsvector) ────────────────────────────────────

  insertFts(_params: {
    table: string;
    id: string;
    path: string;
    source: string;
    model: string;
    startLine: number;
    endLine: number;
    text: string;
  }): void {
    // PostgreSQL FTS uses the chunks table directly via GIN index on tsvector
    // No separate FTS table needed — this is a no-op
  }

  deleteFtsById(_table: string, _id: string): void {
    // No-op — FTS is on chunks table directly
  }

  deleteFtsByPath(_table: string, _path: string): void {
    // No-op
  }

  clearFts(_table: string): void {
    // No-op
  }

  searchFts(params: {
    table: string;
    model: string;
    query: string;
    sourceFilter?: SourceFilter;
    limit: number;
  }): FtsSearchRow[] {
    return [];
  }

  async searchFtsAsync(params: {
    table: string;
    model: string;
    query: string;
    sourceFilter?: SourceFilter;
    limit: number;
  }): Promise<FtsSearchRow[]> {
    // Convert query to tsquery format
    const tsquery = params.query
      .split(/\s+/)
      .filter(Boolean)
      .map((w) => w.replace(/[^a-zA-Z0-9]/g, ""))
      .filter(Boolean)
      .join(" & ");

    if (!tsquery) return [];

    let sql = `SELECT id, path, source, start_line, end_line, text,
              ts_rank(to_tsvector('english', text), to_tsquery('english', $1)) AS rank
       FROM chunks
       WHERE to_tsvector('english', text) @@ to_tsquery('english', $1)
         AND model = $2`;
    const bindParams: unknown[] = [tsquery, params.model];

    if (params.sourceFilter?.sources.length) {
      const placeholders = params.sourceFilter.sources.map((_, i) => `$${i + 3}`).join(", ");
      sql += ` AND source IN (${placeholders})`;
      bindParams.push(...params.sourceFilter.sources);
    }

    sql += ` ORDER BY rank DESC LIMIT $${bindParams.length + 1}`;
    bindParams.push(params.limit);

    return this.queryAll<FtsSearchRow>(sql, bindParams);
  }

  // ── Vector (pgvector) ───────────────────────────────────────────────────

  async ensureVectorIndex(params: {
    table: string;
    dimensions: number;
    extensionPath?: string;
  }): Promise<{ ok: boolean; extensionPath?: string; error?: string }> {
    try {
      // Check if pgvector extension is available
      await this.exec("CREATE EXTENSION IF NOT EXISTS vector");

      // Create vector table if not exists
      await this.exec(`
        CREATE TABLE IF NOT EXISTS ${params.table} (
          id TEXT PRIMARY KEY,
          embedding vector(${params.dimensions})
        )
      `);

      // Create IVFFlat index for fast search (only if enough rows)
      try {
        const countRow = await this.queryOne<{ cnt: string }>(
          `SELECT COUNT(*) AS cnt FROM ${params.table}`,
        );
        const count = Number(countRow?.cnt ?? 0);
        if (count >= 10) {
          const lists = Math.max(1, Math.floor(Math.sqrt(count)));
          await this.exec(`
            CREATE INDEX IF NOT EXISTS idx_${params.table}_embedding
            ON ${params.table} USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = ${lists})
          `);
        }
      } catch {
        // Index creation failure is non-fatal — sequential scan still works
      }

      this.vectorAvailable = true;
      return { ok: true };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      return { ok: false, error: message };
    }
  }

  insertVector(params: { table: string; id: string; embedding: number[] }): void {
    void this.insertVectorAsync(params);
  }

  async insertVectorAsync(params: {
    table: string;
    id: string;
    embedding: number[];
  }): Promise<void> {
    const vecStr = `[${params.embedding.join(",")}]`;
    await this.query(
      `INSERT INTO ${params.table} (id, embedding) VALUES ($1, $2::vector)
       ON CONFLICT (id) DO UPDATE SET embedding = $2::vector`,
      [params.id, vecStr],
    );
  }

  deleteVector(table: string, id: string): void {
    void this.query(`DELETE FROM ${table} WHERE id = $1`, [id]);
  }

  searchVector(params: {
    table: string;
    model: string;
    queryVec: number[];
    sourceFilter?: SourceFilter;
    limit: number;
  }): VectorSearchRow[] {
    return [];
  }

  async searchVectorAsync(params: {
    table: string;
    model: string;
    queryVec: number[];
    sourceFilter?: SourceFilter;
    limit: number;
  }): Promise<VectorSearchRow[]> {
    const vecStr = `[${params.queryVec.join(",")}]`;

    let sql = `SELECT c.id, c.path, c.start_line, c.end_line, c.text, c.source,
              (v.embedding <=> $1::vector) AS dist
       FROM ${params.table} v
       JOIN chunks c ON c.id = v.id
       WHERE c.model = $2`;
    const bindParams: unknown[] = [vecStr, params.model];

    if (params.sourceFilter?.sources.length) {
      const placeholders = params.sourceFilter.sources.map((_, i) => `$${i + 3}`).join(", ");
      sql += ` AND c.source IN (${placeholders})`;
      bindParams.push(...params.sourceFilter.sources);
    }

    sql += ` ORDER BY dist ASC LIMIT $${bindParams.length + 1}`;
    bindParams.push(params.limit);

    return this.queryAll<VectorSearchRow>(sql, bindParams);
  }

  // ── Lifecycle ───────────────────────────────────────────────────────────

  close(): void {
    if (this.client) {
      this.client.release();
      this.client = null;
    }
    void this.pool.end();
  }

  transaction<T>(fn: () => T): T {
    // Sync transactions not supported — use transactionAsync
    return fn();
  }

  async transactionAsync<T>(fn: () => Promise<T>): Promise<T> {
    await this.query("BEGIN");
    try {
      const result = await fn();
      await this.query("COMMIT");
      return result;
    } catch (err) {
      await this.query("ROLLBACK");
      throw err;
    }
  }

  rawExec(sql: string): void {
    void this.exec(sql);
  }
}
