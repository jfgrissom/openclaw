/**
 * SQLite implementation of MemoryDbBackend.
 *
 * This wraps the existing SQLite behavior with zero functional changes.
 * All SQL is preserved exactly as-is from the original manager.ts.
 */

import type { DatabaseSync } from "node:sqlite";
import path from "node:path";
import type {
  DbCacheRow,
  DbChunkRow,
  DbFileRow,
  FtsSearchRow,
  MemoryDbBackend,
  SourceFilter,
  VectorSearchRow,
} from "./db-backend.js";
import { resolveUserPath } from "../utils.js";
import { ensureDir } from "./internal.js";
import { ensureMemoryIndexSchema } from "./memory-schema.js";
import { loadSqliteVecExtension } from "./sqlite-vec.js";
import { requireNodeSqlite } from "./sqlite.js";

const vectorToBlob = (embedding: number[]): Buffer =>
  Buffer.from(new Float32Array(embedding).buffer);

export class SqliteBackend implements MemoryDbBackend {
  readonly type = "sqlite" as const;
  private db: DatabaseSync;
  private readonly dbPath: string;
  private readonly allowExtension: boolean;

  constructor(params: { dbPath: string; allowExtension: boolean }) {
    this.dbPath = resolveUserPath(params.dbPath);
    this.allowExtension = params.allowExtension;
    this.db = this.openDb();
  }

  private openDb(): DatabaseSync {
    const dir = path.dirname(this.dbPath);
    ensureDir(dir);
    const { DatabaseSync } = requireNodeSqlite();
    return new DatabaseSync(this.dbPath, { allowExtension: this.allowExtension });
  }

  // ── Schema ──────────────────────────────────────────────────────────────

  ensureSchema(params: { embeddingCacheTable: string; ftsTable: string; ftsEnabled: boolean }): {
    ftsAvailable: boolean;
    ftsError?: string;
  } {
    return ensureMemoryIndexSchema({
      db: this.db,
      embeddingCacheTable: params.embeddingCacheTable,
      ftsTable: params.ftsTable,
      ftsEnabled: params.ftsEnabled,
    });
  }

  // ── Meta ────────────────────────────────────────────────────────────────

  getMeta(key: string): string | undefined {
    const row = this.db.prepare("SELECT value FROM meta WHERE key = ?").get(key) as
      | { value: string }
      | undefined;
    return row?.value;
  }

  setMeta(key: string, value: string): void {
    this.db.prepare("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)").run(key, value);
  }

  // ── Files ───────────────────────────────────────────────────────────────

  getFile(filePath: string): DbFileRow | undefined {
    return this.db
      .prepare("SELECT path, source, hash, mtime, size FROM files WHERE path = ?")
      .get(filePath) as DbFileRow | undefined;
  }

  upsertFile(row: DbFileRow): void {
    this.db
      .prepare(
        "INSERT OR REPLACE INTO files (path, source, hash, mtime, size) VALUES (?, ?, ?, ?, ?)",
      )
      .run(row.path, row.source, row.hash, row.mtime, row.size);
  }

  deleteFile(filePath: string): void {
    this.db.prepare("DELETE FROM files WHERE path = ?").run(filePath);
  }

  listFiles(source?: string): DbFileRow[] {
    if (source) {
      return this.db
        .prepare("SELECT path, source, hash, mtime, size FROM files WHERE source = ?")
        .all(source) as DbFileRow[];
    }
    return this.db
      .prepare("SELECT path, source, hash, mtime, size FROM files")
      .all() as DbFileRow[];
  }

  countFiles(sourceFilter?: SourceFilter): number {
    if (sourceFilter?.sources.length) {
      const placeholders = sourceFilter.sources.map(() => "?").join(", ");
      const row = this.db
        .prepare(`SELECT COUNT(*) AS cnt FROM files WHERE source IN (${placeholders})`)
        .get(...sourceFilter.sources) as { cnt: number };
      return row.cnt;
    }
    const row = this.db.prepare("SELECT COUNT(*) AS cnt FROM files").get() as { cnt: number };
    return row.cnt;
  }

  // ── Chunks ──────────────────────────────────────────────────────────────

  getChunk(id: string): DbChunkRow | undefined {
    return this.db.prepare("SELECT * FROM chunks WHERE id = ?").get(id) as DbChunkRow | undefined;
  }

  getChunksByPath(chunkPath: string): DbChunkRow[] {
    return this.db.prepare("SELECT * FROM chunks WHERE path = ?").all(chunkPath) as DbChunkRow[];
  }

  upsertChunk(row: DbChunkRow): void {
    this.db
      .prepare(
        `INSERT OR REPLACE INTO chunks (id, path, source, start_line, end_line, hash, model, text, embedding, updated_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(
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
      );
  }

  deleteChunksByPath(chunkPath: string): void {
    this.db.prepare("DELETE FROM chunks WHERE path = ?").run(chunkPath);
  }

  deleteChunk(id: string): void {
    this.db.prepare("DELETE FROM chunks WHERE id = ?").run(id);
  }

  listChunks(params: { model: string; sourceFilter?: SourceFilter }): DbChunkRow[] {
    if (params.sourceFilter?.sources.length) {
      const placeholders = params.sourceFilter.sources.map(() => "?").join(", ");
      return this.db
        .prepare(`SELECT * FROM chunks WHERE model = ? AND source IN (${placeholders})`)
        .all(params.model, ...params.sourceFilter.sources) as DbChunkRow[];
    }
    return this.db
      .prepare("SELECT * FROM chunks WHERE model = ?")
      .all(params.model) as DbChunkRow[];
  }

  countChunks(sourceFilter?: SourceFilter): number {
    if (sourceFilter?.sources.length) {
      const placeholders = sourceFilter.sources.map(() => "?").join(", ");
      const row = this.db
        .prepare(`SELECT COUNT(*) AS cnt FROM chunks WHERE source IN (${placeholders})`)
        .get(...sourceFilter.sources) as { cnt: number };
      return row.cnt;
    }
    const row = this.db.prepare("SELECT COUNT(*) AS cnt FROM chunks").get() as { cnt: number };
    return row.cnt;
  }

  countChunksBySource(): Array<{ source: string; files: number; chunks: number }> {
    return this.db
      .prepare(
        `SELECT source, COUNT(DISTINCT path) AS files, COUNT(*) AS chunks
         FROM chunks GROUP BY source`,
      )
      .all() as Array<{ source: string; files: number; chunks: number }>;
  }

  // ── Embedding Cache ─────────────────────────────────────────────────────

  getCachedEmbedding(params: {
    provider: string;
    model: string;
    providerKey: string;
    hash: string;
  }): DbCacheRow | undefined {
    return this.db
      .prepare(
        `SELECT * FROM embedding_cache
         WHERE provider = ? AND model = ? AND provider_key = ? AND hash = ?`,
      )
      .get(params.provider, params.model, params.providerKey, params.hash) as
      | DbCacheRow
      | undefined;
  }

  upsertCachedEmbedding(row: DbCacheRow): void {
    this.db
      .prepare(
        `INSERT OR REPLACE INTO embedding_cache
         (provider, model, provider_key, hash, embedding, dims, updated_at)
         VALUES (?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(
        row.provider,
        row.model,
        row.provider_key,
        row.hash,
        row.embedding,
        row.dims,
        row.updated_at,
      );
  }

  countCacheEntries(): number {
    const row = this.db.prepare("SELECT COUNT(*) AS cnt FROM embedding_cache").get() as {
      cnt: number;
    };
    return row.cnt;
  }

  trimCache(params: { maxEntries: number; table: string }): void {
    this.db
      .prepare(
        `DELETE FROM ${params.table}
         WHERE rowid IN (
           SELECT rowid FROM ${params.table}
           ORDER BY updated_at ASC
           LIMIT MAX(0, (SELECT COUNT(*) FROM ${params.table}) - ?)
         )`,
      )
      .run(params.maxEntries);
  }

  seedCacheFrom(sourceBackend: MemoryDbBackend, table: string): void {
    // Only works SQLite → SQLite (for atomic reindex swap)
    if (sourceBackend.type !== "sqlite") return;
    const source = sourceBackend as SqliteBackend;
    const rows = source.db.prepare(`SELECT * FROM ${table}`).all() as DbCacheRow[];
    for (const row of rows) {
      this.upsertCachedEmbedding(row);
    }
  }

  // ── FTS ─────────────────────────────────────────────────────────────────

  insertFts(params: {
    table: string;
    id: string;
    path: string;
    source: string;
    model: string;
    startLine: number;
    endLine: number;
    text: string;
  }): void {
    this.db
      .prepare(
        `INSERT INTO ${params.table} (text, id, path, source, model, start_line, end_line)
         VALUES (?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(
        params.text,
        params.id,
        params.path,
        params.source,
        params.model,
        params.startLine,
        params.endLine,
      );
  }

  deleteFtsById(table: string, id: string): void {
    this.db.prepare(`DELETE FROM ${table} WHERE id = ?`).run(id);
  }

  deleteFtsByPath(table: string, chunkPath: string): void {
    this.db.prepare(`DELETE FROM ${table} WHERE path = ?`).run(chunkPath);
  }

  clearFts(table: string): void {
    this.db.exec(`DELETE FROM ${table}`);
  }

  searchFts(params: {
    table: string;
    model: string;
    query: string;
    sourceFilter?: SourceFilter;
    limit: number;
  }): FtsSearchRow[] {
    let sql =
      `SELECT id, path, source, start_line, end_line, text,\n` +
      `       bm25(${params.table}) AS rank\n` +
      `  FROM ${params.table}\n` +
      ` WHERE ${params.table} MATCH ? AND model = ?`;
    const bindParams: unknown[] = [params.query, params.model];

    if (params.sourceFilter?.sources.length) {
      const placeholders = params.sourceFilter.sources.map(() => "?").join(", ");
      sql += ` AND source IN (${placeholders})`;
      bindParams.push(...params.sourceFilter.sources);
    }

    sql += ` ORDER BY rank ASC LIMIT ?`;
    bindParams.push(params.limit);

    return this.db.prepare(sql).all(...bindParams) as FtsSearchRow[];
  }

  // ── Vector ──────────────────────────────────────────────────────────────

  async ensureVectorIndex(params: {
    table: string;
    dimensions: number;
    extensionPath?: string;
  }): Promise<{ ok: boolean; extensionPath?: string; error?: string }> {
    const result = await loadSqliteVecExtension({
      db: this.db,
      extensionPath: params.extensionPath,
    });
    if (!result.ok) return result;

    try {
      this.db.exec(
        `CREATE VIRTUAL TABLE IF NOT EXISTS ${params.table}
         USING vec0(id TEXT PRIMARY KEY, embedding float[${params.dimensions}])`,
      );
    } catch (err) {
      // Table may already exist with different dimensions — that's fine
    }

    return result;
  }

  insertVector(params: { table: string; id: string; embedding: number[] }): void {
    this.db
      .prepare(`INSERT OR REPLACE INTO ${params.table} (id, embedding) VALUES (?, ?)`)
      .run(params.id, vectorToBlob(params.embedding));
  }

  deleteVector(table: string, id: string): void {
    this.db.prepare(`DELETE FROM ${table} WHERE id = ?`).run(id);
  }

  searchVector(params: {
    table: string;
    model: string;
    queryVec: number[];
    sourceFilter?: SourceFilter;
    limit: number;
  }): VectorSearchRow[] {
    let sourceSql = "";
    const sourceParams: string[] = [];
    if (params.sourceFilter?.sources.length) {
      const placeholders = params.sourceFilter.sources.map(() => "?").join(", ");
      sourceSql = ` AND c.source IN (${placeholders})`;
      sourceParams.push(...params.sourceFilter.sources);
    }

    return this.db
      .prepare(
        `SELECT c.id, c.path, c.start_line, c.end_line, c.text,\n` +
          `       c.source,\n` +
          `       vec_distance_cosine(v.embedding, ?) AS dist\n` +
          `  FROM ${params.table} v\n` +
          `  JOIN chunks c ON c.id = v.id\n` +
          ` WHERE c.model = ?${sourceSql}\n` +
          ` ORDER BY dist ASC\n` +
          ` LIMIT ?`,
      )
      .all(
        vectorToBlob(params.queryVec),
        params.model,
        ...sourceParams,
        params.limit,
      ) as VectorSearchRow[];
  }

  // ── Lifecycle ───────────────────────────────────────────────────────────

  close(): void {
    try {
      this.db.close();
    } catch {}
  }

  transaction<T>(fn: () => T): T {
    // node:sqlite doesn't have a built-in transaction API on DatabaseSync,
    // but we can use exec for BEGIN/COMMIT/ROLLBACK
    this.db.exec("BEGIN");
    try {
      const result = fn();
      this.db.exec("COMMIT");
      return result;
    } catch (err) {
      this.db.exec("ROLLBACK");
      throw err;
    }
  }

  rawExec(sql: string): void {
    this.db.exec(sql);
  }

  // ── Internal (for SqliteBackend-specific operations) ────────────────────

  /** Expose the raw DatabaseSync for operations not yet abstracted */
  get rawDb(): DatabaseSync {
    return this.db;
  }

  /** Re-open the database (used during atomic reindex swap) */
  reopen(): void {
    try {
      this.db.close();
    } catch {}
    this.db = this.openDb();
  }
}
