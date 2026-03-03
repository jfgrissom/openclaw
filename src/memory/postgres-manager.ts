/**
 * PostgreSQL-backed MemorySearchManager.
 *
 * Strategy Pattern: Implements the same MemorySearchManager interface as
 * MemoryIndexManager (SQLite), selected via config.memory.backend = "postgres".
 *
 * Reuses pure functions from internal.ts for chunking, hashing, file discovery.
 * Uses the same EmbeddingProvider system for generating embeddings.
 * Stores everything in PostgreSQL with pgvector for similarity search.
 *
 * SOLID:
 * - SRP: Only handles Postgres-backed memory operations
 * - OCP: Doesn't modify existing MemoryIndexManager
 * - LSP: Fully substitutable for MemoryIndexManager
 * - ISP: Implements exactly MemorySearchManager
 * - DIP: Depends on MemorySearchManager interface, not SQLite
 */

import fsSync from "node:fs";
import fs from "node:fs/promises";
import path from "node:path";
import chokidar, { type FSWatcher } from "chokidar";
import pg from "pg";
import { resolveAgentDir, resolveAgentWorkspaceDir } from "../agents/agent-scope.js";
import type { ResolvedMemorySearchConfig } from "../agents/memory-search.js";
import { resolveMemorySearchConfig } from "../agents/memory-search.js";
import type { OpenClawConfig } from "../config/config.js";
import { createSubsystemLogger } from "../logging/subsystem.js";
import { truncateUtf16Safe } from "../utils.js";
import { createEmbeddingProvider, type EmbeddingProvider } from "./embeddings.js";
import {
  buildFileEntry,
  chunkMarkdown,
  cosineSimilarity,
  hashText,
  isMemoryPath,
  listMemoryFiles,
  normalizeExtraMemoryPaths,
  parseEmbedding,
  type MemoryFileEntry,
} from "./internal.js";
import type {
  MemoryEmbeddingProbeResult,
  MemoryProviderStatus,
  MemorySearchManager,
  MemorySearchResult,
  MemorySource,
  MemorySyncProgressUpdate,
} from "./types.js";

const log = createSubsystemLogger("memory-pg");

const SNIPPET_MAX_CHARS = 700;
const PG_INDEX_CACHE = new Map<string, PostgresMemoryManager>();

export class PostgresMemoryManager implements MemorySearchManager {
  private readonly cacheKey: string;
  private readonly cfg: OpenClawConfig;
  private readonly agentId: string;
  private readonly workspaceDir: string;
  private readonly settings: ResolvedMemorySearchConfig;
  private provider: EmbeddingProvider;
  private pool: pg.Pool;
  private readonly sources: Set<MemorySource>;
  private watcher: FSWatcher | null = null;
  private dirty = false;
  private syncing: Promise<void> | null = null;
  private closed = false;
  private vectorReady = false;

  static async get(params: {
    cfg: OpenClawConfig;
    agentId: string;
    connectionString: string;
  }): Promise<PostgresMemoryManager | null> {
    const { cfg, agentId, connectionString } = params;
    const settings = resolveMemorySearchConfig(cfg, agentId);
    if (!settings) {
      return null;
    }
    const workspaceDir = resolveAgentWorkspaceDir(cfg, agentId);
    const key = `pg:${agentId}:${workspaceDir}:${connectionString}`;
    const existing = PG_INDEX_CACHE.get(key);
    if (existing) {
      return existing;
    }
    const providerResult = await createEmbeddingProvider({
      config: cfg,
      agentDir: resolveAgentDir(cfg, agentId),
      provider: settings.provider,
      remote: settings.remote,
      model: settings.model,
      fallback: settings.fallback,
      local: settings.local,
    });

    if (!providerResult.provider) {
      log.warn(
        `No embedding provider available for PostgreSQL memory backend: ${providerResult.providerUnavailableReason ?? "unknown reason"}`,
      );
      return null;
    }

    const manager = new PostgresMemoryManager({
      cacheKey: key,
      cfg,
      agentId,
      workspaceDir,
      settings,
      provider: providerResult.provider,
      connectionString,
    });
    await manager.init();
    PG_INDEX_CACHE.set(key, manager);
    return manager;
  }

  private constructor(params: {
    cacheKey: string;
    cfg: OpenClawConfig;
    agentId: string;
    workspaceDir: string;
    settings: ResolvedMemorySearchConfig;
    provider: EmbeddingProvider;
    connectionString: string;
  }) {
    this.cacheKey = params.cacheKey;
    this.cfg = params.cfg;
    this.agentId = params.agentId;
    this.workspaceDir = params.workspaceDir;
    this.settings = params.settings;
    this.provider = params.provider;
    this.sources = new Set(params.settings.sources);
    this.pool = new pg.Pool({
      connectionString: params.connectionString,
      max: 5,
      idleTimeoutMillis: 30_000,
    });
  }

  private async init(): Promise<void> {
    await this.ensureSchema();
    this.ensureWatcher();
    this.dirty = this.sources.has("memory");
  }

  // ── Schema ──────────────────────────────────────────────────────────────

  private async ensureSchema(): Promise<void> {
    const client = await this.pool.connect();
    try {
      await client.query(`
        CREATE TABLE IF NOT EXISTS meta (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        )
      `);
      await client.query(`
        CREATE TABLE IF NOT EXISTS files (
          path TEXT PRIMARY KEY,
          source TEXT NOT NULL DEFAULT 'memory',
          hash TEXT NOT NULL,
          mtime BIGINT NOT NULL,
          size BIGINT NOT NULL
        )
      `);
      await client.query(`
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
        )
      `);
      await client.query(`CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path)`);
      await client.query(`CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source)`);
      await client.query(`
        CREATE INDEX IF NOT EXISTS idx_chunks_fts
        ON chunks USING gin(to_tsvector('english', text))
      `);
      await client.query(`
        CREATE TABLE IF NOT EXISTS embedding_cache (
          provider TEXT NOT NULL,
          model TEXT NOT NULL,
          provider_key TEXT NOT NULL,
          hash TEXT NOT NULL,
          embedding TEXT NOT NULL,
          dims INTEGER,
          updated_at BIGINT NOT NULL,
          PRIMARY KEY (provider, model, provider_key, hash)
        )
      `);

      // pgvector
      try {
        await client.query("CREATE EXTENSION IF NOT EXISTS vector");
        await client.query(`
          CREATE TABLE IF NOT EXISTS chunks_vec (
            id TEXT PRIMARY KEY,
            embedding vector(768)
          )
        `);
        this.vectorReady = true;
        log.info("pgvector initialized successfully");
      } catch (err) {
        log.warn(`pgvector not available: ${String(err)}`);
      }
    } finally {
      client.release();
    }
  }

  // ── Search (the critical interface) ─────────────────────────────────────

  async search(
    query: string,
    opts?: {
      maxResults?: number;
      minScore?: number;
      sessionKey?: string;
    },
  ): Promise<MemorySearchResult[]> {
    if (this.dirty) {
      void this.sync({ reason: "search" }).catch((err) => {
        log.warn(`memory sync failed (search): ${String(err)}`);
      });
    }

    const cleaned = query.trim();
    if (!cleaned) {
      return [];
    }

    const minScore = opts?.minScore ?? this.settings.query.minScore;
    const maxResults = opts?.maxResults ?? this.settings.query.maxResults;

    // Generate query embedding
    const queryVec = await this.embedText(cleaned);
    const hasVector = queryVec.some((v) => v !== 0);

    // Vector search via pgvector
    let vectorResults: MemorySearchResult[] = [];
    if (hasVector && this.vectorReady) {
      vectorResults = await this.searchVectorPg(queryVec, maxResults * 2);
    } else if (hasVector) {
      // Fallback: brute-force cosine similarity
      vectorResults = await this.searchVectorBruteForce(queryVec, maxResults * 2);
    }

    // FTS search via PostgreSQL tsvector
    let ftsResults: MemorySearchResult[] = [];
    if (this.settings.query.hybrid.enabled) {
      ftsResults = await this.searchFtsPg(cleaned, maxResults * 2);
    }

    // Merge results
    if (!this.settings.query.hybrid.enabled) {
      return vectorResults.filter((r) => r.score >= minScore).slice(0, maxResults);
    }

    const merged = this.mergeResults(vectorResults, ftsResults);
    return merged.filter((r) => r.score >= minScore).slice(0, maxResults);
  }

  private async searchVectorPg(queryVec: number[], limit: number): Promise<MemorySearchResult[]> {
    const vecStr = `[${queryVec.join(",")}]`;
    const sourceFilter = this.buildSourceFilter();

    let sql = `
      SELECT c.id, c.path, c.start_line, c.end_line, c.text, c.source,
             (1 - (v.embedding <=> $1::vector)) AS score
      FROM chunks_vec v
      JOIN chunks c ON c.id = v.id
      WHERE c.model = $2`;
    const params: unknown[] = [vecStr, this.provider.model];

    if (sourceFilter.length > 0) {
      const placeholders = sourceFilter.map((_, i) => `$${i + 3}`).join(", ");
      sql += ` AND c.source IN (${placeholders})`;
      params.push(...sourceFilter);
    }

    sql += ` ORDER BY v.embedding <=> $1::vector ASC LIMIT $${params.length + 1}`;
    params.push(limit);

    const result = await this.pool.query(sql, params);
    return result.rows.map((row: Record<string, unknown>) => ({
      path: row.path as string,
      startLine: row.start_line as number,
      endLine: row.end_line as number,
      score: Number(row.score),
      snippet: truncateUtf16Safe(row.text as string, SNIPPET_MAX_CHARS),
      source: row.source as MemorySource,
    }));
  }

  private async searchVectorBruteForce(
    queryVec: number[],
    limit: number,
  ): Promise<MemorySearchResult[]> {
    const sourceFilter = this.buildSourceFilter();
    let sql = `SELECT id, path, start_line, end_line, text, embedding, source
               FROM chunks WHERE model = $1`;
    const params: unknown[] = [this.provider.model];

    if (sourceFilter.length > 0) {
      const placeholders = sourceFilter.map((_, i) => `$${i + 2}`).join(", ");
      sql += ` AND source IN (${placeholders})`;
      params.push(...sourceFilter);
    }

    const result = await this.pool.query(sql, params);
    const scored = result.rows
      .map((row: Record<string, unknown>) => {
        const embedding = parseEmbedding(row.embedding as string);
        const score = cosineSimilarity(queryVec, embedding);
        return { row, score };
      })
      .filter((entry) => Number.isFinite(entry.score))
      .toSorted((a, b) => b.score - a.score)
      .slice(0, limit);

    return scored.map((entry) => ({
      path: entry.row.path as string,
      startLine: entry.row.start_line as number,
      endLine: entry.row.end_line as number,
      score: entry.score,
      snippet: truncateUtf16Safe(entry.row.text as string, SNIPPET_MAX_CHARS),
      source: entry.row.source as MemorySource,
    }));
  }

  private async searchFtsPg(query: string, limit: number): Promise<MemorySearchResult[]> {
    const tsquery = query
      .split(/\s+/)
      .filter(Boolean)
      .map((w) => w.replace(/[^a-zA-Z0-9]/g, ""))
      .filter(Boolean)
      .join(" & ");

    if (!tsquery) {
      return [];
    }

    const sourceFilter = this.buildSourceFilter();
    let sql = `
      SELECT id, path, source, start_line, end_line, text,
             ts_rank(to_tsvector('english', text), to_tsquery('english', $1)) AS rank
      FROM chunks
      WHERE to_tsvector('english', text) @@ to_tsquery('english', $1)
        AND model = $2`;
    const params: unknown[] = [tsquery, this.provider.model];

    if (sourceFilter.length > 0) {
      const placeholders = sourceFilter.map((_, i) => `$${i + 3}`).join(", ");
      sql += ` AND source IN (${placeholders})`;
      params.push(...sourceFilter);
    }

    sql += ` ORDER BY rank DESC LIMIT $${params.length + 1}`;
    params.push(limit);

    const result = await this.pool.query(sql, params);
    return result.rows.map((row: Record<string, unknown>) => ({
      path: row.path as string,
      startLine: row.start_line as number,
      endLine: row.end_line as number,
      score: Math.min(1, Number(row.rank) * 2), // Normalize ts_rank
      snippet: truncateUtf16Safe(row.text as string, SNIPPET_MAX_CHARS),
      source: row.source as MemorySource,
    }));
  }

  private mergeResults(
    vector: MemorySearchResult[],
    fts: MemorySearchResult[],
  ): MemorySearchResult[] {
    const vWeight = this.settings.query.hybrid.vectorWeight;
    const tWeight = this.settings.query.hybrid.textWeight;
    const seen = new Map<string, MemorySearchResult>();

    for (const r of vector) {
      const key = `${r.path}:${r.startLine}`;
      seen.set(key, { ...r, score: r.score * vWeight });
    }

    for (const r of fts) {
      const key = `${r.path}:${r.startLine}`;
      const existing = seen.get(key);
      if (existing) {
        existing.score += r.score * tWeight;
      } else {
        seen.set(key, { ...r, score: r.score * tWeight });
      }
    }

    return Array.from(seen.values()).toSorted((a, b) => b.score - a.score);
  }

  // ── readFile (filesystem — identical logic to SQLite manager) ───────────

  async readFile(params: {
    relPath: string;
    from?: number;
    lines?: number;
  }): Promise<{ text: string; path: string }> {
    const rawPath = params.relPath.trim();
    if (!rawPath) {
      throw new Error("path required");
    }

    const absPath = path.isAbsolute(rawPath)
      ? path.resolve(rawPath)
      : path.resolve(this.workspaceDir, rawPath);
    const relPath = path.relative(this.workspaceDir, absPath).replace(/\\/g, "/");

    const inWorkspace =
      relPath.length > 0 && !relPath.startsWith("..") && !path.isAbsolute(relPath);
    const allowedWorkspace = inWorkspace && isMemoryPath(relPath);

    let allowedAdditional = false;
    if (!allowedWorkspace && this.settings.extraPaths.length > 0) {
      const additionalPaths = normalizeExtraMemoryPaths(
        this.workspaceDir,
        this.settings.extraPaths,
      );
      for (const additionalPath of additionalPaths) {
        try {
          const stat = await fs.lstat(additionalPath);
          if (stat.isSymbolicLink()) {
            continue;
          }
          if (stat.isDirectory()) {
            if (absPath === additionalPath || absPath.startsWith(`${additionalPath}${path.sep}`)) {
              allowedAdditional = true;
              break;
            }
            continue;
          }
          if (stat.isFile() && absPath === additionalPath && absPath.endsWith(".md")) {
            allowedAdditional = true;
            break;
          }
        } catch {}
      }
    }

    if (!allowedWorkspace && !allowedAdditional) {
      throw new Error("path required");
    }
    if (!absPath.endsWith(".md")) {
      throw new Error("path required");
    }

    const stat = await fs.lstat(absPath);
    if (stat.isSymbolicLink() || !stat.isFile()) {
      throw new Error("path required");
    }

    const content = await fs.readFile(absPath, "utf-8");
    if (!params.from && !params.lines) {
      return { text: content, path: relPath };
    }

    const lines = content.split("\n");
    const start = Math.max(1, params.from ?? 1);
    const count = Math.max(1, params.lines ?? lines.length);
    const slice = lines.slice(start - 1, start - 1 + count);
    return { text: slice.join("\n"), path: relPath };
  }

  // ── Sync (index files into Postgres) ────────────────────────────────────

  async sync(params?: {
    reason?: string;
    force?: boolean;
    progress?: (update: MemorySyncProgressUpdate) => void;
  }): Promise<void> {
    if (this.syncing) {
      return this.syncing;
    }
    this.syncing = this.runSync(params).finally(() => {
      this.syncing = null;
    });
    return this.syncing;
  }

  private async runSync(params?: {
    reason?: string;
    force?: boolean;
    progress?: (update: MemorySyncProgressUpdate) => void;
  }): Promise<void> {
    const memoryFiles = await listMemoryFiles(this.workspaceDir, this.settings.extraPaths);
    const filesToProcess: MemoryFileEntry[] = [];

    for (const absPath of memoryFiles) {
      try {
        const entry = await buildFileEntry(absPath, this.workspaceDir);
        if (!entry) {
          continue;
        }
        // Check if file has changed
        const existing = await this.pool.query("SELECT hash FROM files WHERE path = $1", [
          entry.path,
        ]);
        if (!params?.force && existing.rows.length > 0 && existing.rows[0].hash === entry.hash) {
          continue;
        }
        filesToProcess.push(entry);
      } catch {
        continue;
      }
    }

    if (filesToProcess.length === 0) {
      this.dirty = false;
      return;
    }

    log.info(`Syncing ${filesToProcess.length} files to PostgreSQL`);
    const progress = params?.progress;
    let completed = 0;

    for (const fileEntry of filesToProcess) {
      try {
        await this.indexFile(fileEntry);
        completed++;
        progress?.({ completed, total: filesToProcess.length, label: fileEntry.path });
      } catch (err) {
        log.warn(`Failed to index ${fileEntry.path}: ${String(err)}`);
      }
    }

    // Clean up files that no longer exist
    const currentPaths = new Set(
      memoryFiles.map((absPath) => path.relative(this.workspaceDir, absPath).replace(/\\/g, "/")),
    );
    const dbFiles = await this.pool.query("SELECT path FROM files WHERE source = 'memory'");
    for (const row of dbFiles.rows) {
      if (!currentPaths.has(row.path)) {
        await this.pool.query("DELETE FROM chunks WHERE path = $1", [row.path]);
        await this.pool
          .query("DELETE FROM chunks_vec WHERE id IN (SELECT id FROM chunks WHERE path = $1)", [
            row.path,
          ])
          .catch(() => {});
        await this.pool.query("DELETE FROM files WHERE path = $1", [row.path]);
      }
    }

    this.dirty = false;
    log.info(`Sync complete: ${completed}/${filesToProcess.length} files indexed`);
  }

  private async indexFile(fileEntry: MemoryFileEntry): Promise<void> {
    const content = await fs.readFile(fileEntry.absPath, "utf-8");
    const chunks = chunkMarkdown(content, {
      tokens: this.settings.chunking.tokens,
      overlap: this.settings.chunking.overlap,
    });

    // Delete old chunks for this file
    const oldChunks = await this.pool.query("SELECT id FROM chunks WHERE path = $1", [
      fileEntry.path,
    ]);
    for (const row of oldChunks.rows) {
      await this.pool.query("DELETE FROM chunks_vec WHERE id = $1", [row.id]).catch(() => {});
    }
    await this.pool.query("DELETE FROM chunks WHERE path = $1", [fileEntry.path]);

    // Generate embeddings for all chunks
    const texts = chunks.map((c) => c.text);
    const embeddings = await this.embedBatch(texts);

    // Insert new chunks + vectors
    const client = await this.pool.connect();
    try {
      await client.query("BEGIN");

      // Upsert file record
      await client.query(
        `INSERT INTO files (path, source, hash, mtime, size)
         VALUES ($1, 'memory', $2, $3, $4)
         ON CONFLICT (path) DO UPDATE SET hash = $2, mtime = $3, size = $4`,
        [fileEntry.path, fileEntry.hash, Math.floor(fileEntry.mtimeMs), fileEntry.size],
      );

      for (let i = 0; i < chunks.length; i++) {
        const chunk = chunks[i];
        const embedding = embeddings[i] ?? [];
        const chunkId = hashText(`${fileEntry.path}:${chunk.startLine}:${chunk.hash}`);

        await client.query(
          `INSERT INTO chunks (id, path, source, start_line, end_line, hash, model, text, embedding, updated_at)
           VALUES ($1, $2, 'memory', $3, $4, $5, $6, $7, $8, $9)
           ON CONFLICT (id) DO UPDATE SET
             text = $7, embedding = $8, hash = $5, updated_at = $9`,
          [
            chunkId,
            fileEntry.path,
            chunk.startLine,
            chunk.endLine,
            chunk.hash,
            this.provider.model,
            chunk.text,
            JSON.stringify(embedding),
            Date.now(),
          ],
        );

        // Insert vector for pgvector search
        if (embedding.length > 0 && this.vectorReady) {
          const vecStr = `[${embedding.join(",")}]`;
          await client
            .query(
              `INSERT INTO chunks_vec (id, embedding)
             VALUES ($1, $2::vector)
             ON CONFLICT (id) DO UPDATE SET embedding = $2::vector`,
              [chunkId, vecStr],
            )
            .catch((err) => {
              log.warn(`Failed to insert vector for ${chunkId}: ${String(err)}`);
            });
        }
      }

      await client.query("COMMIT");
    } catch (err) {
      await client.query("ROLLBACK");
      throw err;
    } finally {
      client.release();
    }
  }

  // ── Embedding helpers ───────────────────────────────────────────────────

  private async embedText(text: string): Promise<number[]> {
    try {
      const result = await this.provider.embedQuery(text);
      return result;
    } catch (err) {
      log.warn(`Embedding failed: ${String(err)}`);
      return [];
    }
  }

  private async embedBatch(texts: string[]): Promise<number[][]> {
    const results: number[][] = [];
    // Process in batches to respect rate limits
    for (const text of texts) {
      results.push(await this.embedText(text));
    }
    return results;
  }

  // ── Status ──────────────────────────────────────────────────────────────

  status(): MemoryProviderStatus {
    // Return sync-safe status (counts queried async on demand)
    return {
      backend: "builtin", // Report as builtin for compatibility
      provider: this.provider.id,
      model: this.provider.model,
      files: 0, // Async — use statusAsync() for accurate counts
      chunks: 0,
      dirty: this.dirty,
      workspaceDir: this.workspaceDir,
      sources: Array.from(this.sources),
      custom: { database: "postgres" },
    };
  }

  async statusAsync(): Promise<MemoryProviderStatus> {
    const base = this.status();
    try {
      const files = await this.pool.query("SELECT COUNT(*) AS c FROM files");
      const chunks = await this.pool.query("SELECT COUNT(*) AS c FROM chunks");
      base.files = Number(files.rows[0]?.c ?? 0);
      base.chunks = Number(chunks.rows[0]?.c ?? 0);
    } catch {}
    return base;
  }

  async probeEmbeddingAvailability(): Promise<MemoryEmbeddingProbeResult> {
    try {
      await this.embedText("ping");
      return { ok: true };
    } catch (err) {
      return { ok: false, error: err instanceof Error ? err.message : String(err) };
    }
  }

  async probeVectorAvailability(): Promise<boolean> {
    return this.vectorReady;
  }

  async close(): Promise<void> {
    if (this.closed) {
      return;
    }
    this.closed = true;
    if (this.watcher) {
      await this.watcher.close();
      this.watcher = null;
    }
    await this.pool.end();
    PG_INDEX_CACHE.delete(this.cacheKey);
  }

  // ── File watcher ────────────────────────────────────────────────────────

  private ensureWatcher(): void {
    if (this.watcher || !this.sources.has("memory")) {
      return;
    }

    const memDir = path.join(this.workspaceDir, "memory");
    const memFile = path.join(this.workspaceDir, "MEMORY.md");
    const watchPaths = [memDir, memFile].filter((p) => {
      try {
        return fsSync.existsSync(p);
      } catch {
        return false;
      }
    });

    if (watchPaths.length === 0) {
      return;
    }

    this.watcher = chokidar.watch(watchPaths, {
      ignoreInitial: true,
      persistent: false,
      awaitWriteFinish: { stabilityThreshold: 500 },
    });

    this.watcher.on("all", () => {
      this.dirty = true;
    });
  }

  // ── Helpers ─────────────────────────────────────────────────────────────

  private buildSourceFilter(): string[] {
    return Array.from(this.sources);
  }
}
