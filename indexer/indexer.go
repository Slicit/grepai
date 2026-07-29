package indexer

import (
	"context"
	"fmt"
	"log"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/sync/errgroup"

	"github.com/yoanbernabeu/grepai/embedder"
	"github.com/yoanbernabeu/grepai/framework"
	"github.com/yoanbernabeu/grepai/store"
)

type Indexer struct {
	root          string
	store         store.VectorStore
	embedder      embedder.Embedder
	chunker       *Chunker
	scanner       *Scanner
	processor     *framework.ProcessorRegistry
	lastIndexTime time.Time
}

type IndexStats struct {
	FilesIndexed  int
	FilesSkipped  int
	ChunksCreated int
	FilesRemoved  int
	Duration      time.Duration
	ScannedFiles  []FileMeta // All files found during scan (for reuse by callers)
}

// ProgressInfo contains progress information for indexing
type ProgressInfo struct {
	Current     int    // Current file number (1-indexed)
	Total       int    // Total number of files
	CurrentFile string // Path of current file being processed
}

// ProgressCallback is called for each file during indexing
type ProgressCallback func(info ProgressInfo)

// BatchProgressInfo contains progress information for batch embedding
type BatchProgressInfo struct {
	BatchIndex      int // Current batch index (0-indexed)
	TotalBatches    int // Total number of batches
	CompletedChunks int // Number of chunks completed so far
	TotalChunks     int // Total number of chunks to embed
	// Provisional is true while the scan/decide phase is still running
	// (see IndexAllWithBatchProgress): because embedding overlaps with
	// scanning, TotalChunks only reflects what's been discovered so far
	// and will keep growing as later waves of files are decided and
	// queued. It defaults to false (matching prior behavior for any
	// existing caller that doesn't set it) and is only set true by
	// IndexAllWithBatchProgress itself while more waves may still be
	// coming. Once the scan/decide phase finishes, no more waves can be
	// added, Provisional goes back to false, and TotalChunks stops
	// changing -- only then does CompletedChunks/TotalChunks represent
	// real completion. Callers rendering a percentage or "done" state
	// should treat a Provisional total as not-yet-final, since a mid-run
	// wave completing at its own 100% is not the same as the whole run
	// being done.
	Provisional bool
	Retrying    bool // True if this is a retry attempt
	Attempt     int  // Retry attempt number (1-indexed, 0 if not retrying)
	StatusCode  int  // HTTP status code when retrying (429 = rate limited, 5xx = server error)
}

// BatchProgressCallback is called for batch embedding progress and retry visibility
type BatchProgressCallback func(info BatchProgressInfo)

func NewIndexer(
	root string,
	st store.VectorStore,
	emb embedder.Embedder,
	chunker *Chunker,
	scanner *Scanner,
	lastIndexTime time.Time,
	processors ...*framework.ProcessorRegistry,
) *Indexer {
	var processor *framework.ProcessorRegistry
	if len(processors) > 0 {
		processor = processors[0]
	}

	return &Indexer{
		root:          root,
		store:         st,
		embedder:      emb,
		chunker:       chunker,
		scanner:       scanner,
		processor:     processor,
		lastIndexTime: lastIndexTime,
	}
}

// IndexAll performs a full index of the project (no progress reporting)
func (idx *Indexer) IndexAll(ctx context.Context) (*IndexStats, error) {
	return idx.IndexAllWithProgress(ctx, nil)
}

// IndexAllWithProgress performs a full index with progress reporting
func (idx *Indexer) IndexAllWithProgress(ctx context.Context, onProgress ProgressCallback) (*IndexStats, error) {
	return idx.IndexAllWithBatchProgress(ctx, onProgress, nil)
}

// IndexAllWithBatchProgress performs a full index with both file and batch progress reporting.
// When the embedder implements BatchEmbedder, files are processed in parallel using cross-file batching.
func (idx *Indexer) IndexAllWithBatchProgress(ctx context.Context, onProgress ProgressCallback, onBatchProgress BatchProgressCallback) (*IndexStats, error) {
	start := time.Now()
	stats := &IndexStats{}

	// Guarantee a final flush to durable storage no matter how this
	// function returns (success, error, or a canceled context) -- this is
	// what makes a run stopped partway through resumable rather than lost:
	// combined with the periodic checkpoint flushes below, a crash leaves
	// at most one checkpoint interval's worth of progress unsaved, not the
	// entire run. Cheap/no-op on Postgres and Qdrant; see checkpoint above.
	cp := newCheckpoint(ctx, idx)
	defer func() {
		if err := idx.store.Persist(ctx); err != nil {
			log.Printf("Warning: failed to persist index on exit: %v", err)
		}
	}()

	// Scan all files (metadata-only first pass)
	fileMetas, skipped, err := idx.scanner.ScanMetadata()
	if err != nil {
		return nil, fmt.Errorf("failed to scan files: %w", err)
	}
	stats.FilesSkipped = len(skipped)
	stats.ScannedFiles = fileMetas
	log.Printf("Scan: found %d indexable files (%d skipped as minified/too large/unreadable during scan)", len(fileMetas), len(skipped))

	// Get every existing document in one bulk read, instead of one
	// GetDocument round trip per scanned file below. On the common path of
	// restarting `grepai watch` against a project that is already fully
	// indexed and unchanged on disk, this is the difference between a single
	// store call and N of them (N network round trips on remote backends
	// like Postgres/Qdrant), for a phase whose only job is confirming that
	// nothing needs to happen.
	existingDocs, err := idx.store.GetAllDocuments(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to list documents: %w", err)
	}
	log.Printf("Store: %d existing documents loaded for comparison", len(existingDocs))

	existingMap := make(map[string]bool, len(existingDocs))
	for path := range existingDocs {
		existingMap[path] = true
	}

	// Every scanned file is accounted for exactly once: it either still exists
	// (and is removed from existingMap below) or it was deleted from disk and
	// stays in existingMap so it gets removed from the index further down.
	for _, fileMeta := range fileMetas {
		delete(existingMap, fileMeta.Path)
	}

	// Decide which files need (re)indexing, and start embedding them as soon
	// as they're found -- instead of waiting for every file in the repo to
	// be decided first. Deciding a single file (reading it, hashing it,
	// comparing to its existing document) has no cross-file dependency, so
	// a producer goroutine keeps deciding files on the same bounded worker
	// pool as before, streaming each file that needs (re)indexing into
	// filesCh; this goroutine (the consumer) accumulates them into waves
	// and embeds each wave as soon as it fills up. That overlap is the
	// whole point: embedding is almost always the slowest, network-bound
	// phase, so it can now run concurrently with scanning/deciding the rest
	// of the repository instead of sitting idle until scanning finishes.
	//
	// Two things deliberately still wait for the full scan, because they
	// inherently need it: detecting files deleted from disk (existingMap,
	// built above from the complete file list -- you can't know a file is
	// gone until you've seen everything that remains) and the final removal
	// loop below. Also, batching (FormBatches) only packs chunks within a
	// wave, not across the whole run, so a wave boundary can occasionally
	// split what would otherwise be one larger, marginally more efficient
	// batch. Both are minor compared to eliminating the scan-then-embed
	// stall.
	//
	// filesCh is sized to hold every file in the repo, not just one wave's
	// worth. This is deliberate: the old, fully-sequential implementation
	// already held the entire filesToIndex slice in memory at once, so
	// this doesn't raise peak memory usage versus before -- but it does
	// guarantee the producer below can never block trying to send. With a
	// smaller (e.g. embedWaveSize-sized) buffer, a wave that's slow to
	// embed -- for example because a store's cache lookup is hanging or
	// erroring repeatedly (see cacheLookupTimeout) -- fills the channel and
	// stalls every decide-phase worker mid-send, which freezes the scan
	// progress counter too, not just embedding. That made a downstream
	// backend problem look like scanning itself had hung. Sizing the
	// buffer to the whole run means the decide/scan phase always keeps
	// running at full speed and reports real progress, regardless of how
	// slow or stuck embedding gets.
	filesChCap := len(fileMetas)
	if filesChCap < 1 {
		filesChCap = 1
	}
	filesCh := make(chan FileInfo, filesChCap)

	var (
		statsMu   sync.Mutex
		decideErr error // written once, before filesCh is closed; safe to read after draining filesCh (see below)
	)

	// decideComplete flips to true once every file has been decided (the
	// decide goroutine's g.Wait() below has returned) -- read by
	// cumulativeBatchProgress further down to tell callers whether a given
	// wave's TotalChunks is final or still provisional (see
	// BatchProgressInfo.Provisional). queuedForEmbedding/reusedUnchangedFile
	// track why each file was or wasn't queued, purely for the summary log
	// once decide finishes.
	var (
		decideComplete      atomic.Bool
		queuedForEmbedding  atomic.Int64
		reusedUnchangedFile atomic.Int64
	)

	go func() {
		defer close(filesCh)

		g, gctx := errgroup.WithContext(ctx)
		g.SetLimit(scanWorkerLimit())
		var completed atomic.Int64

		for i := range fileMetas {
			fileMeta := fileMetas[i]
			doc := existingDocs[fileMeta.Path]
			g.Go(func() error {
				decision := idx.decideFileScan(gctx, fileMeta, doc)

				if onProgress != nil {
					n := completed.Add(1)
					onProgress(ProgressInfo{
						Current:     int(n),
						Total:       len(fileMetas),
						CurrentFile: fileMeta.Path,
					})
				}

				switch {
				case decision.countAsSkipped:
					statsMu.Lock()
					stats.FilesSkipped++
					statsMu.Unlock()
				case decision.file == nil:
					// Unchanged (same content hash, already has chunks) --
					// reused from the existing index without re-embedding.
					reusedUnchangedFile.Add(1)
				}
				if decision.file != nil {
					queuedForEmbedding.Add(1)
					select {
					case filesCh <- *decision.file:
					case <-gctx.Done():
						return gctx.Err()
					}
				}
				return nil
			})
		}

		// decideErr is written here, strictly before the deferred close(filesCh)
		// runs -- and close() happens-before the consumer's range loop below
		// observes the channel as closed. That ordering (Go's channel-close
		// happens-before guarantee) is what makes reading decideErr after the
		// loop safe without extra synchronization.
		decideErr = g.Wait()
		decideComplete.Store(true)

		statsMu.Lock()
		skippedSoFar := stats.FilesSkipped
		statsMu.Unlock()
		log.Printf("Scan/decide complete: %d files need (re)indexing, %d unchanged (reused), %d skipped",
			queuedForEmbedding.Load(), reusedUnchangedFile.Load(), skippedSoFar)
	}()

	// Wave progress is reported cumulatively across waves (prior waves'
	// completed/total chunk counts are added to each new wave's numbers) so
	// a progress bar driven by onBatchProgress keeps climbing instead of
	// resetting to 0% every time a new wave starts embedding.
	var priorChunksCompleted, priorChunksTotal atomic.Int64
	cumulativeBatchProgress := func(wave BatchProgressCallback) BatchProgressCallback {
		if wave == nil {
			return nil
		}
		return func(info BatchProgressInfo) {
			info.CompletedChunks += int(priorChunksCompleted.Load())
			info.TotalChunks += int(priorChunksTotal.Load())
			// See BatchProgressInfo.Provisional: TotalChunks above can
			// still grow (a later wave hasn't been decided/queued yet) as
			// long as scan/decide is still running, so callers shouldn't
			// treat completed==total as the whole run finishing while
			// this is true.
			info.Provisional = !decideComplete.Load()
			wave(info)
		}
	}

	batchEmbedder, hasBatchEmbedder := idx.embedder.(embedder.BatchEmbedder)
	var sequentialIndexed atomic.Int64

	// Both paths below save each file's chunks+document as soon as that file
	// is fully embedded (see indexFilesBatched and IndexFile), and
	// checkpoint periodically via cp -- so if embedding stops on an error or
	// a canceled context, the stats returned still reflect everything that
	// was actually saved, and that work is not lost even though this
	// function is returning early.
	var embedErr error
	stopEmbedding := false

	var waveNum atomic.Int64

	flushWave := func(wave []FileInfo) {
		if len(wave) == 0 || stopEmbedding {
			return
		}

		n := waveNum.Add(1)
		waveStart := time.Now()
		log.Printf("Embedding wave %d: %d files queued for embedding", n, len(wave))

		if hasBatchEmbedder {
			indexed, chunks, err := idx.indexFilesBatched(ctx, wave, batchEmbedder, cumulativeBatchProgress(onBatchProgress), cp)
			statsMu.Lock()
			stats.FilesIndexed += indexed
			stats.ChunksCreated += chunks
			statsMu.Unlock()
			priorChunksCompleted.Add(int64(chunks))
			priorChunksTotal.Add(int64(chunks))
			if err != nil {
				embedErr = err
				stopEmbedding = true
				log.Printf("Embedding wave %d failed after %d files / %d chunks (%s): %v",
					n, indexed, chunks, time.Since(waveStart).Round(time.Millisecond), err)
				return
			}
			log.Printf("Embedding wave %d complete: %d files indexed, %d chunks embedded (%s)",
				n, indexed, chunks, time.Since(waveStart).Round(time.Millisecond))
			return
		}

		// Sequential path for embedders that don't implement BatchEmbedder
		// (e.g. LM Studio, Synthetic, OpenRouter).
		var waveIndexed, waveChunks int
		for _, file := range wave {
			if ctx.Err() != nil {
				stopEmbedding = true
				log.Printf("Embedding wave %d interrupted after %d files / %d chunks (%s): %v",
					n, waveIndexed, waveChunks, time.Since(waveStart).Round(time.Millisecond), ctx.Err())
				return
			}
			chunks, err := idx.IndexFile(ctx, file)
			if err != nil {
				log.Printf("Failed to index %s: %v", file.Path, err)
				continue
			}
			statsMu.Lock()
			stats.FilesIndexed++
			stats.ChunksCreated += chunks
			statsMu.Unlock()
			cp.fileSaved()
			waveIndexed++
			waveChunks += chunks

			cur := sequentialIndexed.Add(1)
			if onBatchProgress != nil {
				onBatchProgress(BatchProgressInfo{
					BatchIndex:      int(cur) - 1,
					TotalBatches:    int(cur),
					CompletedChunks: int(cur),
					TotalChunks:     int(cur),
					Provisional:     !decideComplete.Load(),
				})
			}
		}
		log.Printf("Embedding wave %d complete: %d files indexed, %d chunks embedded (%s)",
			n, waveIndexed, waveChunks, time.Since(waveStart).Round(time.Millisecond))
	}

	wave := make([]FileInfo, 0, embedWaveSize)
	for file := range filesCh {
		if stopEmbedding {
			// Keep draining so the producer goroutine (blocked sending on a
			// full channel) can finish rather than leak.
			continue
		}
		wave = append(wave, file)
		if len(wave) >= embedWaveSize {
			flushWave(wave)
			wave = wave[:0]
		}
	}
	flushWave(wave)

	if decideErr != nil {
		stats.Duration = time.Since(start)
		return stats, decideErr
	}
	if embedErr != nil {
		stats.Duration = time.Since(start)
		return stats, embedErr
	}

	// Remove deleted files
	for path := range existingMap {
		if err := idx.RemoveFile(ctx, path); err != nil {
			log.Printf("Failed to remove %s: %v", path, err)
			continue
		}
		stats.FilesRemoved++
	}

	stats.Duration = time.Since(start)
	return stats, nil
}

// fileScanDecision is the outcome of deciding whether a single scanned file
// needs (re)indexing.
type fileScanDecision struct {
	// file is non-nil when the file needs (re)indexing.
	file *FileInfo
	// countAsSkipped mirrors the original sequential bookkeeping: mtime-gated
	// skips and unreadable/binary/oversized files count toward
	// stats.FilesSkipped, but files that are unchanged (same content hash)
	// do not -- matching the pre-existing (sequential) behavior exactly.
	countAsSkipped bool
}

// decideFileScan fetches a file's existing document (if any) and determines
// whether it needs (re)indexing, following the same rules as the original
// sequential loop: an mtime fast-path gate, then a content-hash comparison
// for files that need a closer look. It is safe to call concurrently for
// different files -- it only reads from the store and the filesystem.
func (idx *Indexer) decideFileScan(ctx context.Context, fileMeta FileMeta, doc *store.Document) fileScanDecision {
	// doc comes from a single bulk fetch done once up front (see
	// IndexAllWithBatchProgress), not a per-file store call -- used by both
	// the mod-time gate and hash check below.

	// Skip files modified before lastIndexTime -- but only if they have chunks.
	// Files with no chunks need re-indexing even if their mod_time is old
	// (e.g., a prior indexing run created the document but failed to embed).
	if !idx.lastIndexTime.IsZero() && doc != nil && len(doc.ChunkIDs) > 0 {
		fileModTime := time.Unix(fileMeta.ModTime, 0)
		if fileModTime.Before(idx.lastIndexTime) || fileModTime.Equal(idx.lastIndexTime) {
			return fileScanDecision{countAsSkipped: true}
		}
	}

	// Load file content and hash only after metadata filtering.
	file, err := idx.scanner.ScanFile(fileMeta.Path)
	if err != nil {
		log.Printf("Failed to scan %s: %v", fileMeta.Path, err)
		return fileScanDecision{countAsSkipped: true}
	}
	if file == nil {
		return fileScanDecision{countAsSkipped: true}
	}

	if doc != nil && doc.Hash == file.Hash && len(doc.ChunkIDs) > 0 {
		return fileScanDecision{} // File unchanged and has chunks
	}

	return fileScanDecision{file: file}
}

// checkpointInterval is how many newly-saved files trigger a checkpoint
// flush to durable storage during a run (see checkpoint below).
const checkpointInterval = 200

// embedWaveSize is how many decided (needs-reindexing) files are
// accumulated before starting to embed them, in IndexAllWithBatchProgress.
// Smaller waves start embedding sooner (better overlap with the rest of the
// scan/decide phase still running); larger waves let FormBatches pack
// slightly more efficient batches. 200 mirrors checkpointInterval's
// granularity as a reasonable default for both.
const embedWaveSize = 200

// cacheLookupTimeout bounds how long a single content-addressed cache
// lookup (store.EmbeddingCache.LookupByContentHash) may take. For a remote
// backend like Qdrant this is a network call; without a bound, a slow or
// wedged connection can stall indefinitely. A timeout here is treated the
// same as any other lookup error (see lookupCachedEmbeddingWithTimeout):
// it's just a cache miss, so that chunk gets re-embedded instead of reused.
// Declared as a var (not const) so tests can temporarily shorten it to
// exercise the timeout path without a real 10-second wait.
var cacheLookupTimeout = 10 * time.Second

// lookupCachedEmbeddingWithTimeout wraps store.EmbeddingCache.LookupByContentHash
// with cacheLookupTimeout, so a single slow or unresponsive lookup can't
// block whichever caller is waiting on it forever. Used by both the
// per-wave pre-fill in indexFilesBatched and the per-file lookup in
// lookupCachedEmbeddings.
func lookupCachedEmbeddingWithTimeout(ctx context.Context, cache store.EmbeddingCache, contentHash string) ([]float32, bool, error) {
	lookupCtx, cancel := context.WithTimeout(ctx, cacheLookupTimeout)
	defer cancel()
	return cache.LookupByContentHash(lookupCtx, contentHash)
}

// checkpoint periodically flushes the store to durable storage during a
// long indexing run, so a run that's stopped (Ctrl+C, crash, OOM kill) part
// of the way through loses at most one checkpoint interval's worth of
// progress instead of everything indexed so far.
//
// This matters most for GOBStore: SaveChunks/SaveDocument only mutate its
// in-memory maps, and nothing reaches disk until Persist is called, so
// without periodic checkpointing here a run that dies at, say, 50% through
// a big repository has written zero of that progress anywhere -- the next
// run starts completely from scratch. Postgres and Qdrant already commit
// each SaveChunks/SaveDocument call durably as it happens, so Persist is a
// cheap no-op for them (see store/postgres.go, store/qdrant.go); calling it
// here is harmless for those backends, not just for GOB.
type checkpoint struct {
	idx        *Indexer
	ctx        context.Context
	mu         sync.Mutex
	sinceFlush int
}

func newCheckpoint(ctx context.Context, idx *Indexer) *checkpoint {
	return &checkpoint{idx: idx, ctx: ctx}
}

// fileSaved records that one more file's chunks and document were just
// saved, flushing to durable storage once checkpointInterval files have
// accumulated since the last flush. Safe to call concurrently.
func (c *checkpoint) fileSaved() {
	if c == nil {
		return
	}
	c.mu.Lock()
	c.sinceFlush++
	flush := c.sinceFlush >= checkpointInterval
	if flush {
		c.sinceFlush = 0
	}
	c.mu.Unlock()

	if flush {
		if err := c.idx.store.Persist(c.ctx); err != nil {
			log.Printf("Warning: failed to checkpoint index progress: %v", err)
		}
	}
}

// scanWorkerLimit returns the number of concurrent workers to use when
// deciding which files need (re)indexing (store lookups + optional hashing).
// This phase mixes store I/O (which may be a network round trip for remote
// backends like Postgres/Qdrant) with local file hashing, so it benefits from
// more concurrency than pure CPU-bound work would need.
func scanWorkerLimit() int {
	n := runtime.GOMAXPROCS(0) * 4
	if n < 8 {
		return 8
	}
	if n > 64 {
		return 64
	}
	return n
}

// fileChunkData holds chunking information for a single file during batch processing.
type fileChunkData struct {
	fileIndex  int // Index in the files slice (for result mapping)
	file       FileInfo
	chunkInfos []ChunkInfo
	lineMap    []int
	source     string
}

// prepareFileChunks processes files by deleting existing chunks and creating new chunks.
// Returns the file data for storage and the file chunks for embedding.
// preparedFileChunks holds the chunking outcome for a single file, produced
// concurrently by prepareFileChunks.
type preparedFileChunks struct {
	fd fileChunkData
	fc embedder.FileChunks
	ok bool // false if the file produced no chunks and should be dropped
}

// prepareFileChunks processes files by deleting existing chunks and creating new chunks.
// Returns the file data for storage and the file chunks for embedding.
//
// Each file is independent (its own store delete + chunk generation), so this
// runs concurrently across a bounded worker pool. On large repositories this
// overlaps per-file store round trips (relevant for remote backends like
// Postgres/Qdrant) with the CPU cost of chunking instead of paying both costs
// strictly one file at a time.
func (idx *Indexer) prepareFileChunks(
	ctx context.Context,
	files []FileInfo,
) ([]fileChunkData, []embedder.FileChunks, error) {
	results := make([]preparedFileChunks, len(files))

	g, gctx := errgroup.WithContext(ctx)
	g.SetLimit(scanWorkerLimit())

	for i := range files {
		file := files[i]
		g.Go(func() error {
			if err := idx.store.DeleteByFile(gctx, file.Path); err != nil {
				return fmt.Errorf("failed to delete existing chunks for %s: %w", file.Path, err)
			}

			embedContent, lineMap := idx.embeddingContent(gctx, file)
			chunkInfos := idx.chunker.ChunkWithContext(file.Path, embedContent)
			if len(chunkInfos) == 0 {
				return nil
			}

			contents := make([]string, len(chunkInfos))
			for j, c := range chunkInfos {
				contents[j] = c.EmbedContent
			}

			results[i] = preparedFileChunks{
				fd: fileChunkData{
					fileIndex:  i,
					file:       file,
					chunkInfos: chunkInfos,
					lineMap:    lineMap,
					source:     file.Content,
				},
				fc: embedder.FileChunks{
					FileIndex: i,
					Chunks:    contents,
				},
				ok: true,
			}
			return nil
		})
	}

	if err := g.Wait(); err != nil {
		return nil, nil, err
	}

	// Collect results in original file order for deterministic output.
	fileData := make([]fileChunkData, 0, len(files))
	fileChunks := make([]embedder.FileChunks, 0, len(files))
	for _, r := range results {
		if !r.ok {
			continue
		}
		fileData = append(fileData, r.fd)
		fileChunks = append(fileChunks, r.fc)
	}

	return fileData, fileChunks, nil
}

// createStoreChunks creates store.Chunk objects from chunk info and embeddings.
func createStoreChunks(chunkInfos []ChunkInfo, embeddings [][]float32, now time.Time) ([]store.Chunk, []string) {
	chunks := make([]store.Chunk, len(chunkInfos))
	chunkIDs := make([]string, len(chunkInfos))

	for i, info := range chunkInfos {
		chunks[i] = store.Chunk{
			ID:          info.ID,
			FilePath:    info.FilePath,
			StartLine:   info.StartLine,
			EndLine:     info.EndLine,
			Content:     info.Content,
			Vector:      embeddings[i],
			Hash:        info.Hash,
			ContentHash: info.ContentHash,
			UpdatedAt:   now,
		}
		chunkIDs[i] = info.ID
	}

	return chunks, chunkIDs
}

// saveFileData saves chunks and document metadata for a single file.
func (idx *Indexer) saveFileData(ctx context.Context, fd fileChunkData, chunks []store.Chunk, chunkIDs []string) error {
	if err := idx.store.SaveChunks(ctx, chunks); err != nil {
		return fmt.Errorf("failed to save chunks for %s: %w", fd.file.Path, err)
	}

	doc := store.Document{
		Path:     fd.file.Path,
		Hash:     fd.file.Hash,
		ModTime:  time.Unix(fd.file.ModTime, 0),
		ChunkIDs: chunkIDs,
	}

	if err := idx.store.SaveDocument(ctx, doc); err != nil {
		return fmt.Errorf("failed to save document for %s: %w", fd.file.Path, err)
	}

	return nil
}

// wrapBatchProgress creates an embedder.BatchProgress callback from BatchProgressCallback.
func wrapBatchProgress(onProgress BatchProgressCallback) embedder.BatchProgress {
	if onProgress == nil {
		return nil
	}
	return func(batchIndex, totalBatches, completedChunks, totalChunks int, retrying bool, attempt int, statusCode int) {
		onProgress(BatchProgressInfo{
			BatchIndex:      batchIndex,
			TotalBatches:    totalBatches,
			CompletedChunks: completedChunks,
			TotalChunks:     totalChunks,
			Retrying:        retrying,
			Attempt:         attempt,
			StatusCode:      statusCode,
		})
	}
}

// indexFilesBatched indexes multiple files using cross-file batch embedding.
// It collects chunks from all files, forms batches, embeds them in parallel,
// then maps results back and stores them.
func (idx *Indexer) indexFilesBatched(
	ctx context.Context,
	files []FileInfo,
	batchEmb embedder.BatchEmbedder,
	onProgress BatchProgressCallback,
	cp *checkpoint,
) (filesIndexed int, chunksCreated int, err error) {
	fileData, fileChunks, err := idx.prepareFileChunks(ctx, files)
	if err != nil {
		return 0, 0, err
	}

	if len(fileChunks) == 0 {
		return 0, 0, nil
	}

	totalChunksThisWave := 0
	for _, fc := range fileChunks {
		totalChunksThisWave += len(fc.Chunks)
	}
	log.Printf("Chunked %d files into %d chunks, checking embedding cache before sending to the embedder", len(fileChunks), totalChunksThisWave)

	// Check embedding cache for content-addressed deduplication
	cache, hasCache := idx.store.(store.EmbeddingCache)
	var totalCacheHits atomic.Int64

	// Pre-fill cached embeddings and filter out fully-cached files
	type preFilled struct {
		fdIndex   int
		vectors   [][]float32
		allCached bool
	}

	// cacheChecked[i] holds the outcome of checking fileData[i]'s chunks
	// against the cache. Looked up concurrently (bounded by
	// scanWorkerLimit()) instead of one file, one chunk, one blocking store
	// round trip at a time: on a remote backend like Qdrant, a wave of up
	// to embedWaveSize files each with several chunks meant hundreds of
	// sequential network calls before embedding could even start for files
	// that weren't cached at all. Each lookup is also bounded by
	// cacheLookupTimeout, so a single slow or wedged store connection can't
	// stall this indefinitely -- it's treated as a cache miss (the same
	// fallback already used for a lookup error) and that file just gets
	// re-embedded instead of reused from cache.
	type cacheCheck struct {
		vecs      [][]float32
		allCached bool
	}
	cacheChecked := make([]cacheCheck, len(fileData))

	if hasCache {
		g, gctx := errgroup.WithContext(ctx)
		g.SetLimit(scanWorkerLimit())
		for i := range fileData {
			i := i
			fd := fileData[i]
			g.Go(func() error {
				vecs := make([][]float32, len(fd.chunkInfos))
				allCached := true
				for j, chunk := range fd.chunkInfos {
					if chunk.ContentHash == "" {
						allCached = false
						continue
					}
					vec, found, err := lookupCachedEmbeddingWithTimeout(gctx, cache, chunk.ContentHash)
					if err != nil {
						log.Printf("Warning: cache lookup failed: %v", err)
						allCached = false
						continue
					}
					if found {
						vecs[j] = vec
						totalCacheHits.Add(1)
					} else {
						allCached = false
					}
				}
				cacheChecked[i] = cacheCheck{vecs: vecs, allCached: allCached}
				return nil
			})
		}
		// Cache lookups are a pure optimization (skip re-embedding content
		// that's already embedded elsewhere) -- a failure is already
		// handled per-lookup above as a cache miss, so every goroutine here
		// always returns nil and this can never itself fail the run.
		_ = g.Wait()
	}

	var preFilledFiles []preFilled
	var remainingFileData []fileChunkData
	var remainingFileChunks []embedder.FileChunks

	for i, fd := range fileData {
		if !hasCache {
			remainingFileData = append(remainingFileData, fd)
			remainingFileChunks = append(remainingFileChunks, fileChunks[i])
			continue
		}

		checked := cacheChecked[i]
		if checked.allCached {
			preFilledFiles = append(preFilledFiles, preFilled{fdIndex: i, vectors: checked.vecs, allCached: true})
		} else {
			remainingFileData = append(remainingFileData, fd)
			remainingFileChunks = append(remainingFileChunks, fileChunks[i])
		}
	}

	if hits := totalCacheHits.Load(); hits > 0 {
		log.Printf("Reused %d cached embeddings across %d files", hits, len(preFilledFiles))
	}

	// Save fully-cached files immediately.
	now := time.Now()
	for _, pf := range preFilledFiles {
		fd := fileData[pf.fdIndex]
		idx.remapChunksToSource(fd.chunkInfos, fd.file.Path, fd.source, fd.lineMap)
		chunks, chunkIDs := createStoreChunks(fd.chunkInfos, pf.vectors, now)
		if err := idx.saveFileData(ctx, fd, chunks, chunkIDs); err != nil {
			return filesIndexed, chunksCreated, err
		}
		filesIndexed++
		chunksCreated += len(chunks)
		cp.fileSaved()
	}

	// Embed remaining (non-cached) files. Each file's chunks+document are
	// saved as soon as every chunk belonging to that file has an embedding
	// back -- via the onBatchDone callback below -- instead of waiting for
	// every batch across the whole run to finish first. Without this, a run
	// stopped or crashed partway through a large batch-embedded repository
	// would lose every file that hadn't been saved yet, even ones whose
	// embeddings had already come back successfully; see BatchResultCallback.
	//
	// A single file's chunks can be split across more than one batch --
	// batches are packed by size/token limit, not by file boundary -- so
	// completion is tracked per file across however many batches its chunks
	// landed in, not per batch.
	if len(remainingFileChunks) > 0 {
		batches := embedder.FormBatches(remainingFileChunks)

		// fileIndexToPos maps a file's original index (i.e. its position in
		// the `files` slice passed to this function, as carried by
		// embedder.FileChunks.FileIndex/BatchEntry.FileIndex) to its
		// position in remainingFileData/remainingFileChunks.
		fileIndexToPos := make(map[int]int, len(remainingFileData))
		for pos, fd := range remainingFileData {
			fileIndexToPos[fd.fileIndex] = pos
		}

		pending := make([][][]float32, len(remainingFileData)) // per-file embeddings, filled in as batches complete
		remaining := make([]int, len(remainingFileData))       // chunks still awaiting an embedding, per file
		for pos, fd := range remainingFileData {
			pending[pos] = make([][]float32, len(fd.chunkInfos))
			remaining[pos] = len(fd.chunkInfos)
		}

		var (
			mu          sync.Mutex
			savedFiles  int
			savedChunks int
			saveErr     error
		)

		onBatchDone := func(result embedder.BatchResult) {
			if result.BatchIndex < 0 || result.BatchIndex >= len(batches) {
				return
			}
			batch := batches[result.BatchIndex]

			// Bookkeeping (which files just became complete) happens under
			// the lock; the actual saves (I/O) happen after releasing it so
			// a slow store call for one file doesn't block bookkeeping for
			// batches completing concurrently on other files.
			var readyToSave []int
			mu.Lock()
			for i, entry := range batch.Entries {
				pos, ok := fileIndexToPos[entry.FileIndex]
				if !ok || i >= len(result.Embeddings) {
					continue
				}
				vecs := pending[pos]
				if entry.ChunkIndex < 0 || entry.ChunkIndex >= len(vecs) {
					continue
				}
				vecs[entry.ChunkIndex] = result.Embeddings[i]
				remaining[pos]--
				if remaining[pos] == 0 {
					readyToSave = append(readyToSave, pos)
				}
			}
			mu.Unlock()

			for _, pos := range readyToSave {
				fd := remainingFileData[pos]
				vecs := pending[pos]

				complete := true
				for _, v := range vecs {
					if v == nil {
						complete = false
						break
					}
				}
				if !complete {
					// Shouldn't happen (every chunk's batch reported success
					// for remaining[pos] to reach 0), but don't save a
					// partial/corrupt document if it somehow does -- the
					// next run will pick this file up again.
					log.Printf("Warning: %s completed with missing embeddings, will retry next run", fd.file.Path)
					continue
				}

				idx.remapChunksToSource(fd.chunkInfos, fd.file.Path, fd.source, fd.lineMap)
				chunks, chunkIDs := createStoreChunks(fd.chunkInfos, vecs, time.Now())
				if err := idx.saveFileData(ctx, fd, chunks, chunkIDs); err != nil {
					mu.Lock()
					if saveErr == nil {
						saveErr = err
					}
					mu.Unlock()
					log.Printf("Failed to save %s: %v", fd.file.Path, err)
					continue
				}

				mu.Lock()
				savedFiles++
				savedChunks += len(chunks)
				mu.Unlock()
				cp.fileSaved()
			}
		}

		_, embedErr := batchEmb.EmbedBatches(ctx, batches, wrapBatchProgress(onProgress), onBatchDone)

		mu.Lock()
		filesIndexed += savedFiles
		chunksCreated += savedChunks
		finalSaveErr := saveErr
		mu.Unlock()

		if embedErr != nil {
			return filesIndexed, chunksCreated, fmt.Errorf("failed to embed batches: %w", embedErr)
		}
		if finalSaveErr != nil {
			return filesIndexed, chunksCreated, finalSaveErr
		}
	}

	return filesIndexed, chunksCreated, nil
}

// maxReChunkAttempts is the maximum number of times we'll try to re-chunk
// before giving up on a file.
const maxReChunkAttempts = 3

// IndexFile indexes a single file
func (idx *Indexer) IndexFile(ctx context.Context, file FileInfo) (int, error) {
	// Remove existing chunks for this file
	if err := idx.store.DeleteByFile(ctx, file.Path); err != nil {
		return 0, fmt.Errorf("failed to delete existing chunks: %w", err)
	}

	embedContent, lineMap := idx.embeddingContent(ctx, file)

	// Chunk the file
	chunkInfos := idx.chunker.ChunkWithContext(file.Path, embedContent)
	if len(chunkInfos) == 0 {
		return 0, nil
	}

	// Check embedding cache for content-addressed deduplication
	cachedVectors, cacheHits := idx.lookupCachedEmbeddings(ctx, chunkInfos)
	if cacheHits > 0 {
		log.Printf("Reused %d cached embeddings for %s", cacheHits, file.Path)
	}

	// Separate cached and uncached chunks
	var uncachedChunks []ChunkInfo
	for i, chunk := range chunkInfos {
		if _, ok := cachedVectors[i]; !ok {
			uncachedChunks = append(uncachedChunks, chunk)
		}
	}

	// Embed only uncached chunks
	var uncachedVectors [][]float32
	var finalUncachedChunks []ChunkInfo
	if len(uncachedChunks) > 0 {
		var err error
		uncachedVectors, finalUncachedChunks, err = idx.embedWithReChunking(ctx, uncachedChunks)
		if err != nil {
			return 0, fmt.Errorf("failed to embed chunks: %w", err)
		}
	}

	// Merge cached and freshly embedded results
	// If re-chunking happened, the final chunks may differ from original
	// In that case, we use the re-chunked results plus the cached ones
	var vectors [][]float32
	var finalChunks []ChunkInfo

	if cacheHits == 0 {
		// No cache hits - use embedding results directly
		vectors = uncachedVectors
		finalChunks = finalUncachedChunks
	} else if len(uncachedChunks) == 0 {
		// All cached - build vectors and chunks from cache
		vectors = make([][]float32, len(chunkInfos))
		for i := range chunkInfos {
			vectors[i] = cachedVectors[i]
		}
		finalChunks = chunkInfos
	} else {
		// Mix of cached and uncached - merge results
		// Note: if re-chunking changed uncached chunks, we can't easily merge
		// with the original indices. Fall back to simple merge.
		vectors = make([][]float32, 0, len(chunkInfos))
		finalChunks = make([]ChunkInfo, 0, len(chunkInfos))

		uncachedIdx := 0
		for i, chunk := range chunkInfos {
			if vec, ok := cachedVectors[i]; ok {
				vectors = append(vectors, vec)
				finalChunks = append(finalChunks, chunk)
			} else {
				// Check if re-chunking happened (uncachedVectors may have different length)
				if uncachedIdx < len(uncachedVectors) && uncachedIdx < len(finalUncachedChunks) {
					vectors = append(vectors, uncachedVectors[uncachedIdx])
					finalChunks = append(finalChunks, finalUncachedChunks[uncachedIdx])
					uncachedIdx++
				}
			}
		}
		// If re-chunking produced extra sub-chunks, append them
		for ; uncachedIdx < len(uncachedVectors); uncachedIdx++ {
			vectors = append(vectors, uncachedVectors[uncachedIdx])
			finalChunks = append(finalChunks, finalUncachedChunks[uncachedIdx])
		}
	}

	idx.remapChunksToSource(finalChunks, file.Path, file.Content, lineMap)

	// Create store chunks
	now := time.Now()
	chunks := make([]store.Chunk, len(finalChunks))
	chunkIDs := make([]string, len(finalChunks))

	for i, info := range finalChunks {
		chunks[i] = store.Chunk{
			ID:          info.ID,
			FilePath:    info.FilePath,
			StartLine:   info.StartLine,
			EndLine:     info.EndLine,
			Content:     info.Content,
			Vector:      vectors[i],
			Hash:        info.Hash,
			ContentHash: info.ContentHash,
			UpdatedAt:   now,
		}
		chunkIDs[i] = info.ID
	}

	// Save chunks
	if err := idx.store.SaveChunks(ctx, chunks); err != nil {
		return 0, fmt.Errorf("failed to save chunks: %w", err)
	}

	// Save document metadata
	doc := store.Document{
		Path:     file.Path,
		Hash:     file.Hash,
		ModTime:  time.Unix(file.ModTime, 0),
		ChunkIDs: chunkIDs,
	}

	if err := idx.store.SaveDocument(ctx, doc); err != nil {
		return 0, fmt.Errorf("failed to save document: %w", err)
	}

	return len(chunks), nil
}

// embedWithReChunking attempts to embed chunks, automatically re-chunking
// any chunks that exceed the embedder's context limit.
func (idx *Indexer) embedWithReChunking(ctx context.Context, chunks []ChunkInfo) ([][]float32, []ChunkInfo, error) {
	currentChunks := chunks
	var allVectors [][]float32
	var finalChunks []ChunkInfo

	for attempt := 0; attempt < maxReChunkAttempts; attempt++ {
		contents := make([]string, len(currentChunks))
		for i, c := range currentChunks {
			if c.EmbedContent != "" {
				contents[i] = c.EmbedContent
			} else {
				contents[i] = c.Content
			}
		}

		vectors, err := idx.embedder.EmbedBatch(ctx, contents)
		if err == nil {
			// Success! Append all results
			allVectors = append(allVectors, vectors...)
			finalChunks = append(finalChunks, currentChunks...)
			return allVectors, finalChunks, nil
		}

		// Check if it's a context length error
		ctxErr := embedder.AsContextLengthError(err)
		if ctxErr == nil {
			// Not a context length error, return the original error
			return nil, nil, err
		}

		// Re-chunk the problematic chunk
		failedIndex := ctxErr.ChunkIndex
		if failedIndex < 0 || failedIndex >= len(currentChunks) {
			return nil, nil, fmt.Errorf("invalid chunk index %d from context length error", failedIndex)
		}

		failedChunk := currentChunks[failedIndex]
		log.Printf("Re-chunking %s chunk %d (attempt %d/%d): context limit exceeded",
			failedChunk.FilePath, failedIndex, attempt+1, maxReChunkAttempts)

		// Embed all chunks before the failed one (they should work)
		if failedIndex > 0 {
			beforeContents := make([]string, failedIndex)
			for i := 0; i < failedIndex; i++ {
				if currentChunks[i].EmbedContent != "" {
					beforeContents[i] = currentChunks[i].EmbedContent
				} else {
					beforeContents[i] = currentChunks[i].Content
				}
			}
			beforeVectors, err := idx.embedder.EmbedBatch(ctx, beforeContents)
			if err != nil {
				return nil, nil, fmt.Errorf("failed to embed chunks before failed index: %w", err)
			}
			allVectors = append(allVectors, beforeVectors...)
			finalChunks = append(finalChunks, currentChunks[:failedIndex]...)
		}

		// Re-chunk the failed chunk
		subChunks := idx.chunker.ReChunk(failedChunk, failedIndex)
		if len(subChunks) == 0 {
			return nil, nil, fmt.Errorf("re-chunking produced no chunks for %s", failedChunk.FilePath)
		}

		log.Printf("Split chunk into %d sub-chunks", len(subChunks))

		// Prepare for next iteration: sub-chunks + remaining chunks
		currentChunks = append(subChunks, currentChunks[failedIndex+1:]...)
	}

	return nil, nil, fmt.Errorf("exceeded maximum re-chunk attempts (%d) for file", maxReChunkAttempts)
}

func (idx *Indexer) embeddingContent(ctx context.Context, file FileInfo) (string, []int) {
	if idx.processor == nil {
		return file.Content, nil
	}

	res, err := idx.processor.TransformForEmbedding(ctx, file.Path, file.Content)
	if err != nil {
		log.Printf("Warning: framework embedding transform failed for %s: %v", file.Path, err)
		return file.Content, nil
	}
	framework.LogWarningsOnce(res.Warnings)
	if res.Text == "" {
		return file.Content, nil
	}
	return res.Text, res.GeneratedToSourceLine
}

func (idx *Indexer) remapChunksToSource(chunks []ChunkInfo, filePath, source string, lineMap []int) {
	for i := range chunks {
		startLine, endLine := framework.RemapLineRange(lineMap, chunks[i].StartLine, chunks[i].EndLine)
		snippet := framework.SourceSnippet(source, startLine, endLine)
		if snippet == "" {
			continue
		}
		chunks[i].StartLine = startLine
		chunks[i].EndLine = endLine
		chunks[i].Content = fmt.Sprintf("File: %s\n\n%s", filePath, snippet)
	}
}

// lookupCachedEmbeddings checks if the store implements EmbeddingCache and returns
// cached vectors for chunks with matching content hashes. The returned map maps
// chunk index to cached vector. Chunks not in the map need fresh embedding.
func (idx *Indexer) lookupCachedEmbeddings(ctx context.Context, chunks []ChunkInfo) (map[int][]float32, int) {
	cache, ok := idx.store.(store.EmbeddingCache)
	if !ok {
		return nil, 0
	}

	cached := make(map[int][]float32)
	for i, chunk := range chunks {
		if chunk.ContentHash == "" {
			continue
		}
		vec, found, err := lookupCachedEmbeddingWithTimeout(ctx, cache, chunk.ContentHash)
		if err != nil {
			log.Printf("Warning: cache lookup failed for content hash %s: %v", chunk.ContentHash[:8], err)
			continue
		}
		if found {
			cached[i] = vec
		}
	}

	return cached, len(cached)
}

// RemoveFile removes a file from the index
func (idx *Indexer) RemoveFile(ctx context.Context, path string) error {
	if err := idx.store.DeleteByFile(ctx, path); err != nil {
		return fmt.Errorf("failed to delete chunks: %w", err)
	}

	if err := idx.store.DeleteDocument(ctx, path); err != nil {
		return fmt.Errorf("failed to delete document: %w", err)
	}

	return nil
}

// NeedsReindex checks if a file needs reindexing
func (idx *Indexer) NeedsReindex(ctx context.Context, path string, hash string) (bool, error) {
	doc, err := idx.store.GetDocument(ctx, path)
	if err != nil {
		return false, err
	}

	if doc == nil {
		return true, nil
	}

	// Reindex if hash changed OR if document has no chunks (prior indexing failed)
	if doc.Hash != hash || len(doc.ChunkIDs) == 0 {
		return true, nil
	}

	return false, nil
}
