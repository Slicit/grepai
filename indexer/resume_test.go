package indexer

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/yoanbernabeu/grepai/embedder"
)

// persistCountingStore wraps mockStore and counts Persist calls, so tests
// can verify checkpointing actually flushes to durable storage at the
// expected cadence instead of only once at the very end of a run.
type persistCountingStore struct {
	*mockStore
	persistCount atomic.Int64
}

func newPersistCountingStore() *persistCountingStore {
	return &persistCountingStore{mockStore: newMockStore()}
}

func (p *persistCountingStore) Persist(ctx context.Context) error {
	p.persistCount.Add(1)
	return p.mockStore.Persist(ctx)
}

// TestCheckpoint_FlushesAtInterval verifies the checkpoint helper flushes the
// store every checkpointInterval files, not just once at the end -- this is
// what bounds how much progress a stopped or crashed run can lose on
// backends (like GOBStore) where nothing reaches disk until Persist runs.
func TestCheckpoint_FlushesAtInterval(t *testing.T) {
	st := newPersistCountingStore()
	idx := &Indexer{store: st}
	cp := newCheckpoint(context.Background(), idx)

	for i := 0; i < checkpointInterval-1; i++ {
		cp.fileSaved()
	}
	if got := st.persistCount.Load(); got != 0 {
		t.Fatalf("expected 0 persists before reaching the interval, got %d", got)
	}

	cp.fileSaved() // this is the checkpointInterval-th file saved
	if got := st.persistCount.Load(); got != 1 {
		t.Fatalf("expected 1 persist right after reaching the interval, got %d", got)
	}

	for i := 0; i < checkpointInterval; i++ {
		cp.fileSaved()
	}
	if got := st.persistCount.Load(); got != 2 {
		t.Fatalf("expected 2 persists after a second full interval, got %d", got)
	}
}

// TestCheckpoint_NilSafe verifies a nil *checkpoint (as used by callers that
// don't need checkpointing, e.g. in unit tests) is safe to call fileSaved on.
func TestCheckpoint_NilSafe(t *testing.T) {
	var cp *checkpoint
	cp.fileSaved() // must not panic
}

// TestIndexFilesBatched_PartialBatchFailureSavesCompletedFiles is the core
// regression test for resumability on the batch-embedder path (e.g. OpenAI):
// if the embedder fails partway through a run -- simulating a crash, a lost
// connection, or an exhausted retry budget -- every file whose batch had
// already completed successfully must still end up saved in the store, not
// discarded just because a later batch failed.
//
// FormBatches packs up to embedder.MaxBatchSize (2000) chunks per batch, so
// this test uses enough single-chunk synthetic files to force more than one
// batch -- entirely in memory, no disk I/O, since indexFilesBatched operates
// on already-loaded FileInfo.
func TestIndexFilesBatched_PartialBatchFailureSavesCompletedFiles(t *testing.T) {
	const fileCount = embedder.MaxBatchSize + 500 // guarantees 2 batches: 2000 + 500

	ignoreMatcher, err := NewIgnoreMatcher(t.TempDir(), []string{}, "")
	if err != nil {
		t.Fatalf("failed to create ignore matcher: %v", err)
	}

	mockStore := newMockStore()
	chunker := NewChunker(512, 50)
	idx := &Indexer{
		root:    "/virtual",
		store:   mockStore,
		chunker: chunker,
		scanner: NewScanner("/virtual", ignoreMatcher),
	}

	files := make([]FileInfo, fileCount)
	for i := range files {
		files[i] = FileInfo{
			Path:    fmt.Sprintf("file_%04d.go", i),
			Content: "package main\n\nfunc f() int { return 1 }\n",
			Hash:    fmt.Sprintf("hash-%04d", i),
			ModTime: time.Now().Unix(),
		}
	}

	// Fail once the first batch has completed -- batch 0 (the first 2000
	// files) succeeds and must be saved; batch 1 (the remaining 500) never
	// gets a chance to embed.
	mockEmb := &mockBatchEmbedder{failAfter: 1}
	cp := newCheckpoint(context.Background(), idx)

	filesIndexed, chunksCreated, err := idx.indexFilesBatched(context.Background(), files, mockEmb, nil, cp)
	if err == nil {
		t.Fatal("expected an error from the simulated interruption")
	}
	if filesIndexed == 0 {
		t.Fatal("expected some files to be indexed before the simulated interruption, got 0")
	}
	if filesIndexed >= fileCount {
		t.Fatalf("expected fewer than %d files indexed (interruption should have stopped it), got %d", fileCount, filesIndexed)
	}
	if chunksCreated == 0 {
		t.Fatal("expected some chunks to be created before the interruption")
	}

	// Every file reported as indexed must actually be present in the store
	// with both its chunks and its document record -- that's what makes it
	// safe to skip on the next run instead of silently missing.
	mockStore.mu.Lock()
	savedDocs := len(mockStore.documents)
	savedChunks := len(mockStore.chunks)
	mockStore.mu.Unlock()

	if savedDocs != filesIndexed {
		t.Fatalf("expected %d documents saved, got %d", filesIndexed, savedDocs)
	}
	if savedChunks != chunksCreated {
		t.Fatalf("expected %d chunks saved, got %d", chunksCreated, savedChunks)
	}

	// The files NOT indexed must have no document at all, so a subsequent
	// run's decideFileScan correctly treats them as needing (re)indexing.
	var missing int
	for _, f := range files {
		mockStore.mu.Lock()
		_, ok := mockStore.documents[f.Path]
		mockStore.mu.Unlock()
		if !ok {
			missing++
		}
	}
	if missing != fileCount-filesIndexed {
		t.Fatalf("expected %d files without a document, got %d", fileCount-filesIndexed, missing)
	}
}

// countingCancelingEmbedder wraps mockEmbedder and cancels the provided
// cancel func once EmbedBatch has been called stopAfter times -- simulating
// a process that dies (Ctrl+C, crash, OOM kill) right after that many files
// have been embedded, for the non-batch (sequential IndexFile) path used by
// embedders like Ollama.
type countingCancelingEmbedder struct {
	*mockEmbedder
	mu        sync.Mutex
	calls     int
	stopAfter int
	cancel    context.CancelFunc
}

func (c *countingCancelingEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	c.mu.Lock()
	c.calls++
	shouldStop := c.calls >= c.stopAfter
	c.mu.Unlock()

	vectors, err := c.mockEmbedder.EmbedBatch(ctx, texts)
	if shouldStop {
		c.cancel()
	}
	return vectors, err
}

// TestIndexAllWithBatchProgress_ResumesAfterInterruption is the end-to-end
// regression test matching the reported scenario directly: an indexing run
// is stopped partway through (here, ~50%, via context cancellation once
// half the files have been embedded), and a second run against the same
// store picks up exactly where the first left off -- re-embedding only the
// files that didn't make it, not the whole project again.
func TestIndexAllWithBatchProgress_ResumesAfterInterruption(t *testing.T) {
	tmpDir := t.TempDir()
	const fileCount = 20
	createGoFixtureFiles(t, tmpDir, fileCount)

	ignoreMatcher, err := NewIgnoreMatcher(tmpDir, []string{}, "")
	if err != nil {
		t.Fatalf("failed to create ignore matcher: %v", err)
	}

	mockStore := newMockStore()
	scanner := NewScanner(tmpDir, ignoreMatcher)
	chunker := NewChunker(512, 50)

	ctx, cancel := context.WithCancel(context.Background())
	embedder1 := &countingCancelingEmbedder{
		mockEmbedder: newMockEmbedder(),
		stopAfter:    fileCount / 2,
		cancel:       cancel,
	}

	idx1 := NewIndexer(tmpDir, mockStore, embedder1, chunker, scanner, time.Time{})
	stats1, err := idx1.IndexAllWithProgress(ctx, nil)

	// Interruption should surface as an error (context canceled) or, if the
	// cancellation lands between file iterations, simply stop early -- both
	// are acceptable here; what matters is that it didn't quietly index
	// everything and that some -- but not all -- files got saved.
	if stats1 == nil {
		t.Fatal("expected non-nil stats even though the run was interrupted")
	}
	if stats1.FilesIndexed == 0 {
		t.Fatal("expected some files indexed before interruption, got 0")
	}
	if stats1.FilesIndexed >= fileCount {
		t.Fatalf("expected fewer than %d files indexed given the simulated interruption, got %d", fileCount, stats1.FilesIndexed)
	}
	firstRunIndexed := stats1.FilesIndexed
	t.Logf("first run: indexed %d/%d files before interruption (err=%v)", firstRunIndexed, fileCount, err)

	mockStore.mu.Lock()
	docsAfterFirstRun := len(mockStore.documents)
	mockStore.mu.Unlock()
	if docsAfterFirstRun != firstRunIndexed {
		t.Fatalf("expected %d documents saved after first run, got %d", firstRunIndexed, docsAfterFirstRun)
	}

	// Second run: fresh (uncanceled) context, fresh embedder instance so we
	// can independently count how many files it actually had to embed.
	embedder2 := newMockEmbedder()
	idx2 := NewIndexer(tmpDir, mockStore, embedder2, chunker, scanner, time.Time{})

	stats2, err := idx2.IndexAllWithProgress(context.Background(), nil)
	if err != nil {
		t.Fatalf("second run failed: %v", err)
	}

	// The whole point: the second run should only have had to (re)index the
	// files the first run didn't get to, not all fileCount of them again.
	if stats2.FilesIndexed != fileCount-firstRunIndexed {
		t.Fatalf("expected second run to index the remaining %d files, indexed %d instead",
			fileCount-firstRunIndexed, stats2.FilesIndexed)
	}

	mockStore.mu.Lock()
	finalDocs := len(mockStore.documents)
	mockStore.mu.Unlock()
	if finalDocs != fileCount {
		t.Fatalf("expected all %d files present after resuming, got %d", fileCount, finalDocs)
	}
}
