package indexer

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// blockingCacheStore wraps mockStore and additionally implements
// store.EmbeddingCache with a LookupByContentHash that blocks until either
// the test closes unblock (simulating a store that's slow or wedged, e.g.
// a Qdrant connection that never returns) or its context is canceled/times
// out. It also tracks how many lookups run concurrently, so tests can prove
// lookups are no longer issued one at a time.
type blockingCacheStore struct {
	*mockStore

	unblock chan struct{}

	calls         atomic.Int32
	current       atomic.Int32
	maxConcurrent atomic.Int32
}

func newBlockingCacheStore() *blockingCacheStore {
	return &blockingCacheStore{
		mockStore: newMockStore(),
		unblock:   make(chan struct{}),
	}
}

func (b *blockingCacheStore) LookupByContentHash(ctx context.Context, contentHash string) ([]float32, bool, error) {
	b.calls.Add(1)
	c := b.current.Add(1)
	defer b.current.Add(-1)

	for {
		old := b.maxConcurrent.Load()
		if c <= old {
			break
		}
		if b.maxConcurrent.CompareAndSwap(old, c) {
			break
		}
	}

	select {
	case <-b.unblock:
		return nil, false, nil // resolves as a cache miss once released
	case <-ctx.Done():
		return nil, false, ctx.Err()
	}
}

// TestIndexAllWithBatchProgress_ScanProgressCompletesDespiteStuckCacheLookups
// is the regression test for the bug this change fixes: previously, the
// results channel connecting the decide-phase producer to the embedding
// consumer was sized to hold only one wave's worth of files (embedWaveSize).
// If a wave's embedding got stuck -- e.g. because a store's content-hash
// cache lookup (see indexFilesBatched) was hanging, which is exactly what a
// wedged or misbehaving Qdrant connection looks like -- every decide-phase
// worker eventually blocked trying to enqueue a file into the full channel,
// which froze the scan/decide progress counter too, not just embedding.
// That made a downstream store problem look like scanning itself had
// stalled part-way through (e.g. "stuck at file N and not moving").
//
// This test simulates exactly that: a store whose cache lookups block
// indefinitely until the test releases them, well past when decide-phase
// should have finished. It asserts that scan/decide progress (onProgress
// reaching Current == Total) completes on its own even while the first
// wave's embedding is still stuck waiting on the cache -- proving the
// decide/scan phase is no longer coupled to how far behind embedding gets.
func TestIndexAllWithBatchProgress_ScanProgressCompletesDespiteStuckCacheLookups(t *testing.T) {
	tmpDir := t.TempDir()
	// 3 waves: after wave 1 is drained into flushWave (and gets stuck
	// there), the channel buffer must be large enough for the producer to
	// decide and enqueue every remaining file (waves 2 and 3) without ever
	// blocking on a full channel. With the old, wave-sized buffer, decide
	// on wave 3 would still stall until wave 1 unstuck -- this test needs
	// more than 2 waves' worth of files for that gap to be observable.
	const totalFiles = embedWaveSize * 3
	writeTinyFiles(t, tmpDir, totalFiles)

	blockingStore := newBlockingCacheStore()
	mockEmb := newMockBatchEmbedder()

	ignoreMatcher, err := NewIgnoreMatcher(tmpDir, []string{}, "")
	if err != nil {
		t.Fatalf("failed to create ignore matcher: %v", err)
	}
	scanner := NewScanner(tmpDir, ignoreMatcher)
	chunker := NewChunker(512, 50)
	idx := NewIndexer(tmpDir, blockingStore, mockEmb, chunker, scanner, time.Time{})

	var (
		mu           sync.Mutex
		decideDone   bool
		decideDoneAt time.Time
	)
	decideCompleteCh := make(chan struct{})

	runErrCh := make(chan error, 1)
	go func() {
		_, err := idx.IndexAllWithBatchProgress(context.Background(),
			func(info ProgressInfo) {
				if info.Current == info.Total {
					mu.Lock()
					if !decideDone {
						decideDone = true
						decideDoneAt = time.Now()
						close(decideCompleteCh)
					}
					mu.Unlock()
				}
			}, nil)
		runErrCh <- err
	}()

	// Scan/decide over totalFiles tiny real files should finish in well
	// under a second; give it a generous margin. It must NOT need to wait
	// for the (indefinitely) stuck cache lookups in wave 1's embedding.
	select {
	case <-decideCompleteCh:
		// good: decide-phase finished on its own.
	case <-time.After(5 * time.Second):
		t.Fatal("scan/decide phase did not complete -- it appears to be blocked on the stuck embedding wave, " +
			"which is the regression this test guards against")
	}

	// The run itself should still be blocked at this point (wave 1 hasn't
	// been released yet), confirming decide-phase really did finish
	// independently rather than the whole run having somehow completed
	// some other way.
	select {
	case err := <-runErrCh:
		t.Fatalf("expected IndexAllWithBatchProgress to still be running (blocked on the stuck cache lookup), but it already returned (err=%v)", err)
	case <-time.After(50 * time.Millisecond):
		// expected: still running.
	}

	if blockingStore.calls.Load() == 0 {
		t.Fatal("expected at least one cache lookup to have been attempted")
	}
	if got := blockingStore.maxConcurrent.Load(); got < 2 {
		t.Errorf("expected cache lookups to run concurrently (>=2 at once), max observed concurrency was %d", got)
	}

	// Release the stuck lookups so the run can finish and the test can
	// clean up without leaking goroutines.
	close(blockingStore.unblock)

	select {
	case err := <-runErrCh:
		if err != nil {
			t.Fatalf("IndexAllWithBatchProgress failed after releasing cache lookups: %v", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("IndexAllWithBatchProgress did not finish after releasing the stuck cache lookups")
	}

	mu.Lock()
	defer mu.Unlock()
	if decideDoneAt.IsZero() {
		t.Fatal("expected decide-phase completion to have been recorded")
	}
}

// TestLookupCachedEmbeddingWithTimeout_BoundsAWedgedLookup verifies that a
// single content-hash cache lookup that never returns is bounded by
// cacheLookupTimeout rather than blocking forever -- this is what keeps a
// wedged store connection from being able to stall indexing indefinitely
// (see indexFilesBatched and lookupCachedEmbeddings, both of which are
// treated as an ordinary cache miss on any lookup error, including this
// timeout).
func TestLookupCachedEmbeddingWithTimeout_BoundsAWedgedLookup(t *testing.T) {
	originalTimeout := cacheLookupTimeout
	cacheLookupTimeout = 50 * time.Millisecond
	defer func() { cacheLookupTimeout = originalTimeout }()

	store := newBlockingCacheStore() // never unblocked in this test

	start := time.Now()
	_, found, err := lookupCachedEmbeddingWithTimeout(context.Background(), store, "somehash")
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("expected a timeout error from a lookup that never returns, got nil")
	}
	if found {
		t.Error("expected found=false alongside the timeout error")
	}
	if elapsed > time.Second {
		t.Errorf("expected the lookup to be bounded by cacheLookupTimeout (~50ms), took %v", elapsed)
	}
}

// TestIndexFilesBatched_CacheLookupsRunConcurrently verifies that the
// per-wave cache pre-fill step in indexFilesBatched issues its lookups
// concurrently (bounded by scanWorkerLimit()) rather than one file, one
// chunk, one blocking store round trip at a time -- which is what made a
// wave of many files each with a store-backed cache lookup take as long as
// the sum of every individual lookup, rather than roughly the slowest one.
func TestIndexFilesBatched_CacheLookupsRunConcurrently(t *testing.T) {
	const fileCount = 40
	files := make([]FileInfo, fileCount)
	for i := 0; i < fileCount; i++ {
		files[i] = FileInfo{
			Path:    fmt.Sprintf("file%d.go", i),
			Content: fmt.Sprintf("package main\n\nfunc f%d() int { return %d }\n", i, i),
			Hash:    fmt.Sprintf("hash%d", i),
		}
	}

	ignoreMatcher, err := NewIgnoreMatcher(t.TempDir(), []string{}, "")
	if err != nil {
		t.Fatalf("failed to create ignore matcher: %v", err)
	}

	blockingStore := newBlockingCacheStore()
	mockEmb := newMockBatchEmbedder()
	idx := &Indexer{
		root:    "/virtual",
		store:   blockingStore,
		chunker: NewChunker(512, 50),
		scanner: NewScanner("/virtual", ignoreMatcher),
	}
	cp := newCheckpoint(context.Background(), idx)

	done := make(chan struct{})
	go func() {
		_, _, _ = idx.indexFilesBatched(context.Background(), files, mockEmb, nil, cp)
		close(done)
	}()

	// Give the pre-fill loop time to reach steady-state concurrency before
	// checking (and before releasing it).
	time.Sleep(100 * time.Millisecond)
	close(blockingStore.unblock)

	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("indexFilesBatched did not finish after releasing cache lookups")
	}

	if got := blockingStore.maxConcurrent.Load(); got < 2 {
		t.Errorf("expected cache lookups to run concurrently (>=2 at once) across %d files, max observed concurrency was %d", fileCount, got)
	}
}
