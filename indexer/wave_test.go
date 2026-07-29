package indexer

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/yoanbernabeu/grepai/embedder"
)

// waveTrackingEmbedder wraps mockBatchEmbedder to additionally record, for
// every EmbedBatches call, the wall-clock time it was entered and (if
// configured) sleep for a fixed delay before doing the normal mock work.
// Tests use this to observe when embedding actually starts relative to how
// far the scan/decide phase has progressed -- the thing under test.
type waveTrackingEmbedder struct {
	mockBatchEmbedder

	mu         sync.Mutex
	callStarts []time.Time
}

func (w *waveTrackingEmbedder) EmbedBatches(ctx context.Context, batches []embedder.Batch, progress embedder.BatchProgress, onBatchDone embedder.BatchResultCallback) ([]embedder.BatchResult, error) {
	w.mu.Lock()
	w.callStarts = append(w.callStarts, time.Now())
	w.mu.Unlock()
	return w.mockBatchEmbedder.EmbedBatches(ctx, batches, progress, onBatchDone)
}

func (w *waveTrackingEmbedder) callCount() int {
	w.mu.Lock()
	defer w.mu.Unlock()
	return len(w.callStarts)
}

func (w *waveTrackingEmbedder) firstCallStart() time.Time {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.callStarts[0]
}

// writeTinyFiles creates n minimal, distinct, indexable Go files in dir so a
// real Scanner + real decideFileScan pass has actual (if small) work to do
// per file -- these tests are specifically about the interaction between
// scanning/deciding and embedding, so they need to go through the real
// scan path rather than the synthetic in-memory FileInfo used elsewhere
// (e.g. resume_test.go), which bypasses scanning entirely.
func writeTinyFiles(t *testing.T, dir string, n int) {
	t.Helper()
	for i := 0; i < n; i++ {
		path := filepath.Join(dir, fmt.Sprintf("file%04d.go", i))
		content := fmt.Sprintf("package main\n\nfunc f%04d() int { return %d }\n", i, i)
		if err := os.WriteFile(path, []byte(content), 0644); err != nil {
			t.Fatalf("failed to write %s: %v", path, err)
		}
	}
}

// TestIndexAllWithBatchProgress_EmbeddingStartsBeforeScanCompletes is the
// core regression test for overlapping scan/decide with embedding: it
// proves the first embedding call is made before every file in the repo
// has finished being decided, rather than only after the entire scan/decide
// pass completes (which is what the old, fully sequential implementation
// always did -- this same assertion would fail against that code, since it
// never invoked the embedder until decisions for every file were in hand).
//
// enough files are used (comfortably more than one embedWaveSize) that the
// first wave reliably fills and starts embedding while the decide-phase
// worker pool is still working through the rest.
func TestIndexAllWithBatchProgress_EmbeddingStartsBeforeScanCompletes(t *testing.T) {
	tmpDir := t.TempDir()
	// A large multiple of embedWaveSize (not just enough for a couple of
	// waves) so decide-phase's total real work -- reading, hashing, and
	// deciding every file -- is clearly larger than the fixed, roughly
	// constant per-wave dispatch overhead (forming/logging/dispatching wave
	// 1). That asymmetry is what makes the margin between "wave 1 starts
	// embedding" and "decide-phase finishes" grow with file count, so this
	// stays robust under system load/scheduling jitter instead of being a
	// tight race between two roughly-equal durations.
	const totalFiles = embedWaveSize * 25
	writeTinyFiles(t, tmpDir, totalFiles)

	mockStore := newMockStore()
	mockEmb := &waveTrackingEmbedder{}

	ignoreMatcher, err := NewIgnoreMatcher(tmpDir, []string{}, "")
	if err != nil {
		t.Fatalf("failed to create ignore matcher: %v", err)
	}
	scanner := NewScanner(tmpDir, ignoreMatcher)
	chunker := NewChunker(512, 50)
	idx := NewIndexer(tmpDir, mockStore, mockEmb, chunker, scanner, time.Time{})

	var (
		decideMu       sync.Mutex
		decidePhaseEnd time.Time
	)

	stats, err := idx.IndexAllWithBatchProgress(context.Background(),
		func(info ProgressInfo) {
			if info.Current == info.Total {
				decideMu.Lock()
				decidePhaseEnd = time.Now()
				decideMu.Unlock()
			}
		}, nil)
	if err != nil {
		t.Fatalf("IndexAllWithBatchProgress failed: %v", err)
	}

	if stats.FilesIndexed != totalFiles {
		t.Fatalf("expected %d files indexed, got %d", totalFiles, stats.FilesIndexed)
	}

	if mockEmb.callCount() < 2 {
		t.Fatalf("expected multiple EmbedBatches calls (multiple waves) for %d files, got %d", totalFiles, mockEmb.callCount())
	}

	decideMu.Lock()
	endOfDecide := decidePhaseEnd
	decideMu.Unlock()

	if endOfDecide.IsZero() {
		t.Fatal("expected the decide-phase completion (Current == Total) to have been observed")
	}

	firstEmbed := mockEmb.firstCallStart()
	if !firstEmbed.Before(endOfDecide) {
		t.Errorf("expected the first embedding call to start before scanning/deciding finished for every file, "+
			"but first embed call was at %v and scan/decide finished at %v -- embedding is not overlapping with scanning",
			firstEmbed, endOfDecide)
	}
}

// TestIndexAllWithBatchProgress_MultipleWavesIndexEveryFileExactlyOnce
// verifies that splitting a large run across multiple embedding waves (see
// embedWaveSize) doesn't lose or double-process any file: every file that
// needs indexing ends up saved exactly once, regardless of which wave its
// decision happened to land in.
func TestIndexAllWithBatchProgress_MultipleWavesIndexEveryFileExactlyOnce(t *testing.T) {
	tmpDir := t.TempDir()
	const totalFiles = embedWaveSize*2 + embedWaveSize/2 // guarantees 3 waves: full, full, partial
	writeTinyFiles(t, tmpDir, totalFiles)

	mockStore := newMockStore()
	mockEmb := &waveTrackingEmbedder{}

	ignoreMatcher, err := NewIgnoreMatcher(tmpDir, []string{}, "")
	if err != nil {
		t.Fatalf("failed to create ignore matcher: %v", err)
	}
	scanner := NewScanner(tmpDir, ignoreMatcher)
	chunker := NewChunker(512, 50)
	idx := NewIndexer(tmpDir, mockStore, mockEmb, chunker, scanner, time.Time{})

	stats, err := idx.IndexAllWithBatchProgress(context.Background(), nil, nil)
	if err != nil {
		t.Fatalf("IndexAllWithBatchProgress failed: %v", err)
	}

	if stats.FilesIndexed != totalFiles {
		t.Fatalf("expected %d files indexed, got %d", totalFiles, stats.FilesIndexed)
	}

	if got := mockEmb.callCount(); got != 3 {
		t.Fatalf("expected exactly 3 EmbedBatches calls (one per wave) for %d files, got %d", totalFiles, got)
	}

	mockStore.mu.Lock()
	defer mockStore.mu.Unlock()
	if len(mockStore.documents) != totalFiles {
		t.Fatalf("expected exactly %d documents saved (no loss or duplication across waves), got %d", totalFiles, len(mockStore.documents))
	}
}

// TestIndexAllWithBatchProgress_LaterWaveFailureStopsButPreservesEarlierWork
// verifies that if embedding fails partway through a multi-wave run (e.g.
// the embedder's connection drops during wave 2), files from waves that
// already completed successfully are still saved and reflected in the
// returned stats -- consistent with the resumable-indexing contract
// (see resume_test.go) now that a run can span more than one embedding
// wave rather than always being a single EmbedBatches call.
func TestIndexAllWithBatchProgress_LaterWaveFailureStopsButPreservesEarlierWork(t *testing.T) {
	tmpDir := t.TempDir()
	const totalFiles = embedWaveSize * 3
	writeTinyFiles(t, tmpDir, totalFiles)

	mockStore := newMockStore()
	mockEmb := &waveTrackingEmbedder{}
	// Each wave of embedWaveSize single-chunk-ish tiny files forms just one
	// or two batches; failing after the first batch of the second
	// EmbedBatches call (i.e. partway through wave 2) reliably fails that
	// wave while leaving wave 1 (already fully returned) untouched.
	mockEmb.failAfter = 0 // overridden dynamically below via a wrapping type

	failingEmb := &waveFailingEmbedder{waveTrackingEmbedder: mockEmb, failOnCall: 2}

	ignoreMatcher, err := NewIgnoreMatcher(tmpDir, []string{}, "")
	if err != nil {
		t.Fatalf("failed to create ignore matcher: %v", err)
	}
	scanner := NewScanner(tmpDir, ignoreMatcher)
	chunker := NewChunker(512, 50)
	idx := NewIndexer(tmpDir, mockStore, failingEmb, chunker, scanner, time.Time{})

	stats, err := idx.IndexAllWithBatchProgress(context.Background(), nil, nil)
	if err == nil {
		t.Fatal("expected an error from the simulated wave-2 failure")
	}

	if stats.FilesIndexed == 0 {
		t.Fatal("expected wave 1's files to still be indexed despite wave 2 failing")
	}
	if stats.FilesIndexed >= totalFiles {
		t.Fatalf("expected fewer than all %d files indexed (later waves should have stopped), got %d", totalFiles, stats.FilesIndexed)
	}

	mockStore.mu.Lock()
	savedDocs := len(mockStore.documents)
	mockStore.mu.Unlock()
	if savedDocs != stats.FilesIndexed {
		t.Errorf("expected saved documents (%d) to match stats.FilesIndexed (%d)", savedDocs, stats.FilesIndexed)
	}
}

// waveFailingEmbedder fails entirely on the Nth call to EmbedBatches
// (1-indexed), simulating an interruption partway through a specific wave
// of a multi-wave run, while earlier waves succeed normally.
type waveFailingEmbedder struct {
	*waveTrackingEmbedder
	failOnCall int
	calls      atomic.Int32
}

func (w *waveFailingEmbedder) EmbedBatches(ctx context.Context, batches []embedder.Batch, progress embedder.BatchProgress, onBatchDone embedder.BatchResultCallback) ([]embedder.BatchResult, error) {
	n := w.calls.Add(1)
	if int(n) == w.failOnCall {
		w.waveTrackingEmbedder.mu.Lock()
		w.waveTrackingEmbedder.callStarts = append(w.waveTrackingEmbedder.callStarts, time.Now())
		w.waveTrackingEmbedder.mu.Unlock()
		return nil, fmt.Errorf("simulated failure on embedding call %d", n)
	}
	return w.waveTrackingEmbedder.EmbedBatches(ctx, batches, progress, onBatchDone)
}
