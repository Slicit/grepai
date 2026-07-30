package indexer

import (
	"context"
	"sync"
	"testing"
	"time"
)

// TestIndexAllWithBatchProgress_ProvisionalUntilDecideCompletes is the
// regression test for the fix to a misleading progress display: because
// embedding overlaps with scan/decide, TotalFiles isn't the grand total
// for the whole run until scan/decide has finished discovering every file
// that needs (re)indexing. The Provisional flag is what tells a caller
// whether the reported total is still growing or final.
//
// This test asserts: at least one progress update during a multi-wave run
// has Provisional == true (proving the flag is actually used, not just
// always false), and every update belonging to the final wave has
// Provisional == false (proving the flag correctly reflects "no more
// waves are coming" once scan/decide has finished).
func TestIndexAllWithBatchProgress_ProvisionalUntilDecideCompletes(t *testing.T) {
	tmpDir := t.TempDir()
	// A large multiple of embedWaveSize, same reasoning as
	// TestIndexAllWithBatchProgress_EmbeddingStartsBeforeScanCompletes:
	// decide-phase's total real work needs to clearly outlast wave 1's
	// dispatch+embed overhead so this doesn't degrade into a tight race
	// under system load.
	const totalFiles = embedWaveSize * 25
	writeTinyFiles(t, tmpDir, totalFiles)

	mockStore := newMockStore()
	// No artificial delay here: we want wave 1's very first progress
	// callback (fired essentially as soon as EmbedBatches is entered) to
	// land before decide-phase finishes for all totalFiles -- the same
	// margin property TestIndexAllWithBatchProgress_EmbeddingStartsBeforeScanCompletes
	// relies on. Adding delay here would be counterproductive: it delays
	// the callback itself, which could push it past decide-phase
	// completion instead of before it.
	mockEmb := newMockBatchEmbedder()

	ignoreMatcher, err := NewIgnoreMatcher(tmpDir, []string{}, "")
	if err != nil {
		t.Fatalf("failed to create ignore matcher: %v", err)
	}
	scanner := NewScanner(tmpDir, ignoreMatcher)
	chunker := NewChunker(512, 50)
	idx := NewIndexer(tmpDir, mockStore, mockEmb, chunker, scanner, time.Time{})

	var (
		mu               sync.Mutex
		sawProvisional   bool
		lastWasFinal     bool
		observedAnything bool
	)

	_, err = idx.IndexAllWithBatchProgress(context.Background(), nil, func(info BatchProgressInfo) {
		mu.Lock()
		defer mu.Unlock()
		observedAnything = true
		if info.Provisional {
			sawProvisional = true
		}
		lastWasFinal = !info.Provisional
	})
	if err != nil {
		t.Fatalf("IndexAllWithBatchProgress failed: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if !observedAnything {
		t.Fatal("expected at least one BatchProgressInfo update")
	}
	if !sawProvisional {
		t.Error("expected at least one update to be Provisional (scan/decide still running, total may grow)")
	}
	if !lastWasFinal {
		t.Error("expected the last observed update to be non-provisional (scan/decide finished, total is final)")
	}
}

// TestIndexAllWithBatchProgress_TotalFilesNeverDecreases guards against a
// regression in the progress counting: the reported TotalFiles must never
// go down between successive updates, even though it's allowed to grow
// (that's the whole point of overlapping scan/decide with embedding).
func TestIndexAllWithBatchProgress_TotalFilesNeverDecreases(t *testing.T) {
	tmpDir := t.TempDir()
	const totalFiles = embedWaveSize * 3
	writeTinyFiles(t, tmpDir, totalFiles)

	mockStore := newMockStore()
	mockEmb := newMockBatchEmbedder()

	ignoreMatcher, err := NewIgnoreMatcher(tmpDir, []string{}, "")
	if err != nil {
		t.Fatalf("failed to create ignore matcher: %v", err)
	}
	scanner := NewScanner(tmpDir, ignoreMatcher)
	chunker := NewChunker(512, 50)
	idx := NewIndexer(tmpDir, mockStore, mockEmb, chunker, scanner, time.Time{})

	var (
		mu       sync.Mutex
		lastSeen int
	)

	_, err = idx.IndexAllWithBatchProgress(context.Background(), nil, func(info BatchProgressInfo) {
		mu.Lock()
		defer mu.Unlock()
		if info.TotalFiles < lastSeen {
			t.Errorf("TotalFiles decreased: was %d, now %d", lastSeen, info.TotalFiles)
		}
		lastSeen = info.TotalFiles
	})
	if err != nil {
		t.Fatalf("IndexAllWithBatchProgress failed: %v", err)
	}
}
