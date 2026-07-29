package indexer

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"sync/atomic"
	"testing"
	"time"
)

// writeTree creates numDirs top-level subdirectories under dir, each
// containing a single file named "file.go", so tests have several
// independent top-level entries for walkConcurrent to shard across.
func writeTree(t *testing.T, dir string, numDirs int) {
	t.Helper()
	for i := 0; i < numDirs; i++ {
		sub := filepath.Join(dir, fmt.Sprintf("dir%d", i))
		if err := os.MkdirAll(sub, 0755); err != nil {
			t.Fatalf("failed to create %s: %v", sub, err)
		}
		content := fmt.Sprintf("package dir%d\n\nfunc F() int { return %d }\n", i, i)
		if err := os.WriteFile(filepath.Join(sub, "file.go"), []byte(content), 0644); err != nil {
			t.Fatalf("failed to write file in %s: %v", sub, err)
		}
	}
}

// TestWalkConcurrent_RunsTopLevelEntriesInParallel is the core regression
// test for the scan-phase parallelism feature: it proves walkConcurrent
// actually processes multiple top-level directories at the same time
// rather than one at a time, by having visitFile block briefly and
// tracking how many calls were in flight simultaneously.
func TestWalkConcurrent_RunsTopLevelEntriesInParallel(t *testing.T) {
	tmpDir := t.TempDir()
	const numDirs = 4
	writeTree(t, tmpDir, numDirs)

	ignoreMatcher, err := NewIgnoreMatcher(tmpDir, []string{}, "")
	if err != nil {
		t.Fatalf("failed to create ignore matcher: %v", err)
	}

	var current, maxConcurrent atomic.Int32
	visit := func(path, relPath string, d fs.DirEntry) (*struct{}, string, error) {
		c := current.Add(1)
		defer current.Add(-1)
		for {
			old := maxConcurrent.Load()
			if c <= old {
				break
			}
			if maxConcurrent.CompareAndSwap(old, c) {
				break
			}
		}
		time.Sleep(50 * time.Millisecond)
		return &struct{}{}, "", nil
	}

	results, skipped, err := walkConcurrent(tmpDir, numDirs, ignoreMatcher, visit)
	if err != nil {
		t.Fatalf("walkConcurrent failed: %v", err)
	}
	if len(results) != numDirs {
		t.Fatalf("expected %d results, got %d", numDirs, len(results))
	}
	if len(skipped) != 0 {
		t.Fatalf("expected no skipped files, got %d", len(skipped))
	}

	if got := maxConcurrent.Load(); got < 2 {
		t.Errorf("expected walkConcurrent to process multiple top-level entries concurrently (>=2 at once), max observed concurrency was %d", got)
	}
}

// TestWalkConcurrent_WorkersOneRunsSequentially is the control test for the
// above: with workers=1, concurrency must never exceed 1, proving the
// workers parameter genuinely bounds concurrency rather than the walk
// always running in parallel regardless of the setting.
func TestWalkConcurrent_WorkersOneRunsSequentially(t *testing.T) {
	tmpDir := t.TempDir()
	const numDirs = 3
	writeTree(t, tmpDir, numDirs)

	ignoreMatcher, err := NewIgnoreMatcher(tmpDir, []string{}, "")
	if err != nil {
		t.Fatalf("failed to create ignore matcher: %v", err)
	}

	var current, maxConcurrent atomic.Int32
	visit := func(path, relPath string, d fs.DirEntry) (*struct{}, string, error) {
		c := current.Add(1)
		defer current.Add(-1)
		for {
			old := maxConcurrent.Load()
			if c <= old {
				break
			}
			if maxConcurrent.CompareAndSwap(old, c) {
				break
			}
		}
		time.Sleep(20 * time.Millisecond)
		return nil, "", nil
	}

	if _, _, err := walkConcurrent(tmpDir, 1, ignoreMatcher, visit); err != nil {
		t.Fatalf("walkConcurrent failed: %v", err)
	}

	if got := maxConcurrent.Load(); got > 1 {
		t.Errorf("expected workers=1 to process entries sequentially (max concurrency 1), got %d", got)
	}
}

// TestWalkConcurrent_NonPositiveWorkersDefaultsToOne verifies a
// non-positive workers value doesn't disable the walk or panic (e.g. via
// errgroup.SetLimit(0), which would allow zero concurrent goroutines and
// deadlock); it should behave the same as workers=1.
func TestWalkConcurrent_NonPositiveWorkersDefaultsToOne(t *testing.T) {
	tmpDir := t.TempDir()
	writeTree(t, tmpDir, 3)

	ignoreMatcher, err := NewIgnoreMatcher(tmpDir, []string{}, "")
	if err != nil {
		t.Fatalf("failed to create ignore matcher: %v", err)
	}

	visit := func(path, relPath string, d fs.DirEntry) (*struct{}, string, error) {
		return &struct{}{}, "", nil
	}

	results, _, err := walkConcurrent(tmpDir, 0, ignoreMatcher, visit)
	if err != nil {
		t.Fatalf("walkConcurrent failed with workers=0: %v", err)
	}
	if len(results) != 3 {
		t.Fatalf("expected 3 results, got %d", len(results))
	}
}

// scanMetaPaths runs ScanMetadata and returns just the sorted list of
// relative paths found, for equivalence comparisons across worker counts.
func scanMetaPaths(t *testing.T, s *Scanner) []string {
	t.Helper()
	files, _, err := s.ScanMetadata()
	if err != nil {
		t.Fatalf("ScanMetadata failed: %v", err)
	}
	paths := make([]string, len(files))
	for i, f := range files {
		paths[i] = f.Path
	}
	sort.Strings(paths)
	return paths
}

// TestScanner_ScanMetadata_ConcurrentMatchesSequential is the correctness
// guard for the walk-parallelization refactor: ScanMetadata must find
// exactly the same set of files regardless of how many worker goroutines
// are used to walk the tree. This is what proves sharding by top-level
// subdirectory didn't drop, duplicate, or miscategorize any files compared
// to the old single-goroutine filepath.WalkDir behavior.
func TestScanner_ScanMetadata_ConcurrentMatchesSequential(t *testing.T) {
	tmpDir := t.TempDir()
	writeTree(t, tmpDir, 6)
	// A couple of nested subdirectories and a root-level file too, so the
	// comparison covers more than just "one file per top-level dir".
	if err := os.MkdirAll(filepath.Join(tmpDir, "dir0", "nested"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(tmpDir, "dir0", "nested", "deep.go"), []byte("package nested\n"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(tmpDir, "root.go"), []byte("package main\n"), 0644); err != nil {
		t.Fatal(err)
	}

	ignoreMatcher, err := NewIgnoreMatcher(tmpDir, []string{}, "")
	if err != nil {
		t.Fatalf("failed to create ignore matcher: %v", err)
	}

	sequential := scanMetaPaths(t, NewScanner(tmpDir, ignoreMatcher).WithScanWorkers(1))
	concurrent := scanMetaPaths(t, NewScanner(tmpDir, ignoreMatcher).WithScanWorkers(8))

	if len(sequential) == 0 {
		t.Fatal("expected at least one file to be found")
	}
	if len(sequential) != len(concurrent) {
		t.Fatalf("sequential found %d files, concurrent found %d: sequential=%v concurrent=%v",
			len(sequential), len(concurrent), sequential, concurrent)
	}
	for i := range sequential {
		if sequential[i] != concurrent[i] {
			t.Fatalf("file set differs at index %d: sequential=%v concurrent=%v", i, sequential, concurrent)
		}
	}
}

// TestScanner_Scan_ConcurrentMatchesSequential is the Scan() (content +
// hash) equivalent of the above: file set, hashes, and content must be
// identical regardless of worker count.
func TestScanner_Scan_ConcurrentMatchesSequential(t *testing.T) {
	tmpDir := t.TempDir()
	writeTree(t, tmpDir, 5)

	ignoreMatcher, err := NewIgnoreMatcher(tmpDir, []string{}, "")
	if err != nil {
		t.Fatalf("failed to create ignore matcher: %v", err)
	}

	seqFiles, _, err := NewScanner(tmpDir, ignoreMatcher).WithScanWorkers(1).Scan()
	if err != nil {
		t.Fatalf("Scan (workers=1) failed: %v", err)
	}
	concFiles, _, err := NewScanner(tmpDir, ignoreMatcher).WithScanWorkers(8).Scan()
	if err != nil {
		t.Fatalf("Scan (workers=8) failed: %v", err)
	}

	toMap := func(files []FileInfo) map[string]FileInfo {
		m := make(map[string]FileInfo, len(files))
		for _, f := range files {
			m[f.Path] = f
		}
		return m
	}

	seqMap := toMap(seqFiles)
	concMap := toMap(concFiles)

	if len(seqMap) == 0 {
		t.Fatal("expected at least one file to be found")
	}
	if len(seqMap) != len(concMap) {
		t.Fatalf("sequential found %d files, concurrent found %d", len(seqMap), len(concMap))
	}
	for path, seqInfo := range seqMap {
		concInfo, ok := concMap[path]
		if !ok {
			t.Errorf("file %s found sequentially but not concurrently", path)
			continue
		}
		if seqInfo.Hash != concInfo.Hash || seqInfo.Content != concInfo.Content {
			t.Errorf("file %s differs between sequential and concurrent scan", path)
		}
	}
}

// TestScanner_ScanMetadata_RespectsIgnoredTopLevelDir verifies that an
// entire ignored top-level subdirectory is skipped before ever being
// dispatched to a worker goroutine, not just filtered out after walking
// it -- matching the old behavior of ShouldSkipDir returning
// filepath.SkipDir before descending.
func TestScanner_ScanMetadata_RespectsIgnoredTopLevelDir(t *testing.T) {
	tmpDir := t.TempDir()
	writeTree(t, tmpDir, 3)

	ignoreMatcher, err := NewIgnoreMatcher(tmpDir, []string{"dir1"}, "")
	if err != nil {
		t.Fatalf("failed to create ignore matcher: %v", err)
	}

	scanner := NewScanner(tmpDir, ignoreMatcher).WithScanWorkers(4)
	files, _, err := scanner.ScanMetadata()
	if err != nil {
		t.Fatalf("ScanMetadata failed: %v", err)
	}

	for _, f := range files {
		if f.Path == filepath.Join("dir1", "file.go") {
			t.Fatalf("expected dir1 to be ignored entirely, but found %s", f.Path)
		}
	}
	if len(files) != 2 {
		t.Fatalf("expected 2 files (dir0 and dir2), got %d: %v", len(files), files)
	}
}

// TestScanner_WithScanWorkers_DefaultAndOverride verifies the default
// worker count and that WithScanWorkers ignores non-positive values,
// mirroring the ignore-invalid-value contract used elsewhere in this
// codebase (e.g. WithOllamaParallelism, WithOllamaTimeout).
func TestScanner_WithScanWorkers_DefaultAndOverride(t *testing.T) {
	ignoreMatcher, err := NewIgnoreMatcher(t.TempDir(), []string{}, "")
	if err != nil {
		t.Fatalf("failed to create ignore matcher: %v", err)
	}

	s := NewScanner(t.TempDir(), ignoreMatcher)
	if s.workers != DefaultScanWorkers {
		t.Errorf("expected default workers %d, got %d", DefaultScanWorkers, s.workers)
	}

	s.WithScanWorkers(8)
	if s.workers != 8 {
		t.Errorf("expected workers 8 after override, got %d", s.workers)
	}

	s.WithScanWorkers(0)
	if s.workers != 8 {
		t.Errorf("expected non-positive WithScanWorkers to be ignored (workers still 8), got %d", s.workers)
	}

	s.WithScanWorkers(-3)
	if s.workers != 8 {
		t.Errorf("expected negative WithScanWorkers to be ignored (workers still 8), got %d", s.workers)
	}
}
