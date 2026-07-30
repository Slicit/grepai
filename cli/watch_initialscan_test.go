package cli

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/yoanbernabeu/grepai/config"
	"github.com/yoanbernabeu/grepai/indexer"
	"github.com/yoanbernabeu/grepai/store"
	"github.com/yoanbernabeu/grepai/trace"
	"github.com/yoanbernabeu/grepai/watcher"
)

type noOpEmbedder struct{}

func (e *noOpEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	return []float32{0.1, 0.2, 0.3}, nil
}

func (e *noOpEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	vectors := make([][]float32, len(texts))
	for i := range texts {
		vectors[i] = []float32{0.1, 0.2, 0.3}
	}
	return vectors, nil
}

func (e *noOpEmbedder) Dimensions() int {
	return 3
}

func (e *noOpEmbedder) Close() error {
	return nil
}

type countingEmbedder struct {
	noOpEmbedder
	embedCalls      int
	embedBatchCalls int
}

func (e *countingEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	e.embedCalls++
	return e.noOpEmbedder.Embed(ctx, text)
}

func (e *countingEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	e.embedBatchCalls++
	return e.noOpEmbedder.EmbedBatch(ctx, texts)
}

func TestRunInitialScan_SkipsSymbolExtractionWhenContentHashMatches(t *testing.T) {
	ctx := context.Background()
	projectRoot := t.TempDir()

	srcPath := filepath.Join(projectRoot, "main.go")
	srcContent := "package main\n\nfunc real() {}\n"
	if err := os.WriteFile(srcPath, []byte(srcContent), 0644); err != nil {
		t.Fatalf("failed to create source file: %v", err)
	}

	ignoreMatcher, err := indexer.NewIgnoreMatcher(projectRoot, []string{}, "")
	if err != nil {
		t.Fatalf("failed to create ignore matcher: %v", err)
	}

	scanner := indexer.NewScanner(projectRoot, ignoreMatcher)
	chunker := indexer.NewChunker(512, 50)
	vecStore := store.NewGOBStore(filepath.Join(projectRoot, "index.gob"))
	idx := indexer.NewIndexer(projectRoot, vecStore, &noOpEmbedder{}, chunker, scanner, time.Now().Add(1*time.Hour))

	symbolStore := trace.NewGOBSymbolStore(filepath.Join(projectRoot, "symbols.gob"))
	if err := symbolStore.Load(ctx); err != nil {
		t.Fatalf("failed to load symbol store: %v", err)
	}
	defer symbolStore.Close()

	fileInfo, err := scanner.ScanFile("main.go")
	if err != nil {
		t.Fatalf("failed to scan file: %v", err)
	}
	if fileInfo == nil {
		t.Fatal("expected scanned file info")
	}

	sentinel := []trace.Symbol{
		{
			Name:     "sentinel",
			Kind:     trace.KindFunction,
			File:     "main.go",
			Line:     1,
			Language: "go",
		},
	}
	if err := symbolStore.SaveFileWithContentHash(ctx, fileInfo.Path, fileInfo.Hash, sentinel, nil, fileInfo.Size, fileInfo.ModTime); err != nil {
		t.Fatalf("failed to seed symbol store: %v", err)
	}

	extractor := trace.NewRegexExtractor()
	if _, err := runInitialScan(ctx, idx, scanner, extractor, symbolStore, []string{".go"}, time.Time{}, true, nil, nil); err != nil {
		t.Fatalf("runInitialScan failed: %v", err)
	}

	sentinelSymbols, err := symbolStore.LookupSymbol(ctx, "sentinel")
	if err != nil {
		t.Fatalf("failed to lookup sentinel symbol: %v", err)
	}
	if len(sentinelSymbols) == 0 {
		t.Fatal("expected seeded sentinel symbol to remain when hash matches")
	}

	realSymbols, err := symbolStore.LookupSymbol(ctx, "real")
	if err != nil {
		t.Fatalf("failed to lookup real symbol: %v", err)
	}
	if len(realSymbols) != 0 {
		t.Fatalf("expected real symbol extraction to be skipped, found %d symbols", len(realSymbols))
	}
}

func TestRunInitialScan_SkipsIndexedFileByLastIndexTime(t *testing.T) {
	ctx := context.Background()
	projectRoot := t.TempDir()

	srcPath := filepath.Join(projectRoot, "main.go")
	srcContent := "package main\n\nfunc real() {}\n"
	if err := os.WriteFile(srcPath, []byte(srcContent), 0644); err != nil {
		t.Fatalf("failed to create source file: %v", err)
	}

	ignoreMatcher, err := indexer.NewIgnoreMatcher(projectRoot, []string{}, "")
	if err != nil {
		t.Fatalf("failed to create ignore matcher: %v", err)
	}

	emb := &countingEmbedder{}
	scanner := indexer.NewScanner(projectRoot, ignoreMatcher)
	chunker := indexer.NewChunker(512, 50)
	vecStore := store.NewGOBStore(filepath.Join(projectRoot, "index.gob"))
	idx := indexer.NewIndexer(projectRoot, vecStore, emb, chunker, scanner, time.Now().Add(1*time.Hour))

	// Seed a document with ChunkIDs so the lastIndexTime gate can skip it.
	// The new logic requires doc != nil && len(doc.ChunkIDs) > 0 to skip.
	if err := vecStore.SaveDocument(ctx, store.Document{
		Path:     "main.go",
		Hash:     "seeded",
		ChunkIDs: []string{"c1"},
	}); err != nil {
		t.Fatalf("failed to seed document: %v", err)
	}

	symbolStore := trace.NewGOBSymbolStore(filepath.Join(projectRoot, "symbols.gob"))
	defer symbolStore.Close()

	if err := symbolStore.SaveFile(ctx, "main.go", []trace.Symbol{
		{
			Name:     "sentinel",
			Kind:     trace.KindFunction,
			File:     "main.go",
			Line:     1,
			Language: "go",
		},
	}, nil); err != nil {
		t.Fatalf("failed to seed symbol store: %v", err)
	}

	lastIndexTime := time.Now().Add(1 * time.Hour)
	extractor := trace.NewRegexExtractor()
	if _, err := runInitialScan(ctx, idx, scanner, extractor, symbolStore, []string{".go"}, lastIndexTime, true, nil, nil); err != nil {
		t.Fatalf("runInitialScan failed: %v", err)
	}

	sentinelSymbols, err := symbolStore.LookupSymbol(ctx, "sentinel")
	if err != nil {
		t.Fatalf("failed to lookup sentinel symbol: %v", err)
	}
	if len(sentinelSymbols) == 0 {
		t.Fatal("expected sentinel symbol to remain when file is skipped by lastIndexTime")
	}

	realSymbols, err := symbolStore.LookupSymbol(ctx, "real")
	if err != nil {
		t.Fatalf("failed to lookup real symbol: %v", err)
	}
	if len(realSymbols) != 0 {
		t.Fatalf("expected real symbol extraction to be skipped, found %d symbols", len(realSymbols))
	}

	if emb.embedCalls != 0 || emb.embedBatchCalls != 0 {
		t.Fatalf("expected no embedding calls for skipped startup path, got embed=%d embedBatch=%d", emb.embedCalls, emb.embedBatchCalls)
	}
}

func TestHandleFileEvent_SkipsUnchangedFile(t *testing.T) {
	ctx := context.Background()
	projectRoot := t.TempDir()

	srcPath := filepath.Join(projectRoot, "main.go")
	srcContent := "package main\n\nfunc real() {}\n"
	if err := os.WriteFile(srcPath, []byte(srcContent), 0644); err != nil {
		t.Fatalf("failed to create source file: %v", err)
	}

	ignoreMatcher, err := indexer.NewIgnoreMatcher(projectRoot, []string{}, "")
	if err != nil {
		t.Fatalf("failed to create ignore matcher: %v", err)
	}

	emb := &countingEmbedder{}
	scanner := indexer.NewScanner(projectRoot, ignoreMatcher)
	chunker := indexer.NewChunker(512, 50)
	vecStore := store.NewGOBStore(filepath.Join(projectRoot, "index.gob"))
	idx := indexer.NewIndexer(projectRoot, vecStore, emb, chunker, scanner, time.Time{})

	fileInfo, err := scanner.ScanFile("main.go")
	if err != nil || fileInfo == nil {
		t.Fatalf("failed to scan source file: %v", err)
	}
	if err := vecStore.SaveDocument(ctx, store.Document{
		Path:     "main.go",
		Hash:     fileInfo.Hash,
		ModTime:  time.Unix(fileInfo.ModTime, 0),
		ChunkIDs: []string{"c1"},
	}); err != nil {
		t.Fatalf("failed to seed document: %v", err)
	}

	symbolStore := trace.NewGOBSymbolStore(filepath.Join(projectRoot, "symbols.gob"))
	defer symbolStore.Close()
	if err := symbolStore.SaveFile(ctx, "main.go", []trace.Symbol{
		{
			Name:     "sentinel",
			Kind:     trace.KindFunction,
			File:     "main.go",
			Line:     1,
			Language: "go",
		},
	}, nil); err != nil {
		t.Fatalf("failed to seed symbol store: %v", err)
	}

	cfg := config.DefaultConfig()
	lastWrite := time.Time{}
	handleFileEvent(
		ctx,
		idx,
		scanner,
		trace.NewRegexExtractor(),
		symbolStore,
		nil,
		nil,
		[]string{".go"},
		projectRoot,
		cfg,
		&lastWrite,
		nil,
		watcher.FileEvent{Type: watcher.EventModify, Path: "main.go"},
		nil,
		nil,
	)

	if emb.embedCalls != 0 || emb.embedBatchCalls != 0 {
		t.Fatalf("expected unchanged file to skip embedding, got embed=%d embedBatch=%d", emb.embedCalls, emb.embedBatchCalls)
	}

	realSymbols, err := symbolStore.LookupSymbol(ctx, "real")
	if err != nil {
		t.Fatalf("failed to lookup real symbol: %v", err)
	}
	if len(realSymbols) != 0 {
		t.Fatalf("expected no new symbol extraction for unchanged file, got %d", len(realSymbols))
	}

	if !cfg.Watch.LastIndexTime.IsZero() {
		t.Fatalf("expected config last index time to remain zero on skip, got %v", cfg.Watch.LastIndexTime)
	}
}

func TestHandleWorkspaceFileEvent_SkipsUnchangedFile(t *testing.T) {
	ctx := context.Background()
	tmpRoot := t.TempDir()
	origWD, err := os.Getwd()
	if err != nil {
		t.Fatalf("failed to get cwd: %v", err)
	}
	if err := os.Chdir(tmpRoot); err != nil {
		t.Fatalf("failed to chdir to temp root: %v", err)
	}
	defer func() {
		_ = os.Chdir(origWD)
	}()

	projectPath := "proj"
	projectRoot := filepath.Join(tmpRoot, projectPath)
	if err := os.MkdirAll(filepath.Join(projectRoot, "proj"), 0755); err != nil {
		t.Fatalf("failed to create project dirs: %v", err)
	}

	srcPath := filepath.Join(projectRoot, "proj", "main.go")
	srcContent := "package main\n\nfunc real() {}\n"
	if err := os.WriteFile(srcPath, []byte(srcContent), 0644); err != nil {
		t.Fatalf("failed to create source file: %v", err)
	}

	ignoreMatcher, err := indexer.NewIgnoreMatcher(projectPath, []string{}, "")
	if err != nil {
		t.Fatalf("failed to create ignore matcher: %v", err)
	}
	scanner := indexer.NewScanner(projectPath, ignoreMatcher)
	fileInfo, err := scanner.ScanFile("proj/main.go")
	if err != nil || fileInfo == nil {
		t.Fatalf("failed to scan source file: %v", err)
	}

	st := store.NewGOBStore(filepath.Join(projectRoot, "workspace-index.gob"))
	workspaceName := "ws"
	projectName := "proj"
	prefixedPath := workspaceName + "/" + projectName + "/proj/main.go"
	if err := st.SaveDocument(ctx, store.Document{
		Path:     prefixedPath,
		Hash:     fileInfo.Hash,
		ModTime:  time.Unix(fileInfo.ModTime, 0),
		ChunkIDs: []string{"c1"},
	}); err != nil {
		t.Fatalf("failed to seed workspace document: %v", err)
	}

	emb := &countingEmbedder{}
	wrappedStore := &projectPrefixStore{
		store:         st,
		workspaceName: workspaceName,
		projectName:   projectName,
		projectPath:   projectPath,
	}
	chunker := indexer.NewChunker(512, 64)
	idx := indexer.NewIndexer(projectPath, wrappedStore, emb, chunker, scanner, time.Time{})
	extractor := trace.NewRegexExtractor()
	cfg := config.DefaultConfig()
	var lastConfigWrite time.Time

	handleFileEvent(ctx, idx, scanner, extractor, nil, nil, wrappedStore, nil, projectPath, cfg, &lastConfigWrite, nil, watcher.FileEvent{
		Type: watcher.EventModify,
		Path: "proj/main.go",
	}, nil, nil)

	if emb.embedCalls != 0 || emb.embedBatchCalls != 0 {
		t.Fatalf("expected unchanged workspace file to skip embedding, got embed=%d embedBatch=%d", emb.embedCalls, emb.embedBatchCalls)
	}

	stats, err := st.GetStats(ctx)
	if err != nil {
		t.Fatalf("failed to get stats: %v", err)
	}
	if stats.TotalFiles != 1 {
		t.Fatalf("expected workspace docs to remain unchanged, got total files %d", stats.TotalFiles)
	}
}

func TestHandleFileEvent_IndexesChangedFileAndUpdatesSymbols(t *testing.T) {
	ctx := context.Background()
	projectRoot := t.TempDir()

	srcPath := filepath.Join(projectRoot, "main.go")
	srcContent := "package main\n\nfunc real() {}\n"
	if err := os.WriteFile(srcPath, []byte(srcContent), 0644); err != nil {
		t.Fatalf("failed to create source file: %v", err)
	}

	ignoreMatcher, err := indexer.NewIgnoreMatcher(projectRoot, []string{}, "")
	if err != nil {
		t.Fatalf("failed to create ignore matcher: %v", err)
	}

	emb := &countingEmbedder{}
	scanner := indexer.NewScanner(projectRoot, ignoreMatcher)
	chunker := indexer.NewChunker(512, 50)
	vecStore := store.NewGOBStore(filepath.Join(projectRoot, "index.gob"))
	idx := indexer.NewIndexer(projectRoot, vecStore, emb, chunker, scanner, time.Time{})

	// Seed old hash to force NeedsReindex == true.
	fileInfo, err := scanner.ScanFile("main.go")
	if err != nil || fileInfo == nil {
		t.Fatalf("failed to scan source file: %v", err)
	}
	if err := vecStore.SaveDocument(ctx, store.Document{
		Path:    "main.go",
		Hash:    "old-hash",
		ModTime: time.Unix(fileInfo.ModTime, 0),
	}); err != nil {
		t.Fatalf("failed to seed old document: %v", err)
	}

	symbolStore := trace.NewGOBSymbolStore(filepath.Join(projectRoot, "symbols.gob"))
	defer symbolStore.Close()

	cfg := config.DefaultConfig()
	lastWrite := time.Time{}
	handleFileEvent(
		ctx,
		idx,
		scanner,
		trace.NewRegexExtractor(),
		symbolStore,
		nil,
		nil,
		[]string{".go"},
		projectRoot,
		cfg,
		&lastWrite,
		nil,
		watcher.FileEvent{Type: watcher.EventModify, Path: "main.go"},
		nil,
		nil,
	)

	if emb.embedCalls == 0 && emb.embedBatchCalls == 0 {
		t.Fatal("expected changed file to trigger embedding")
	}
	if cfg.Watch.LastIndexTime.IsZero() {
		t.Fatal("expected changed file to update config last index time")
	}
	if lastWrite.IsZero() {
		t.Fatal("expected last config write timestamp to be updated")
	}

	realSymbols, err := symbolStore.LookupSymbol(ctx, "real")
	if err != nil {
		t.Fatalf("failed to lookup real symbol: %v", err)
	}
	if len(realSymbols) == 0 {
		t.Fatal("expected symbols to be extracted and saved for changed file")
	}
}

func TestHandleFileEvent_DeleteRemovesIndexAndSymbols(t *testing.T) {
	ctx := context.Background()
	projectRoot := t.TempDir()

	ignoreMatcher, err := indexer.NewIgnoreMatcher(projectRoot, []string{}, "")
	if err != nil {
		t.Fatalf("failed to create ignore matcher: %v", err)
	}

	emb := &countingEmbedder{}
	scanner := indexer.NewScanner(projectRoot, ignoreMatcher)
	chunker := indexer.NewChunker(512, 50)
	vecStore := store.NewGOBStore(filepath.Join(projectRoot, "index.gob"))
	idx := indexer.NewIndexer(projectRoot, vecStore, emb, chunker, scanner, time.Time{})

	if err := vecStore.SaveDocument(ctx, store.Document{
		Path: "main.go",
		Hash: "hash",
	}); err != nil {
		t.Fatalf("failed to seed document: %v", err)
	}

	symbolStore := trace.NewGOBSymbolStore(filepath.Join(projectRoot, "symbols.gob"))
	defer symbolStore.Close()
	if err := symbolStore.SaveFile(ctx, "main.go", []trace.Symbol{
		{
			Name:     "sentinel",
			Kind:     trace.KindFunction,
			File:     "main.go",
			Line:     1,
			Language: "go",
		},
	}, nil); err != nil {
		t.Fatalf("failed to seed symbol store: %v", err)
	}

	cfg := config.DefaultConfig()
	lastWrite := time.Time{}
	handleFileEvent(
		ctx,
		idx,
		scanner,
		trace.NewRegexExtractor(),
		symbolStore,
		nil,
		nil,
		[]string{".go"},
		projectRoot,
		cfg,
		&lastWrite,
		nil,
		watcher.FileEvent{Type: watcher.EventDelete, Path: "main.go"},
		nil,
		nil,
	)

	doc, err := vecStore.GetDocument(ctx, "main.go")
	if err != nil {
		t.Fatalf("failed to read document: %v", err)
	}
	if doc != nil {
		t.Fatal("expected document to be deleted on delete event")
	}
	if symbolStore.IsFileIndexed("main.go") {
		t.Fatal("expected symbols to be deleted on delete event")
	}
}

func TestEmitInitialStatsSnapshot_ReportsExistingTotals(t *testing.T) {
	ctx := context.Background()
	projectRoot := t.TempDir()

	vecStore := store.NewGOBStore(filepath.Join(projectRoot, "index.gob"))
	if err := vecStore.SaveChunks(ctx, []store.Chunk{
		{ID: "chunk-1", FilePath: "main.go"},
		{ID: "chunk-2", FilePath: "main.go"},
	}); err != nil {
		t.Fatalf("failed to seed chunks: %v", err)
	}
	if err := vecStore.SaveDocument(ctx, store.Document{
		Path:     "main.go",
		Hash:     "hash",
		ChunkIDs: []string{"chunk-1", "chunk-2"},
	}); err != nil {
		t.Fatalf("failed to seed document: %v", err)
	}

	symbolStore := trace.NewGOBSymbolStore(filepath.Join(projectRoot, "symbols.gob"))
	defer symbolStore.Close()
	if err := symbolStore.SaveFile(ctx, "main.go", []trace.Symbol{
		{
			Name:     "Foo",
			Kind:     trace.KindFunction,
			File:     "main.go",
			Line:     1,
			Language: "go",
		},
		{
			Name:     "Bar",
			Kind:     trace.KindFunction,
			File:     "main.go",
			Line:     5,
			Language: "go",
		},
	}, nil); err != nil {
		t.Fatalf("failed to seed symbol store: %v", err)
	}

	var got watchStatsDelta
	calls := 0
	emitInitialStatsSnapshot(ctx, vecStore, symbolStore, projectRoot, func(_ string, delta watchStatsDelta) {
		calls++
		got = delta
	})

	if calls != 1 {
		t.Fatalf("stats callback calls = %d, want 1", calls)
	}
	if got.ChunksCreated != 2 {
		t.Fatalf("chunks created = %d, want 2", got.ChunksCreated)
	}
	if got.FilesIndexed != 1 {
		t.Fatalf("files indexed = %d, want 1", got.FilesIndexed)
	}
	if got.SymbolsFound != 2 {
		t.Fatalf("symbols found = %d, want 2", got.SymbolsFound)
	}
	if !got.Snapshot {
		t.Fatal("expected snapshot delta to be marked as Snapshot")
	}
}

// TestRunInitialScan_RestartWithNoChangesSkipsEveryFileViaFastSkip is the
// regression test for the "restarting the watcher can take an hour even
// though nothing changed" symptom: the symbol-index build phase used to
// re-read and re-hash every traced file sequentially on every restart, and
// on a repo with many files (or a slow filesystem) that dominated startup
// time even when nothing needed re-indexing.
//
// This builds a repo with many files, runs runInitialScan once (a cold
// build), then runs it again against the same symbolStore/scanner (a
// simulated restart with zero changes) and asserts every file's symbols
// survive untouched and no new extraction happens -- i.e. FastSkip (backed
// by the per-file size/mtime recorded during the first run) correctly
// short-circuits every single file on the second pass without needing
// lastIndexTime at all (passed as zero here, matching a fresh process that
// hasn't loaded any config watermark yet).
func TestRunInitialScan_RestartWithNoChangesSkipsEveryFileViaFastSkip(t *testing.T) {
	ctx := context.Background()
	projectRoot := t.TempDir()

	const fileCount = 40
	for i := 0; i < fileCount; i++ {
		path := filepath.Join(projectRoot, fmt.Sprintf("file%03d.go", i))
		content := fmt.Sprintf("package main\n\nfunc F%03d() {}\n", i)
		if err := os.WriteFile(path, []byte(content), 0644); err != nil {
			t.Fatalf("failed to create %s: %v", path, err)
		}
	}

	ignoreMatcher, err := indexer.NewIgnoreMatcher(projectRoot, []string{}, "")
	if err != nil {
		t.Fatalf("failed to create ignore matcher: %v", err)
	}
	scanner := indexer.NewScanner(projectRoot, ignoreMatcher)
	chunker := indexer.NewChunker(512, 50)
	vecStore := store.NewGOBStore(filepath.Join(projectRoot, "index.gob"))
	idx := indexer.NewIndexer(projectRoot, vecStore, &noOpEmbedder{}, chunker, scanner, time.Time{})

	symbolStore := trace.NewGOBSymbolStore(filepath.Join(projectRoot, "symbols.gob"))
	if err := symbolStore.Load(ctx); err != nil {
		t.Fatalf("failed to load symbol store: %v", err)
	}
	defer symbolStore.Close()

	extractor := trace.NewRegexExtractor()

	// Cold build: every file needs real extraction.
	if _, err := runInitialScan(ctx, idx, scanner, extractor, symbolStore, []string{".go"}, time.Time{}, true, nil, nil); err != nil {
		t.Fatalf("first runInitialScan failed: %v", err)
	}

	for i := 0; i < fileCount; i++ {
		name := fmt.Sprintf("F%03d", i)
		syms, err := symbolStore.LookupSymbol(ctx, name)
		if err != nil {
			t.Fatalf("LookupSymbol(%s) failed: %v", name, err)
		}
		if len(syms) == 0 {
			t.Fatalf("expected %s to be indexed after the first run", name)
		}
	}

	// Confirm FastSkip actually recorded every file -- if it hadn't, the
	// second run below would silently fall through to a real re-hash for
	// everything and this test would still pass, defeating its purpose.
	for i := 0; i < fileCount; i++ {
		relPath := fmt.Sprintf("file%03d.go", i)
		fileInfo, err := scanner.ScanFile(relPath)
		if err != nil || fileInfo == nil {
			t.Fatalf("failed to scan %s for verification: %v", relPath, err)
		}
		if !symbolStore.FastSkip(relPath, fileInfo.Size, fileInfo.ModTime) {
			t.Fatalf("expected FastSkip to match %s after the first run recorded its size/mtime", relPath)
		}
	}

	// Seed a sentinel per file so we can detect any unwanted re-extraction:
	// re-extraction would call SaveFileWithContentHash, which replaces a
	// file's whole symbol set -- so if the sentinel survives, that file was
	// never re-extracted on the "restart" below.
	for i := 0; i < fileCount; i++ {
		relPath := fmt.Sprintf("file%03d.go", i)
		hash, ok := symbolStore.GetFileContentHash(relPath)
		if !ok {
			t.Fatalf("expected a stored content hash for %s", relPath)
		}
		sentinel := []trace.Symbol{{
			Name:     fmt.Sprintf("Sentinel%03d", i),
			Kind:     trace.KindFunction,
			File:     relPath,
			Line:     1,
			Language: "go",
		}}
		fileInfo, err := scanner.ScanFile(relPath)
		if err != nil || fileInfo == nil {
			t.Fatalf("failed to scan %s: %v", relPath, err)
		}
		if err := symbolStore.SaveFileWithContentHash(ctx, relPath, hash, sentinel, nil, fileInfo.Size, fileInfo.ModTime); err != nil {
			t.Fatalf("failed to seed sentinel for %s: %v", relPath, err)
		}
	}

	// Simulated restart: same files, same store, zero changes, and
	// lastIndexTime passed as zero (as if the config watermark were
	// unavailable) -- FastSkip must be the thing that saves this, not the
	// legacy lastIndexTime gate.
	if _, err := runInitialScan(ctx, idx, scanner, extractor, symbolStore, []string{".go"}, time.Time{}, true, nil, nil); err != nil {
		t.Fatalf("second runInitialScan failed: %v", err)
	}

	for i := 0; i < fileCount; i++ {
		relPath := fmt.Sprintf("file%03d.go", i)
		name := fmt.Sprintf("Sentinel%03d", i)
		syms, err := symbolStore.LookupSymbol(ctx, name)
		if err != nil {
			t.Fatalf("LookupSymbol(%s) failed: %v", name, err)
		}
		if len(syms) == 0 {
			t.Fatalf("expected sentinel for %s to survive the no-op restart (FastSkip should have prevented re-extraction)", relPath)
		}
	}
}
