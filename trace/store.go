package trace

import (
	"context"
	"encoding/gob"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/yoanbernabeu/grepai/internal/fileutil"
)

// GOBSymbolStore implements SymbolStore using GOB encoding.
type GOBSymbolStore struct {
	indexPath         string
	lockPath          string
	index             *SymbolIndex
	fileIndex         map[string]bool
	fileContentHashes map[string]string
	// fileMeta records, per file, the size and mtime last observed when its
	// content hash was computed. FastSkip uses this to answer "has this file
	// possibly changed?" without opening it -- see FastSkip for why that
	// matters (avoiding a full disk read + SHA-256 for every traced file on
	// every restart is the difference between a restart taking seconds and
	// one taking an hour on a large repo or a slow filesystem).
	fileMeta map[string]fileMetaRecord
	mu       sync.RWMutex
}

// fileMetaRecord is the (size, mtime) pair FastSkip compares against to
// decide, with zero I/O, whether a file needs to be re-read at all.
type fileMetaRecord struct {
	Size    int64
	ModTime int64
}

type gobSymbolData struct {
	Index             SymbolIndex
	FileIndex         map[string]bool
	FileContentHashes map[string]string
	FileMeta          map[string]fileMetaRecord
}

// NewGOBSymbolStore creates a new GOB-based symbol store.
func NewGOBSymbolStore(indexPath string) *GOBSymbolStore {
	return &GOBSymbolStore{
		indexPath: indexPath,
		lockPath:  indexPath + ".lock",
		index: &SymbolIndex{
			Symbols:    make(map[string][]Symbol),
			References: make(map[string][]Reference),
			CallGraph:  []CallEdge{},
			Version:    1,
		},
		fileIndex:         make(map[string]bool),
		fileContentHashes: make(map[string]string),
		fileMeta:          make(map[string]fileMetaRecord),
	}
}

// Load reads the index from storage.
func (s *GOBSymbolStore) Load(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	lockFile, err := os.OpenFile(s.lockPath, os.O_CREATE|os.O_RDWR, 0644)
	if err != nil {
		return s.loadUnlocked()
	}
	defer lockFile.Close()
	if err := fileutil.FlockShared(lockFile, false); err != nil {
		return s.loadUnlocked()
	}
	defer func() {
		_ = fileutil.Funlock(lockFile)
	}()

	return s.loadUnlocked()
}

func (s *GOBSymbolStore) loadUnlocked() error {
	file, err := os.Open(s.indexPath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("failed to open symbol index: %w", err)
	}
	defer file.Close()

	var data gobSymbolData
	if err := gob.NewDecoder(file).Decode(&data); err != nil {
		return fmt.Errorf("failed to decode symbol index: %w", err)
	}

	s.index = &data.Index
	s.fileIndex = data.FileIndex
	s.fileContentHashes = data.FileContentHashes
	s.fileMeta = data.FileMeta

	if s.index.Symbols == nil {
		s.index.Symbols = make(map[string][]Symbol)
	}
	if s.index.References == nil {
		s.index.References = make(map[string][]Reference)
	}
	if s.index.CallGraph == nil {
		s.index.CallGraph = []CallEdge{}
	}
	if s.fileIndex == nil {
		s.fileIndex = make(map[string]bool)
	}
	if s.fileContentHashes == nil {
		s.fileContentHashes = make(map[string]string)
	}
	// Older on-disk indexes (encoded before FileMeta existed) simply omit
	// the field -- gob leaves it as nil rather than erroring, since decoding
	// matches by field name and tolerates the receiving struct having a
	// field the encoded data doesn't. Every file falls back to the existing
	// lastIndexTime-based gate below until it's next saved, at which point
	// FastSkip starts working for it.
	if s.fileMeta == nil {
		s.fileMeta = make(map[string]fileMetaRecord)
	}

	return nil
}

// Persist writes the index to storage.
func (s *GOBSymbolStore) Persist(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if err := fileutil.EnsureParentDir(s.indexPath); err != nil {
		return fmt.Errorf("failed to prepare symbol index directory: %w", err)
	}

	lockFile, err := os.OpenFile(s.lockPath, os.O_CREATE|os.O_RDWR, 0644)
	if err != nil {
		return s.persistUnlocked()
	}
	defer lockFile.Close()
	if err := fileutil.FlockExclusive(lockFile, false); err != nil {
		return s.persistUnlocked()
	}
	defer func() {
		_ = fileutil.Funlock(lockFile)
	}()

	return s.persistUnlocked()
}

func (s *GOBSymbolStore) persistUnlocked() error {
	s.index.UpdatedAt = time.Now()
	data := gobSymbolData{
		Index:             *s.index,
		FileIndex:         s.fileIndex,
		FileContentHashes: s.fileContentHashes,
		FileMeta:          s.fileMeta,
	}

	tmpFile, err := os.CreateTemp(filepath.Dir(s.indexPath), filepath.Base(s.indexPath)+".tmp-*")
	if err != nil {
		return fmt.Errorf("failed to create symbol index temp file: %w", err)
	}

	tmpPath := tmpFile.Name()
	cleanupTemp := true
	defer func() {
		if cleanupTemp {
			_ = os.Remove(tmpPath)
		}
	}()

	if err := gob.NewEncoder(tmpFile).Encode(data); err != nil {
		_ = tmpFile.Close()
		return fmt.Errorf("failed to encode symbol index: %w", err)
	}
	if err := tmpFile.Sync(); err != nil {
		_ = tmpFile.Close()
		return fmt.Errorf("failed to sync symbol index temp file: %w", err)
	}
	if err := tmpFile.Close(); err != nil {
		return fmt.Errorf("failed to close symbol index temp file: %w", err)
	}
	if err := fileutil.ReplaceFileAtomically(tmpPath, s.indexPath); err != nil {
		return fmt.Errorf("failed to replace symbol index file: %w", err)
	}
	cleanupTemp = false

	return nil
}

// SaveFile persists symbols and references for a file.
func (s *GOBSymbolStore) SaveFile(ctx context.Context, filePath string, symbols []Symbol, refs []Reference) error {
	return s.SaveFileWithContentHash(ctx, filePath, "", symbols, refs, 0, 0)
}

// SaveFileWithContentHash persists symbols/references for a file and tracks
// the current file content hash for future cache checks. size and modTime
// are the values observed when contentHash was computed (pass 0, 0 if
// unknown, e.g. from SaveFile below) -- they seed FastSkip so a future
// restart can skip this file without reading it, as long as neither has
// changed since.
func (s *GOBSymbolStore) SaveFileWithContentHash(ctx context.Context, filePath string, contentHash string, symbols []Symbol, refs []Reference, size int64, modTime int64) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Remove old entries for this file first
	s.deleteFileUnlocked(filePath)

	// Add new symbols
	for _, sym := range symbols {
		s.index.Symbols[sym.Name] = append(s.index.Symbols[sym.Name], sym)
	}

	// Add new references
	for _, ref := range refs {
		s.index.References[ref.SymbolName] = append(s.index.References[ref.SymbolName], ref)
	}

	// Build call graph edges
	for _, ref := range refs {
		if ref.CallerName != "" && ref.CallerName != "<top-level>" {
			s.index.CallGraph = append(s.index.CallGraph, CallEdge{
				Caller:   ref.CallerName,
				Callee:   ref.SymbolName,
				File:     ref.File,
				Line:     ref.Line,
				CallType: "direct",
			})
		}
	}

	s.fileIndex[filePath] = true
	if contentHash != "" {
		s.fileContentHashes[filePath] = contentHash
		s.fileMeta[filePath] = fileMetaRecord{Size: size, ModTime: modTime}
	} else {
		delete(s.fileContentHashes, filePath)
		delete(s.fileMeta, filePath)
	}
	return nil
}

// DeleteFile removes all symbols and references for a file.
func (s *GOBSymbolStore) DeleteFile(ctx context.Context, filePath string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.deleteFileUnlocked(filePath)
	return nil
}

func (s *GOBSymbolStore) deleteFileUnlocked(filePath string) {
	// Remove symbols from this file
	for name, symbols := range s.index.Symbols {
		filtered := make([]Symbol, 0, len(symbols))
		for _, sym := range symbols {
			if sym.File != filePath {
				filtered = append(filtered, sym)
			}
		}
		if len(filtered) == 0 {
			delete(s.index.Symbols, name)
		} else {
			s.index.Symbols[name] = filtered
		}
	}

	// Remove references from this file
	for name, refs := range s.index.References {
		filtered := make([]Reference, 0, len(refs))
		for _, ref := range refs {
			if ref.File != filePath {
				filtered = append(filtered, ref)
			}
		}
		if len(filtered) == 0 {
			delete(s.index.References, name)
		} else {
			s.index.References[name] = filtered
		}
	}

	// Remove call graph edges from this file
	filtered := make([]CallEdge, 0, len(s.index.CallGraph))
	for _, edge := range s.index.CallGraph {
		if edge.File != filePath {
			filtered = append(filtered, edge)
		}
	}
	s.index.CallGraph = filtered

	delete(s.fileIndex, filePath)
	delete(s.fileContentHashes, filePath)
	delete(s.fileMeta, filePath)
}

// LookupSymbol finds symbol definitions by name.
func (s *GOBSymbolStore) LookupSymbol(ctx context.Context, name string) ([]Symbol, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	symbols := s.index.Symbols[name]
	if symbols == nil {
		return []Symbol{}, nil
	}
	return symbols, nil
}

// LookupCallers finds all references/callers of a symbol.
func (s *GOBSymbolStore) LookupCallers(ctx context.Context, symbolName string) ([]Reference, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	refs := s.index.References[symbolName]
	if refs == nil {
		return []Reference{}, nil
	}
	return filterByReferenceKinds(refs, RefKindCall, ""), nil
}

// LookupCallees finds all symbols called by a function.
func (s *GOBSymbolStore) LookupCallees(ctx context.Context, symbolName string, file string) ([]Reference, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var callees []Reference
	seen := make(map[string]bool)

	for _, edge := range s.index.CallGraph {
		if edge.Caller == symbolName {
			key := fmt.Sprintf("%s:%d", edge.File, edge.Line)
			if seen[key] {
				continue
			}
			seen[key] = true

			// Find reference details
			if refs, ok := s.index.References[edge.Callee]; ok {
				for _, ref := range refs {
					if !isCallReference(ref) {
						continue
					}
					if ref.CallerName == symbolName && ref.File == edge.File && ref.Line == edge.Line {
						callees = append(callees, ref)
						break
					}
				}
			}

			// If no reference found, create a minimal one
			if !seen[key] || len(callees) == 0 {
				callees = append(callees, Reference{
					SymbolName: edge.Callee,
					File:       edge.File,
					Line:       edge.Line,
					CallerName: symbolName,
				})
			}
		}
	}
	return callees, nil
}

// LookupReaders finds property/data readers for a symbol name.
func (s *GOBSymbolStore) LookupReaders(ctx context.Context, symbolName string) ([]Reference, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	refs := s.index.References[symbolName]
	if refs == nil {
		return []Reference{}, nil
	}
	return filterByReferenceKinds(refs, RefKindRead), nil
}

// LookupWriters finds property/data writers for a symbol name.
func (s *GOBSymbolStore) LookupWriters(ctx context.Context, symbolName string) ([]Reference, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	refs := s.index.References[symbolName]
	if refs == nil {
		return []Reference{}, nil
	}
	return filterByReferenceKinds(refs, RefKindWrite), nil
}

func filterByReferenceKinds(refs []Reference, kinds ...string) []Reference {
	if len(refs) == 0 {
		return []Reference{}
	}

	allowed := make(map[string]bool, len(kinds))
	for _, kind := range kinds {
		allowed[kind] = true
	}

	filtered := make([]Reference, 0, len(refs))
	for _, ref := range refs {
		if allowed[ref.Kind] {
			filtered = append(filtered, ref)
			continue
		}
		// Backward compatibility with older indices where kind wasn't persisted.
		if ref.Kind == "" && allowed[RefKindCall] {
			filtered = append(filtered, ref)
		}
	}

	return filtered
}

func isCallReference(ref Reference) bool {
	return ref.Kind == "" || ref.Kind == RefKindCall
}

// GetCallGraph builds a call graph from a starting symbol.
func (s *GOBSymbolStore) GetCallGraph(ctx context.Context, symbolName string, depth int) (*CallGraph, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	graph := &CallGraph{
		Root:  symbolName,
		Nodes: make(map[string]Symbol),
		Edges: []CallEdge{},
		Depth: depth,
	}

	// BFS to build graph up to depth
	visited := make(map[string]bool)
	type queueItem struct {
		name  string
		depth int
	}
	queue := []queueItem{{symbolName, 0}}
	edgeSeen := make(map[string]bool)

	shouldTraverse := func(name string, isRoot bool) bool {
		symbols := s.index.Symbols[name]
		if len(symbols) == 0 {
			return false
		}
		// Root is explicitly requested by user and may be ambiguous.
		if isRoot {
			return true
		}
		// Avoid exploding through name-collided symbols (e.g. Load, Init).
		return len(symbols) == 1
	}
	isDeclarationSelfEdge := func(edge CallEdge) bool {
		if edge.Caller != edge.Callee {
			return false
		}
		for _, sym := range s.index.Symbols[edge.Caller] {
			if sym.File == edge.File && sym.Line == edge.Line {
				return true
			}
		}
		return false
	}

	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]

		if visited[current.name] || current.depth > depth {
			continue
		}
		visited[current.name] = true

		// Add node
		if symbols, ok := s.index.Symbols[current.name]; ok && len(symbols) > 0 {
			graph.Nodes[current.name] = symbols[0]
		}

		// Find edges (both callers and callees)
		for _, edge := range s.index.CallGraph {
			if edge.Caller == current.name {
				if isDeclarationSelfEdge(edge) {
					continue
				}
				edgeKey := fmt.Sprintf("%s->%s", edge.Caller, edge.Callee)
				if !edgeSeen[edgeKey] {
					graph.Edges = append(graph.Edges, edge)
					edgeSeen[edgeKey] = true
				}
				if !visited[edge.Callee] && shouldTraverse(edge.Callee, false) {
					queue = append(queue, queueItem{edge.Callee, current.depth + 1})
				}
			}
			if current.depth == 0 && edge.Callee == current.name {
				if isDeclarationSelfEdge(edge) {
					continue
				}
				edgeKey := fmt.Sprintf("%s->%s", edge.Caller, edge.Callee)
				if !edgeSeen[edgeKey] {
					graph.Edges = append(graph.Edges, edge)
					edgeSeen[edgeKey] = true
				}
				// Ensure caller node is present in the graph.
				if _, exists := graph.Nodes[edge.Caller]; !exists {
					if syms, ok := s.index.Symbols[edge.Caller]; ok && len(syms) > 0 {
						graph.Nodes[edge.Caller] = syms[0]
					}
				}
			}
		}
	}

	return graph, nil
}

// GetSymbolsForFile returns all symbols defined in a specific file.
func (s *GOBSymbolStore) GetSymbolsForFile(ctx context.Context, filePath string) ([]Symbol, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var result []Symbol
	for _, symbols := range s.index.Symbols {
		for _, sym := range symbols {
			if sym.File == filePath {
				result = append(result, sym)
			}
		}
	}
	return result, nil
}

// GetCallEdges returns all call graph edges.
func (s *GOBSymbolStore) GetCallEdges(ctx context.Context) ([]CallEdge, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	edges := make([]CallEdge, len(s.index.CallGraph))
	copy(edges, s.index.CallGraph)
	return edges, nil
}

// Close shuts down the store.
func (s *GOBSymbolStore) Close() error {
	return s.Persist(context.Background())
}

// GetStats returns statistics about the symbol index.
func (s *GOBSymbolStore) GetStats(ctx context.Context) (*SymbolStats, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	totalSymbols := 0
	for _, syms := range s.index.Symbols {
		totalSymbols += len(syms)
	}

	totalRefs := 0
	for _, refs := range s.index.References {
		totalRefs += len(refs)
	}

	var size int64
	if info, err := os.Stat(s.indexPath); err == nil {
		size = info.Size()
	}

	return &SymbolStats{
		TotalSymbols:    totalSymbols,
		TotalReferences: totalRefs,
		TotalFiles:      len(s.fileIndex),
		IndexSize:       size,
		LastUpdated:     s.index.UpdatedAt,
	}, nil
}

// IsFileIndexed checks if a file has been indexed.
func (s *GOBSymbolStore) IsFileIndexed(filePath string) bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.fileIndex[filePath]
}

// GetFileContentHash returns the stored content hash for a file when available.
func (s *GOBSymbolStore) GetFileContentHash(filePath string) (string, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	hash, ok := s.fileContentHashes[filePath]
	return hash, ok
}

// RefreshFileMeta updates the recorded (size, mtime) for an already-indexed
// file without touching its symbols, references, or content hash. Callers
// use this when a file was read and its content hash still matched what
// was already stored (so nothing about its symbols changed) but its mtime
// had drifted from what FastSkip last saw -- for example after a git
// checkout, rsync, or bind-mount remount rewrites mtimes repo-wide without
// touching content. Without this, such a file would keep failing FastSkip
// and paying a real (if now-parallelized) read+hash on every subsequent
// restart, even though its content never actually changes again. A no-op
// if the file isn't currently indexed.
func (s *GOBSymbolStore) RefreshFileMeta(filePath string, size, modTime int64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.fileIndex[filePath] {
		return
	}
	s.fileMeta[filePath] = fileMetaRecord{Size: size, ModTime: modTime}
}

// FastSkip reports whether filePath can be skipped without reading it: true
// only when the file is already indexed and its size and modification time
// exactly match what was recorded the last time its content hash was
// computed (see SaveFileWithContentHash). This is a zero-I/O check -- no
// file is opened, nothing is hashed -- so callers can use it as the very
// first gate before anything more expensive, regardless of how large the
// file is or how slow the filesystem is.
//
// This is deliberately independent of any global "last index time"
// watermark: a single event that rewrites every file's mtime (a git
// checkout, an rsync, a bind-mount remount) invalidates a watermark-based
// gate for the entire repository at once, forcing a full re-read of every
// file. Per-file size+mtime, persisted alongside the hash it was computed
// from, survives that: as long as this exact file's size and mtime are
// unchanged from the last time it was actually read, its content can be
// trusted unchanged too, no matter what happened to the rest of the repo or
// to any watermark.
func (s *GOBSymbolStore) FastSkip(filePath string, size, modTime int64) bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if !s.fileIndex[filePath] {
		return false
	}
	meta, ok := s.fileMeta[filePath]
	return ok && meta.Size == size && meta.ModTime == modTime
}
