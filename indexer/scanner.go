package indexer

import (
	"crypto/sha256"
	"encoding/hex"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"unicode/utf8"

	"golang.org/x/sync/errgroup"
)

const (
	maxFileSize = 1 * 1024 * 1024 // 1 MB
)

// MinifiedPatterns lists patterns for minified files to skip by default
var MinifiedPatterns = []string{
	".min.js",
	".min.css",
	".bundle.js",
	".bundle.css",
}

// isMinifiedFile checks if a file is a minified file based on naming patterns
func isMinifiedFile(path string) bool {
	lowerPath := strings.ToLower(path)
	for _, pattern := range MinifiedPatterns {
		if strings.HasSuffix(lowerPath, pattern) {
			return true
		}
	}
	return false
}

// SupportedExtensions lists file extensions to index
var SupportedExtensions = map[string]bool{
	".go":     true,
	".js":     true,
	".ts":     true,
	".jsx":    true,
	".tsx":    true,
	".py":     true,
	".rb":     true,
	".java":   true,
	".c":      true,
	".cpp":    true,
	".cc":     true,
	".h":      true,
	".hpp":    true,
	".cs":     true,
	".php":    true,
	".rs":     true,
	".swift":  true,
	".kt":     true,
	".scala":  true,
	".vue":    true,
	".svelte": true,
	".html":   true,
	".css":    true,
	".scss":   true,
	".less":   true,
	".sql":    true,
	".sh":     true,
	".bash":   true,
	".zsh":    true,
	".yaml":   true,
	".yml":    true,
	".json":   true,
	".xml":    true,
	".md":     true,
	".txt":    true,
	".toml":   true,
	".ini":    true,
	".cfg":    true,
	".conf":   true,
	".env":    true,
	".lua":    true,
	".r":      true,
	".R":      true,
	".dart":   true,
	".ex":     true,
	".exs":    true,
	".erl":    true,
	".clj":    true,
	".hs":     true,
	".ml":     true,
	".fs":     true,
	".elm":    true,
	".nim":    true,
	".zig":    true,
	".proto":  true,
	".tf":     true,
	".hcl":    true,
	".pas":    true, // Pascal source file
	".dpr":    true, // Delphi project file
}

type FileInfo struct {
	Path    string
	Size    int64
	ModTime int64
	Hash    string
	Content string
}

type FileMeta struct {
	Path    string
	Size    int64
	ModTime int64
}

// DefaultScanWorkers is the default number of goroutines used to walk the
// directory tree concurrently (see Scanner.WithScanWorkers). It matches
// config.DefaultScanWorkers; kept as a separate constant here so this
// package doesn't need to import config.
const DefaultScanWorkers = 2

type Scanner struct {
	root    string
	ignore  *IgnoreMatcher
	workers int
}

func NewScanner(root string, ignore *IgnoreMatcher) *Scanner {
	return &Scanner{
		root:    root,
		ignore:  ignore,
		workers: DefaultScanWorkers,
	}
}

// WithScanWorkers overrides the number of concurrent goroutines used to
// walk the directory tree, one per top-level subdirectory of root (see
// walkConcurrent). Non-positive values are ignored, leaving the default in
// place. The default is deliberately small (DefaultScanWorkers) rather than
// scaled to GOMAXPROCS: directory walking is bounded by filesystem
// syscalls (stat/readdir), not CPU, and sharding is by top-level
// subdirectory, so raising this only helps if the tree actually has that
// many top-level subdirectories to spread work across -- a repo with one
// giant subdirectory and many small ones won't benefit much from a higher
// value since one goroutine still ends up walking most of the tree.
func (s *Scanner) WithScanWorkers(workers int) *Scanner {
	if workers > 0 {
		s.workers = workers
	}
	return s
}

// ScanMetadata scans indexable files and returns only file metadata.
// It avoids reading file contents and hash computation for a faster first pass.
func (s *Scanner) ScanMetadata() ([]FileMeta, []string, error) {
	return walkConcurrent(s.root, s.workers, s.ignore, func(path, relPath string, d fs.DirEntry) (*FileMeta, string, error) {
		// Skip ignored files
		if s.ignore.ShouldIgnore(relPath) {
			return nil, "", nil
		}

		// Check extension
		ext := strings.ToLower(filepath.Ext(path))
		if !SupportedExtensions[ext] {
			return nil, "", nil
		}

		// Skip minified files
		if isMinifiedFile(relPath) {
			return nil, relPath + " (minified)", nil
		}

		info, err := d.Info()
		if err != nil {
			return nil, "", nil
		}

		// Skip large files
		if info.Size() > maxFileSize {
			return nil, relPath + " (too large)", nil
		}

		return &FileMeta{
			Path:    relPath,
			Size:    info.Size(),
			ModTime: info.ModTime().Unix(),
		}, "", nil
	})
}

func (s *Scanner) Scan() ([]FileInfo, []string, error) {
	return walkConcurrent(s.root, s.workers, s.ignore, func(path, relPath string, d fs.DirEntry) (*FileInfo, string, error) {
		// Skip ignored files
		if s.ignore.ShouldIgnore(relPath) {
			return nil, "", nil
		}

		// Check extension
		ext := strings.ToLower(filepath.Ext(path))
		if !SupportedExtensions[ext] {
			return nil, "", nil
		}

		// Skip minified files
		if isMinifiedFile(relPath) {
			return nil, relPath + " (minified)", nil
		}

		info, err := d.Info()
		if err != nil {
			return nil, "", nil
		}

		// Skip large files
		if info.Size() > maxFileSize {
			return nil, relPath + " (too large)", nil
		}

		// Read file content
		content, err := os.ReadFile(path)
		if err != nil {
			return nil, "", nil
		}

		// Skip binary files
		if !utf8.Valid(content) || containsNull(content) {
			return nil, "", nil
		}

		// Calculate hash
		hash := sha256.Sum256(content)

		return &FileInfo{
			Path:    relPath,
			Size:    info.Size(),
			ModTime: info.ModTime().Unix(),
			Hash:    hex.EncodeToString(hash[:]),
			Content: string(content),
		}, "", nil
	})
}

// walkConcurrent walks the directory tree rooted at root, distributing
// top-level entries (subdirectories and any files directly under root)
// across up to workers goroutines -- one goroutine walks each top-level
// entry's own subtree to completion independently, so no synchronization
// is needed within a single entry's walk, only when merging each
// goroutine's results into the shared slices at the end.
//
// For every directory encountered (including top-level ones), ignore.
// ShouldSkipDir is consulted first, exactly as the old single-threaded
// filepath.WalkDir callback did, so .grepaiignore negation semantics are
// unchanged. For every file encountered, visitFile is called with its
// absolute path, path relative to root, and fs.DirEntry; it must be safe
// to call concurrently (it should not mutate any shared state -- return
// values are what get collected) since it runs in parallel across
// subtrees. It returns:
//   - result: non-nil if the file should be included in the returned slice
//   - skippedReason: non-empty if the file should instead be recorded in
//     the returned skipped-files list (e.g. "path (minified)")
//   - err: any error is treated as "skip this file", matching the old
//     WalkDir callback's tolerance of individual file errors
//
// Note: unlike top-level subdirectories, one giant subdirectory alongside
// many small ones won't parallelize well under this scheme, since a
// single goroutine walks each top-level entry's entire subtree -- this is
// a simple sharding strategy, not a work-stealing walker.
func walkConcurrent[T any](
	root string,
	workers int,
	ignore *IgnoreMatcher,
	visitFile func(path, relPath string, d fs.DirEntry) (result *T, skippedReason string, err error),
) ([]T, []string, error) {
	if workers < 1 {
		workers = 1
	}

	entries, err := os.ReadDir(root)
	if err != nil {
		return nil, nil, err
	}

	var (
		mu      sync.Mutex
		results []T
		skipped []string
		walkErr error
	)

	collect := func(subResults []T, subSkipped []string, err error) {
		mu.Lock()
		defer mu.Unlock()
		results = append(results, subResults...)
		skipped = append(skipped, subSkipped...)
		if err != nil && walkErr == nil {
			walkErr = err
		}
	}

	// walkEntry walks a single top-level entry (file or directory) of root
	// to completion and returns everything found under it, without
	// touching any shared state directly -- the caller merges results via
	// collect(), so appends to the shared slices only ever happen once per
	// entry, not once per file.
	walkEntry := func(entryPath string) ([]T, []string, error) {
		var localResults []T
		var localSkipped []string
		werr := filepath.WalkDir(entryPath, func(path string, d fs.DirEntry, err error) error {
			if err != nil {
				return nil // Skip files we can't access
			}

			relPath, relErr := filepath.Rel(root, path)
			if relErr != nil {
				return nil
			}

			// Handle directories: use ShouldSkipDir to respect .grepaiignore negations
			if d.IsDir() {
				if ignore.ShouldSkipDir(relPath) {
					return filepath.SkipDir
				}
				return nil // Descend into the directory
			}

			result, skippedReason, ferr := visitFile(path, relPath, d)
			if ferr != nil {
				return nil
			}
			if skippedReason != "" {
				localSkipped = append(localSkipped, skippedReason)
				return nil
			}
			if result != nil {
				localResults = append(localResults, *result)
			}
			return nil
		})
		return localResults, localSkipped, werr
	}

	g := new(errgroup.Group)
	g.SetLimit(workers)

	for _, entry := range entries {
		entry := entry

		if entry.IsDir() {
			if ignore.ShouldSkipDir(entry.Name()) {
				continue
			}
		}

		entryPath := filepath.Join(root, entry.Name())
		g.Go(func() error {
			sub, subSkipped, werr := walkEntry(entryPath)
			collect(sub, subSkipped, werr)
			return nil // never fail the group; per-entry errors are merged into walkErr instead
		})
	}

	_ = g.Wait()

	return results, skipped, walkErr
}

func (s *Scanner) ScanFile(relPath string) (*FileInfo, error) {
	absPath := filepath.Join(s.root, relPath)

	// Skip minified files
	if isMinifiedFile(relPath) {
		return nil, nil
	}

	info, err := os.Stat(absPath)
	if err != nil {
		return nil, err
	}

	if info.Size() > maxFileSize {
		return nil, nil // Skip large files
	}

	content, err := os.ReadFile(absPath)
	if err != nil {
		return nil, err
	}

	if !utf8.Valid(content) || containsNull(content) {
		return nil, nil // Skip binary files
	}

	hash := sha256.Sum256(content)

	return &FileInfo{
		Path:    relPath,
		Size:    info.Size(),
		ModTime: info.ModTime().Unix(),
		Hash:    hex.EncodeToString(hash[:]),
		Content: string(content),
	}, nil
}

func containsNull(data []byte) bool {
	for _, b := range data {
		if b == 0 {
			return true
		}
	}
	return false
}

func HashFile(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()

	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}

	return hex.EncodeToString(h.Sum(nil)), nil
}
