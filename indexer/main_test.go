package indexer

import (
	"io"
	"log"
	"os"
	"testing"
)

// TestMain discards the standard logger's output for the whole package's
// test run. IndexAllWithBatchProgress and friends now emit a fair amount of
// informational log.Printf output (scan/decide summaries, per-wave file and
// chunk counts, cache hits, etc.) so operators running `grepai watch` can
// see what's happening -- but under `go test -v`, stderr is captured
// through a pipe, and writes to a piped stderr can be considerably slower
// and less predictable than a real terminal. Timing-sensitive tests (e.g.
// TestIndexAllWithBatchProgress_EmbeddingStartsBeforeScanCompletes, which
// asserts an ordering between a decide-phase completion timestamp and a
// wave's first embed-dispatch timestamp) were observed to flake under
// -count repeats specifically because of this added, variable log-write
// latency sitting in the same hot path being timed. Discarding log output
// here removes that variable without weakening what the tests actually
// assert.
func TestMain(m *testing.M) {
	log.SetOutput(io.Discard)
	os.Exit(m.Run())
}
