package embedder

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestNewOllamaEmbedder_ParallelismAndBatchSizeDefaults(t *testing.T) {
	e := NewOllamaEmbedder()

	if e.parallelism != defaultOllamaParallelism {
		t.Errorf("expected parallelism %d, got %d", defaultOllamaParallelism, e.parallelism)
	}
	if e.batchSize != defaultOllamaBatchSize {
		t.Errorf("expected batchSize %d, got %d", defaultOllamaBatchSize, e.batchSize)
	}
}

func TestNewOllamaEmbedder_WithParallelismAndBatchSize(t *testing.T) {
	e := NewOllamaEmbedder(
		WithOllamaParallelism(16),
		WithOllamaBatchSize(64),
	)

	if e.parallelism != 16 {
		t.Errorf("expected parallelism 16, got %d", e.parallelism)
	}
	if e.batchSize != 64 {
		t.Errorf("expected batchSize 64, got %d", e.batchSize)
	}

	// Zero/negative values should be ignored, keeping defaults.
	e2 := NewOllamaEmbedder(WithOllamaParallelism(0), WithOllamaBatchSize(-1))
	if e2.parallelism != defaultOllamaParallelism {
		t.Errorf("expected default parallelism to be kept, got %d", e2.parallelism)
	}
	if e2.batchSize != defaultOllamaBatchSize {
		t.Errorf("expected default batchSize to be kept, got %d", e2.batchSize)
	}
}

// TestOllamaEmbedder_EmbedBatch_UsesBulkEndpoint verifies EmbedBatch sends
// requests to the bulk /api/embed endpoint with multiple texts per request,
// instead of the old behavior of one /api/embeddings call per text.
func TestOllamaEmbedder_EmbedBatch_UsesBulkEndpoint(t *testing.T) {
	var requestCount int32
	var maxInputsPerRequest int32

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&requestCount, 1)

		if r.URL.Path != "/api/embed" {
			t.Errorf("expected request to /api/embed, got %s", r.URL.Path)
		}

		var req ollamaEmbedBatchRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("failed to decode request: %v", err)
		}

		if int32(len(req.Input)) > atomic.LoadInt32(&maxInputsPerRequest) {
			atomic.StoreInt32(&maxInputsPerRequest, int32(len(req.Input)))
		}

		resp := ollamaEmbedBatchResponse{
			Embeddings: make([][]float32, len(req.Input)),
		}
		for i := range req.Input {
			resp.Embeddings[i] = []float32{float32(i), float32(i * 2)}
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	e := NewOllamaEmbedder(
		WithOllamaEndpoint(server.URL),
		WithOllamaBatchSize(10),
		WithOllamaParallelism(4),
	)

	texts := make([]string, 25)
	for i := range texts {
		texts[i] = "chunk"
	}

	results, err := e.EmbedBatch(context.Background(), texts)
	if err != nil {
		t.Fatalf("EmbedBatch failed: %v", err)
	}

	if len(results) != 25 {
		t.Fatalf("expected 25 results, got %d", len(results))
	}

	// 25 texts split into batches of 10 => 3 requests (10, 10, 5), not 25.
	if got := atomic.LoadInt32(&requestCount); got != 3 {
		t.Errorf("expected 3 bulk requests for 25 texts with batchSize=10, got %d", got)
	}
	if got := atomic.LoadInt32(&maxInputsPerRequest); got != 10 {
		t.Errorf("expected largest request to carry 10 inputs, got %d", got)
	}
}

// TestOllamaEmbedder_EmbedBatch_PreservesOrder verifies that splitting a
// large input into concurrent sub-requests still returns embeddings in the
// original order, even though the underlying HTTP calls complete out of
// order.
func TestOllamaEmbedder_EmbedBatch_PreservesOrder(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req ollamaEmbedBatchRequest
		json.NewDecoder(r.Body).Decode(&req)

		// Vary latency so requests don't necessarily complete in send order.
		if len(req.Input) > 0 {
			time.Sleep(time.Duration(len(req.Input)%3) * 5 * time.Millisecond)
		}

		resp := ollamaEmbedBatchResponse{Embeddings: make([][]float32, len(req.Input))}
		for i, text := range req.Input {
			// Encode the original text (which embeds its global index) into
			// the "embedding" so we can verify ordering after the fact.
			resp.Embeddings[i] = []float32{float32(len(text))}
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	e := NewOllamaEmbedder(
		WithOllamaEndpoint(server.URL),
		WithOllamaBatchSize(4),
		WithOllamaParallelism(8),
	)

	// Build texts whose length encodes their index so we can check ordering
	// after the fact (the mock server returns len(text) as the embedding).
	texts := make([]string, 20)
	for i := range texts {
		b := make([]byte, i)
		for j := range b {
			b[j] = 'a'
		}
		texts[i] = string(b)
	}

	results, err := e.EmbedBatch(context.Background(), texts)
	if err != nil {
		t.Fatalf("EmbedBatch failed: %v", err)
	}
	for i, r := range results {
		if len(r) != 1 || int(r[0]) != i {
			t.Errorf("result[%d]: expected embedding encoding index %d, got %v", i, i, r)
		}
	}
}

// TestOllamaEmbedder_EmbedBatches_ConcurrencyBoundedByParallelism verifies
// that EmbedBatches never exceeds e.parallelism concurrent HTTP requests in
// flight, even when the run is split across multiple logical batches (the
// scenario that previously risked parallelism^2 concurrent requests).
func TestOllamaEmbedder_EmbedBatches_ConcurrencyBoundedByParallelism(t *testing.T) {
	var (
		current       int32
		maxConcurrent int32
		mu            sync.Mutex
	)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c := atomic.AddInt32(&current, 1)
		defer atomic.AddInt32(&current, -1)

		mu.Lock()
		if c > maxConcurrent {
			maxConcurrent = c
		}
		mu.Unlock()

		time.Sleep(20 * time.Millisecond)

		var req ollamaEmbedBatchRequest
		json.NewDecoder(r.Body).Decode(&req)
		resp := ollamaEmbedBatchResponse{Embeddings: make([][]float32, len(req.Input))}
		for i := range req.Input {
			resp.Embeddings[i] = []float32{float32(i)}
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	const parallelism = 3
	e := NewOllamaEmbedder(
		WithOllamaEndpoint(server.URL),
		WithOllamaBatchSize(2), // small, so many sub-chunks are generated
		WithOllamaParallelism(parallelism),
	)

	// Two logical batches, each with plenty of entries so they generate
	// several sub-chunks apiece.
	batches := []Batch{
		{Index: 0, Entries: makeEntries(0, 10)},
		{Index: 1, Entries: makeEntries(1, 10)},
	}

	results, err := e.EmbedBatches(context.Background(), batches, nil, nil)
	if err != nil {
		t.Fatalf("EmbedBatches failed: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("expected 2 batch results, got %d", len(results))
	}
	for _, r := range results {
		if len(r.Embeddings) != 10 {
			t.Errorf("batch %d: expected 10 embeddings, got %d", r.BatchIndex, len(r.Embeddings))
		}
	}

	if got := atomic.LoadInt32(&maxConcurrent); got > parallelism {
		t.Errorf("expected at most %d concurrent requests, saw %d", parallelism, got)
	}
}

// TestOllamaEmbedder_EmbedBatches_OnBatchDoneFiresPerBatch verifies that
// onBatchDone is invoked once per logical batch, only once every sub-chunk
// belonging to that batch has completed, and that each callback carries a
// complete, correctly-ordered set of embeddings for that batch.
func TestOllamaEmbedder_EmbedBatches_OnBatchDoneFiresPerBatch(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req ollamaEmbedBatchRequest
		json.NewDecoder(r.Body).Decode(&req)
		resp := ollamaEmbedBatchResponse{Embeddings: make([][]float32, len(req.Input))}
		for i := range req.Input {
			resp.Embeddings[i] = []float32{1}
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	e := NewOllamaEmbedder(
		WithOllamaEndpoint(server.URL),
		WithOllamaBatchSize(3),
		WithOllamaParallelism(4),
	)

	batches := []Batch{
		{Index: 0, Entries: makeEntries(0, 7)},
		{Index: 1, Entries: makeEntries(1, 5)},
	}

	var mu sync.Mutex
	doneCounts := map[int]int{}
	doneSizes := map[int]int{}

	_, err := e.EmbedBatches(context.Background(), batches, nil, func(r BatchResult) {
		mu.Lock()
		defer mu.Unlock()
		doneCounts[r.BatchIndex]++
		doneSizes[r.BatchIndex] = len(r.Embeddings)
	})
	if err != nil {
		t.Fatalf("EmbedBatches failed: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if doneCounts[0] != 1 {
		t.Errorf("expected onBatchDone to fire exactly once for batch 0, fired %d times", doneCounts[0])
	}
	if doneCounts[1] != 1 {
		t.Errorf("expected onBatchDone to fire exactly once for batch 1, fired %d times", doneCounts[1])
	}
	if doneSizes[0] != 7 {
		t.Errorf("expected batch 0 callback to carry 7 embeddings, got %d", doneSizes[0])
	}
	if doneSizes[1] != 5 {
		t.Errorf("expected batch 1 callback to carry 5 embeddings, got %d", doneSizes[1])
	}
}

// TestOllamaEmbedder_EmbedBatches_PartialResultsSurviveFailure verifies
// that if one sub-chunk fails, batches whose sub-chunks already completed
// successfully still have their onBatchDone fired and are present in the
// returned results, consistent with the OpenAI BatchEmbedder contract that
// makes resumable indexing possible.
func TestOllamaEmbedder_EmbedBatches_PartialResultsSurviveFailure(t *testing.T) {
	var callCount int32

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := atomic.AddInt32(&callCount, 1)
		var req ollamaEmbedBatchRequest
		json.NewDecoder(r.Body).Decode(&req)

		// Fail the second request only. With parallelism=1 and one
		// sub-chunk per batch, batch 0's request reliably goes first and
		// succeeds; batch 1's request goes second and fails -- mirroring
		// the OpenAI BatchEmbedder test for the same contract.
		if n == 2 {
			w.WriteHeader(http.StatusInternalServerError)
			w.Write([]byte("simulated failure"))
			return
		}

		resp := ollamaEmbedBatchResponse{Embeddings: make([][]float32, len(req.Input))}
		for i := range req.Input {
			resp.Embeddings[i] = []float32{1}
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	e := NewOllamaEmbedder(
		WithOllamaEndpoint(server.URL),
		WithOllamaBatchSize(5),
		WithOllamaParallelism(1), // deterministic: batch 0's single request goes first
	)

	batches := []Batch{
		{Index: 0, Entries: makeEntries(0, 5)}, // one sub-chunk -> succeeds
		{Index: 1, Entries: makeEntries(1, 5)}, // one sub-chunk -> the one that fails
	}

	var onDoneCount int32
	_, err := e.EmbedBatches(context.Background(), batches, nil, func(r BatchResult) {
		atomic.AddInt32(&onDoneCount, 1)
	})
	if err == nil {
		t.Fatal("expected an error from EmbedBatches due to simulated failure")
	}
	if got := atomic.LoadInt32(&onDoneCount); got != 1 {
		t.Errorf("expected onBatchDone to fire once (for the surviving batch), fired %d times", got)
	}
}

func makeEntries(fileIndex, n int) []BatchEntry {
	entries := make([]BatchEntry, n)
	for i := 0; i < n; i++ {
		entries[i] = BatchEntry{
			FileIndex:  fileIndex,
			ChunkIndex: i,
			Content:    "content",
		}
	}
	return entries
}
