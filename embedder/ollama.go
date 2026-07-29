package embedder

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/sync/errgroup"
)

const (
	defaultOllamaEndpoint = "http://localhost:11434"
	defaultOllamaModel    = "nomic-embed-text"
	nomicEmbedDimensions  = 768

	// defaultOllamaParallelism is how many /api/embed requests may be in
	// flight at once by default. Ollama has no rate limiter of its own, so
	// unlike OpenAI this is a fixed value rather than an adaptive one --
	// callers with GPU headroom to spare should raise it via
	// WithOllamaParallelism / the embedder.parallelism config field.
	defaultOllamaParallelism = 4

	// defaultOllamaBatchSize is the number of texts bundled into a single
	// /api/embed request. Ollama's batch endpoint embeds every input in one
	// call, which is what lets the underlying model actually batch work on
	// the GPU instead of processing one prompt at a time. Combined with
	// parallelism (multiple such requests in flight concurrently), this is
	// what lets grepai saturate a GPU during indexing.
	defaultOllamaBatchSize = 32

	// defaultOllamaPerTextTimeout is added to a request's timeout budget for
	// every text included in that /api/embed call. The old implementation
	// used one fixed http.Client timeout (120s) shared by every request
	// regardless of size; that was fine when each call embedded a single
	// text (same cost as a one-off curl), but once batching sends
	// defaultOllamaBatchSize texts per request, a CPU-only Ollama install
	// -- which typically processes a batch close to linearly rather than
	// getting GPU-style batch speedup -- can easily blow past a fixed 120s
	// budget for a full batch even though a single-text request completes
	// almost instantly. Scaling the timeout with the number of texts in the
	// specific request keeps small requests fast-failing while giving large
	// batches a proportionally larger budget.
	defaultOllamaPerTextTimeout = 10 * time.Second

	// defaultOllamaMinTimeout is the floor applied to every request
	// regardless of how many texts it contains, covering fixed overhead
	// (connection setup, the model being loaded into memory on first use,
	// etc.) that doesn't scale with batch size.
	defaultOllamaMinTimeout = 60 * time.Second
)

type OllamaEmbedder struct {
	endpoint    string
	model       string
	dimensions  int
	parallelism int
	batchSize   int
	client      *http.Client

	// requestTimeout, if set (>0) via WithOllamaTimeout, overrides the
	// default per-batch-size timeout scaling below with one fixed value
	// applied to every request regardless of size.
	requestTimeout time.Duration
	// perTextTimeout and minTimeout implement the default scaling
	// behavior: timeoutFor(n) == minTimeout + n*perTextTimeout. See
	// defaultOllamaPerTextTimeout / defaultOllamaMinTimeout for why this
	// scaling exists.
	perTextTimeout time.Duration
	minTimeout     time.Duration
}

// ollamaEmbedRequest targets Ollama's legacy /api/embeddings endpoint, which
// only accepts a single prompt per call. Kept only for Embed(), which embeds
// one string at a time; bulk paths use ollamaEmbedBatchRequest instead.
type ollamaEmbedRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
}

type ollamaEmbedResponse struct {
	Embedding []float32 `json:"embedding"`
}

// ollamaEmbedBatchRequest targets Ollama's newer /api/embed endpoint, which
// accepts multiple inputs in a single request and embeds them together --
// letting the model batch on GPU -- instead of requiring one HTTP round trip
// per text. Supported since Ollama v0.1.35+.
type ollamaEmbedBatchRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

type ollamaEmbedBatchResponse struct {
	Embeddings [][]float32 `json:"embeddings"`
}

type OllamaOption func(*OllamaEmbedder)

func WithOllamaEndpoint(endpoint string) OllamaOption {
	return func(e *OllamaEmbedder) {
		e.endpoint = endpoint
	}
}

func WithOllamaModel(model string) OllamaOption {
	return func(e *OllamaEmbedder) {
		e.model = model
	}
}
func WithOllamaDimensions(dimensions int) OllamaOption {
	return func(e *OllamaEmbedder) {
		e.dimensions = dimensions
	}
}

// WithOllamaParallelism sets how many /api/embed requests may be in flight
// at once. Each request already carries a batch of texts (see
// WithOllamaBatchSize); this controls how many such requests run
// concurrently, which is what lets grepai actually keep a GPU busy instead
// of sending one request at a time and waiting for it before sending the
// next.
func WithOllamaParallelism(parallelism int) OllamaOption {
	return func(e *OllamaEmbedder) {
		if parallelism > 0 {
			e.parallelism = parallelism
		}
	}
}

// WithOllamaBatchSize sets the maximum number of texts bundled into a
// single /api/embed request. Larger batches mean fewer HTTP round trips and
// let Ollama pack more work into each call to the model. If a request
// fails because the combined input is too large for the model's context,
// lower this value.
func WithOllamaBatchSize(batchSize int) OllamaOption {
	return func(e *OllamaEmbedder) {
		if batchSize > 0 {
			e.batchSize = batchSize
		}
	}
}

// WithOllamaTimeout overrides the HTTP request timeout with one fixed
// value applied to every request regardless of how many texts it contains,
// replacing the default behavior of scaling the timeout with batch size
// (see defaultOllamaPerTextTimeout / defaultOllamaMinTimeout). Use this if
// the default scaling still isn't generous enough for a particularly slow
// CPU-only setup, or to pin a tighter bound on a fast one.
func WithOllamaTimeout(timeout time.Duration) OllamaOption {
	return func(e *OllamaEmbedder) {
		if timeout > 0 {
			e.requestTimeout = timeout
		}
	}
}

func NewOllamaEmbedder(opts ...OllamaOption) *OllamaEmbedder {
	e := &OllamaEmbedder{
		endpoint:       defaultOllamaEndpoint,
		model:          defaultOllamaModel,
		dimensions:     nomicEmbedDimensions,
		parallelism:    defaultOllamaParallelism,
		batchSize:      defaultOllamaBatchSize,
		perTextTimeout: defaultOllamaPerTextTimeout,
		minTimeout:     defaultOllamaMinTimeout,
		// No client-level Timeout: per-request deadlines are applied via
		// context in timeoutFor/Embed/embedRequest instead, so the budget
		// can scale with how many texts a given request actually carries
		// rather than using one fixed value for every request size.
		client: &http.Client{},
	}

	for _, opt := range opts {
		opt(e)
	}

	return e
}

// timeoutFor returns the HTTP request timeout budget for a request
// embedding n texts. If requestTimeout was set via WithOllamaTimeout, that
// fixed value is used for every request regardless of n. Otherwise the
// timeout scales with n: minTimeout + n*perTextTimeout.
func (e *OllamaEmbedder) timeoutFor(n int) time.Duration {
	if e.requestTimeout > 0 {
		return e.requestTimeout
	}
	return e.minTimeout + time.Duration(n)*e.perTextTimeout
}

func (e *OllamaEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	reqBody := ollamaEmbedRequest{
		Model:  e.model,
		Prompt: text,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	reqCtx, cancel := context.WithTimeout(ctx, e.timeoutFor(1))
	defer cancel()

	url := fmt.Sprintf("%s/api/embeddings", e.endpoint)
	req, err := http.NewRequestWithContext(reqCtx, http.MethodPost, url, bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request to Ollama: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		bodyStr := string(body)

		// Check for context length error (Ollama returns 500 with specific message)
		if resp.StatusCode == http.StatusInternalServerError &&
			strings.Contains(bodyStr, "exceeds the context length") {
			estimatedTokens := len(text) / 4 // rough estimate: 4 chars per token
			return nil, NewContextLengthError(0, estimatedTokens, 0, bodyStr)
		}

		return nil, fmt.Errorf("Ollama returned status %d: %s", resp.StatusCode, bodyStr)
	}

	var result ollamaEmbedResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	if len(result.Embedding) == 0 {
		return nil, fmt.Errorf("Ollama returned empty embedding")
	}

	return result.Embedding, nil
}

// embedRequest sends a single /api/embed call for up to e.batchSize texts.
// Callers (EmbedBatch, EmbedBatches) are responsible for splitting larger
// inputs into chunks of that size before calling this.
func (e *OllamaEmbedder) embedRequest(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	reqBody := ollamaEmbedBatchRequest{
		Model: e.model,
		Input: texts,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	reqCtx, cancel := context.WithTimeout(ctx, e.timeoutFor(len(texts)))
	defer cancel()

	url := fmt.Sprintf("%s/api/embed", e.endpoint)
	req, err := http.NewRequestWithContext(reqCtx, http.MethodPost, url, bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request to Ollama: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		bodyStr := string(body)

		// Check for context length error (Ollama returns 500 with specific message)
		if resp.StatusCode == http.StatusInternalServerError &&
			strings.Contains(bodyStr, "exceeds the context length") {
			totalChars := 0
			for _, t := range texts {
				totalChars += len(t)
			}
			estimatedTokens := totalChars / 4 // rough estimate: 4 chars per token
			return nil, NewContextLengthError(0, estimatedTokens, 0, bodyStr)
		}

		return nil, fmt.Errorf("Ollama returned status %d: %s", resp.StatusCode, bodyStr)
	}

	var result ollamaEmbedBatchResponse
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	if len(result.Embeddings) != len(texts) {
		return nil, fmt.Errorf("expected %d embeddings, got %d", len(texts), len(result.Embeddings))
	}

	return result.Embeddings, nil
}

// EmbedBatch embeds all texts, splitting them into chunks of at most
// e.batchSize and sending those chunks to Ollama concurrently (bounded by
// e.parallelism). This replaces the old behavior of issuing one
// /api/embeddings call per text, sequentially, which is what kept GPU
// utilization pinned near-idle regardless of any parallelism setting.
func (e *OllamaEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	type chunk struct {
		start int
		texts []string
	}

	var chunks []chunk
	for start := 0; start < len(texts); start += e.batchSize {
		end := start + e.batchSize
		if end > len(texts) {
			end = len(texts)
		}
		chunks = append(chunks, chunk{start: start, texts: texts[start:end]})
	}

	results := make([][]float32, len(texts))
	g, gctx := errgroup.WithContext(ctx)
	g.SetLimit(e.parallelism)

	for _, c := range chunks {
		c := c
		g.Go(func() error {
			embeddings, err := e.embedRequest(gctx, c.texts)
			if err != nil {
				// Wrap ContextLengthError with correct chunk index
				if ctxErr := AsContextLengthError(err); ctxErr != nil {
					ctxErr.ChunkIndex = c.start
					return ctxErr
				}
				return fmt.Errorf("failed to embed texts %d-%d: %w", c.start, c.start+len(c.texts)-1, err)
			}
			copy(results[c.start:c.start+len(embeddings)], embeddings)
			return nil
		})
	}

	if err := g.Wait(); err != nil {
		return nil, err
	}

	return results, nil
}

func (e *OllamaEmbedder) Dimensions() int {
	return e.dimensions
}

func (e *OllamaEmbedder) Close() error {
	return nil
}

// Ping checks if Ollama is reachable
func (e *OllamaEmbedder) Ping(ctx context.Context) error {
	url := fmt.Sprintf("%s/api/tags", e.endpoint)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	resp, err := e.client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to reach Ollama at %s: %w", e.endpoint, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("Ollama returned status %d", resp.StatusCode)
	}

	return nil
}

// EmbedBatches implements the BatchEmbedder interface, which lets the
// indexer drive Ollama through its incremental-save path (see
// indexer.indexFilesBatched) the same way it drives OpenAI: as soon as a
// logical batch's embeddings are ready, onBatchDone fires so the indexer can
// persist the corresponding files without waiting for the rest of the run.
//
// Every batch's texts are flattened into batchSize-sized sub-requests up
// front, and a single worker pool (bounded by e.parallelism) drives ALL of
// them for the whole call -- regardless of how many logical batches the
// indexer formed. Without this flattening, bounding concurrency separately
// at both the per-batch level and the per-request level would multiply
// (parallelism x parallelism concurrent requests instead of parallelism).
func (e *OllamaEmbedder) EmbedBatches(ctx context.Context, batches []Batch, progress BatchProgress, onBatchDone BatchResultCallback) ([]BatchResult, error) {
	if len(batches) == 0 {
		return nil, nil
	}

	type subChunk struct {
		batchIdx int
		start    int
		texts    []string
	}

	var subChunks []subChunk
	batchEmbeddings := make([][][]float32, len(batches))
	batchRemaining := make([]int32, len(batches))
	totalChunks := 0

	for bi, batch := range batches {
		totalChunks += batch.Size()
		contents := batch.Contents()
		batchEmbeddings[bi] = make([][]float32, len(contents))
		if len(contents) == 0 {
			continue
		}
		var n int32
		for start := 0; start < len(contents); start += e.batchSize {
			end := start + e.batchSize
			if end > len(contents) {
				end = len(contents)
			}
			subChunks = append(subChunks, subChunk{batchIdx: bi, start: start, texts: contents[start:end]})
			n++
		}
		batchRemaining[bi] = n
	}

	results := make([]BatchResult, len(batches))
	var resultsMu sync.Mutex
	var completedChunks atomic.Int64

	g, gctx := errgroup.WithContext(ctx)
	g.SetLimit(e.parallelism)

	for _, sc := range subChunks {
		sc := sc
		g.Go(func() error {
			embeddings, err := e.embedRequest(gctx, sc.texts)
			if err != nil {
				if ctxErr := AsContextLengthError(err); ctxErr != nil {
					ctxErr.ChunkIndex = sc.start
					return ctxErr
				}
				return fmt.Errorf("batch %d texts %d-%d: %w", sc.batchIdx, sc.start, sc.start+len(sc.texts)-1, err)
			}

			copy(batchEmbeddings[sc.batchIdx][sc.start:sc.start+len(embeddings)], embeddings)

			newCompleted := completedChunks.Add(int64(len(embeddings)))
			if progress != nil {
				progress(sc.batchIdx, len(batches), int(newCompleted), totalChunks, false, 0, 0)
			}

			if atomic.AddInt32(&batchRemaining[sc.batchIdx], -1) == 0 {
				result := BatchResult{
					BatchIndex: sc.batchIdx,
					Embeddings: batchEmbeddings[sc.batchIdx],
				}
				resultsMu.Lock()
				results[sc.batchIdx] = result
				resultsMu.Unlock()
				if onBatchDone != nil {
					onBatchDone(result)
				}
			}
			return nil
		})
	}

	err := g.Wait()
	return results, err
}
