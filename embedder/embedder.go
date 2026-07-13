package embedder

import "context"

// Embedder defines the interface for text embedding providers
type Embedder interface {
	// Embed converts text into a vector embedding
	Embed(ctx context.Context, text string) ([]float32, error)

	// EmbedBatch converts multiple texts into vector embeddings
	EmbedBatch(ctx context.Context, texts []string) ([][]float32, error)

	// Dimensions returns the vector dimension size for this embedder
	Dimensions() int

	// Close cleanly shuts down the embedder
	Close() error
}

// BatchProgress is a callback for reporting batch embedding progress.
// It receives the batch index, total batches, chunk progress info, and optional retry information.
// completedChunks and totalChunks track overall progress across all batches.
// statusCode is the HTTP status code when retrying (429 = rate limited, 5xx = server error).
type BatchProgress func(batchIndex, totalBatches, completedChunks, totalChunks int, retrying bool, attempt int, statusCode int)

// BatchResultCallback is invoked once per batch, as soon as that batch's
// embeddings are ready -- before EmbedBatches has finished processing every
// other batch in the run. Callers use this to persist each file's chunks as
// soon as all of that file's chunks have embeddings, rather than waiting
// for the entire (potentially very large) run to finish before saving
// anything. This is what lets an indexing run be safely stopped and resumed:
// without it, a crash partway through a large batch-embedded run would lose
// every file that hadn't been saved yet, even ones whose embeddings had
// already come back successfully.
//
// The callback may be invoked concurrently from multiple goroutines (one per
// in-flight batch) and must be safe for concurrent use.
type BatchResultCallback func(BatchResult)

// BatchEmbedder extends Embedder with cross-file batch embedding capabilities.
// Providers that support advanced batching (like OpenAI) implement this interface
// to enable parallel processing of multiple batches.
type BatchEmbedder interface {
	Embedder

	// EmbedBatches processes multiple batches of chunks concurrently.
	// It returns results mapped back to their source files. If any batch
	// fails, it returns an error alongside whatever results *did* complete
	// successfully -- callers should not assume a non-nil error means no
	// work was done, since onBatchDone (if provided) will already have been
	// called for every batch that succeeded before the failure.
	// The progress callback is called for each batch completion or retry attempt.
	// onBatchDone, if non-nil, is called synchronously for each batch as it
	// completes successfully, before EmbedBatches returns.
	EmbedBatches(ctx context.Context, batches []Batch, progress BatchProgress, onBatchDone BatchResultCallback) ([]BatchResult, error)
}
