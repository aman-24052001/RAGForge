#pragma once
#include "chunker.h"
#include "index.h"
#include "mmr.h"
#include <string>
#include <vector>
#include <memory>

// NOTE: Embeddings are injected from Python (sentence-transformers).
// C++ handles: chunking, HNSW indexing, BM25, hybrid scoring, MMR.

struct QueryResult {
    std::string chunk_text;
    std::string doc_id;
    int chunk_index;
    float relevance_score;
    float diversity_rank;
};

struct IndexStats {
    size_t total_chunks;
    std::vector<std::string> doc_ids;
};

class RAGEngineCore {
public:
    RAGEngineCore(int dim,
                  int chunk_size    = 512,
                  int chunk_overlap = 64,
                  float mmr_lambda  = 0.6f);

    // Add pre-embedded chunks (embeddings computed in Python)
    void add_chunks(const std::vector<std::string>& texts,
                    const std::vector<std::string>& doc_ids,
                    const std::vector<int>& chunk_indices,
                    const std::vector<Embedding>& embeddings);

    // Query with a pre-computed query embedding
    std::vector<QueryResult> query(const Embedding& query_emb,
                                   const std::string& query_text,
                                   int top_k = 20,
                                   int top_n = 5);

    // Chunk raw text → returns chunk texts (Python embeds them, then calls add_chunks)
    std::vector<std::string> chunk_text(const std::string& text, const std::string& doc_id);

    void save(const std::string& path);
    void load(const std::string& path);
    void clear();
    IndexStats stats() const;

private:
    std::unique_ptr<Chunker>     chunker_;
    std::unique_ptr<VectorIndex> index_;
    std::unique_ptr<MMRReranker> mmr_;
    std::vector<std::string>     doc_ids_;
    int dim_;
};
