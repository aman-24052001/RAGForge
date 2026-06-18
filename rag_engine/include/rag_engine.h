#pragma once
#include "chunker.h"
#include "embedder.h"
#include "index.h"
#include "mmr.h"
#include <string>
#include <vector>
#include <memory>

struct QueryResult {
    std::string chunk_text;
    std::string doc_id;
    int chunk_index;
    float relevance_score;   // hybrid cosine+BM25
    float diversity_rank;    // MMR rank (0 = most relevant+diverse)
};

struct IndexStats {
    size_t total_chunks;
    std::vector<std::string> doc_ids;
};

class RAGEngine {
public:
    RAGEngine(const std::string& model_path,
              int chunk_size = 512,
              int chunk_overlap = 64,
              float mmr_lambda = 0.6f);

    // Ingest raw text with a doc identifier
    int ingest(const std::string& text, const std::string& doc_id);

    // Query: returns top_n diverse, relevant chunks
    std::vector<QueryResult> query(const std::string& query_text,
                                   int top_k = 20,
                                   int top_n = 5);

    void save_index(const std::string& path);
    void load_index(const std::string& path);
    IndexStats stats() const;

private:
    std::unique_ptr<Chunker>    chunker_;
    std::unique_ptr<Embedder>   embedder_;
    std::unique_ptr<VectorIndex> index_;
    std::unique_ptr<MMRReranker> mmr_;
    std::vector<std::string>    doc_ids_;
};
