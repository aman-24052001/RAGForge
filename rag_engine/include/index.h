#pragma once
#include "chunker.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

using Embedding = std::vector<float>;

struct IndexedChunk {
    Chunk chunk;
    Embedding embedding;
    size_t hnsw_id;
};

struct SearchResult {
    IndexedChunk item;
    float score;        // cosine similarity
    float bm25_score;   // BM25 lexical score
    float final_score;  // hybrid weighted
};

class VectorIndex {
public:
    VectorIndex(int dim, int M = 16, int ef_construction = 200);
    ~VectorIndex();

    void add(const IndexedChunk& ic);
    std::vector<SearchResult> search(const Embedding& query_emb,
                                      const std::string& query_text,
                                      int top_k = 20) const;
    void save(const std::string& path) const;
    void load(const std::string& path);
    size_t size() const { return id_to_chunk_.size(); }
    void clear();

private:
    struct HNSWState;
    std::unique_ptr<HNSWState> hnsw_;
    int dim_;

    std::unordered_map<size_t, IndexedChunk> id_to_chunk_;
    std::unordered_map<std::string, std::vector<size_t>> inverted_index_;
    std::unordered_map<size_t, std::unordered_map<std::string, int>> term_freq_;
    std::unordered_map<std::string, int> doc_freq_;
    double avg_dl_ = 0.0;
    size_t next_id_ = 0;

    float cosine(const Embedding& a, const Embedding& b) const;
    float bm25(const std::string& query, size_t chunk_id,
               float k1 = 1.5f, float b = 0.75f) const;
    std::vector<std::string> tokenize(const std::string& q) const;
    void index_chunk(size_t id, const std::string& text);
    void rebuild_hnsw();
};
