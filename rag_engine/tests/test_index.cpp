#include "../include/rag_engine_core.h"
#include <cassert>
#include <iostream>
#include <random>

std::vector<float> rand_emb(int dim, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> e(dim);
    float norm = 0.0f;
    for (auto& v : e) { v = dist(rng); norm += v*v; }
    norm = std::sqrt(norm);
    for (auto& v : e) v /= norm;
    return e;
}

int main() {
    const int DIM = 32;
    RAGEngineCore engine(DIM, 50, 10, 0.6f);

    // Chunk
    std::string text =
        "HNSW is a graph-based ANN algorithm. "
        "BM25 is a lexical ranking function. "
        "MMR ensures diversity in results. "
        "Vector search uses embeddings. "
        "Hybrid retrieval combines both approaches.";

    auto chunks = engine.chunk_text(text, "doc_test");
    assert(!chunks.empty());
    std::cout << "Chunks: " << chunks.size() << "\n";

    // Add with random embeddings
    std::vector<std::string> doc_ids(chunks.size(), "doc_test");
    std::vector<int> indices;
    std::vector<std::vector<float>> embs;
    for (int i = 0; i < (int)chunks.size(); ++i) {
        indices.push_back(i);
        embs.push_back(rand_emb(DIM, i + 42));
    }
    engine.add_chunks(chunks, doc_ids, indices, embs);

    auto stats = engine.stats();
    assert(stats.total_chunks == chunks.size());
    std::cout << "Indexed: " << stats.total_chunks << " chunks\n";

    // Query
    auto q_emb = rand_emb(DIM, 999);
    auto results = engine.query(q_emb, "what is HNSW?", 10, 3);
    assert(!results.empty());
    for (const auto& r : results) {
        std::cout << "  [" << r.doc_id << " c" << r.chunk_index
                  << "] score=" << r.relevance_score
                  << " : " << r.chunk_text.substr(0, 40) << "\n";
        assert(r.relevance_score >= 0.0f && r.relevance_score <= 1.0f);
    }

    std::cout << "PASS: index+query test\n";
    return 0;
}
