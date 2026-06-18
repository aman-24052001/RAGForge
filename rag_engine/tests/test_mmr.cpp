#include "../include/mmr.h"
#include <cassert>
#include <iostream>
#include <random>

// Generate random unit embedding of given dim
Embedding rand_emb(int dim, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    Embedding e(dim);
    float norm = 0.0f;
    for (auto& v : e) { v = dist(rng); norm += v * v; }
    norm = std::sqrt(norm);
    for (auto& v : e) v /= norm;
    return e;
}

int main() {
    const int DIM = 16;
    MMRReranker mmr(0.6f);

    // Create 10 candidates with varying similarity to query
    Embedding query = rand_emb(DIM, 42);
    std::vector<SearchResult> cands;
    for (int i = 0; i < 10; ++i) {
        SearchResult sr;
        sr.item.embedding = rand_emb(DIM, i * 7 + 1);
        sr.final_score = 1.0f - (float)i * 0.07f; // descending relevance
        sr.score = sr.final_score;
        sr.bm25_score = 0.0f;
        sr.item.chunk.text = "chunk " + std::to_string(i);
        sr.item.chunk.doc_id = "doc";
        sr.item.chunk.chunk_index = i;
        cands.push_back(sr);
    }

    auto result = mmr.rerank(cands, query, 5);
    assert(result.size() == 5);

    std::cout << "MMR selected chunks:\n";
    for (const auto& r : result)
        std::cout << "  " << r.item.chunk.text
                  << " score=" << r.final_score << "\n";

    std::cout << "PASS: MMR test\n";
    return 0;
}
