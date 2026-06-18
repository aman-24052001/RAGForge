#include "index.h"
#include "hnswlib/hnswlib.h"
#include "nlohmann/json.hpp"
#include <cmath>
#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <fstream>

struct VectorIndex::HNSWState {
    hnswlib::L2Space space;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> alg;
    size_t max_elements;
    int M, ef;

    HNSWState(int dim, size_t max_el, int M_, int ef_)
        : space(dim), max_elements(max_el), M(M_), ef(ef_)
    {
        alg = std::make_unique<hnswlib::HierarchicalNSW<float>>(&space, max_el, M_, ef_);
        alg->ef_ = 50;
    }
};

VectorIndex::VectorIndex(int dim, int M, int ef_construction)
    : dim_(dim)
{
    hnsw_ = std::make_unique<HNSWState>(dim, 100000, M, ef_construction);
}

VectorIndex::~VectorIndex() = default;

void VectorIndex::clear() {
    id_to_chunk_.clear();
    inverted_index_.clear();
    term_freq_.clear();
    doc_freq_.clear();
    avg_dl_ = 0.0;
    next_id_ = 0;
    hnsw_ = std::make_unique<HNSWState>(
        dim_, hnsw_->max_elements, hnsw_->M, hnsw_->ef);
}

float VectorIndex::cosine(const Embedding& a, const Embedding& b) const {
    float dot = 0.0f;
    for (int i = 0; i < dim_; ++i) dot += a[i] * b[i];
    return (dot + 1.0f) / 2.0f; // normalize from [-1,1] to [0,1]
}

std::vector<std::string> VectorIndex::tokenize(const std::string& q) const {
    std::vector<std::string> tokens;
    std::istringstream ss(q);
    std::string w;
    while (ss >> w) {
        std::transform(w.begin(), w.end(), w.begin(), ::tolower);
        while (!w.empty() && !std::isalnum((unsigned char)w.back())) w.pop_back();
        if (w.size() > 1) tokens.push_back(w);
    }
    return tokens;
}

void VectorIndex::index_chunk(size_t id, const std::string& text) {
    auto tokens = tokenize(text);
    std::unordered_map<std::string, int> tf;
    for (const auto& t : tokens) tf[t]++;
    term_freq_[id] = tf;
    for (const auto& [term, cnt] : tf) {
        inverted_index_[term].push_back(id);
        doc_freq_[term]++;
    }
    // Recompute avg doc length
    double total = 0;
    for (const auto& [_, m] : term_freq_)
        for (const auto& [__, c] : m) total += c;
    avg_dl_ = total / std::max((size_t)1, term_freq_.size());
}

float VectorIndex::bm25(const std::string& query, size_t chunk_id,
                         float k1, float b) const {
    auto tokens = tokenize(query);
    auto it = term_freq_.find(chunk_id);
    if (it == term_freq_.end()) return 0.0f;
    const auto& tf_map = it->second;
    int dl = 0;
    for (const auto& [_, c] : tf_map) dl += c;
    float score = 0.0f;
    size_t N = id_to_chunk_.size();
    for (const auto& term : tokens) {
        auto tf_it = tf_map.find(term);
        if (tf_it == tf_map.end()) continue;
        float tf = tf_it->second;
        auto df_it = doc_freq_.find(term);
        int df = (df_it != doc_freq_.end()) ? df_it->second : 1;
        float idf = std::log((N - df + 0.5f) / (df + 0.5f) + 1.0f);
        float tf_norm = tf * (k1 + 1.0f) /
            (tf + k1 * (1.0f - b + b * dl / (float)std::max(1.0, avg_dl_)));
        score += idf * tf_norm;
    }
    return score;
}

void VectorIndex::add(const IndexedChunk& ic) {
    size_t id = next_id_++;
    hnsw_->alg->addPoint(ic.embedding.data(), id);
    IndexedChunk stored = ic;
    stored.hnsw_id = id;
    id_to_chunk_[id] = stored;
    index_chunk(id, ic.chunk.text);
}

std::vector<SearchResult> VectorIndex::search(
    const Embedding& query_emb,
    const std::string& query_text,
    int top_k) const
{
    if (id_to_chunk_.empty()) return {};
    int k = std::min((int)(top_k * 2), (int)id_to_chunk_.size());

    // ANN search
    auto pq = hnsw_->alg->searchKnn(query_emb.data(), k);

    // BM25 candidates
    std::unordered_map<size_t, float> bm25_scores;
    for (const auto& term : tokenize(query_text)) {
        auto it = inverted_index_.find(term);
        if (it == inverted_index_.end()) continue;
        for (size_t cid : it->second)
            bm25_scores[cid] = bm25(query_text, cid);
    }

    float max_bm25 = 0.0f;
    for (const auto& [_, s] : bm25_scores) max_bm25 = std::max(max_bm25, s);

    std::unordered_map<size_t, SearchResult> merged;

    while (!pq.empty()) {
        auto [dist, id] = pq.top(); pq.pop();
        auto it = id_to_chunk_.find(id);
        if (it == id_to_chunk_.end()) continue;
        float cos_sim = std::max(0.0f, 1.0f - dist);
        SearchResult sr;
        sr.item = it->second;
        sr.score = cos_sim;
        sr.bm25_score = bm25_scores.count(id) ? bm25_scores[id] : 0.0f;
        float nb = (max_bm25 > 0) ? sr.bm25_score / max_bm25 : 0.0f;
        sr.final_score = 0.6f * cos_sim + 0.4f * nb;
        merged[id] = sr;
    }

    for (const auto& [id, bs] : bm25_scores) {
        if (merged.count(id)) continue;
        auto it = id_to_chunk_.find(id);
        if (it == id_to_chunk_.end()) continue;
        SearchResult sr;
        sr.item = it->second;
        sr.score = cosine(query_emb, it->second.embedding);
        sr.bm25_score = bs;
        float nb = (max_bm25 > 0) ? bs / max_bm25 : 0.0f;
        sr.final_score = 0.6f * sr.score + 0.4f * nb;
        merged[id] = sr;
    }

    std::vector<SearchResult> out;
    out.reserve(merged.size());
    for (auto& [_, sr] : merged) out.push_back(sr);
    std::sort(out.begin(), out.end(),
        [](const SearchResult& a, const SearchResult& b) {
            return a.final_score > b.final_score;
        });
    if ((int)out.size() > top_k) out.resize(top_k);
    return out;
}

void VectorIndex::save(const std::string& path) const {
    hnsw_->alg->saveIndex(path + ".hnsw");
    nlohmann::json j;
    for (const auto& [id, ic] : id_to_chunk_) {
        j["chunks"][std::to_string(id)] = {
            {"text",        ic.chunk.text},
            {"doc_id",      ic.chunk.doc_id},
            {"chunk_index", ic.chunk.chunk_index},
            {"start_char",  ic.chunk.start_char},
            {"end_char",    ic.chunk.end_char},
            {"embedding",   ic.embedding}
        };
    }
    j["next_id"] = next_id_;
    j["avg_dl"]  = avg_dl_;
    std::ofstream f(path + ".meta.json");
    f << j.dump();
}

void VectorIndex::load(const std::string& path) {
    hnsw_->alg->loadIndex(path + ".hnsw", &hnsw_->space, 100000);
    std::ifstream f(path + ".meta.json");
    auto j = nlohmann::json::parse(f);
    next_id_ = j["next_id"];
    avg_dl_  = j["avg_dl"];
    for (auto& [sid, v] : j["chunks"].items()) {
        size_t id = std::stoul(sid);
        IndexedChunk ic;
        ic.chunk.text        = v["text"];
        ic.chunk.doc_id      = v["doc_id"];
        ic.chunk.chunk_index = v["chunk_index"];
        ic.chunk.start_char  = v["start_char"];
        ic.chunk.end_char    = v["end_char"];
        ic.embedding         = v["embedding"].get<Embedding>();
        ic.hnsw_id           = id;
        id_to_chunk_[id]     = ic;
        index_chunk(id, ic.chunk.text);
    }
}
