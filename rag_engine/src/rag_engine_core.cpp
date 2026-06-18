#include "rag_engine_core.h"
#include <stdexcept>

RAGEngineCore::RAGEngineCore(int dim, int chunk_size, int chunk_overlap, float mmr_lambda)
    : dim_(dim)
{
    chunker_ = std::make_unique<Chunker>(chunk_size, chunk_overlap);
    index_   = std::make_unique<VectorIndex>(dim);
    mmr_     = std::make_unique<MMRReranker>(mmr_lambda);
}

std::vector<std::string> RAGEngineCore::chunk_text(
    const std::string& text, const std::string& doc_id)
{
    auto chunks = chunker_->chunk(text, doc_id);
    std::vector<std::string> texts;
    texts.reserve(chunks.size());
    for (const auto& c : chunks) texts.push_back(c.text);
    return texts;
}

void RAGEngineCore::add_chunks(
    const std::vector<std::string>& texts,
    const std::vector<std::string>& doc_ids,
    const std::vector<int>& chunk_indices,
    const std::vector<Embedding>& embeddings)
{
    if (texts.size() != embeddings.size())
        throw std::invalid_argument("texts and embeddings must have same length");

    for (size_t i = 0; i < texts.size(); ++i) {
        IndexedChunk ic;
        ic.chunk.text        = texts[i];
        ic.chunk.doc_id      = doc_ids[i];
        ic.chunk.chunk_index = chunk_indices[i];
        ic.chunk.start_char  = 0;
        ic.chunk.end_char    = (int)texts[i].size();
        ic.embedding         = embeddings[i];
        index_->add(ic);
    }

    if (!doc_ids.empty()) {
        const auto& did = doc_ids[0];
        bool found = false;
        for (const auto& d : doc_ids_) if (d == did) { found = true; break; }
        if (!found) doc_ids_.push_back(did);
    }
}

std::vector<QueryResult> RAGEngineCore::query(
    const Embedding& query_emb,
    const std::string& query_text,
    int top_k, int top_n)
{
    if (index_->size() == 0) return {};

    auto candidates = index_->search(query_emb, query_text, top_k);
    auto diverse    = mmr_->rerank(candidates, query_emb, top_n);

    std::vector<QueryResult> results;
    results.reserve(diverse.size());
    for (int i = 0; i < (int)diverse.size(); ++i) {
        QueryResult r;
        r.chunk_text      = diverse[i].item.chunk.text;
        r.doc_id          = diverse[i].item.chunk.doc_id;
        r.chunk_index     = diverse[i].item.chunk.chunk_index;
        r.relevance_score = diverse[i].final_score;
        r.diversity_rank  = (float)i;
        results.push_back(r);
    }
    return results;
}

void RAGEngineCore::save(const std::string& path) { index_->save(path); }
void RAGEngineCore::load(const std::string& path) { index_->load(path); }

void RAGEngineCore::clear() {
    index_->clear();
    doc_ids_.clear();
}

IndexStats RAGEngineCore::stats() const {
    IndexStats s;
    s.total_chunks = index_->size();
    s.doc_ids      = doc_ids_;
    return s;
}
