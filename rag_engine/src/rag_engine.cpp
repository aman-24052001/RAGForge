#include "rag_engine.h"
#include <stdexcept>
#include <algorithm>

RAGEngine::RAGEngine(const std::string& model_path,
                     int chunk_size,
                     int chunk_overlap,
                     float mmr_lambda)
{
    chunker_  = std::make_unique<Chunker>(chunk_size, chunk_overlap);
    embedder_ = std::make_unique<Embedder>(model_path);
    index_    = std::make_unique<VectorIndex>(embedder_->dim());
    mmr_      = std::make_unique<MMRReranker>(mmr_lambda);
}

int RAGEngine::ingest(const std::string& text, const std::string& doc_id) {
    if (text.empty()) throw std::invalid_argument("Empty text for doc: " + doc_id);

    auto chunks = chunker_->chunk(text, doc_id);
    std::vector<std::string> texts;
    texts.reserve(chunks.size());
    for (const auto& c : chunks) texts.push_back(c.text);

    auto embeddings = embedder_->embed_batch(texts);

    for (size_t i = 0; i < chunks.size(); ++i) {
        IndexedChunk ic;
        ic.chunk = chunks[i];
        ic.embedding = embeddings[i];
        index_->add(ic);
    }

    doc_ids_.push_back(doc_id);
    return (int)chunks.size();
}

std::vector<QueryResult> RAGEngine::query(const std::string& query_text,
                                           int top_k,
                                           int top_n) {
    if (index_->size() == 0) return {};

    // 1. Embed query
    auto query_emb = embedder_->embed(query_text);

    // 2. Hybrid retrieval (ANN + BM25)
    auto candidates = index_->search(query_emb, query_text, top_k);

    // 3. MMR diversity reranking
    auto diverse = mmr_->rerank(candidates, query_emb, top_n);

    // 4. Build output
    std::vector<QueryResult> results;
    results.reserve(diverse.size());
    for (int i = 0; i < (int)diverse.size(); ++i) {
        QueryResult r;
        r.chunk_text     = diverse[i].item.chunk.text;
        r.doc_id         = diverse[i].item.chunk.doc_id;
        r.chunk_index    = diverse[i].item.chunk.chunk_index;
        r.relevance_score = diverse[i].final_score;
        r.diversity_rank  = (float)i;
        results.push_back(r);
    }
    return results;
}

void RAGEngine::save_index(const std::string& path) {
    index_->save(path);
}

void RAGEngine::load_index(const std::string& path) {
    index_->load(path);
}

IndexStats RAGEngine::stats() const {
    IndexStats s;
    s.total_chunks = index_->size();
    s.doc_ids = doc_ids_;
    return s;
}
