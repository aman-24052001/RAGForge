#include "mmr.h"
#include <algorithm>
#include <cmath>
#include <limits>

MMRReranker::MMRReranker(float lambda) : lambda_(lambda) {}

float MMRReranker::cosine_sim(const Embedding& a, const Embedding& b) const {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12f);
}

std::vector<SearchResult> MMRReranker::rerank(
    const std::vector<SearchResult>& candidates,
    const Embedding& query_emb,
    int top_n) const
{
    if (candidates.empty()) return {};
    int n = std::min(top_n, (int)candidates.size());

    std::vector<bool> selected(candidates.size(), false);
    std::vector<SearchResult> result;
    result.reserve(n);

    for (int iter = 0; iter < n; ++iter) {
        float best_score = -std::numeric_limits<float>::infinity();
        int best_idx = -1;

        for (int i = 0; i < (int)candidates.size(); ++i) {
            if (selected[i]) continue;

            // Relevance term: similarity to query
            float rel = lambda_ * candidates[i].final_score;

            // Redundancy term: max similarity to already selected
            float max_sim = 0.0f;
            for (const auto& sel : result) {
                float sim = cosine_sim(candidates[i].item.embedding,
                                       sel.item.embedding);
                max_sim = std::max(max_sim, sim);
            }
            float red = (1.0f - lambda_) * max_sim;

            float mmr = rel - red;
            if (mmr > best_score) {
                best_score = mmr;
                best_idx = i;
            }
        }

        if (best_idx >= 0) {
            selected[best_idx] = true;
            result.push_back(candidates[best_idx]);
        }
    }

    return result;
}
