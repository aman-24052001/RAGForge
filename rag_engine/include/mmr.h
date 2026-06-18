#pragma once
#include "index.h"
#include <vector>

// Maximal Marginal Relevance:
// Selects results that are both relevant to query AND diverse from each other.
// λ controls relevance vs diversity trade-off:
//   λ=1.0 → pure relevance (no diversity)
//   λ=0.5 → balanced (default)
//   λ=0.0 → pure diversity

class MMRReranker {
public:
    explicit MMRReranker(float lambda = 0.6f);

    std::vector<SearchResult> rerank(
        const std::vector<SearchResult>& candidates,
        const Embedding& query_emb,
        int top_n = 5
    ) const;

private:
    float lambda_;
    float cosine_sim(const Embedding& a, const Embedding& b) const;
};
