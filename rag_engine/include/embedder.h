#pragma once
#include <string>
#include <vector>
#include <memory>

// Forward declare ONNX session to avoid heavy include in header
namespace Ort { class Session; class Env; class SessionOptions; }

using Embedding = std::vector<float>;

class Embedder {
public:
    explicit Embedder(const std::string& model_path);
    ~Embedder();

    Embedding embed(const std::string& text);
    std::vector<Embedding> embed_batch(const std::vector<std::string>& texts);

    int dim() const { return dim_; }

private:
    struct OrtState;
    std::unique_ptr<OrtState> ort_;
    int dim_;
    int max_seq_len_ = 256;

    std::vector<int64_t> tokenize(const std::string& text);
    Embedding mean_pool(const std::vector<float>& hidden, int seq_len);
    void normalize(Embedding& emb);
};
