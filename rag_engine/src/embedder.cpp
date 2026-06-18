#include "embedder.h"
#include <onnxruntime_cxx_api.h>
#include <cmath>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <unordered_map>

// ---- Minimal BPE-style whitespace tokenizer for MiniLM ----
// Production: replace with full WordPiece tokenizer or call Python for tokenization.
// For demo we use a whitespace tokenizer with a vocab file.

struct OrtState_impl {
    Ort::Env env;
    Ort::SessionOptions opts;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;
    OrtState_impl() : env(ORT_LOGGING_LEVEL_WARNING, "ragforge") {}
};

// Embedder stores OrtState via pimpl
struct Embedder::OrtState {
    OrtState_impl impl;
    std::unordered_map<std::string, int> vocab;
    int unk_id = 100;
    int cls_id = 101;
    int sep_id = 102;
    int pad_id = 0;
};

Embedder::Embedder(const std::string& model_path) {
    ort_ = std::make_unique<OrtState>();
    ort_->impl.opts.SetIntraOpNumThreads(4);
    ort_->impl.opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    ort_->impl.session = std::make_unique<Ort::Session>(
        ort_->impl.env,
        model_path.c_str(),
        ort_->impl.opts
    );

    // Load vocab.txt from same directory as model
    std::string vocab_path = model_path.substr(0, model_path.rfind('/')) + "/vocab.txt";
    std::ifstream vf(vocab_path);
    if (!vf.is_open()) throw std::runtime_error("Cannot open vocab.txt at: " + vocab_path);
    std::string token;
    int idx = 0;
    while (std::getline(vf, token)) {
        ort_->vocab[token] = idx++;
    }

    // Determine embedding dim from model output shape
    auto out_info = ort_->impl.session->GetOutputTypeInfo(0);
    auto& shape = out_info.GetTensorTypeAndShapeInfo().GetShape();
    dim_ = (shape.size() >= 3) ? (int)shape[2] : 384;
}

Embedder::~Embedder() = default;

std::vector<int64_t> Embedder::tokenize(const std::string& text) {
    // Simple whitespace tokenization with WordPiece fallback to [UNK]
    std::vector<int64_t> ids;
    ids.push_back(ort_->cls_id);
    std::istringstream ss(text);
    std::string word;
    while (ss >> word && (int)ids.size() < max_seq_len_ - 1) {
        // lowercase
        std::transform(word.begin(), word.end(), word.begin(), ::tolower);
        // strip punctuation suffix for basic matching
        while (!word.empty() && !std::isalnum(word.back())) word.pop_back();
        auto it = ort_->vocab.find(word);
        ids.push_back(it != ort_->vocab.end() ? it->second : ort_->unk_id);
    }
    ids.push_back(ort_->sep_id);
    // Pad to max_seq_len_
    while ((int)ids.size() < max_seq_len_)
        ids.push_back(ort_->pad_id);
    return ids;
}

Embedding Embedder::mean_pool(const std::vector<float>& hidden, int seq_len) {
    // hidden shape: [1, seq_len, dim_]
    Embedding pooled(dim_, 0.0f);
    for (int t = 0; t < seq_len; ++t) {
        for (int d = 0; d < dim_; ++d) {
            pooled[d] += hidden[t * dim_ + d];
        }
    }
    float norm = (float)seq_len;
    for (auto& v : pooled) v /= norm;
    return pooled;
}

void Embedder::normalize(Embedding& emb) {
    float sq = 0.0f;
    for (auto v : emb) sq += v * v;
    float inv = 1.0f / (std::sqrt(sq) + 1e-12f);
    for (auto& v : emb) v *= inv;
}

Embedding Embedder::embed(const std::string& text) {
    auto ids = tokenize(text);
    int seq_len = (int)ids.size();

    // Build attention mask (1 for real tokens, 0 for pad)
    std::vector<int64_t> mask(seq_len, 0);
    for (int i = 0; i < seq_len; ++i)
        mask[i] = (ids[i] != ort_->pad_id) ? 1 : 0;

    std::vector<int64_t> token_type(seq_len, 0);

    std::array<int64_t, 2> shape = {1, seq_len};
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<Ort::Value> inputs;
    inputs.push_back(Ort::Value::CreateTensor<int64_t>(mem, ids.data(), ids.size(), shape.data(), 2));
    inputs.push_back(Ort::Value::CreateTensor<int64_t>(mem, mask.data(), mask.size(), shape.data(), 2));
    inputs.push_back(Ort::Value::CreateTensor<int64_t>(mem, token_type.data(), token_type.size(), shape.data(), 2));

    const char* in_names[] = {"input_ids", "attention_mask", "token_type_ids"};
    const char* out_names[] = {"last_hidden_state"};

    auto outputs = ort_->impl.session->Run(
        Ort::RunOptions{nullptr}, in_names, inputs.data(), 3, out_names, 1);

    auto* data = outputs[0].GetTensorData<float>();
    int actual_seq = (int)(std::count(mask.begin(), mask.end(), 1));
    auto emb = mean_pool(std::vector<float>(data, data + seq_len * dim_), actual_seq);
    normalize(emb);
    return emb;
}

std::vector<Embedding> Embedder::embed_batch(const std::vector<std::string>& texts) {
    std::vector<Embedding> result;
    result.reserve(texts.size());
    for (const auto& t : texts)
        result.push_back(embed(t));
    return result;
}
