#pragma once
#include <string>
#include <vector>

struct Chunk {
    std::string text;
    std::string doc_id;
    int chunk_index;
    int start_char;
    int end_char;
};

class Chunker {
public:
    Chunker(int chunk_size = 512, int overlap = 64);
    std::vector<Chunk> chunk(const std::string& text, const std::string& doc_id);

private:
    int chunk_size_;
    int overlap_;
    std::vector<std::string> split_sentences(const std::string& text);
};
