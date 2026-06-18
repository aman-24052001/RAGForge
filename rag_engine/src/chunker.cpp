#include "chunker.h"
#include <sstream>
#include <regex>
#include <numeric>

Chunker::Chunker(int chunk_size, int overlap)
    : chunk_size_(chunk_size), overlap_(overlap) {}

std::vector<std::string> Chunker::split_sentences(const std::string& text) {
    // Split on sentence boundaries: ". ", "! ", "? ", "\n\n"
    std::vector<std::string> sentences;
    std::regex sent_re(R"((?<=[.!?])\s+|(?<=\n)\n+)");
    std::sregex_token_iterator it(text.begin(), text.end(), sent_re, -1);
    std::sregex_token_iterator end;
    for (; it != end; ++it) {
        std::string s = it->str();
        if (!s.empty()) sentences.push_back(s);
    }
    if (sentences.empty()) sentences.push_back(text);
    return sentences;
}

std::vector<Chunk> Chunker::chunk(const std::string& text, const std::string& doc_id) {
    auto sentences = split_sentences(text);
    std::vector<Chunk> chunks;

    int i = 0;
    int chunk_idx = 0;
    int char_pos = 0;

    while (i < (int)sentences.size()) {
        std::string current;
        int start = char_pos;
        int j = i;

        // Build a chunk up to chunk_size_ characters
        while (j < (int)sentences.size() &&
               (int)(current.size() + sentences[j].size()) <= chunk_size_) {
            current += sentences[j] + " ";
            j++;
        }

        // If no sentence fit (sentence > chunk_size), force-add one
        if (j == i) {
            current = sentences[i].substr(0, chunk_size_);
            j = i + 1;
        }

        // Trim trailing space
        if (!current.empty() && current.back() == ' ')
            current.pop_back();

        Chunk c;
        c.text = current;
        c.doc_id = doc_id;
        c.chunk_index = chunk_idx++;
        c.start_char = start;
        c.end_char = start + (int)current.size();
        chunks.push_back(c);

        // Advance with overlap: step back overlap_ chars worth of sentences
        int back_chars = 0;
        int back = j - 1;
        while (back > i && back_chars < overlap_) {
            back_chars += sentences[back].size();
            back--;
        }
        i = (back > i) ? back + 1 : j;
        char_pos += (int)current.size() + 1;
    }

    return chunks;
}
