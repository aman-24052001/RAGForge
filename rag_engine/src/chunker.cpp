#include "chunker.h"
#include <cctype>
#include <algorithm>

Chunker::Chunker(int chunk_size, int overlap)
    : chunk_size_(chunk_size), overlap_(overlap) {}

// Manual sentence splitter - no std::regex dependency
std::vector<std::string> Chunker::split_sentences(const std::string& text) {
    std::vector<std::string> sentences;
    std::string cur;
    cur.reserve(128);

    for (size_t i = 0; i < text.size(); ++i) {
        char c = text[i];
        cur += c;

        bool is_boundary = false;
        if ((c == '.' || c == '!' || c == '?') &&
            i + 1 < text.size() && (text[i+1] == ' ' || text[i+1] == '\n')) {
            is_boundary = true;
        }
        if (c == '\n' && i + 1 < text.size() && text[i+1] == '\n') {
            is_boundary = true;
        }

        if (is_boundary && !cur.empty()) {
            // Trim
            size_t e = cur.find_last_not_of(" \t\n\r");
            if (e != std::string::npos)
                sentences.push_back(cur.substr(0, e + 1));
            cur.clear();
        }
    }
    if (!cur.empty()) {
        size_t e = cur.find_last_not_of(" \t\n\r");
        if (e != std::string::npos)
            sentences.push_back(cur.substr(0, e + 1));
    }
    if (sentences.empty() && !text.empty())
        sentences.push_back(text);
    return sentences;
}

std::vector<Chunk> Chunker::chunk(const std::string& text, const std::string& doc_id) {
    auto sentences = split_sentences(text);
    std::vector<Chunk> chunks;
    int chunk_idx = 0;
    int char_pos  = 0;
    int i = 0;

    while (i < (int)sentences.size()) {
        std::string current;
        int start = char_pos;
        int j = i;

        while (j < (int)sentences.size() &&
               (int)(current.size() + sentences[j].size() + 1) <= chunk_size_) {
            if (!current.empty()) current += ' ';
            current += sentences[j];
            j++;
        }
        if (j == i) {
            // Single sentence larger than chunk_size - force include it
            current = sentences[i].substr(0, chunk_size_);
            j = i + 1;
        }

        Chunk c;
        c.text        = current;
        c.doc_id      = doc_id;
        c.chunk_index = chunk_idx++;
        c.start_char  = start;
        c.end_char    = start + (int)current.size();
        chunks.push_back(c);

        // Overlap: step back
        int back_chars = 0;
        int back = j - 1;
        while (back > i && back_chars < overlap_) {
            back_chars += (int)sentences[back].size();
            back--;
        }
        i = (back > i) ? back + 1 : j;
        char_pos += (int)current.size() + 1;
    }
    return chunks;
}
