#include "../include/chunker.h"
#include <cassert>
#include <iostream>

int main() {
    Chunker c(100, 20);
    std::string text =
        "The quick brown fox jumps over the lazy dog. "
        "This is a second sentence for testing. "
        "Here comes a third one. And a fourth. And a fifth. "
        "The sixth sentence ends the paragraph.\n\n"
        "New paragraph begins here. It has some content too.";

    auto chunks = c.chunk(text, "test_doc");

    assert(!chunks.empty());
    for (const auto& ch : chunks) {
        assert(!ch.text.empty());
        assert((int)ch.text.size() <= 120); // allow slight overflow on forced splits
        std::cout << "[chunk " << ch.chunk_index << "] " << ch.text.size()
                  << " chars: " << ch.text.substr(0, 60) << "...\n";
    }

    std::cout << "\nTotal chunks: " << chunks.size() << "\n";
    std::cout << "PASS: chunker test\n";
    return 0;
}
