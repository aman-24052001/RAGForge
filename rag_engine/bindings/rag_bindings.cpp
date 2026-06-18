#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../include/rag_engine_core.h"

namespace py = pybind11;

PYBIND11_MODULE(ragforge_core, m) {
    m.doc() = "RAGForge C++ core — chunking, HNSW+BM25 hybrid retrieval, MMR reranking. "
              "Embeddings are provided by Python (sentence-transformers).";

    py::class_<QueryResult>(m, "QueryResult")
        .def_readonly("chunk_text",      &QueryResult::chunk_text)
        .def_readonly("doc_id",          &QueryResult::doc_id)
        .def_readonly("chunk_index",     &QueryResult::chunk_index)
        .def_readonly("relevance_score", &QueryResult::relevance_score)
        .def_readonly("diversity_rank",  &QueryResult::diversity_rank);

    py::class_<IndexStats>(m, "IndexStats")
        .def_readonly("total_chunks", &IndexStats::total_chunks)
        .def_readonly("doc_ids",      &IndexStats::doc_ids);

    py::class_<RAGEngineCore>(m, "RAGEngineCore")
        .def(py::init<int, int, int, float>(),
             py::arg("dim"),
             py::arg("chunk_size")    = 512,
             py::arg("chunk_overlap") = 64,
             py::arg("mmr_lambda")    = 0.6f,
             "Create engine. dim = embedding dimension (e.g. 384 for MiniLM).")
        .def("chunk_text", &RAGEngineCore::chunk_text,
             py::arg("text"), py::arg("doc_id"),
             "Chunk raw text. Returns list of chunk strings. "
             "Embed them in Python, then call add_chunks.")
        .def("add_chunks", &RAGEngineCore::add_chunks,
             py::arg("texts"), py::arg("doc_ids"),
             py::arg("chunk_indices"), py::arg("embeddings"),
             "Index pre-embedded chunks.")
        .def("query", &RAGEngineCore::query,
             py::arg("query_emb"), py::arg("query_text"),
             py::arg("top_k") = 20, py::arg("top_n") = 5,
             "Retrieve top_n diverse chunks. query_emb = list[float].")
        .def("save",  &RAGEngineCore::save,  py::arg("path"))
        .def("load",  &RAGEngineCore::load,  py::arg("path"))
        .def("clear", &RAGEngineCore::clear)
        .def("stats", &RAGEngineCore::stats);
}
