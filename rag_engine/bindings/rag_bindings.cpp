#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../include/rag_engine.h"

namespace py = pybind11;

PYBIND11_MODULE(ragforge_core, m) {
    m.doc() = "RAGForge C++ core — chunking, HNSW+BM25 hybrid retrieval, MMR reranking";

    py::class_<QueryResult>(m, "QueryResult")
        .def_readonly("chunk_text",      &QueryResult::chunk_text)
        .def_readonly("doc_id",          &QueryResult::doc_id)
        .def_readonly("chunk_index",     &QueryResult::chunk_index)
        .def_readonly("relevance_score", &QueryResult::relevance_score)
        .def_readonly("diversity_rank",  &QueryResult::diversity_rank)
        .def("__repr__", [](const QueryResult& r) {
            return "<QueryResult doc=" + r.doc_id +
                   " score=" + std::to_string(r.relevance_score) +
                   " chunk=" + std::to_string(r.chunk_index) + ">";
        });

    py::class_<IndexStats>(m, "IndexStats")
        .def_readonly("total_chunks", &IndexStats::total_chunks)
        .def_readonly("doc_ids",      &IndexStats::doc_ids);

    py::class_<RAGEngine>(m, "RAGEngine")
        .def(py::init<const std::string&, int, int, float>(),
             py::arg("model_path"),
             py::arg("chunk_size")    = 512,
             py::arg("chunk_overlap") = 64,
             py::arg("mmr_lambda")    = 0.6f)
        .def("ingest", &RAGEngine::ingest,
             py::arg("text"), py::arg("doc_id"),
             "Chunk, embed and index a document. Returns chunk count.")
        .def("query", &RAGEngine::query,
             py::arg("query_text"),
             py::arg("top_k") = 20,
             py::arg("top_n") = 5,
             "Retrieve top_n diverse, relevant chunks for query.")
        .def("save_index", &RAGEngine::save_index, py::arg("path"))
        .def("load_index", &RAGEngine::load_index, py::arg("path"))
        .def("stats", &RAGEngine::stats);
}
