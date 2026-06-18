cmake_minimum_required(VERSION 3.18)
project(ragforge VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# ── Dependencies ──────────────────────────────────────────────────────────────

# pybind11 (fetched via cmake or system)
find_package(pybind11 REQUIRED)

# ONNX Runtime
# Expects: ONNXRUNTIME_ROOT env var or default /usr/local/onnxruntime
set(ORT_ROOT $ENV{ONNXRUNTIME_ROOT})
if(NOT ORT_ROOT)
    set(ORT_ROOT "/usr/local/onnxruntime")
endif()
find_library(ORT_LIB onnxruntime PATHS ${ORT_ROOT}/lib REQUIRED)
set(ORT_INCLUDE ${ORT_ROOT}/include)

# hnswlib (header-only, fetched via FetchContent)
include(FetchContent)
FetchContent_Declare(
    hnswlib
    GIT_REPOSITORY https://github.com/nmslib/hnswlib.git
    GIT_TAG        v0.8.0
)
FetchContent_MakeAvailable(hnswlib)

# nlohmann/json (header-only)
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(json)

# ── Core library ──────────────────────────────────────────────────────────────
add_library(ragforge_lib STATIC
    src/chunker.cpp
    src/embedder.cpp
    src/index.cpp
    src/mmr.cpp
    src/rag_engine.cpp
)

target_include_directories(ragforge_lib PUBLIC
    include
    ${ORT_INCLUDE}
    ${hnswlib_SOURCE_DIR}
)

target_link_libraries(ragforge_lib PUBLIC
    ${ORT_LIB}
    nlohmann_json::nlohmann_json
)

# ── pybind11 module ───────────────────────────────────────────────────────────
pybind11_add_module(ragforge_core
    bindings/rag_bindings.cpp
)

target_link_libraries(ragforge_core PRIVATE ragforge_lib)

# Output .so next to backend/
set_target_properties(ragforge_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../backend
)

# ── Tests ─────────────────────────────────────────────────────────────────────
add_executable(test_chunker tests/test_chunker.cpp)
target_link_libraries(test_chunker PRIVATE ragforge_lib)

add_executable(test_mmr tests/test_mmr.cpp)
target_link_libraries(test_mmr PRIVATE ragforge_lib)
