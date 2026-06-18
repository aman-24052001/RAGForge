cmake_minimum_required(VERSION 3.15)
project(ragforge VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
# Required for shared library (.so) linkage
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 REQUIRED)

set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

add_library(ragforge_lib STATIC
    src/chunker.cpp
    src/index.cpp
    src/mmr.cpp
    src/rag_engine_core.cpp
)

target_include_directories(ragforge_lib PUBLIC
    include
    ${THIRD_PARTY_DIR}
)

# Force -fPIC on static lib so it can be linked into shared .so
set_target_properties(ragforge_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

pybind11_add_module(ragforge_core
    bindings/rag_bindings.cpp
)
target_link_libraries(ragforge_core PRIVATE ragforge_lib)
set_target_properties(ragforge_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../backend
)

add_executable(test_chunker tests/test_chunker.cpp)
target_link_libraries(test_chunker PRIVATE ragforge_lib)

add_executable(test_mmr tests/test_mmr.cpp)
target_link_libraries(test_mmr PRIVATE ragforge_lib)

add_executable(test_index tests/test_index.cpp)
target_link_libraries(test_index PRIVATE ragforge_lib)
