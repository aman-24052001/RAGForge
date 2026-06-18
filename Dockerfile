# ── Stage 1: Build C++ engine ─────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    cmake build-essential python3-dev git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install pybind11 --break-system-packages

# Copy only what's needed for C++ build
COPY rag_engine/ ./rag_engine/

# CMake outputs .so to ../backend relative to rag_engine/
# which means /build/backend/ragforge_core*.so
RUN mkdir -p backend && \
    cd rag_engine && mkdir -p build && cd build && \
    cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -Dpybind11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())") && \
    make -j$(nproc) ragforge_core && \
    echo "✓ C++ build done:" && ls -lh /build/backend/ragforge_core*.so

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./

# Pull compiled .so from builder
COPY --from=builder /build/backend/ragforge_core*.so ./

RUN python3 -c "import ragforge_core; print('✓ ragforge_core loaded:', ragforge_core.__doc__[:40])" \
    || echo "⚠ ragforge_core not available — Python fallback will be used"

ENV INDEX_PATH=/tmp/ragforge_index
ENV CHUNK_SIZE=512
ENV CHUNK_OVERLAP=64
ENV MMR_LAMBDA=0.6

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}
