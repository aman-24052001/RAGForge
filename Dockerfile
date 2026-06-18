FROM python:3.11-slim

WORKDIR /app

# System deps for building C++ (optional — Python fallback works without)
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake build-essential libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY backend/ ./backend/
COPY models/ ./models/

# If C++ .so is pre-built, copy it
# COPY backend/ragforge_core*.so ./backend/

ENV EMBEDDING_MODEL_PATH=/app/models/minilm/model.onnx
ENV INDEX_PATH=/tmp/ragforge_index
ENV CHUNK_SIZE=512
ENV CHUNK_OVERLAP=64
ENV MMR_LAMBDA=0.6

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
