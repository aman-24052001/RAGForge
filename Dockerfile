FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./

# ragforge_core.so is pre-built by CI and committed to repo
# If present: C++ engine runs. If absent: Python fallback activates.
# The .so is NOT gitignored — CI commits it after building.

ENV INDEX_PATH=/tmp/ragforge_index
ENV CHUNK_SIZE=512
ENV CHUNK_OVERLAP=64
ENV MMR_LAMBDA=0.6

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}
