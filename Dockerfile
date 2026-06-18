FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./

ENV INDEX_PATH=/tmp/ragforge_index
ENV CHUNK_SIZE=512
ENV CHUNK_OVERLAP=64
ENV MMR_LAMBDA=0.6

# Render sets $PORT dynamically — must use shell form to expand it
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}
