FROM python:3.11-slim

WORKDIR /app

# Install deps
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ ./

ENV INDEX_PATH=/tmp/ragforge_index
ENV CHUNK_SIZE=512
ENV CHUNK_OVERLAP=64
ENV MMR_LAMBDA=0.6

EXPOSE 10000

# Run from /app (where main.py, rag_wrapper.py etc. live directly)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
