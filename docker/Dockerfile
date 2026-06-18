FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/

ENV INDEX_PATH=/tmp/ragforge_index
ENV CHUNK_SIZE=512
ENV CHUNK_OVERLAP=64
ENV MMR_LAMBDA=0.6

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
