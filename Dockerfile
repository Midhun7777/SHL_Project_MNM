FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Do NOT load SentenceTransformer here — Render free-tier builds OOM above 512MB.
# Model weights are cached on first /chat (or by background warm in app/main.py).

COPY app/ app/
COPY data/catalog.json data/catalog.json
COPY data/traces.json data/traces.json

ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false
ENV OMP_NUM_THREADS=1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
