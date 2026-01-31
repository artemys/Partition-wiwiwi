FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    musescore3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY server /app/server

ENV TABSERVER_DATA_DIR=/app/server/data \
    TABSERVER_DB_URL=sqlite:////app/server/data/tabscore.db \
    TABSERVER_REDIS_URL=redis://redis:6379/0 \
    TABSERVER_PUBLIC_URL=http://localhost:8000

EXPOSE 8000

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
