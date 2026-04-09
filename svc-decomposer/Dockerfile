FROM python:3.12-slim

WORKDIR /app

# System deps for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install shared package first (changes less often)
COPY shared/ /app/shared/
RUN pip install --no-cache-dir /app/shared

# Install decomposer package
COPY svc-decomposer/ /app/svc-decomposer/
RUN cd /app/svc-decomposer && pip install --no-cache-dir .

ENV OHSHEET_BLOB_ROOT=/app/blob

CMD ["celery", "-A", "decomposer.celery_app", "worker", "-Q", "decomposer", "-c", "1", "--loglevel=info"]
