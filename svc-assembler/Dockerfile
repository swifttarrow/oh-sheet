FROM python:3.12-slim

WORKDIR /app

# Install shared package first
COPY shared/ /app/shared/
RUN pip install --no-cache-dir /app/shared

# Install assembler package
COPY svc-assembler/ /app/svc-assembler/
RUN cd /app/svc-assembler && pip install --no-cache-dir .

ENV OHSHEET_BLOB_ROOT=/app/blob

CMD ["celery", "-A", "assembler.celery_app", "worker", "-Q", "assembler", "-c", "1", "--loglevel=info"]
