FROM python:3.12-slim

WORKDIR /app

# Install shared package first
COPY shared/ /app/shared/
RUN pip install --no-cache-dir /app/shared

# Install the main backend package (needed for ArrangeService stub)
COPY pyproject.toml /app/pyproject.toml
COPY backend/ /app/backend/
RUN pip install --no-cache-dir /app

# Install assembler package
COPY svc-assembler/ /app/svc-assembler/
RUN pip install --no-cache-dir --no-deps /app/svc-assembler

ENV OHSHEET_BLOB_ROOT=/app/blob

CMD ["celery", "-A", "assembler.celery_app", "worker", "-Q", "assembler", "-c", "1", "--loglevel=info"]
