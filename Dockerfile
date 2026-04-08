# ---- Stage 1: Build Flutter web app ----
FROM ghcr.io/cirruslabs/flutter:stable AS flutter-build

WORKDIR /app/frontend
COPY frontend/ .
RUN flutter pub get
# Empty API_BASE_URL → client uses same-origin relative URLs (/v1/...)
RUN flutter build web --release --dart-define=API_BASE_URL=

# ---- Stage 2: Python runtime ----
# Python 3.12 for forward-compatibility with torch (MT3 model) which
# does not yet support 3.13.
FROM python:3.12-slim

WORKDIR /app

# System deps for yt-dlp (ffmpeg) and general health
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install shared package first (changes less often)
COPY shared/ shared/
RUN pip install --no-cache-dir ./shared

# Install Python package (no ML deps to keep image small)
COPY pyproject.toml .
COPY backend/ backend/
RUN pip install --no-cache-dir .

# Copy Flutter web build into a directory the backend will serve
COPY --from=flutter-build /app/frontend/build/web /app/static

# Cloud Run sets PORT; default to 8080
ENV PORT=8080

EXPOSE ${PORT}

CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}
