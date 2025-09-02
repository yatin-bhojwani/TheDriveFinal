FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=180

WORKDIR /workspace

# Minimal system deps; keep lean
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workspace/requirements.txt

# More resilient pip install
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install \
      --retries 5 \
      --timeout 180 \
      --progress-bar off \
      --prefer-binary \
      -r /workspace/requirements.txt

COPY app /workspace/app

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
