# syntax=docker/dockerfile:1
# Two targets from one file:
#   docker build --target api -t rag-api .     # FastAPI service  (port 8000)
#   docker build --target ui  -t rag-ui  .     # Streamlit UI     (port 8501)

FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache
WORKDIR /app
# CPU-only PyTorch: the default wheel bundles CUDA and adds ~2 GB the container never uses.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch
RUN useradd --create-home app && mkdir -p /opt/hf-cache /app/data/index && chown -R app /opt/hf-cache /app

# ---------------------------------------------------------------- API
FROM base AS api
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt
COPY rag/ rag/
# Bake model weights into the image: fast cold starts, and no download at runtime.
RUN python -m rag.warmup && chown -R app /opt/hf-cache
COPY api.py ingest.py ./
COPY docs/ docs/
ENV HF_HUB_OFFLINE=1
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

# ---------------------------------------------------------------- UI
FROM base AS ui
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY rag/ rag/
RUN python -m rag.warmup && chown -R app /opt/hf-cache
COPY app.py ./
ENV HF_HUB_OFFLINE=1
USER app
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
