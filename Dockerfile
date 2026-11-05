# DocForensics web app. Build:  docker build -t docforensics .
# Run:    docker run -p 8501:8501 -v docfor_models:/app/models docforensics     then open http://localhost:8501
# The optional second OCR engine (RapidOCR) is included by default; build without it:  --build-arg WITH_RAPIDOCR=0
# NOTE: written for the repository's requirements; not built in the development environment (no Docker there).
FROM python:3.10-slim

ARG WITH_RAPIDOCR=1
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 STREAMLIT_SERVER_HEADLESS=true STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-optional.txt ./
RUN pip install -r requirements.txt && if [ "$WITH_RAPIDOCR" = "1" ]; then pip install -r requirements-optional.txt; fi

COPY . .
# The semantic model (~70 MB) is downloaded on first use into /app/models; mount a volume there to keep it.
VOLUME ["/app/models", "/app/output"]
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
