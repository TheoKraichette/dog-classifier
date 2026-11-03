# syntax=docker/dockerfile:1
# Image multi-cibles : `--target backend` (API FastAPI) ou `--target frontend`
# (interface Streamlit). Le docker-compose.yml construit les deux.

# --------------------------------------------------------------------------
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

# --------------------------------------------------------------------------
FROM base AS backend

# libgomp1 : OpenMP, requis par certains noyaux TensorFlow sur image slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY webapp/backend/requirements.txt ./requirements.backend.txt
RUN pip install --no-cache-dir -r requirements.backend.txt

COPY models/class_indices.json ./models/class_indices.json
COPY webapp/backend ./webapp/backend

ENV MODEL_PATH=models/final_transfer_model.h5 \
    CLASS_INDICES_PATH=models/class_indices.json

EXPOSE 8000
CMD ["uvicorn", "webapp.backend.app:app", "--host", "0.0.0.0", "--port", "8000"]

# --------------------------------------------------------------------------
FROM base AS frontend

COPY webapp/frontend/requirements.txt ./requirements.frontend.txt
RUN pip install --no-cache-dir -r requirements.frontend.txt

COPY webapp/frontend ./webapp/frontend

ENV BACKEND_URL=http://backend:8000

EXPOSE 8501
CMD ["streamlit", "run", "webapp/frontend/frontend_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--browser.gatherUsageStats=false"]
