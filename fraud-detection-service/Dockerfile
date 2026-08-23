### builder
FROM python:3.11-slim AS builder

# set the working directory = /app
WORKDIR /app

# prevent python from writing bytecode files during build
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# install build tools if native C/C++ extensions need compilation
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# create isolated virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# install dependencies into virtualenv
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

### runtime image
FROM python:3.11-slim AS runner

WORKDIR /app

# environment configurations
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    ARTIFACT_DIR=artifacts \
    PATH="/opt/venv/bin:$PATH"

# installl minimal runtime libraries required by C++ extensions (OpenMP for LightGBM/XGBoost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# copy pre-compiled virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# copy application source code and artifacts
COPY inference.py main.py ./
COPY artifacts/ ./artifacts/

# run as non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

# exec from avoids extra shell process overhead
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1"]