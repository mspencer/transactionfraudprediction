# slim python = parent image
FROM python:3.11-slim

# python environment variables for optimization
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    ARTIFACT_DIR=artifacts

# set the working directory = /app
WORKDIR /app

# copy dependency definitions
COPY requirements.txt .
# copy scripting files
COPY inference.py main.py ./
# copy the artifacts directory
COPY artifacts/ ./artifacts/

# install required dependencies, then python packages
RUN pip install --no-cache-dir -r requirements.txt

# expose the default Cloud Run port
EXPOSE 8080

# run uvicorn pointing to the app inside the api package
# Cloud Run injects the PORT env var automatically ${PORT}
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]