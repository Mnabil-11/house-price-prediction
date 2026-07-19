# Small, official Python base image (Debian-slim keeps the final image lean)
FROM python:3.11-slim

# All following instructions run relative to /app inside the container
WORKDIR /app

# Copy just the dependency list first (see explanation: Docker layer caching)
COPY requirements.txt .
# xgboost's wheel pulls in nvidia-nccl-cu12 (~300MB) for multi-GPU training,
# which this CPU-only inference API never uses. Installing and removing it
# in the same layer keeps it out of the final image entirely.
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y --no-input nvidia-nccl-cu12

# Now copy only what the API actually needs to run
COPY models/ ./models/
COPY api/ ./api/

# main.py resolves the model path as ../models relative to itself,
# so the working directory has to be /app/api for that to line up.
WORKDIR /app/api

EXPOSE 8000

# Shell form (not exec array form) so $PORT is actually substituted -- cloud
# hosts like Render assign their own port via this env var. Defaults to 8000
# for local `docker run`, where PORT usually isn't set.
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
