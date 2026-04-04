# ── OpsForge Dockerfile ───────────────────────────────────────────────────
FROM python:3.11-slim

# Metadata
LABEL name="OpsForge" \
      description="Autonomous Operations Control Environment (OpenEnv)" \
      version="1.0.0"

# Create non-root user for safety
RUN useradd --create-home --shell /bin/bash opsforge
WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY opsforge/ ./opsforge/
COPY tasks/    ./tasks/
COPY inference.py .
COPY openenv.yaml .

# Give ownership to non-root user
RUN chown -R opsforge:opsforge /app
USER opsforge

# Default environment variables (override with -e flags)
ENV API_BASE_URL=https://api.openai.com/v1
ENV MODEL_NAME=gpt-4o-mini
ENV HF_TOKEN=""

# Default command: run the hard task with the LLM agent
CMD ["python", "inference.py", "--task", "hard"]
