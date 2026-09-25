# --- Stage 1: Builder ---
# Use a slim Python base image for a smaller footprint.
FROM python:3.12-slim AS builder

# Set the working directory inside the container.
WORKDIR /app

# Install system dependencies needed by gdown, Pillow, OpenCV, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to ensure it's the latest version.
RUN pip install --no-cache-dir --upgrade pip

# PyTorch first, from the CPU wheel index: the server image has no GPU, and
# the default PyPI wheels (also on arm64) pull several GB of CUDA libraries
# that would only make the image slower to pull and start. Override for a
# GPU host: --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu126
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch torchvision --index-url ${TORCH_INDEX}

# Copy the pre-compiled requirements file first to leverage Docker's layer caching.
COPY requirements.txt .

# Install the exact Python dependencies from the locked requirements file.
# This step is fast because no dependency resolution is needed.
RUN pip install --no-cache-dir -r requirements.txt

# Download and unzip the DCNN model needed by AtomisticMicroscopyAnalysisAgent.
# This avoids downloading it every time the container runs.
ENV DCNN_MODEL_GDRIVE_ID=16LFMIEADO3XI8uNqiUoKKlrzWlc1_Q-p
ENV DCNN_MODEL_DIR=dcnn_trained
RUN gdown ${DCNN_MODEL_GDRIVE_ID} -O ${DCNN_MODEL_DIR}.zip && \
    unzip ${DCNN_MODEL_DIR}.zip -d ${DCNN_MODEL_DIR} && \
    rm ${DCNN_MODEL_DIR}.zip

# Copy your application source code and project definition.
COPY pyproject.toml .
COPY scilink/ ./scilink/

# Install the scilink package itself (without reinstalling its dependencies).
# This will also pick up the console_scripts entry point from pyproject.toml.
RUN pip install --no-cache-dir --no-deps .


# --- Stage 2: Runtime base (shared by the CLI and the web images) ---
FROM python:3.12-slim AS runtime

# Install the missing system dependency libGL.so.1 required by OpenCV.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create a dedicated, non-root user and group for enhanced security.
RUN addgroup --system scilinkgroup && adduser --system --ingroup scilinkgroup scilinkuser

# Set the home directory for the new user.
ENV HOME=/home/scilinkuser
WORKDIR /home/scilinkuser

# Copy installed Python packages from the builder stage.
COPY --from=builder /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/
# Copy the installed command-line scripts from the builder stage.
COPY --from=builder /usr/local/bin/scilink /usr/local/bin/scilink
COPY --from=builder /usr/local/bin/scilink-web /usr/local/bin/scilink-web

# Copy the pre-downloaded DCNN model from the builder stage.
# The application looks for it in the current working directory.
COPY --from=builder /app/dcnn_trained ./dcnn_trained

# Document the required API keys. You MUST provide these at runtime.
# Example: docker run -e GOOGLE_API_KEY="your-key" ...
ENV GOOGLE_API_KEY=""
ENV FUTUREHOUSE_API_KEY=""
ENV MP_API_KEY=""

# Headless defaults: no display, unbuffered logs.
ENV MPLBACKEND=Agg PYTHONUNBUFFERED=1


# --- Stage 3a: the web server, one container per workspace ---
#
#   docker build --target web -t scilink-web .
#   docker run -p 8422:8422 -e SCILINK_WEB_TOKEN=<secret> \
#              -v /path/to/workspace:/workspace -v scilink-models:/models scilink-web
#
# /workspace is the campaign's volume: sessions/ (the session root), home/
# (SCILINK_HOME: knowledge bases, banked scripts, graduated skills,
# instrument memory, the memory switch), data/ (uploads and mounted
# datasets) and optionally workspace.json (the manifest /api/v1/ops/health
# names). /models is the shared, read-only model cache. The server binds
# 0.0.0.0 and therefore REQUIRES authentication: SCILINK_WEB_TOKEN, or
# override the command with --auth-header/--trusted-proxy behind an
# authenticating proxy. Vendor keys ride the environment
# (AWS_BEARER_TOKEN_BEDROCK + AWS_REGION_NAME, ANTHROPIC_API_KEY, ...) and
# never reach generated scripts.
FROM runtime AS web

ENV SCILINK_HOME=/workspace/home \
    SCILINK_MODELS=/models \
    SCILINK_WORKSPACE=/workspace/workspace.json \
    SCILINK_USAGE_FILE=/workspace/usage.jsonl
RUN mkdir -p /workspace/sessions /workspace/home /workspace/data /models \
    && chown -R scilinkuser:scilinkgroup /workspace /models /home/scilinkuser
VOLUME ["/workspace", "/models"]
EXPOSE 8422
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8422/api/v1/ops/health', timeout=4).status == 200 else 1)"
USER scilinkuser
ENTRYPOINT ["scilink-web"]
CMD ["--host", "0.0.0.0", "--port", "8422", "--session-root", "/workspace/sessions", "--no-open"]


# --- Stage 3b: the CLI (the default target) ---
FROM runtime AS cli

# Persistent memory: graduated + auto-distilled skills live here. Declared as a
# VOLUME so it is easy to persist across container restarts. WITHOUT a mounted
# volume this dir is ephemeral and learned skills are lost when the container
# exits — mount it to keep them, e.g.:
#   docker run -v ~/.scilink:/home/scilinkuser/.scilink scilink ...
# (or set SCILINK_HOME to a mounted path).
RUN mkdir -p /home/scilinkuser/.scilink
VOLUME ["/home/scilinkuser/.scilink"]

# Ensure the non-root user owns all the files in its home directory.
RUN chown -R scilinkuser:scilinkgroup /home/scilinkuser

# Switch to the non-root user. All subsequent commands will run as this user.
USER scilinkuser

# Set the entrypoint to your scilink CLI tool.
ENTRYPOINT ["scilink"]

# Set a default command (shows help if no other command is provided).
CMD ["--help"]
