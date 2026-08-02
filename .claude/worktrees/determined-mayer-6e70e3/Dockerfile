# --- Stage 1: Builder ---
# Use a slim Python base image for a smaller footprint.
FROM python:3.11-slim as builder

# Set the working directory inside the container.
WORKDIR /app

# Install system dependencies needed by gdown, Pillow, OpenCV, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to ensure it's the latest version.
RUN pip install --no-cache-dir --upgrade pip

# Copy the pre-compiled requirements file first to leverage Docker's layer caching.
COPY requirements.txt .

# Install the exact Python dependencies from the locked requirements file.
# This step is fast because no dependency resolution is needed.
RUN pip install --no-cache-dir -r requirements.txt

# Download and unzip the DCNN model needed by AtomisticMicroscopyAnalysisAgent.
# This avoids downloading it every time the container runs.
ENV DCNN_MODEL_GDRIVE_ID=16LFMIEADO3XI8uNqiUoKKlrzWlc1_Q-p
ENV DCNN_MODEL_DIR=dcnn_trained
RUN gdown --id ${DCNN_MODEL_GDRIVE_ID} -O ${DCNN_MODEL_DIR}.zip && \
    unzip ${DCNN_MODEL_DIR}.zip -d ${DCNN_MODEL_DIR} && \
    rm ${DCNN_MODEL_DIR}.zip

# Copy your application source code and project definition.
COPY pyproject.toml .
COPY scilink/ ./scilink/

# Install the scilink package itself (without reinstalling its dependencies).
# This will also pick up the console_scripts entry point from pyproject.toml.
RUN pip install --no-cache-dir --no-deps .


# --- Stage 2: Final Image ---
# Start from a clean, minimal base image for the final product.
FROM python:3.11-slim

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
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
# Copy the installed command-line script from the builder stage.
COPY --from=builder /usr/local/bin/scilink /usr/local/bin/scilink

# Copy the pre-downloaded DCNN model from the builder stage.
# The application looks for it in the current working directory.
COPY --from=builder /app/dcnn_trained ./dcnn_trained

# Document the required API keys. You MUST provide these at runtime.
# Example: docker run -e GOOGLE_API_KEY="your-key" ...
ENV GOOGLE_API_KEY=""
ENV FUTUREHOUSE_API_KEY=""
ENV MP_API_KEY=""

# Ensure the non-root user owns all the files in its home directory.
RUN chown -R scilinkuser:scilinkgroup /home/scilinkuser

# Switch to the non-root user. All subsequent commands will run as this user.
USER scilinkuser

# Set the entrypoint to your scilink CLI tool.
ENTRYPOINT ["scilink"]

# Set a default command (shows help if no other command is provided).
CMD ["--help"]
