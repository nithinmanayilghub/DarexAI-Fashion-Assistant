FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Set path for uv
ENV PATH="/root/.local/bin/:$PATH"

# Set working directory
WORKDIR /app

# Copy dependency configuration files
COPY pyproject.toml uv.lock ./

# Install python dependencies using uv (install globally in the container's python environment)
RUN uv pip install --system -r pyproject.toml

# Copy source code and dataset
COPY src/ ./src/
COPY ML-TASK/ ./ML-TASK/

# Pre-download and cache CLIP weights to prevent runtime container cold starts
RUN python src/download_models.py

# Set Python path to find the src module
ENV PYTHONPATH="/app"

# Expose ports (8501 for Streamlit, 8000 for FastAPI)
EXPOSE 8501 8000

# Streamlit config
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Start the application (default to frontend client)
CMD ["streamlit", "run", "src/app.py"]

