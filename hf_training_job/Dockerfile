FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y git build-essential gcc curl && rm -rf /var/lib/apt/lists/*

# Create user for Hugging Face Spaces
RUN useradd -m -u 1000 user

# Switch to user context
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set working directory
WORKDIR $HOME/app

# Copy requirements
COPY --chown=user requirements.txt $HOME/app/

# Install Python dependencies in one layer for better reproducibility.
# Explicitly uninstall llm_blender if cached from a previous layer,
# as it is incompatible with transformers>=4.38 (TRANSFORMERS_CACHE removed).
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt \
 && pip uninstall -y llm_blender 2>/dev/null || true

# Copy application files
COPY --chown=user . $HOME/app/

# Expose port
EXPOSE 7860

# Run application
CMD ["python", "app.py"]
