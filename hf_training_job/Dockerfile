FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y git build-essential gcc curl

# Create user for Hugging Face Spaces
RUN useradd -m -u 1000 user

# Switch to user context
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set working directory
WORKDIR $HOME/app

# Install PyTorch 2.5.1+ with CUDA 12.4
RUN pip install --no-cache-dir "torch>=2.5.1" "torchvision" "torchaudio" --index-url https://download.pytorch.org/whl/cu124

# Copy requirements
COPY --chown=user requirements.txt $HOME/app/

# Install Unsloth and other dependencies
RUN pip install --no-cache-dir "unsloth @ git+https://github.com/unslothai/unsloth.git" "unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo.git"
RUN pip install --no-cache-dir gradio
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY --chown=user . $HOME/app/

# Expose port
EXPOSE 7860

# Run application
CMD ["python", "app.py"]
