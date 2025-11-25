FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

WORKDIR /app

# Copy src files
COPY knowledge_base.py .
COPY semantic_layer.py .
COPY retrieval_system.py .
COPY augmentation.py .
COPY generation.py .

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy notebook folder
COPY notebook/ ./notebook/

# Copy data folder (if exists)
COPY data/ ./data/

# Expose Jupyter port
EXPOSE 8888
# Expose OLLAMA port
EXPOSE 11434

# Startup script
COPY startup.sh .
RUN chmod +x startup.sh

CMD ["./startup.sh"]