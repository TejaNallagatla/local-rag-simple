#!/bin/bash

echo "Starting Ollama..."
ollama serve &
sleep 3

echo "Pulling model..."
ollama pull llama3.2:3b

echo "Starting Jupyter..."
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --notebook-dir=/app/notebook
```

---

## `.dockerignore`
```
__pycache__/
venv/
*.pyc
.DS_Store
.git
*.ipynb_checkpoints