# Document Classification Assistant

An AI-powered assistant for classifying multi-modal documents into Public, Confidential, Highly Sensitive, or Unsafe categories.

## Features
- Multi-modal preprocessing for text, PDFs, and images with optional OCR.
- Configurable prompt tree library leveraging LangChain/LlamaIndex orchestration patterns.
- Dual-LLM cross verification with safety keyword detection.
- Citation generation and evidence tracking.
- Human-in-the-loop feedback persistence powered by SQLite and SQLAlchemy.
- FastAPI backend with Streamlit dashboard for interactive and batch workflows.
- Automated report generation in JSON and PDF formats.
- Dockerized deployment, comprehensive tests, and formatting/linting via Makefile.

## Getting Started
```bash
python -m venv .datathon_env
source .datathon_env/bin/activate
pip install -r requirements.txt
```

## Local LLMs via Ollama
1. [Install Ollama](https://ollama.com/download) on the host where this backend runs and start the daemon with `ollama serve` (it listens on `http://127.0.0.1:11434` by default).
2. Pull the models referenced in `config/config.yaml`, for example:
   ```bash
   ollama pull llama3.1:8b
   ollama pull llama3.1:13b
   ```
3. Optional: verify the service is reachable with `curl http://127.0.0.1:11434/api/tags`.
4. Adjust `config/config.yaml` if you need a different model name, base URL, or generation options. The default template configures two local models:
   ```yaml
   ollama:
     base_url: http://127.0.0.1:11434
   models:
     primary:
       name: risk-primary-8b
       model: llama3.1:8b
     secondary:
       name: risk-secondary-13b
       model: llama3.1:13b
   ```
5. Restart `uvicorn` (or the Docker container) so the new configuration is loaded. The backend will now call the local Ollama HTTP API for both classifier passes.

### Using Ollama Cloud
1. Authenticate the Ollama CLI (one time):
   ```bash
   ollama signin
   ```
2. Register the cloud models you plan to call (examples):
   ```bash
   ollama pull gpt-oss:120b-cloud
   ollama pull deepseek-v3.1:671b-cloud
   ```
3. Copy your Ollama Cloud API key from https://ollama.com.
4. Export it before starting the backend (or place it in your shell profile):
   ```bash
   export OLLAMA_API_KEY="sk_live_xxx"
   ```
   The default `config/config.yaml` already points to `https://api.ollama.ai` and reads the key via `api_key_env: OLLAMA_API_KEY`, so no further edits are required. If you prefer to hardcode the key instead, replace the env entry with `api_key: "sk_live_xxx"` (only do this for local testing).
5. Restart `uvicorn src.main:app --reload`. The classifier will now send requests to Ollama Cloud using your key while retaining the same dual-model workflow.

### Running the Backend
```bash
uvicorn src.main:app --reload
```

### Running the Dashboard
```bash
streamlit run ui_dashboard/app.py
```

### Tests and Quality Checks
```bash
make test
make lint
make format
```

### Docker
```bash
docker build -t doc-assistant .
docker run -p 8000:8000 doc-assistant
```

## Demo Notebook
See `demo.ipynb` for an end-to-end walkthrough of the sample test cases (TC1–TC5).
