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
