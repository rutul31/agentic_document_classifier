# Repository Agents

This repository hosts an AI-powered document classification assistant.

## Module Responsibilities
- `src/preprocess.py`: Extracts and normalizes text and images from multi-modal documents. Inputs include file paths or raw payloads; outputs include structured `DocumentBundle` objects.
- `src/prompt_tree.py`: Loads and manages prompt trees used for LLM orchestration. Inputs: configuration paths or dictionaries. Outputs: `PromptTree` objects capable of producing prompts per classification stage.
- `src/classifier.py`: Runs dual-LLM classification with safety checks, report generation, and storage. Inputs: `DocumentBundle`, prompt trees, and configuration values. Outputs: `ClassificationResult` objects saved to the persistence layer.
- `src/citations.py`: Builds citation evidence from document segments and images. Inputs: extracted snippets, bounding boxes, and page metadata. Outputs: citation records suitable for reports and UI rendering.
- `src/hitl_feedback.py`: Persists human-in-the-loop feedback and audit data. Inputs: classification identifiers, reviewer notes, and quality scores. Outputs: database updates accessible for analytics.
- `src/utils/logger.py`: Provides structured logging utilities. Inputs: logger names and optional configuration overrides. Outputs: configured `logging.Logger` instances.
- `src/utils/simple_yaml.py`: Supplies an offline-friendly YAML loader. Inputs: file paths or raw YAML strings. Outputs: native Python dictionaries and lists used by configuration consumers.
- `src/main.py`: Exposes the FastAPI backend entry point and orchestrates batch operations. Inputs: runtime configuration and document payloads. Outputs: API responses, scheduled jobs, and report artifacts.
- `ui_dashboard/app.py`: Implements the Streamlit-based business dashboard. Inputs: backend API endpoints and user selections. Outputs: interactive visualizations and report download links.

Follow these guidelines when modifying files within this repository.
