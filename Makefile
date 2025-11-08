.PHONY: install format lint test

install:
pip install -r requirements.txt

format:
black src tests

lint:
flake8 src tests

test:
pytest
