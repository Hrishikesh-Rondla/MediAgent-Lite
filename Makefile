# MediAgent-Lite Makefile
# Usage: make <target>
# Requires: Python 3.11+, dependencies installed via pip install -r requirements.txt

.PHONY: help install test lint format ingest smoke eval app clean

help:
	@echo "MediAgent-Lite build targets:"
	@echo "  install  - Install all dependencies"
	@echo "  test     - Run full test suite (offline, no API keys needed)"
	@echo "  lint     - Run ruff linter"
	@echo "  format   - Run black formatter"
	@echo "  ingest   - Ingest corpus into Qdrant (needs NCBI_EMAIL)"
	@echo "  smoke    - Run 3 sample cases end-to-end (needs API keys)"
	@echo "  eval     - Run MedQA evaluation (needs API keys)"
	@echo "  app      - Launch Streamlit UI"
	@echo "  clean    - Remove generated files"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short

lint:
	ruff check mediagent_lite/ tests/

format:
	black mediagent_lite/ tests/ scripts/

ingest:
	python scripts/ingest.py

smoke:
	python scripts/smoke_test.py

eval:
	python scripts/run_eval.py

app:
	streamlit run mediagent_lite/ui/app.py

clean:
	rd /s /q qdrant_storage 2>nul || true
	rd /s /q data\pubmed_cache 2>nul || true
	rd /s /q __pycache__ 2>nul || true
	find . -name '*.pyc' -delete 2>nul || true
