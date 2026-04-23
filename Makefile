SHELL := /bin/zsh

.DEFAULT_GOAL := help

UV_CACHE_DIR ?= .uv-cache
UV := UV_CACHE_DIR=$(UV_CACHE_DIR) uv
RUN := $(UV) run python
PYTEST := $(UV) run pytest

MAX_PAGES_PER_DOC ?= 60
GRADIO_SERVER_PORT ?= 7860

.PHONY: help install install-lite env download ingest app query eval eval-judge list test clean clean-data clean-corpus

help: ## Show every make target
	@echo "rag-demo-unified"
	@echo
	@grep -E '^[a-zA-Z0-9_.-]+:.*## ' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*## "}; {printf "  %-18s %s\n", $$1, $$2}'

install: ## Full install (includes Docling — needed for ingestion)
	$(UV) sync --extra docling

install-lite: ## Lightweight install (skips Docling; query-only on pre-built index)
	$(UV) sync

env: ## Create .env from .env.example if it doesn't exist
	@if [ -f .env ]; then echo ".env already exists"; else cp .env.example .env; echo "Created .env from .env.example"; fi

download: ## Fetch the corpus PDFs listed in files/corpus_manifest.json
	$(RUN) -m rag_demo.cli download

ingest: ## Run Docling + chunking + Qdrant index build on corpus/
	MAX_PAGES_PER_DOC=$(MAX_PAGES_PER_DOC) $(RUN) -m rag_demo.cli rebuild

app: ## Launch the Gradio UI
	GRADIO_SERVER_PORT=$(GRADIO_SERVER_PORT) $(RUN) -m rag_demo.cli app

list: ## List ingested documents
	$(RUN) -m rag_demo.cli list

query: ## make query Q='your question here'  (multimodal mode)
	@if [ -z "$(Q)" ]; then echo "usage: make query Q='your question here'"; exit 2; fi
	$(RUN) -m rag_demo.cli query "$(Q)"

eval: ## Run extraction + generation eval, one question per doc (no LLM judge)
	$(RUN) -m rag_demo.cli eval --max-questions 1

eval-judge: ## Full eval including the LLM judge (extra OpenAI cost)
	$(RUN) -m rag_demo.cli eval --judge

test: ## Run the pytest suite (offline — no Docling, no OpenAI required)
	$(PYTEST) -q

clean-data: ## Wipe data/ (chunks, page renders, crops, qdrant, manifest)
	rm -rf data/

clean-corpus: ## Wipe downloaded PDFs in corpus/
	find corpus -name '*.pdf' -delete

clean: clean-data clean-corpus ## Wipe everything derived
