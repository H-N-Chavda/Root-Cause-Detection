# Developer entry points. `make help` lists them.
VENV    := .venv
PY      := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip
PKG     := src/causal_bench
SRC     := src tests

.DEFAULT_GOAL := help
.PHONY: help venv install test lint format typecheck check run eda discover clean

help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

venv:  ## Create the local virtual environment
	python3 -m venv $(VENV)

install: venv  ## Install the package in editable mode with the dev extras
	$(PIP) install --upgrade pip
	# lingam and its import-time extras pin scipy<=1.13.1, which has no wheel
	# for python 3.14. The pin is conservative -- both estimators are verified
	# against scipy 1.18.1 by the test suite -- so they go in without their own
	# dependency resolution. See the note in pyproject.toml.
	$(PIP) install --no-deps lingam==1.13.0 graphviz semopy psy pygam autograd
	$(PIP) install -e ".[dev]"
	$(VENV)/bin/pre-commit install || true

test:  ## Run the test suite with coverage
	$(PY) -m pytest

lint:  ## Lint with ruff (no changes written)
	$(VENV)/bin/ruff check $(SRC)
	$(VENV)/bin/ruff format --check $(SRC)

format:  ## Reformat and autofix with ruff
	$(VENV)/bin/ruff format $(SRC)
	$(VENV)/bin/ruff check --fix $(SRC)

typecheck:  ## Type-check with mypy (non-strict)
	$(VENV)/bin/mypy

check: lint typecheck test  ## Lint, type-check and test

run:  ## Run the full benchmark; output goes to results/<timestamp>/
	$(VENV)/bin/causal-bench

discover:  ## Run every algorithm on Tennessee Eastman and score them (~2.5h)
	$(VENV)/bin/causal-bench run \
		--out results/runs \
		--copy-to docs/RUN_REPORT.md

eda:  ## Characterise the Tennessee Eastman data; output goes to results/eda/
	$(VENV)/bin/causal-bench eda \
		--dataset datasetTE.csv \
		--ground-truth TEGroundTruth.txt \
		--reference configs/reference/tennessee_eastman.json \
		--copy-to docs/EDA_REPORT.md

clean:  ## Remove caches and build artefacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist
	find src tests -name __pycache__ -type d -exec rm -rf {} +
