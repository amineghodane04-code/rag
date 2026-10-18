PYTHON = python3
UV = uv

.PHONY: all install run debug clean lint lint-strict

all: run

install:
	$(UV) sync

run:
	$(UV) run $(PYTHON) -m src search "How to configure OpenAI server?" --k 5

debug:
	$(UV) run $(PYTHON) -m pdb -m src search "How to configure OpenAI server?" --k 5

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf data/processed/*

lint:
	flake8 src
	mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	flake8 src
	mypy src --strict
