.PHONY: install test test-mace lint demo relax validate benchmark package clean

PYTHON ?= python
MACE_SITE_PACKAGES ?=

install:
	python -m pip install -e '.[dev]'

test:
	$(PYTHON) -m pytest -q

test-mace:
	@if [ -n "$(MACE_SITE_PACKAGES)" ]; then \
		export PYTHONPATH="$(MACE_SITE_PACKAGES):$$PYTHONPATH"; \
	fi; \
	BATCH_MLIP_RUN_MACE_TESTS=1 $(PYTHON) -m pytest -q -m mace tests/test_mace_optimization.py

lint:
	ruff check batch_mlip atombit_batch src examples tests tools benchmarks

demo:
	batch-mlip make-demo data/demo.extxyz

relax:
	batch-mlip run configs/relax_toy.yaml

validate:
	batch-mlip validate configs/relax_toy.yaml

benchmark:
	python benchmarks/benchmark_scaling.py --output runs/benchmark.json

package:
	python -m build

clean:
	rm -rf build dist .pytest_cache .ruff_cache *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
