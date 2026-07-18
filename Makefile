.PHONY: install test lint demo relax validate benchmark package clean

install:
	python -m pip install -e '.[dev]'

test:
	pytest -q

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
