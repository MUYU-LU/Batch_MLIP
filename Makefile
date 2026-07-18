.PHONY: install test lint demo relax validate benchmark package clean

install:
	python -m pip install -e '.[dev]'

test:
	pytest -q

lint:
	ruff check atombit_batch src examples tests tools benchmarks

demo:
	atombit-batch make-demo data/demo.extxyz

relax:
	atombit-batch run configs/relax_toy.yaml

validate:
	atombit-batch validate configs/relax_toy.yaml

benchmark:
	python benchmarks/benchmark_scaling.py --output runs/benchmark.json

package:
	python -m build

clean:
	rm -rf build dist .pytest_cache .ruff_cache *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
