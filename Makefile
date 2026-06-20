.PHONY: format lint type test test-unit test-interpreter test-gpu coverage bench all

format:
	uv run ruff format src tests benchmarks
	uv run ruff check --fix src tests benchmarks

lint:
	uv run ruff check src tests benchmarks
	uv run ruff format --check src tests benchmarks

type:
	uv run mypy src tests benchmarks

test: test-unit

test-unit:
	uv run pytest -m "not gpu and not interpret"

test-interpreter:
	TRITON_INTERPRET=1 uv run pytest -m interpret

test-gpu:
	uv run pytest -m gpu

coverage:
	uv run pytest -m "not gpu and not interpret" --cov=tklab --cov-report=term-missing

bench:
	uv run tklab-bench

all: lint type test-unit
