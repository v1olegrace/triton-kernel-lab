.PHONY: format lint type test test-unit test-interpreter test-gpu coverage bench all docker-build docker-test-gpu

format:
	uv run --frozen ruff format src tests benchmarks
	uv run --frozen ruff check --fix src tests benchmarks

lint:
	uv run --frozen ruff check src tests benchmarks
	uv run --frozen ruff format --check src tests benchmarks

type:
	uv run --frozen mypy src tests benchmarks

test: test-unit

test-unit:
	uv run --frozen pytest -m "not gpu and not interpret"

test-interpreter:
	TRITON_INTERPRET=1 uv run --frozen pytest -m interpret

test-gpu:
	uv run --frozen pytest -m gpu

coverage:
	uv run --frozen pytest -m "not gpu and not interpret" --cov=tklab --cov-report=term-missing

bench:
	uv run --frozen tklab-bench

all: lint type test-unit

docker-build:
	docker build -t triton-kernel-lab .

docker-test-gpu:
	docker run --rm --gpus all triton-kernel-lab uv run --frozen pytest -m gpu -q
