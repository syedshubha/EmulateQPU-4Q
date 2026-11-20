lint:
    uv run --locked ruff check src tests reproducibility/mesh-ladder/replay.py

test:
    uv run --locked --extra physical python -m unittest discover -s tests

reference:
    uv run --no-project --python 3.11 --with . --with-requirements requirements-reference.txt python -m unittest discover -s tests/references

build:
    uv build
