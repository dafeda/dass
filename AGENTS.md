# Agent Instructions

## Package Management

- Use `uv` as the package manager. Do not use `pip` or `pip install`.
- Run Python commands with `uv run`, e.g. `uv run python`, `uv run pytest`.
- Add dependencies with `uv add <package>`.
- Add dev dependencies with `uv add --group dev <package>`.
