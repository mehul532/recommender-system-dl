# Repository Guidelines

This repository is a starter scaffold for a MovieLens 1M hybrid recommender.

## Working style

- Keep code readable and modular.
- Prefer small functions and simple dataclasses over abstractions.
- Add only the minimum structure needed for the current step.
- Keep data, model, training, inference, and app logic in separate modules.

## Implementation notes

- `src` is the Python package root.
- Avoid heavy logic in `__init__.py` files.
- Keep starter files import-safe even when optional dependencies are not installed.
- Treat `data/`, `models/`, and `reports/` as project storage/output directories.
