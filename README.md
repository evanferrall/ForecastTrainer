# ForecastTrainer

This repository contains utilities and training scripts for forecasting escape room demand using AutoGluon.

## Layout

- `src/forecast_cli/` – reusable library code for feature engineering and preprocessing.
- `forecast_cli/` – CLI entry points and higher level training workflows.
- `conf/` – example configuration files.
- `raw_data/` – small sample dataset for testing.
- `tests/` – unit tests.

## Quick Start

Install the project with Poetry and run tests:

```bash
poetry install
pytest -q
```

Training can be launched via:

```bash
poetry run python forecast_cli/train_escape_room.py --config conf/test_escape_room_config.yaml
```

The default configuration expects an NVIDIA GPU and mixed precision enabled.
