# Forecast Trainer Dependencies

## Python Version
- Python >= 3.10, < 3.12

## Core Dependencies
The project uses Poetry for dependency management. All dependencies are specified in `pyproject.toml` and `poetry.lock`.

### Key Dependencies
- **PyTorch**: >= 2.3.0, < 2.4
- **PyTorch Lightning**: >= 2.2.0, < 2.3
- **NeuralForecast**: 1.7.5
- **Polars**: 0.20.18 (with pyarrow extras)
- **AutoGluon**: 1.3.0
- **HierarchicalForecast**: 1.2.1

### Data Processing & Features
- **Pandas**: 2.0.0
- **OpenMeteo-Requests**: 1.4.0
- **Holidays**: 0.47
- **Workalendar**: 17.0.0
- **Requests-Cache**: 1.2.1
- **Retry-Requests**: 2.0.0

### Configuration & Utilities
- **Hydra-Core**: 1.3.2
- **Pydantic-Settings**: 2.2.1
- **Optuna**: 3.0.0
- **Einops**: 0.8.1
- **Rich**: >= 10.2.2

## Installation

### Using Poetry (Recommended)
```bash
# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install
```

### Using pip
```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## GPU Support
The project supports GPU training through PyTorch and AutoGluon. Make sure you have:
1. CUDA-compatible GPU
2. CUDA toolkit installed
3. cuDNN installed

## Dependencies Archive
A tarball containing all dependencies is available at `forecast_dependencies.tar.gz`. This includes:
- `requirements.txt` with pinned versions
- Additional setup instructions

To extract:
```bash
tar -xzf forecast_dependencies.tar.gz
``` 