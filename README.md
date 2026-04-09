# Movie Recommendation System Scaffold

Starter structure for a hybrid deep learning recommender built on MovieLens 1M.

The project is intentionally small: it gives you clean module boundaries for data loading, model code, training, inference, and a future Streamlit app without implementing the full system yet.

## Project layout

```text
.
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── reports/
├── src/
│   ├── app/
│   ├── data/
│   ├── inference/
│   ├── models/
│   └── training/
└── tests/
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

## What each module is for

- `src/data`: dataset paths and data loading stubs.
- `src/models`: recommender model interface and placeholder behavior.
- `src/training`: training entrypoint that wires the data and model layers together.
- `src/inference`: prediction and recommendation helpers.
- `src/app`: Streamlit app scaffold.

## Next steps

- Replace placeholder data loaders with MovieLens 1M parsing.
- Implement the hybrid model architecture.
- Add evaluation metrics and experiment tracking.
- Expand the Streamlit app once inference is real.
