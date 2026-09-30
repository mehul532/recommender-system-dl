# Hybrid Deep-Learning Recommender (MovieLens 1M)

A complete recommender system on MovieLens 1M (1,000,209 ratings, 6,040 users, 3,883 movies): bias baselines, a PyTorch deep embedding model, and a hybrid model, compared under one offline evaluation protocol with a Streamlit demo app.

## Results

Held-out evaluation (train 805,443 / validation 97,383 / test 97,383):

| Model | Validation RMSE | Test RMSE |
| --- | ---: | ---: |
| Bias baseline | 0.9130 | 0.9306 |
| Deep model | 0.9179 | 0.9367 |
| Hybrid model | 0.9149 | 0.9323 |

The honest finding: the bias baseline still wins on both validation and test. The hybrid model improves on the deep model but does not beat the baseline — a useful reminder that well-tuned simple baselines are hard to beat on dense rating data. Full metrics are tracked in `reports/`: `baseline_metrics.json`, `deep_model_metrics.json`, `hybrid_model_metrics.json`, `model_comparison.json`.

## Architecture

- `src/data/` — MovieLens 1M parsing, dataset config, and train/validation/test splits (`dataset.py`)
- `src/models/` — popularity and user-item bias baselines (`baselines.py`), deep embedding model (`deep_recommender.py`), hybrid model (`hybrid_recommender.py`), shared interface (`recommender.py`)
- `src/training/` — training entry points (`train.py`, `deep_train.py`, `hybrid_train.py`), RMSE / precision@k evaluation (`evaluation.py`), model comparison (`comparison.py`)
- `src/inference/` — prediction and per-user recommendation helpers (`predict.py`)
- `src/app/` — Streamlit demo app (`streamlit_app.py`)

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

Train each model and write its report artifact to `reports/`:

```bash
python -m src.training.train          # popularity + bias baselines
python -m src.training.deep_train     # deep embedding model
python -m src.training.hybrid_train   # hybrid model
```

Launch the Streamlit demo:

```bash
streamlit run src/app/streamlit_app.py
```

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

## License

MIT. See [LICENSE](LICENSE).
