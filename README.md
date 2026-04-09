# Movie Recommendation System Scaffold

Starter structure for a hybrid deep learning recommender built on MovieLens 1M.

The project is intentionally small: it gives you clean module boundaries for data loading, model code, training, inference, and a future Streamlit app without implementing the full system yet.

## Real MovieLens 1M results

Saved report artifacts:

- `reports/baseline_metrics.json`
- `reports/deep_model_metrics.json`
- `reports/model_comparison.json`

Dataset summary from the real run:

- Ratings: 1,000,209
- Users: 6,040
- Movies: 3,883
- Split sizes: train 805,443, validation 97,383, test 97,383

Metrics:

- Popularity Precision@10: val 0.0409, test 0.0399
- User-item bias RMSE: val 0.9130, test 0.9306
- Deep recommender RMSE: val 0.9179, test 0.9367

Current comparison:

- The user-item bias baseline is better than the current deep model by 0.0049 RMSE on validation and 0.0061 RMSE on test.

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
