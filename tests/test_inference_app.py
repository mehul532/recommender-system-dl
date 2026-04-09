"""Tests for real inference helpers and the Streamlit demo app."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.app import streamlit_app
from src.data import DatasetConfig, load_processed_movielens, preprocess_movielens_1m
from src.inference import predict as inference_predict
from src.models import DeepRecommender, HybridDeepRecommender, Recommendation


def test_popularity_recommendations_exclude_seen_movies(tmp_path, monkeypatch) -> None:
    """Popularity recommendations should only include unseen movies."""

    dataset_config = _write_demo_movielens_files(tmp_path)
    preprocess_movielens_1m(dataset_config)

    monkeypatch.setattr(inference_predict, "DEFAULT_DATASET_CONFIG", dataset_config)

    recommendations = inference_predict.recommend_for_user(
        user_id=1,
        model_family="baseline_popularity",
        top_k=5,
    )

    recommended_movie_ids = {item.movie_id for item in recommendations}
    assert recommended_movie_ids == {5, 6}
    assert all(item.score >= 0.0 for item in recommendations)


def test_deep_and_hybrid_recommendations_load_checkpoints(tmp_path, monkeypatch) -> None:
    """Deep and hybrid inference should load saved checkpoints and score unseen items."""

    dataset_config = _write_demo_movielens_files(tmp_path)
    preprocess_movielens_1m(dataset_config)
    data = load_processed_movielens(dataset_config)
    deep_checkpoint_path, hybrid_checkpoint_path = _train_demo_checkpoints(
        tmp_path,
        data=data,
    )

    monkeypatch.setattr(inference_predict, "DEFAULT_DATASET_CONFIG", dataset_config)
    monkeypatch.setitem(
        inference_predict.DEFAULT_MODEL_PATHS,
        "deep",
        deep_checkpoint_path,
    )
    monkeypatch.setitem(
        inference_predict.DEFAULT_MODEL_PATHS,
        "hybrid",
        hybrid_checkpoint_path,
    )

    for model_family in ("deep", "hybrid"):
        model = inference_predict.load_model(model_family=model_family, data=data)
        recommendations = inference_predict.recommend_for_user(
            user_id=1,
            model_family=model_family,
            top_k=2,
            model=model,
            data=data,
        )

        assert len(recommendations) == 2
        assert {item.movie_id for item in recommendations} == {5, 6}
        assert all(1.0 <= item.score <= 5.0 for item in recommendations)


def test_model_availability_and_metric_helpers(tmp_path, monkeypatch) -> None:
    """App helpers should expose model availability and saved metrics cleanly."""

    deep_checkpoint = tmp_path / "models" / "deep_recommender.pt"
    deep_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    deep_checkpoint.write_bytes(b"checkpoint")

    monkeypatch.setitem(streamlit_app.CHECKPOINT_PATHS, "deep", deep_checkpoint)
    monkeypatch.setitem(
        streamlit_app.CHECKPOINT_PATHS,
        "hybrid",
        tmp_path / "models" / "missing_hybrid.pt",
    )

    available_models, unavailable_messages = streamlit_app._get_model_availability()
    assert available_models == ["baseline_popularity", "deep"]
    assert len(unavailable_messages) == 1
    assert "Hybrid Model unavailable" in unavailable_messages[0]

    comparison_report = {
        "available_models": [
            "baseline_user_item_bias",
            "deep_model",
            "hybrid_model",
        ],
        "baseline_user_item_bias": {
            "validation_rmse": 0.91,
            "test_rmse": 0.93,
        },
        "deep_model": {
            "validation_rmse": 0.92,
            "test_rmse": 0.94,
        },
        "hybrid_model": {
            "validation_rmse": 0.915,
            "test_rmse": 0.932,
        },
    }
    rmse_table = streamlit_app._build_rmse_table(comparison_report)
    assert rmse_table["model"].tolist() == [
        "Baseline User-Item Bias",
        "Deep Model",
        "Hybrid Model",
    ]

    baseline_metrics = streamlit_app._build_selected_model_metrics(
        selected_model_family="baseline_popularity",
        baseline_report={
            "baselines": {
                "popularity": {
                    "validation": {"precision_at_10": 0.04},
                    "test": {"precision_at_10": 0.03},
                }
            }
        },
        deep_report=None,
        hybrid_report=None,
    )
    assert baseline_metrics is not None
    assert baseline_metrics["validation_value"] == "0.0400"


def test_run_app_smoke(tmp_path, monkeypatch) -> None:
    """Run the Streamlit app path with a fake Streamlit module."""

    dataset_config = _write_demo_movielens_files(tmp_path)
    preprocess_movielens_1m(dataset_config)
    data = load_processed_movielens(dataset_config)

    monkeypatch.setattr(streamlit_app, "_load_processed_data", lambda: data)
    monkeypatch.setattr(
        streamlit_app,
        "_load_optional_report",
        lambda path: _report_fixture_for_name(path.name),
    )
    monkeypatch.setattr(
        streamlit_app,
        "_get_model_availability",
        lambda: (["baseline_popularity"], ["Deep Model unavailable: missing models/deep_recommender.pt"]),
    )
    monkeypatch.setattr(
        streamlit_app,
        "load_model",
        lambda model_family, data=None: object(),
    )
    monkeypatch.setattr(
        streamlit_app,
        "recommend_for_user",
        lambda user_id, model_family, top_k, model=None, data=None: [
            Recommendation(movie_id=5, score=12.0),
            Recommendation(movie_id=6, score=10.0),
        ],
    )

    fake_streamlit = FakeStreamlit()
    streamlit_app.run_app(streamlit_module=fake_streamlit)

    assert fake_streamlit.errors == []
    assert len(fake_streamlit.dataframes) >= 2
    recommendation_frame = fake_streamlit.dataframes[-1]
    assert recommendation_frame["movie_id"].tolist() == [5, 6]
    assert "popularity_score" in recommendation_frame.columns


def _train_demo_checkpoints(tmp_path, data) -> tuple:
    """Train tiny deep and hybrid models and save checkpoints."""

    num_users = int(data.users["user_idx"].max()) + 1
    num_movies = int(data.movies["movie_idx"].max()) + 1
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    deep_model = DeepRecommender(
        num_users=num_users,
        num_movies=num_movies,
        embedding_dim=8,
        hidden_dim=16,
        dropout=0.1,
        batch_size=4,
        epochs=2,
        early_stopping_patience=2,
    ).fit(data.ratings_train, data.ratings_val)
    deep_checkpoint_path = deep_model.save_checkpoint(models_dir / "deep_test.pt")

    hybrid_model = HybridDeepRecommender(
        num_users=num_users,
        num_movies=num_movies,
        movie_genre_features=data.genre_features,
        embedding_dim=8,
        hidden_dim=16,
        dropout=0.1,
        batch_size=4,
        epochs=2,
        early_stopping_patience=2,
    ).fit(data.ratings_train, data.ratings_val)
    hybrid_checkpoint_path = hybrid_model.save_checkpoint(models_dir / "hybrid_test.pt")

    return deep_checkpoint_path, hybrid_checkpoint_path


def _report_fixture_for_name(report_name: str) -> dict[str, object] | None:
    """Return a small saved-report fixture for the app smoke test."""

    if report_name == "model_comparison.json":
        return {
            "available_models": ["baseline_user_item_bias", "deep_model", "hybrid_model"],
            "baseline_user_item_bias": {
                "validation_rmse": 0.91,
                "test_rmse": 0.93,
            },
            "deep_model": {"validation_rmse": 0.92, "test_rmse": 0.94},
            "hybrid_model": {"validation_rmse": 0.915, "test_rmse": 0.932},
            "best_model": {
                "validation": "baseline_user_item_bias",
                "test": "baseline_user_item_bias",
            },
        }
    if report_name == "baseline_metrics.json":
        return {
            "baselines": {
                "popularity": {
                    "validation": {"precision_at_10": 0.04},
                    "test": {"precision_at_10": 0.03},
                }
            }
        }
    return None


def _write_demo_movielens_files(tmp_path) -> DatasetConfig:
    """Create a small MovieLens-style dataset with unseen movies for one user."""

    raw_dir = tmp_path / "data" / "raw" / "ml-1m"
    processed_dir = tmp_path / "data" / "processed" / "ml-1m"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    (raw_dir / "users.dat").write_text(
        "\n".join(
            [
                "1::F::25::10::48067",
                "2::M::35::16::12345",
            ]
        ),
        encoding="utf-8",
    )
    (raw_dir / "movies.dat").write_text(
        "\n".join(
            [
                "1::Toy Story (1995)::Animation|Children's|Comedy",
                "2::Jumanji (1995)::Adventure|Children's|Fantasy",
                "3::Grumpier Old Men (1995)::Comedy|Romance",
                "4::Waiting to Exhale (1995)::Comedy|Drama",
                "5::Heat (1995)::Action|Crime|Thriller",
                "6::GoldenEye (1995)::Action|Adventure|Thriller|Sci-Fi",
            ]
        ),
        encoding="latin-1",
    )
    (raw_dir / "ratings.dat").write_text(
        "\n".join(
            [
                "1::1::5::1000000001",
                "1::2::4::1000000002",
                "1::3::4::1000000003",
                "1::4::3::1000000004",
                "2::1::4::1000000101",
                "2::2::5::1000000102",
                "2::3::3::1000000103",
                "2::4::4::1000000104",
                "2::5::5::1000000105",
                "2::6::4::1000000106",
            ]
        ),
        encoding="utf-8",
    )

    return DatasetConfig(raw_dir=raw_dir, processed_dir=processed_dir)


@dataclass
class FakeColumn:
    """Minimal Streamlit column stub for smoke testing."""

    metrics: list[tuple[str, object]] = field(default_factory=list)

    def metric(self, label, value) -> None:
        self.metrics.append((label, value))


@dataclass
class FakeSidebar:
    """Minimal Streamlit sidebar stub for smoke testing."""

    messages: list[str] = field(default_factory=list)

    def header(self, _label) -> None:
        return None

    def info(self, message) -> None:
        self.messages.append(str(message))

    def selectbox(self, _label, options, index=0, format_func=None, help=None):
        _ = (format_func, help)
        return options[index]

    def slider(self, _label, min_value, max_value, value):
        _ = (min_value, max_value)
        return value


@dataclass
class FakeStreamlit:
    """Minimal Streamlit module stub for smoke testing."""

    sidebar: FakeSidebar = field(default_factory=FakeSidebar)
    dataframes: list[pd.DataFrame] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def cache_data(self, show_spinner=False):
        _ = show_spinner
        return self._identity_decorator

    def cache_resource(self, show_spinner=False):
        _ = show_spinner
        return self._identity_decorator

    def _identity_decorator(self, function):
        return function

    def set_page_config(self, **kwargs) -> None:
        _ = kwargs

    def title(self, _label) -> None:
        return None

    def caption(self, _label) -> None:
        return None

    def subheader(self, _label) -> None:
        return None

    def write(self, _value) -> None:
        return None

    def warning(self, message) -> None:
        self.warnings.append(str(message))

    def error(self, message) -> None:
        self.errors.append(str(message))

    def dataframe(self, frame, **kwargs) -> None:
        _ = kwargs
        self.dataframes.append(frame.copy())

    def columns(self, count):
        return [FakeColumn() for _ in range(count)]
