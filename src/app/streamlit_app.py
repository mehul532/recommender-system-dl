"""Simple Streamlit demo app for the MovieLens recommenders."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import DatasetConfig, ProcessedMovieLensData, ensure_processed_movielens
from src.inference import load_model, recommend_for_user


MODEL_LABELS = {
    "baseline_popularity": "Baseline Popularity",
    "deep": "Deep Model",
    "hybrid": "Hybrid Model",
}
CHECKPOINT_PATHS = {
    "deep": Path("models/deep_recommender.pt"),
    "hybrid": Path("models/hybrid_recommender.pt"),
}
REPORT_PATHS = {
    "baseline": Path("reports/baseline_metrics.json"),
    "comparison": Path("reports/model_comparison.json"),
    "deep": Path("reports/deep_model_metrics.json"),
    "hybrid": Path("reports/hybrid_model_metrics.json"),
}
GENDER_LABELS = {
    "F": "Female",
    "M": "Male",
}
AGE_LABELS = {
    1: "Under 18",
    18: "18-24",
    25: "25-34",
    35: "35-44",
    45: "45-49",
    50: "50-55",
    56: "56+",
}
OCCUPATION_LABELS = {
    0: "Other / not specified",
    1: "Academic / educator",
    2: "Artist",
    3: "Clerical / admin",
    4: "College / grad student",
    5: "Customer service",
    6: "Doctor / health care",
    7: "Executive / managerial",
    8: "Farmer",
    9: "Homemaker",
    10: "K-12 student",
    11: "Lawyer",
    12: "Programmer",
    13: "Retired",
    14: "Sales / marketing",
    15: "Scientist",
    16: "Self-employed",
    17: "Technician / engineer",
    18: "Tradesman / craftsman",
    19: "Unemployed",
    20: "Writer",
}


def run_app(streamlit_module: Any | None = None) -> None:
    """Launch the Streamlit demo UI."""

    st = streamlit_module or _import_streamlit()

    @st.cache_data(show_spinner=False)
    def load_processed_data() -> ProcessedMovieLensData:
        return _load_processed_data()

    @st.cache_data(show_spinner=False)
    def load_report(report_key: str) -> dict[str, object] | None:
        return _load_optional_report(REPORT_PATHS[report_key])

    @st.cache_resource(show_spinner=False)
    def load_cached_model(model_family: str) -> object:
        return load_model(
            model_family=model_family,
            data=load_processed_data(),
        )

    st.set_page_config(page_title="MovieLens Recommender Demo", layout="wide")
    st.title("MovieLens Recommender Demo")
    st.caption(
        "Recommendations exclude movies the user has already rated and compare "
        "baseline, deep, and hybrid results on saved MovieLens 1M metrics."
    )

    try:
        data = load_processed_data()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return
    except Exception as exc:  # pragma: no cover - defensive app fallback
        st.error(f"Could not load processed MovieLens data: {exc}")
        return

    comparison_report = load_report("comparison")
    baseline_report = load_report("baseline")
    deep_report = load_report("deep")
    hybrid_report = load_report("hybrid")

    available_models, unavailable_messages = _get_model_availability()

    st.sidebar.header("Controls")
    if unavailable_messages:
        st.sidebar.info("\n".join(unavailable_messages))

    selected_model_family = st.sidebar.selectbox(
        "Model",
        options=available_models,
        format_func=lambda family: MODEL_LABELS[family],
    )
    user_ids = data.users["user_id"].astype("int64").tolist()
    selected_user_id = int(
        st.sidebar.selectbox(
            "User ID",
            options=user_ids,
            index=0,
            help="Type in the select box to search for a user ID.",
        )
    )
    top_k = int(st.sidebar.slider("Top N", 5, 20, 10))

    user_profile = _build_user_profile(data, selected_user_id)
    left_column, right_column = st.columns(2)
    _render_user_summary(left_column, user_profile)

    selected_metrics = _build_selected_model_metrics(
        selected_model_family=selected_model_family,
        baseline_report=baseline_report,
        deep_report=deep_report,
        hybrid_report=hybrid_report,
    )
    _render_selected_model_metrics(
        left_column,
        selected_metrics=selected_metrics,
    )
    _render_metrics_panel(
        right_column,
        comparison_report=comparison_report,
    )

    try:
        model = load_cached_model(selected_model_family)
        recommendations = recommend_for_user(
            user_id=selected_user_id,
            model_family=selected_model_family,
            top_k=top_k,
            model=model,
            data=data,
        )
    except FileNotFoundError as exc:
        st.error(str(exc))
        return
    except ValueError as exc:
        st.error(str(exc))
        return

    st.markdown("#### Top Recommendations")
    st.caption(
        "Popularity uses train interaction counts. Deep and hybrid scores are predicted ratings."
    )
    if not recommendations:
        st.warning("No unseen movies are available for this user.")
        return

    recommendation_table = _build_recommendation_table(
        data=data,
        recommendations=recommendations,
        model_family=selected_model_family,
    )
    st.dataframe(recommendation_table, use_container_width=True, hide_index=True)


def _import_streamlit() -> Any:
    """Import Streamlit with a clearer error message."""

    try:
        import streamlit as st
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "Streamlit is not installed. Run `pip install -r requirements.txt` first."
        ) from exc
    return st


def _load_processed_data() -> ProcessedMovieLensData:
    """Load processed MovieLens tables for the demo."""

    return ensure_processed_movielens(DatasetConfig())


def _load_optional_report(report_path: Path) -> dict[str, object] | None:
    """Load a report when it exists."""

    if not report_path.exists():
        return None
    with report_path.open("r", encoding="utf-8") as report_file:
        return json.load(report_file)


def _get_model_availability() -> tuple[list[str], list[str]]:
    """Return model families available for app selection plus warnings."""

    available_models = ["baseline_popularity"]
    unavailable_messages: list[str] = []

    for model_family in ("deep", "hybrid"):
        checkpoint_path = CHECKPOINT_PATHS[model_family]
        if checkpoint_path.exists():
            available_models.append(model_family)
        else:
            unavailable_messages.append(
                f"{MODEL_LABELS[model_family]} unavailable: missing {checkpoint_path}"
            )

    return available_models, unavailable_messages


def _build_user_profile(
    data: ProcessedMovieLensData,
    user_id: int,
) -> dict[str, int | str]:
    """Return a small user summary for the sidebar selection."""

    matched_user = data.users.loc[data.users["user_id"] == user_id]
    if matched_user.empty:
        raise ValueError(f"User ID {user_id} was not found in the processed dataset.")

    all_ratings = pd.concat(
        [data.ratings_train, data.ratings_val, data.ratings_test],
        ignore_index=True,
    )
    rating_count = int((all_ratings["user_id"] == user_id).sum())
    user_row = matched_user.iloc[0]
    return {
        "user_id": int(user_row["user_id"]),
        "gender_label": _gender_label(str(user_row["gender"])),
        "age_label": _age_label(int(user_row["age"])),
        "occupation_label": _occupation_label(int(user_row["occupation"])),
        "rated_movies": rating_count,
    }


def _gender_label(gender_code: str) -> str:
    """Return a human-readable label for a gender code."""

    return GENDER_LABELS.get(gender_code, gender_code)


def _age_label(age_code: int) -> str:
    """Return a human-readable label for a MovieLens age bucket."""

    return AGE_LABELS.get(age_code, f"Age {age_code}")


def _occupation_label(occupation_code: int) -> str:
    """Return a human-readable label for a MovieLens occupation code."""

    return OCCUPATION_LABELS.get(occupation_code, f"Occupation {occupation_code}")


def _model_display_name(model_name: str) -> str:
    """Return a human-readable label for model and report identifiers."""

    labels = {
        "baseline_popularity": "Baseline Popularity",
        "baseline_user_item_bias": "Baseline User-Item Bias",
        "deep": "Deep Model",
        "deep_model": "Deep Model",
        "hybrid": "Hybrid Model",
        "hybrid_model": "Hybrid Model",
    }
    return labels.get(model_name, model_name.replace("_", " ").title())


def _render_user_summary(container: Any, user_profile: dict[str, int | str]) -> None:
    """Render a compact, readable user summary block."""

    container.markdown("#### User Snapshot")
    metric_columns = container.columns(2)
    metric_columns[0].metric("User ID", int(user_profile["user_id"]))
    metric_columns[1].metric("Rated Movies", int(user_profile["rated_movies"]))
    container.markdown(
        f"**{user_profile['gender_label']}**  •  "
        f"**{user_profile['age_label']}**  •  "
        f"**{user_profile['occupation_label']}**"
    )


def _render_metrics_panel(
    container: Any,
    comparison_report: dict[str, object] | None,
) -> None:
    """Render the saved RMSE comparison section."""

    container.markdown("#### Model Performance")

    if comparison_report is None:
        container.warning("Comparison report is missing. Run the training pipelines first.")
        return

    best_model_summary = _build_best_model_summary(comparison_report)
    if best_model_summary is not None:
        container.markdown(f"**Best Saved RMSE**  \n{best_model_summary}")

    rmse_table = _build_rmse_table(comparison_report)
    container.dataframe(rmse_table, use_container_width=True, hide_index=True)


def _render_selected_model_metrics(
    container: Any,
    selected_metrics: dict[str, str] | None,
) -> None:
    """Render the selected-model metrics block."""

    container.markdown("#### Selected Model")
    if selected_metrics is None:
        container.warning("Selected-model metrics are not available yet.")
        return

    container.markdown(f"**{selected_metrics['title']}**")
    metric_columns = container.columns(2)
    metric_columns[0].metric(
        selected_metrics["validation_label"],
        selected_metrics["validation_value"],
    )
    metric_columns[1].metric(
        selected_metrics["test_label"],
        selected_metrics["test_value"],
    )

    checkpoint_path = selected_metrics.get("checkpoint_path")
    if checkpoint_path:
        container.caption(f"Checkpoint: {checkpoint_path}")


def _build_best_model_summary(
    comparison_report: dict[str, object],
) -> str | None:
    """Create a readable best-model summary from the saved comparison report."""

    best_model = comparison_report.get("best_model", {})
    if not isinstance(best_model, dict) or not best_model:
        return None

    validation_model = _model_display_name(str(best_model.get("validation", "n/a")))
    test_model = _model_display_name(str(best_model.get("test", "n/a")))
    if validation_model == test_model:
        return f"{validation_model} on validation and test."
    return (
        f"Validation: {validation_model}. "
        f"Test: {test_model}."
    )


def _build_rmse_table(comparison_report: dict[str, object]) -> pd.DataFrame:
    """Create a compact RMSE comparison table from the saved comparison report."""

    rows: list[dict[str, object]] = []
    for model_family in comparison_report.get("available_models", []):
        display_name = _model_display_name(str(model_family))
        model_metrics = comparison_report.get(model_family, {})
        if not isinstance(model_metrics, dict):
            continue
        rows.append(
            {
                "Model": display_name,
                "_validation_rmse": float(model_metrics["validation_rmse"]),
                "_test_rmse": float(model_metrics["test_rmse"]),
            }
        )

    rmse_frame = pd.DataFrame(rows)
    if rmse_frame.empty:
        return rmse_frame

    rmse_frame = rmse_frame.sort_values(
        ["_validation_rmse", "_test_rmse", "Model"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    rmse_frame["Validation RMSE"] = rmse_frame["_validation_rmse"].map(
        lambda value: f"{value:.4f}"
    )
    rmse_frame["Test RMSE"] = rmse_frame["_test_rmse"].map(
        lambda value: f"{value:.4f}"
    )
    return rmse_frame[["Model", "Validation RMSE", "Test RMSE"]]


def _build_selected_model_metrics(
    selected_model_family: str,
    baseline_report: dict[str, object] | None,
    deep_report: dict[str, object] | None,
    hybrid_report: dict[str, object] | None,
) -> dict[str, str] | None:
    """Return the metric values shown for the selected model family."""

    if selected_model_family == "baseline_popularity":
        if baseline_report is None:
            return None
        popularity_metrics = baseline_report["baselines"]["popularity"]
        return {
            "title": "Baseline Popularity",
            "validation_label": "Validation Precision@10",
            "validation_value": f"{float(popularity_metrics['validation']['precision_at_10']):.4f}",
            "test_label": "Test Precision@10",
            "test_value": f"{float(popularity_metrics['test']['precision_at_10']):.4f}",
        }

    if selected_model_family == "deep":
        if deep_report is None:
            return None
        return {
            "title": "Deep Model",
            "validation_label": "Validation RMSE",
            "validation_value": f"{float(deep_report['validation']['rmse']):.4f}",
            "test_label": "Test RMSE",
            "test_value": f"{float(deep_report['test']['rmse']):.4f}",
            "checkpoint_path": str(deep_report.get("checkpoint_path", "")),
        }

    if selected_model_family == "hybrid":
        if hybrid_report is None:
            return None
        return {
            "title": "Hybrid Model",
            "validation_label": "Validation RMSE",
            "validation_value": f"{float(hybrid_report['validation']['rmse']):.4f}",
            "test_label": "Test RMSE",
            "test_value": f"{float(hybrid_report['test']['rmse']):.4f}",
            "checkpoint_path": str(hybrid_report.get("checkpoint_path", "")),
        }

    return None


def _build_recommendation_table(
    data: ProcessedMovieLensData,
    recommendations: list[Any],
    model_family: str,
) -> pd.DataFrame:
    """Attach movie metadata to the generated recommendations."""

    recommendation_rows = [
        {
            "rank": rank,
            "movie_id": int(item.movie_id),
            "score": float(item.score),
        }
        for rank, item in enumerate(recommendations, start=1)
    ]
    recommendation_frame = pd.DataFrame(recommendation_rows)
    movie_details = data.movies[["movie_id", "title", "genres"]].copy()
    recommendation_frame = recommendation_frame.merge(
        movie_details,
        on="movie_id",
        how="left",
    )
    recommendation_frame["genres"] = recommendation_frame["genres"].map(_format_genres)

    score_column = (
        "Popularity Score"
        if model_family == "baseline_popularity"
        else "Predicted Rating"
    )
    if model_family == "baseline_popularity":
        recommendation_frame[score_column] = recommendation_frame.pop("score").map(
            lambda value: f"{int(round(value))}"
        )
    else:
        recommendation_frame[score_column] = recommendation_frame.pop("score").map(
            lambda value: f"{value:.3f}"
        )

    recommendation_frame = recommendation_frame.rename(
        columns={
            "rank": "Rank",
            "movie_id": "Movie ID",
            "title": "Title",
            "genres": "Genres",
        }
    )
    return recommendation_frame[
        ["Rank", "Movie ID", "Title", "Genres", score_column]
    ]


def _format_genres(genres: str) -> str:
    """Convert a pipe-delimited genre string into a friendlier display value."""

    if pd.isna(genres) or not genres:
        return ""
    return " • ".join(part.strip() for part in str(genres).split("|") if part.strip())


if __name__ == "__main__":
    run_app()
