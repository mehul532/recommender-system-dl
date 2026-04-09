"""Minimal Streamlit app scaffold for the recommender project."""

from __future__ import annotations

from src.inference import recommend_for_user


def run_app() -> None:
    """Launch a placeholder Streamlit UI."""

    try:
        import streamlit as st
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Streamlit is not installed. Run `pip install -r requirements.txt` first."
        ) from exc

    recommendations = recommend_for_user(user_id=1, top_k=5)

    st.set_page_config(page_title="Movie Recommender", layout="wide")
    st.title("Movie Recommender")
    st.write("Starter Streamlit app for a hybrid MovieLens 1M recommender.")
    st.write("Replace the placeholder inference layer with real recommendations.")
    st.write("Current placeholder output:")
    st.json(
        [
            {"movie_id": item.movie_id, "score": item.score}
            for item in recommendations
        ]
    )


if __name__ == "__main__":
    run_app()
