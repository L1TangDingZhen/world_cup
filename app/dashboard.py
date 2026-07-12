from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from worldcup_predictor.compute import DeviceUnavailableError, resolve_device
from worldcup_predictor.evaluation.backtest import time_split_backtest
from worldcup_predictor.ingestion.matches import load_matches
from worldcup_predictor.models.elo_poisson import EloPoissonModel
from worldcup_predictor.simulation.actual_results import (
    load_knockout_winners_from_files,
)
from worldcup_predictor.simulation.tournament import (
    TournamentConfig,
    TournamentSimulator,
)
from worldcup_predictor.workflows.catch_up import catch_up

MODEL_PATH = Path("models/elo_poisson_current.json")
MATCHES_PATH = Path("data/raw/international_results.csv")
SHOOTOUTS_PATH = Path("data/raw/shootouts.csv")
GROUPS_PATH = Path("data/worldcup/groups_2026.csv")
FIXTURES_PATH = Path("data/worldcup/fixtures_2026.csv")


@st.cache_resource
def load_model() -> EloPoissonModel:
    return EloPoissonModel.load(MODEL_PATH)


@st.cache_data
def load_config() -> TournamentConfig:
    return TournamentConfig.from_csv(GROUPS_PATH, FIXTURES_PATH)


@st.cache_data
def load_knockout_winners(raw_mtime: float) -> dict[frozenset[str], str]:
    return load_knockout_winners_from_files(
        raw_path=MATCHES_PATH,
        config=load_config(),
        shootouts_path=SHOOTOUTS_PATH,
    )


@st.cache_data
def run_simulation(
    simulations: int,
    seed: int,
    device: str,
    condition_knockouts: bool,
) -> pd.DataFrame:
    knockout_winners = None
    if condition_knockouts and MATCHES_PATH.is_file():
        knockout_winners = load_knockout_winners(MATCHES_PATH.stat().st_mtime)
    return TournamentSimulator(
        load_model(),
        load_config(),
        random_seed=seed,
        device=device,
        knockout_winners=knockout_winners,
    ).run(simulations=simulations)


def render_data_status() -> None:
    st.sidebar.subheader("Data")
    metadata_path = MATCHES_PATH.with_suffix(MATCHES_PATH.suffix + ".metadata.json")
    if metadata_path.is_file():
        downloaded_at = json.loads(metadata_path.read_text(encoding="utf-8")).get(
            "downloaded_at", ""
        )
        st.sidebar.caption(f"Results downloaded: {downloaded_at[:19]}")
    st.sidebar.caption(f"Model trained through: {load_model().trained_through}")
    fixtures = load_config().fixtures
    with_results = int(
        (fixtures["home_goals"].notna() & fixtures["away_goals"].notna()).sum()
    )
    st.sidebar.caption(f"Group fixtures with results: {with_results}/{len(fixtures)}")
    if st.sidebar.button("Sync data & refit model"):
        with st.spinner("Downloading latest results and refitting the model..."):
            summary = catch_up(
                raw_path=MATCHES_PATH,
                fixtures_path=FIXTURES_PATH,
                model_output=MODEL_PATH,
                shootouts_path=SHOOTOUTS_PATH,
            )
        st.sidebar.success(
            f"Data through {summary.latest_result_date}; model trained through "
            f"{summary.trained_through}."
        )
        load_model.clear()
        st.cache_data.clear()
        st.rerun()


def render_prediction(model: EloPoissonModel, device: str) -> None:
    st.header("Single Match Prediction")
    teams = [team for team, _ in model.rankings()]
    left, right, venue = st.columns(3)
    with left:
        home = st.selectbox("Home team", teams, index=teams.index("Argentina") if "Argentina" in teams else 0)
    with right:
        away_default = teams.index("France") if "France" in teams else min(1, len(teams) - 1)
        away = st.selectbox("Away team", teams, index=away_default)
    with venue:
        neutral = st.checkbox("Neutral venue", value=True)

    if home == away:
        st.warning("Choose two different teams.")
        return

    prediction = model.predict(home, away, neutral_venue=neutral, device=device)
    metrics = st.columns(4)
    metrics[0].metric("Home win", f"{prediction.home_win_prob:.1%}")
    metrics[1].metric("Draw", f"{prediction.draw_prob:.1%}")
    metrics[2].metric("Away win", f"{prediction.away_win_prob:.1%}")
    metrics[3].metric("Most likely", prediction.most_likely_score)

    matrix = pd.DataFrame(prediction.score_matrix)
    matrix.index.name = f"{home} goals"
    matrix.columns.name = f"{away} goals"
    st.dataframe(matrix.style.format("{:.3f}"), use_container_width=True)


def render_rankings(model: EloPoissonModel) -> None:
    st.header("Elo Rankings")
    rankings = pd.DataFrame(model.rankings(), columns=["team", "elo"])
    st.bar_chart(rankings.head(25), x="team", y="elo")
    st.dataframe(rankings, use_container_width=True)


def render_tournament(device: str) -> None:
    st.header("Tournament Simulation")
    simulations = st.slider("Simulations", min_value=100, max_value=5000, value=1000, step=100)
    seed = st.number_input("Random seed", value=42, step=1)
    condition_knockouts = st.checkbox(
        "Use real knockout results already played",
        value=True,
        help="Knockout matches that already happened are pinned to their real "
        "winners; only the remaining matches are simulated.",
    )
    result = run_simulation(simulations, int(seed), device, condition_knockouts)
    st.bar_chart(result.head(20), x="team", y="champion_prob")
    st.bar_chart(result.head(20), x="team", y="group_qualify_prob")
    st.dataframe(result, use_container_width=True)


def render_groups() -> None:
    st.header("Groups and Fixtures")
    config = load_config()
    for group, teams in config.groups.sort_values(["group", "team"]).groupby("group"):
        st.subheader(f"Group {group}")
        st.dataframe(teams[["team", "fifa_ranking"]], hide_index=True, use_container_width=True)
    fixtures = config.fixtures.copy()
    fixtures["date"] = fixtures["date"].dt.date.astype(str)
    st.dataframe(fixtures, use_container_width=True)


def render_backtest() -> None:
    st.header("Backtest")
    cutoff = st.text_input("Cutoff date", value="2024-01-01")
    if st.button("Run backtest"):
        matches = load_matches(MATCHES_PATH, completed_only=True)
        result = time_split_backtest(matches, cutoff=cutoff)
        st.json(result.summary())
        st.line_chart(
            result.calibration,
            x="mean_predicted_probability",
            y="observed_frequency",
        )
        st.dataframe(result.calibration, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="World Cup Predictor", layout="wide")
    st.title("World Cup Predictor")
    device = st.sidebar.selectbox(
        "Compute device",
        options=["auto", "cpu", "cuda"],
        help="auto uses CUDA only when the WC environment has a working CUDA PyTorch installation.",
    )
    try:
        resolved_device = resolve_device(device).name
    except DeviceUnavailableError as exc:
        st.sidebar.error(str(exc))
        st.stop()
    st.sidebar.caption(f"Using: {resolved_device.upper()}")
    render_data_status()
    model = load_model()

    tab_prediction, tab_rankings, tab_groups, tab_tournament, tab_backtest = st.tabs(
        ["Prediction", "Rankings", "Groups", "Tournament", "Backtest"]
    )
    with tab_prediction:
        render_prediction(model, resolved_device)
    with tab_rankings:
        render_rankings(model)
    with tab_groups:
        render_groups()
    with tab_tournament:
        render_tournament(resolved_device)
    with tab_backtest:
        render_backtest()


if __name__ == "__main__":
    main()
