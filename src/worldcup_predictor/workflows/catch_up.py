"""Bring local data, fixtures and the current model up to date in one step.

The system is operated from a personal machine that is regularly switched
off, so instead of a permanently running update service, every simulation is
preceded by this catch-up: download the latest results, fill the fixture
scores, refit the model, and only then simulate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from worldcup_predictor.ingestion.download import (
    SHOOTOUTS_URL,
    download_international_results,
)
from worldcup_predictor.ingestion.matches import load_matches
from worldcup_predictor.models import read_model_version
from worldcup_predictor.models.dixon_coles import DixonColesModel
from worldcup_predictor.models.elo_poisson import EloPoissonModel

DEFAULT_RAW_PATH = Path("data/raw/international_results.csv")
DEFAULT_SHOOTOUTS_PATH = Path("data/raw/shootouts.csv")
DEFAULT_FIXTURES_PATH = Path("data/worldcup/fixtures_2026.csv")
DEFAULT_MODEL_OUTPUT = Path("models/elo_poisson_current.json")

# Dixon-Coles is refit on a rolling window rather than full history; this
# matches the protocol validated in compare_elo_poisson_to_dixon_coles.
DIXON_COLES_TRAINING_WINDOW_DAYS = 3650


@dataclass(frozen=True)
class CatchUpSummary:
    downloaded: bool
    matches: int
    latest_result_date: str
    fixtures_filled_now: int
    fixtures_with_results: int
    fixtures_total: int
    model_output: str | None
    trained_through: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def fill_fixture_results(
    fixtures_path: str | Path,
    matches: pd.DataFrame,
) -> tuple[int, int, int]:
    """Fill missing fixture scores from completed matches.

    Matches are joined on (date, home team, away team); the reversed
    orientation is also accepted in case the fixture list and the data source
    disagree about which side was nominally at home.
    """
    fixtures = pd.read_csv(fixtures_path)
    completed: dict[tuple[str, str, str], tuple[int, int]] = {}
    for match in matches.itertuples(index=False):
        key = (match.date.date().isoformat(), match.home_team, match.away_team)
        completed[key] = (int(match.home_goals), int(match.away_goals))

    filled = 0
    for index, row in fixtures.iterrows():
        if pd.notna(row["home_goals"]) and pd.notna(row["away_goals"]):
            continue
        date = str(row["date"])[:10]
        home = str(row["home_team"]).strip()
        away = str(row["away_team"]).strip()
        if (date, home, away) in completed:
            home_goals, away_goals = completed[(date, home, away)]
        elif (date, away, home) in completed:
            away_goals, home_goals = completed[(date, away, home)]
        else:
            continue
        fixtures.loc[index, "home_goals"] = home_goals
        fixtures.loc[index, "away_goals"] = away_goals
        filled += 1

    if filled:
        for column in ("home_goals", "away_goals"):
            fixtures[column] = fixtures[column].astype("Int64")
        fixtures.to_csv(fixtures_path, index=False)

    with_results = int(
        (fixtures["home_goals"].notna() & fixtures["away_goals"].notna()).sum()
    )
    return filled, with_results, len(fixtures)


def catch_up(
    raw_path: str | Path = DEFAULT_RAW_PATH,
    fixtures_path: str | Path = DEFAULT_FIXTURES_PATH,
    model_output: str | Path = DEFAULT_MODEL_OUTPUT,
    shootouts_path: str | Path = DEFAULT_SHOOTOUTS_PATH,
    offline: bool = False,
    refit: bool = True,
) -> CatchUpSummary:
    """Sync data and fixtures; refit the Elo-Poisson model unless refit=False.

    refit=False keeps the data refresh for callers whose model_output is not
    an Elo-Poisson file (e.g. a Dixon-Coles model that must not be
    overwritten by an Elo refit).
    """
    raw = Path(raw_path)
    downloaded = False
    if not offline:
        download_international_results(raw)
        download_international_results(
            Path(shootouts_path),
            source_url=SHOOTOUTS_URL,
        )
        downloaded = True
    elif not raw.is_file():
        raise FileNotFoundError(
            f"Match data file does not exist: {raw}. "
            "Run without offline mode to download it."
        )

    matches = load_matches(raw, completed_only=True)
    filled, with_results, total = fill_fixture_results(fixtures_path, matches)
    refit_output = None
    trained_through = None
    if refit:
        refit_output, trained_through = _refit_model(model_output, matches)

    return CatchUpSummary(
        downloaded=downloaded,
        matches=len(matches),
        latest_result_date=matches["date"].max().date().isoformat(),
        fixtures_filled_now=filled,
        fixtures_with_results=with_results,
        fixtures_total=total,
        model_output=refit_output,
        trained_through=trained_through,
    )


def _refit_model(
    model_output: str | Path,
    matches: pd.DataFrame,
) -> tuple[str | None, str | None]:
    """Refit the model file in place according to its stored model_version.

    A missing file is created as Elo-Poisson. Dixon-Coles files are refit on
    the rolling training window, keeping their stored configuration. Other
    model versions are left untouched so the catch-up never overwrites a
    model type it cannot rebuild.
    """
    path = Path(model_output)
    version = (
        read_model_version(path) if path.is_file() else EloPoissonModel.model_version
    )
    if version == EloPoissonModel.model_version:
        model = EloPoissonModel().fit(matches)
    elif version == DixonColesModel.model_version:
        existing = DixonColesModel.load(path)
        window_start = matches["date"].max() - pd.Timedelta(
            days=DIXON_COLES_TRAINING_WINDOW_DAYS
        )
        model = DixonColesModel(
            model_config=existing.model_config,
            max_iterations=existing.max_iterations,
        ).fit(matches.loc[matches["date"] >= window_start])
    else:
        return None, None
    model.save(path)
    return str(path), model.trained_through
