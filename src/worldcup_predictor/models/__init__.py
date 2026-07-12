"""Match prediction models."""

import json
from pathlib import Path

from worldcup_predictor.models.bayesian import BayesianHierarchicalModel, PyMCBayesianHierarchicalModel
from worldcup_predictor.models.dixon_coles import DixonColesModel
from worldcup_predictor.models.elo_poisson import EloPoissonModel, MatchPrediction
from worldcup_predictor.models.neural_outcome import NeuralOutcomeModel
from worldcup_predictor.models.rating_v2_poisson import RatingV2PoissonModel
from worldcup_predictor.models.tournament_value import TournamentValueNetwork

# Model classes that expose the common predict()/ratings interface expected
# by the tournament simulators, keyed by the model_version stored in their
# JSON files.
MODEL_CLASSES_BY_VERSION = {
    EloPoissonModel.model_version: EloPoissonModel,
    DixonColesModel.model_version: DixonColesModel,
    BayesianHierarchicalModel.model_version: BayesianHierarchicalModel,
}


def read_model_version(path: str | Path) -> str | None:
    """Return the model_version stored in a model JSON file."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    version = payload.get("model_version")
    return str(version) if version is not None else None


def load_model(path: str | Path):
    """Load a model file, dispatching on its stored model_version."""
    version = read_model_version(path)
    model_class = MODEL_CLASSES_BY_VERSION.get(version)
    if model_class is None:
        supported = ", ".join(sorted(MODEL_CLASSES_BY_VERSION))
        raise ValueError(
            f"Unsupported model_version {version!r} in {path}; "
            f"supported versions: {supported}"
        )
    return model_class.load(path)


__all__ = [
    "BayesianHierarchicalModel",
    "PyMCBayesianHierarchicalModel",
    "DixonColesModel",
    "EloPoissonModel",
    "MatchPrediction",
    "MODEL_CLASSES_BY_VERSION",
    "NeuralOutcomeModel",
    "RatingV2PoissonModel",
    "TournamentValueNetwork",
    "load_model",
    "read_model_version",
]
