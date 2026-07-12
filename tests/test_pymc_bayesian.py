from __future__ import annotations

import numpy as np
import pytest

from worldcup_predictor.ingestion.matches import load_matches
from worldcup_predictor.models.bayesian import PyMCBayesianHierarchicalModel


@pytest.mark.slow
def test_pymc_hierarchical_model_fits_and_predicts() -> None:
    model = PyMCBayesianHierarchicalModel(random_seed=1).fit(
        load_matches("data/examples/synthetic_matches.csv"),
        draws=10,
        tune=10,
        chains=1,
    )
    prediction = model.predict("Atlas", "Comet")

    assert np.asarray(prediction.score_matrix).sum() == pytest.approx(1.0)
