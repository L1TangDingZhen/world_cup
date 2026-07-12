"""Time-based backtesting and probability metrics."""

from worldcup_predictor.evaluation.backtest import BacktestResult, time_split_backtest
from worldcup_predictor.evaluation.comparison import (
    compare_elo_poisson_to_dixon_coles,
    compare_elo_poisson_to_logistic,
)
from worldcup_predictor.evaluation.dixon_coles_backtest import backtest_dixon_coles
from worldcup_predictor.evaluation.metrics import (
    brier_score,
    calibration_table,
    log_loss,
    ranked_probability_score,
)
from worldcup_predictor.evaluation.rating_v2_backtest import backtest_rating_v2_poisson

__all__ = [
    "BacktestResult",
    "brier_score",
    "calibration_table",
    "backtest_dixon_coles",
    "backtest_rating_v2_poisson",
    "compare_elo_poisson_to_dixon_coles",
    "compare_elo_poisson_to_logistic",
    "log_loss",
    "ranked_probability_score",
    "time_split_backtest",
]
