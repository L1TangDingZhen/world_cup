"""Data ingestion and validation."""

from worldcup_predictor.ingestion.download import download_international_results
from worldcup_predictor.ingestion.api_football import ApiFootballClient
from worldcup_predictor.ingestion.matches import load_matches, validate_matches

__all__ = [
    "ApiFootballClient",
    "download_international_results",
    "load_matches",
    "validate_matches",
]
