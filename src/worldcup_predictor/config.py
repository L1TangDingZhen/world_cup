from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class EloConfig:
    initial_rating: float = 1500.0
    rating_scale: float = 400.0
    home_advantage_points: float = 100.0
    default_k_factor: float = 20.0
    competition_k_factors: dict[str, float] = field(
        default_factory=lambda: {
            "world_cup": 60.0,
            "continental_championship": 50.0,
            "world_cup_qualifier": 40.0,
            "continental_qualifier": 40.0,
            "nations_league": 30.0,
            "friendly": 20.0,
        }
    )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "EloConfig":
        return cls(**values)


@dataclass(frozen=True)
class ModelConfig:
    max_goals: int = 10
    time_decay_half_life_days: float = 1095.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "ModelConfig":
        return cls(**values)

