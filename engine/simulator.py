from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Scenario:
    key: str
    label: str
    options: List[str]


class DecisionSimulator:
    """Build scenario space and convert user choices into model-ready state."""

    scenarios = [
        Scenario("career_path", "Job vs Startup", ["Job", "Startup"]),
        Scenario("location", "India vs Abroad", ["India", "Abroad"]),
        Scenario("study_work", "Higher Study vs Work", ["Higher Study", "Work"]),
        Scenario("invest_save", "Invest vs Save", ["Invest", "Save"]),
    ]

    def encode_profile(self, profile: Dict, decisions: Dict[str, str]) -> Dict[str, float]:
        age = float(profile["age"])
        income = float(profile["income"])
        savings = float(profile["savings"])
        stress = float(profile["stress_level"])
        family = float(profile["family_support"])

        startup_bias = 1.0 if decisions["career_path"] == "Startup" else 0.0
        abroad_bias = 1.0 if decisions["location"] == "Abroad" else 0.0
        study_bias = 1.0 if decisions["study_work"] == "Higher Study" else 0.0
        invest_bias = 1.0 if decisions["invest_save"] == "Invest" else 0.0

        return {
            "age_norm": age / 60.0,
            "income_norm": min(income / 5_000_000.0, 1.0),
            "savings_norm": min(savings / 10_000_000.0, 1.0),
            "stress_norm": stress / 10.0,
            "family_norm": family / 10.0,
            "startup_bias": startup_bias,
            "abroad_bias": abroad_bias,
            "study_bias": study_bias,
            "invest_bias": invest_bias,
        }
