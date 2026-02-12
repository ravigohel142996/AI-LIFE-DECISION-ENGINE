from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error


class PolicyNet(torch.nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_features, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class RLDecisionAgent:
    """Tiny policy-gradient style learner for success/happiness action-value approximation."""

    n_features: int
    epochs: int = 120
    lr: float = 0.03

    def __post_init__(self):
        self.model = PolicyNet(self.n_features)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def _reward(self, state: np.ndarray) -> float:
        age, income, savings, stress, family, startup, abroad, study, invest = state
        exploration_reward = 0.4 * startup + 0.25 * abroad + 0.2 * invest
        stability_reward = 0.35 * income + 0.25 * savings + 0.2 * family
        stress_penalty = 0.5 * stress
        study_premium = 0.15 * study * (1 - age)
        raw = 0.45 + exploration_reward + stability_reward + study_premium - stress_penalty
        return float(np.clip(raw, 0, 1))

    def train(self, state: Dict[str, float]) -> Tuple[float, float]:
        x = torch.tensor([list(state.values())], dtype=torch.float32)
        target = torch.tensor([[self._reward(np.array(list(state.values())))]]).float()

        for _ in range(self.epochs):
            pred = self.model(x)
            loss = torch.nn.functional.mse_loss(pred, target)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        with torch.no_grad():
            success_prob = float(self.model(x).item())
        happiness_score = float(np.clip(success_prob * 100 - state["stress_norm"] * 30 + state["family_norm"] * 20, 0, 100))
        return success_prob, happiness_score


class BayesianLifeModel:
    """Bayesian probability estimator based on profile and decisions."""

    def __init__(self):
        self.model = BayesianRidge()
        self._fit_on_synthetic_data()

    def _fit_on_synthetic_data(self):
        rng = np.random.default_rng(7)
        X = rng.uniform(0, 1, (220, 9))
        weights = np.array([0.08, 0.2, 0.17, -0.24, 0.15, 0.12, 0.1, 0.1, 0.08])
        y = np.clip(X @ weights + 0.18 + rng.normal(0, 0.04, 220), 0, 1)
        self.model.fit(X, y)

    def predict(self, state: Dict[str, float]) -> Dict[str, float]:
        x = np.array([list(state.values())])
        mean_pred, std_pred = self.model.predict(x, return_std=True)
        mean = float(np.clip(mean_pred[0], 0, 1))
        uncertainty = float(np.clip(std_pred[0], 0, 0.3))
        return {
            "bayesian_success": mean,
            "confidence": float(np.clip(1 - uncertainty, 0, 1)),
        }


class LifeForecastModel:
    """Simple time-series forecaster for wealth projection."""

    def project_wealth(self, income: float, savings: float, invest_mode: str, years: int = 10) -> pd.DataFrame:
        annual_growth = 0.11 if invest_mode == "Invest" else 0.05
        volatility = 0.07 if invest_mode == "Invest" else 0.025
        rng = np.random.default_rng(21)

        values = [savings]
        for _ in range(1, years + 1):
            contribution = income * 0.35
            shock = rng.normal(annual_growth, volatility)
            next_value = max(values[-1] * (1 + shock) + contribution, 0)
            values.append(next_value)

        df = pd.DataFrame({
            "Year": np.arange(0, years + 1),
            "Projected Wealth": values,
        })
        return df

    @staticmethod
    def forecast_quality(df: pd.DataFrame) -> float:
        trend = df["Projected Wealth"].rolling(2).mean().bfill()
        mse = mean_squared_error(df["Projected Wealth"], trend)
        rmse = float(np.sqrt(mse))
        base = max(df["Projected Wealth"].max(), 1)
        return float(np.clip(1 - (rmse / base), 0, 1))
