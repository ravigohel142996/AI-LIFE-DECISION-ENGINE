from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None

if HAS_TORCH:
    import torch
else:
    torch = None

if HAS_SKLEARN:
    from sklearn.linear_model import BayesianRidge
    from sklearn.metrics import mean_squared_error
else:
    BayesianRidge = None


if HAS_TORCH:

    class PolicyNet(torch.nn.Module):
        def __init__(self, n_features: int):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(n_features, 24),
                torch.nn.GELU(),
                torch.nn.Linear(24, 12),
                torch.nn.GELU(),
                torch.nn.Linear(12, 1),
                torch.nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


@dataclass
class RLDecisionAgent:
    """Policy learner with deterministic fallback if torch is unavailable."""

    n_features: int
    epochs: int = 180
    lr: float = 0.02

    def __post_init__(self):
        self._torch_ready = HAS_TORCH
        if self._torch_ready:
            self.model = PolicyNet(self.n_features)
            self.optim = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    def _reward(self, state: np.ndarray) -> float:
        age, income, savings, stress, family, startup, abroad, study, invest = state
        growth_bias = 0.25 * startup + 0.15 * abroad + 0.14 * study + 0.13 * invest
        resilience = 0.26 * savings + 0.2 * family + 0.12 * income
        stress_penalty = 0.42 * stress
        maturity_factor = 0.08 * (1 - abs(age - 0.45))
        raw = 0.3 + growth_bias + resilience + maturity_factor - stress_penalty
        return float(np.clip(raw, 0, 1))

    def train(self, state: Dict[str, float]) -> Tuple[float, float]:
        values = np.array(list(state.values()), dtype=float)
        target_score = self._reward(values)

        if self._torch_ready:
            x = torch.tensor([values], dtype=torch.float32)
            target = torch.tensor([[target_score]], dtype=torch.float32)
            for _ in range(self.epochs):
                pred = self.model(x)
                loss = torch.nn.functional.mse_loss(pred, target)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
            with torch.no_grad():
                success_prob = float(self.model(x).item())
        else:
            success_prob = target_score

        happiness_score = float(
            np.clip(success_prob * 100 - state["stress_norm"] * 32 + state["family_norm"] * 24, 0, 100)
        )
        return success_prob, happiness_score


class BayesianLifeModel:
    """Bayesian estimator with linear-probabilistic fallback."""

    def __init__(self):
        self._sk_ready = HAS_SKLEARN
        self.weights = np.array([0.07, 0.21, 0.2, -0.23, 0.18, 0.11, 0.09, 0.07, 0.1])
        self.bias = 0.18

        if self._sk_ready:
            self.model = BayesianRidge()
            self._fit_on_synthetic_data()

    def _fit_on_synthetic_data(self):
        rng = np.random.default_rng(7)
        X = rng.uniform(0, 1, (250, 9))
        y = np.clip(X @ self.weights + self.bias + rng.normal(0, 0.04, 250), 0, 1)
        self.model.fit(X, y)

    def predict(self, state: Dict[str, float]) -> Dict[str, float]:
        x = np.array([list(state.values())], dtype=float)
        if self._sk_ready:
            mean_pred, std_pred = self.model.predict(x, return_std=True)
            mean = float(np.clip(mean_pred[0], 0, 1))
            uncertainty = float(np.clip(std_pred[0], 0, 0.35))
        else:
            mean = float(np.clip(x @ self.weights + self.bias, 0, 1).item())
            uncertainty = 0.17
        return {
            "bayesian_success": mean,
            "confidence": float(np.clip(1 - uncertainty, 0, 1)),
        }


class LifeForecastModel:
    """Scenario-aware wealth projection with confidence estimate."""

    def project_wealth(self, income: float, savings: float, invest_mode: str, years: int = 12) -> pd.DataFrame:
        annual_growth = 0.115 if invest_mode == "Invest" else 0.052
        volatility = 0.065 if invest_mode == "Invest" else 0.022
        rng = np.random.default_rng(42)

        values = [float(savings)]
        for _ in range(1, years + 1):
            contribution = income * 0.32
            shock = rng.normal(annual_growth, volatility)
            next_value = max(values[-1] * (1 + shock) + contribution, 0)
            values.append(float(next_value))

        return pd.DataFrame({"Year": np.arange(0, years + 1), "Projected Wealth": values})

    @staticmethod
    def forecast_quality(df: pd.DataFrame) -> float:
        series = df["Projected Wealth"].astype(float)
        if HAS_SKLEARN:
            trend = series.rolling(2).mean().bfill()
            rmse = mean_squared_error(series, trend, squared=False)
        else:
            rmse = float((series.diff().abs().mean() or 0.0) * 0.25)
        scale = max(float(series.max()), 1.0)
        return float(np.clip(1 - (rmse / scale), 0, 1))
