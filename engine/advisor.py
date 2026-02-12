from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Dict, List

HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None

if HAS_SKLEARN:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class AdvisorBot:
    """NLP advisor bot with semantic retrieval and deterministic fallback."""

    def __post_init__(self):
        self.knowledge = [
            "Build a six month emergency fund if stress is high and income is unstable.",
            "If your goal is global leadership, combine higher study abroad with intensive internships.",
            "Startup path is strongest when your skills include product, sales, and resilience.",
            "Investing systematically usually beats idle savings for long horizons.",
            "Family support reduces burnout; actively design around your support system.",
            "Prioritize work experience before a degree if you need immediate cash flow and clarity.",
            "If stress is high, favor one major move per quarter to protect execution quality.",
        ]

        self._sk_ready = HAS_SKLEARN
        if self._sk_ready:
            self.vectorizer = TfidfVectorizer(stop_words="english")
            self.doc_vectors = self.vectorizer.fit_transform(self.knowledge)

    def _fallback_rank(self, summary: str) -> List[int]:
        tokens = {t.strip('.,').lower() for t in summary.split() if len(t) > 3}
        scored = []
        for idx, doc in enumerate(self.knowledge):
            overlap = len(tokens.intersection({t.strip('.,').lower() for t in doc.split()}))
            scored.append((overlap, idx))
        return [idx for _, idx in sorted(scored, reverse=True)[:3]]

    def suggest(self, profile: Dict, decisions: Dict[str, str]) -> List[str]:
        summary = (
            f"Age {profile['age']}, education {profile['education']}, skills {profile['skills']}, "
            f"stress {profile['stress_level']}, goal {profile['career_goals']}, family support {profile['family_support']}, "
            f"career choice {decisions['career_path']}, location {decisions['location']}, "
            f"study/work {decisions['study_work']}, money choice {decisions['invest_save']}"
        )

        if self._sk_ready:
            q_vec = self.vectorizer.transform([summary])
            scores = cosine_similarity(q_vec, self.doc_vectors)[0]
            top_idx = scores.argsort()[::-1][:3]
            recommendations = [self.knowledge[i] for i in top_idx]
        else:
            recommendations = [self.knowledge[i] for i in self._fallback_rank(summary)]

        custom = []
        if profile["stress_level"] >= 7:
            custom.append("Your stress is elevated—reduce simultaneous big bets and schedule weekly recovery time.")
        if decisions["career_path"] == "Startup" and profile["savings"] < 1_000_000:
            custom.append("Consider a hybrid path: keep income while validating startup traction milestones.")
        if decisions["location"] == "Abroad" and profile["family_support"] <= 4:
            custom.append("Before relocating, set up mentors, peer groups, and monthly emotional check-ins.")

        return recommendations + custom
