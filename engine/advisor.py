from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class AdvisorBot:
    """Lightweight NLP advisor using TF-IDF semantic retrieval."""

    def __post_init__(self):
        self.knowledge = [
            "Build a six month emergency fund if stress is high and income is unstable.",
            "If your goal is global leadership, combining higher study abroad with industry internships is powerful.",
            "Startup path is strongest when your skills include product, sales, and resilience under uncertainty.",
            "Investing systematically beats idle savings for long horizons if risk tolerance is moderate to high.",
            "Family support can reduce burnout; design decisions around your emotional support system.",
            "Prioritize work experience before a degree when you need immediate cash flow and clarity.",
        ]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_vectors = self.vectorizer.fit_transform(self.knowledge)

    def suggest(self, profile: Dict, decisions: Dict[str, str]) -> List[str]:
        summary = (
            f"Age {profile['age']}, education {profile['education']}, skills {profile['skills']}, "
            f"stress {profile['stress_level']}, goal {profile['career_goals']}, family support {profile['family_support']}, "
            f"career choice {decisions['career_path']}, location {decisions['location']}, "
            f"study/work {decisions['study_work']}, money choice {decisions['invest_save']}"
        )
        q_vec = self.vectorizer.transform([summary])
        scores = cosine_similarity(q_vec, self.doc_vectors)[0]
        top_idx = scores.argsort()[::-1][:3]

        custom = []
        if profile["stress_level"] >= 7:
            custom.append("Your stress is elevated—reduce simultaneous big bets and build recovery time weekly.")
        if decisions["career_path"] == "Startup" and profile["savings"] < 1_000_000:
            custom.append("Consider a hybrid path: keep a job while validating startup traction milestones.")
        if decisions["location"] == "Abroad" and profile["family_support"] <= 4:
            custom.append("Plan emotional infrastructure before moving abroad: mentors, peer groups, and monthly check-ins.")

        return [self.knowledge[i] for i in top_idx] + custom
