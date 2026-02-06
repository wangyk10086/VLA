"""Prototype LIS-VLA latent planning scaffolding.

This module provides a minimal, self-contained implementation skeleton for
LIS-VLA-style latent imagination + scoring + action selection. It is not wired
into training or deployment yet, but offers a concrete interface for future
integration with OpenVLA backbones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


Latent = List[float]
Action = List[float]


@dataclass
class ActionHead:
    """Action head stub that samples candidate action sequences."""

    horizon: int
    action_dim: int

    def sample_action_sequences(self, num_sequences: int) -> List[List[Action]]:
        """Return a list of action sequences, each length `horizon`.

        This placeholder returns zero vectors to keep the module dependency-free.
        """

        return [
            [[0.0 for _ in range(self.action_dim)] for _ in range(self.horizon)]
            for _ in range(num_sequences)
        ]


@dataclass
class LatentWorldHead:
    """Latent world model stub that rolls forward in latent space."""

    latent_dim: int

    def rollout(self, latent: Latent, actions: Sequence[Action]) -> Latent:
        """Roll forward the latent state given a sequence of actions."""

        if len(latent) != self.latent_dim:
            raise ValueError("Latent dimension mismatch.")
        next_latent = latent[:]
        for step in actions:
            scale = sum(step) if step else 0.0
            next_latent = [value + 0.01 * scale for value in next_latent]
        return next_latent


@dataclass
class VisualAlignmentScorer:
    """Cosine alignment scorer for latent states and goal embeddings."""

    def score(self, latent: Latent, goal_embedding: Latent) -> float:
        if len(latent) != len(goal_embedding):
            raise ValueError("Latent and goal embeddings must match in length.")
        dot = sum(a * b for a, b in zip(latent, goal_embedding))
        norm_a = sum(a * a for a in latent) ** 0.5
        norm_b = sum(b * b for b in goal_embedding) ** 0.5
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)


@dataclass
class LLMGuidedScorer:
    """Placeholder for LLM-based semantic evaluation."""

    def score(self, latent_summary: str, goal: str) -> float:
        """Return a heuristic score based on string overlap.

        In a real system this should call an LLM with a rubric prompt.
        """

        latent_tokens = set(latent_summary.lower().split())
        goal_tokens = set(goal.lower().split())
        if not goal_tokens:
            return 0.0
        return len(latent_tokens & goal_tokens) / len(goal_tokens)


@dataclass
class LatentMCTSPlanner:
    """Latent-space planner that selects the best first action."""

    action_head: ActionHead
    world_head: LatentWorldHead
    visual_scorer: VisualAlignmentScorer
    llm_scorer: LLMGuidedScorer
    num_candidates: int

    def plan(
        self,
        latent: Latent,
        goal_embedding: Latent,
        goal_text: str,
        latent_summaries: Iterable[str],
    ) -> Tuple[Action, float]:
        """Return the best first action and its combined score."""

        candidates = self.action_head.sample_action_sequences(self.num_candidates)
        best_action: Action = []
        best_score = float("-inf")
        summaries = list(latent_summaries)
        if len(summaries) != self.num_candidates:
            raise ValueError("Provide one latent summary per candidate.")

        for sequence, summary in zip(candidates, summaries):
            predicted = self.world_head.rollout(latent, sequence)
            visual_score = self.visual_scorer.score(predicted, goal_embedding)
            llm_score = self.llm_scorer.score(summary, goal_text)
            combined = 0.5 * visual_score + 0.5 * llm_score
            if combined > best_score:
                best_score = combined
                best_action = sequence[0] if sequence else []

        return best_action, best_score


def demo() -> None:
    """Minimal demo with placeholder latents and summaries."""

    action_head = ActionHead(horizon=3, action_dim=4)
    world_head = LatentWorldHead(latent_dim=8)
    visual_scorer = VisualAlignmentScorer()
    llm_scorer = LLMGuidedScorer()
    planner = LatentMCTSPlanner(
        action_head=action_head,
        world_head=world_head,
        visual_scorer=visual_scorer,
        llm_scorer=llm_scorer,
        num_candidates=2,
    )

    latent = [0.1 for _ in range(8)]
    goal_embedding = [0.2 for _ in range(8)]
    goal_text = "place the can into the recycling bin"
    summaries = [
        "robot moves can toward recycling bin",
        "robot drops can on floor",
    ]
    action, score = planner.plan(latent, goal_embedding, goal_text, summaries)
    print("best action:", action)
    print("score:", score)


if __name__ == "__main__":
    demo()
