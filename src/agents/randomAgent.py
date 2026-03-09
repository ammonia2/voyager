from __future__ import annotations
import random

class RandomAgent:
    """Samples random multidiscrete actions."""
    def __init__(self, nMove: int = 3, nTurn: int = 3, nAttack: int = 2):
        self.nMove   = nMove
        self.nTurn   = nTurn
        self.nAttack = nAttack

    def act(self) -> tuple[int, int, int]:
        return (
            random.randint(0, self.nMove - 1),
            random.randint(0, self.nTurn - 1),
            random.randint(0, self.nAttack - 1),
        )