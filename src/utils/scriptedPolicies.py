"""
scriptedPolicies.py
===================
Scripted policy registries for OMIS overhaul.

- Prey policies are the fixed opponents for training/collection/eval.
- Predator policies are kept as reference behavior sets.
"""

import math
import random
from typing import Tuple, Optional, Set, List

ARENA_MIN = 2.0
ARENA_MAX = 18.0
ARENA_CENTER = (10.0, 10.0)

TURN_BINS = 5


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _distance(x1: float, z1: float, x2: float, z2: float) -> float:
    return math.sqrt((x1 - x2) ** 2 + (z1 - z2) ** 2)


def _angle_to_target(self_x: float, self_z: float, self_yaw: float, target_x: float, target_z: float) -> float:
    """Signed yaw delta in degrees, positive means turn right."""
    dx = target_x - self_x
    dz = target_z - self_z
    target_yaw = math.degrees(math.atan2(-dx, dz))
    diff = (target_yaw - self_yaw + 360.0) % 360.0
    if diff > 180.0:
        diff -= 360.0
    return diff


def _turn_idx_from_diff(diff: float, threshold: float = 10.0) -> int:
    if diff > threshold:
        return 1  # right
    if diff < -threshold:
        return 0  # left
    return 2      # none


def _get_pos_yaw(obs: dict) -> Tuple[float, float, float]:
    pos = obs.get("pos", [0.0, 0.0])
    return float(pos[0]), float(pos[1]), float(obs.get("yaw", 0.0))


def _entity_name(entity: dict) -> str:
    return str(entity.get("name", ""))


def _entity_xz(entity: dict) -> Tuple[float, float]:
    return float(entity.get("x", 0.0)), float(entity.get("z", 0.0))


def _nearest_entity(obs: dict, name_filter: Set[str]) -> Optional[dict]:
    sx, sz, _ = _get_pos_yaw(obs)
    best = None
    best_d = float("inf")
    for entity in obs.get("nearbyEntities", []):
        if _entity_name(entity) not in name_filter:
            continue
        ex, ez = _entity_xz(entity)
        d = _distance(sx, sz, ex, ez)
        if d < best_d:
            best = entity
            best_d = d
    return best


def _nearest_predator(obs: dict) -> Optional[dict]:
    return _nearest_entity(obs, {"Predator1", "Predator2"})


def _nearest_prey(obs: dict) -> Optional[dict]:
    return _nearest_entity(obs, {"Prey1"})


def _clamp_to_arena(x: float, z: float) -> Tuple[float, float]:
    return max(ARENA_MIN, min(ARENA_MAX, x)), max(ARENA_MIN, min(ARENA_MAX, z))


# ---------------------------------------------------------------------------
# Predator policy set (kept as reference behaviors)
# ---------------------------------------------------------------------------


class RandomPredatorPolicy:
    name = "random_predator"

    def __call__(self, obs: dict, agent_idx: int) -> Tuple[int, int, int]:
        return random.randint(0, 2), random.randint(0, 2), random.randint(0, 1)


class AggressivePredatorPolicy:
    name = "aggressive_predator"

    def __call__(self, obs: dict, agent_idx: int) -> Tuple[int, int, int]:
        prey = _nearest_prey(obs)
        if prey is None:
            return 0, 2, 1
        sx, sz, syaw = _get_pos_yaw(obs)
        px, pz = _entity_xz(prey)
        diff = _angle_to_target(sx, sz, syaw, px, pz)
        turn = _turn_idx_from_diff(diff)
        attack = 0 if _distance(sx, sz, px, pz) <= 3.5 else 1
        return 0, turn, attack


class ZigzagPredatorPolicy:
    name = "zigzag_predator"

    def __init__(self):
        self._tick = 0
        self._zig = 1

    def __call__(self, obs: dict, agent_idx: int) -> Tuple[int, int, int]:
        self._tick += 1
        if self._tick % 5 == 0:
            self._zig *= -1

        prey = _nearest_prey(obs)
        if prey is None:
            return 0, 2, 1

        sx, sz, syaw = _get_pos_yaw(obs)
        px, pz = _entity_xz(prey)
        diff = _angle_to_target(sx, sz, syaw, px, pz) + 25.0 * self._zig
        turn = _turn_idx_from_diff(diff)
        attack = 0 if _distance(sx, sz, px, pz) <= 3.5 else 1
        return 0, turn, attack


class CautiousPredatorPolicy:
    name = "cautious_predator"

    def __call__(self, obs: dict, agent_idx: int) -> Tuple[int, int, int]:
        prey = _nearest_prey(obs)
        if prey is None:
            return 2, 2, 1

        sx, sz, syaw = _get_pos_yaw(obs)
        px, pz = _entity_xz(prey)
        dist = _distance(sx, sz, px, pz)
        diff = _angle_to_target(sx, sz, syaw, px, pz)
        turn = _turn_idx_from_diff(diff)

        if dist < 2.5:
            return 1, turn, 0
        if dist > 6.0:
            return 0, turn, 1
        return 2, turn, 1


class WallSweepPredatorPolicy:
    name = "wall_sweep_predator"

    def __call__(self, obs: dict, agent_idx: int) -> Tuple[int, int, int]:
        sx, sz, syaw = _get_pos_yaw(obs)
        nearest_wall_x = ARENA_MIN if abs(sx - ARENA_MIN) < abs(sx - ARENA_MAX) else ARENA_MAX
        tx, tz = nearest_wall_x, sz
        diff = _angle_to_target(sx, sz, syaw, tx, tz)
        turn = _turn_idx_from_diff(diff)

        prey = _nearest_prey(obs)
        attack = 1
        if prey is not None:
            px, pz = _entity_xz(prey)
            attack = 0 if _distance(sx, sz, px, pz) <= 3.0 else 1
        return 0, turn, attack


def get_pi_train() -> list:
    """Predator reference policies (legacy compatibility)."""
    return [
        RandomPredatorPolicy(),
        AggressivePredatorPolicy(),
        ZigzagPredatorPolicy(),
        CautiousPredatorPolicy(),
        WallSweepPredatorPolicy(),
        RandomPredatorPolicy(),
        AggressivePredatorPolicy(),
        ZigzagPredatorPolicy(),
        CautiousPredatorPolicy(),
        WallSweepPredatorPolicy(),
    ]


def get_pi_test() -> list:
    """Predator reference policies used as unseen behavior set."""
    return [
        AggressivePredatorPolicy(),
        ZigzagPredatorPolicy(),
        WallSweepPredatorPolicy(),
        CautiousPredatorPolicy(),
    ]


def get_policy_by_index(k: int):
    return get_pi_train()[k]


# ---------------------------------------------------------------------------
# Prey evasion policy set (new Pi_train_prey)
# ---------------------------------------------------------------------------


class RandomPreyPolicy:
    name = "random_prey"

    def __call__(self, obs: dict, agent_idx: int) -> Tuple[int, int]:
        return random.randint(0, 2), random.randint(0, 2)


class FleeNearestPolicy:
    name = "flee_nearest"

    def __call__(self, obs: dict, agent_idx: int) -> Tuple[int, int]:
        predator = _nearest_predator(obs)
        if predator is None:
            return 0, 2
        sx, sz, syaw = _get_pos_yaw(obs)
        px, pz = _entity_xz(predator)
        target_x = sx + (sx - px)
        target_z = sz + (sz - pz)
        diff = _angle_to_target(sx, sz, syaw, target_x, target_z)
        return 0, _turn_idx_from_diff(diff)


class ZigzagFleePolicy:
    name = "zigzag_flee"

    def __init__(self):
        self._tick = 0
        self._zig = 1

    def __call__(self, obs: dict, agent_idx: int) -> Tuple[int, int]:
        self._tick += 1
        if self._tick % 5 == 0:
            self._zig *= -1
        predator = _nearest_predator(obs)
        if predator is None:
            return 0, 2
        sx, sz, syaw = _get_pos_yaw(obs)
        px, pz = _entity_xz(predator)
        target_x = sx + (sx - px)
        target_z = sz + (sz - pz)
        diff = _angle_to_target(sx, sz, syaw, target_x, target_z) + 25.0 * self._zig
        return 0, _turn_idx_from_diff(diff)


class WallHugPolicy:
    name = "wall_hug"

    def __call__(self, obs: dict, agent_idx: int) -> Tuple[int, int]:
        sx, sz, syaw = _get_pos_yaw(obs)
        candidates = [
            (ARENA_MIN, sz),
            (ARENA_MAX, sz),
            (sx, ARENA_MIN),
            (sx, ARENA_MAX),
        ]
        tx, tz = min(candidates, key=lambda p: _distance(sx, sz, p[0], p[1]))
        diff = _angle_to_target(sx, sz, syaw, tx, tz)
        return 0, _turn_idx_from_diff(diff)


class CenterHoldPolicy:
    name = "center_hold"

    def __call__(self, obs: dict, agent_idx: int) -> Tuple[int, int]:
        sx, sz, syaw = _get_pos_yaw(obs)
        predator = _nearest_predator(obs)
        if predator is None:
            diff = _angle_to_target(sx, sz, syaw, ARENA_CENTER[0], ARENA_CENTER[1])
            return 0, _turn_idx_from_diff(diff)

        px, pz = _entity_xz(predator)
        if _distance(sx, sz, px, pz) < 4.0:
            tx = sx + (sx - px)
            tz = sz + (sz - pz)
        else:
            tx, tz = ARENA_CENTER
        diff = _angle_to_target(sx, sz, syaw, tx, tz)
        return 0, _turn_idx_from_diff(diff)


class SpinPolicy:
    name = "spin"

    def __init__(self):
        self._right = True

    def __call__(self, obs: dict, agent_idx: int) -> Tuple[int, int]:
        turn = 1 if self._right else 0
        self._right = not self._right
        return 0, turn


class StochasticFleePolicy:
    name = "stochastic_flee"

    def __call__(self, obs: dict, agent_idx: int) -> Tuple[int, int]:
        if random.random() < 0.2:
            return random.randint(0, 2), random.randint(0, 2)
        return FleeNearestPolicy()(obs, agent_idx)


class CornerToCornerPolicy:
    name = "corner_to_corner"
    CORNERS = [(2.0, 2.0), (2.0, 18.0), (18.0, 18.0), (18.0, 2.0)]

    def __init__(self):
        self._corner_idx = 0

    def __call__(self, obs: dict, agent_idx: int) -> Tuple[int, int]:
        sx, sz, syaw = _get_pos_yaw(obs)
        cx, cz = self.CORNERS[self._corner_idx]
        if _distance(sx, sz, cx, cz) < 2.0:
            self._corner_idx = (self._corner_idx + 1) % len(self.CORNERS)
            cx, cz = self.CORNERS[self._corner_idx]
        diff = _angle_to_target(sx, sz, syaw, cx, cz)
        return 0, _turn_idx_from_diff(diff)


class ReactiveDodgePolicy:
    name = "reactive_dodge"

    def __call__(self, obs: dict, agent_idx: int) -> Tuple[int, int]:
        predator = _nearest_predator(obs)
        if predator is None:
            return 0, 2

        sx, sz, syaw = _get_pos_yaw(obs)
        px, pz = _entity_xz(predator)
        vec_x = sx - px
        vec_z = sz - pz
        perp_x, perp_z = -vec_z, vec_x
        tx = sx + perp_x
        tz = sz + perp_z
        tx, tz = _clamp_to_arena(tx, tz)
        diff = _angle_to_target(sx, sz, syaw, tx, tz)
        return 0, _turn_idx_from_diff(diff)


class SlowFleePolicy:
    name = "slow_flee"

    def __call__(self, obs: dict, agent_idx: int) -> Tuple[int, int]:
        predator = _nearest_predator(obs)
        if predator is None:
            return 1, 2
        sx, sz, syaw = _get_pos_yaw(obs)
        px, pz = _entity_xz(predator)
        tx = sx + (sx - px)
        tz = sz + (sz - pz)
        diff = _angle_to_target(sx, sz, syaw, tx, tz)
        return 1, _turn_idx_from_diff(diff)


def get_pi_train_prey() -> list:
    """Primary 10-policy prey evasion set for OMIS training/collection."""
    return [
        RandomPreyPolicy(),
        FleeNearestPolicy(),
        ZigzagFleePolicy(),
        WallHugPolicy(),
        CenterHoldPolicy(),
        SpinPolicy(),
        StochasticFleePolicy(),
        CornerToCornerPolicy(),
        ReactiveDodgePolicy(),
        SlowFleePolicy(),
    ]


def get_pi_test_prey() -> list:
    """Unseen prey variants for generalization evaluation."""
    return [
        StochasticFleePolicy(),
        ReactiveDodgePolicy(),
        SlowFleePolicy(),
        CornerToCornerPolicy(),
        WallHugPolicy(),
    ]


def get_prey_policy_by_index(k: int):
    return get_pi_train_prey()[k]


# ---------------------------------------------------------------------------
# Action conversion helpers
# ---------------------------------------------------------------------------


def get_flat_action(pred_action: Tuple[int, int, int]) -> int:
    """Legacy predator 18-action flatten: (move, turn3, attack) -> [0, 17]."""
    move, turn, attack = pred_action
    return int(move) * 6 + int(turn) * 2 + int(attack)


def decode_flat_action(flat_idx: int) -> Tuple[int, int, int]:
    """Legacy predator 18-action decode."""
    action = int(flat_idx)
    attack = action % 2
    turn = (action // 2) % 3
    move = action // 6
    return move, turn, attack


def encode_pred_action_30(move: int, turn_bin: int, attack: int) -> int:
    """Predator action flatten for OMIS embedding space [0, 29]."""
    return int(move) * (TURN_BINS * 2) + int(turn_bin) * 2 + int(attack)


def decode_pred_action_30(flat_idx: int) -> Tuple[int, int, int]:
    """Predator action decode from OMIS embedding space [0, 29]."""
    action = int(flat_idx)
    attack = action % 2
    turn_bin = (action // 2) % TURN_BINS
    move = action // (TURN_BINS * 2)
    return move, turn_bin, attack


def encode_prey_action_9(move: int, turn: int) -> int:
    """Prey action flatten (move, turn) -> [0, 8]."""
    return int(move) * 3 + int(turn)


def decode_prey_action_9(flat_idx: int) -> Tuple[int, int]:
    """Prey action decode from [0, 8] to (move, turn)."""
    action = int(flat_idx)
    turn = action % 3
    move = action // 3
    return move, turn


def turn3_to_turn_bin5(turn_idx: int) -> int:
    """Map discrete 3-turn command index to closest 5-bin turn index."""
    mapping = {0: 0, 1: 4, 2: 2}
    return mapping[int(turn_idx)]


def turn_bin5_to_cont(turn_bin: int) -> float:
    idx = max(0, min(TURN_BINS - 1, int(turn_bin)))
    return (idx / (TURN_BINS - 1)) * 2.0 - 1.0


def convert_actions_flat_to_tuple(
    actions_flat: List[int],
    predator_indices=[0, 1],
    prey_indices=[2],
) -> List[Tuple]:
    """
    Legacy converter from old 18-flat actions to env tuple format.

    Predators become (move, turn_cont, attack) with turn mapped from 3 discrete values.
    Prey becomes (move, turn_idx, 0).
    """
    actions_tuple = []
    for agent_idx, flat_action in enumerate(actions_flat):
        move, turn, attack = decode_flat_action(int(flat_action))
        if agent_idx in predator_indices:
            turn_bin = turn3_to_turn_bin5(turn)
            actions_tuple.append((move, turn_bin5_to_cont(turn_bin), attack))
        else:
            actions_tuple.append((move, turn, 0))
    return actions_tuple
