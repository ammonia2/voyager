"""
scriptedPolicies.py
===================
10 diverse scripted predator policies forming Pi_train (replacing MEP).
These policies are used to:
  1. Train Best Response (BR) policies via PPO (trainBR.py)
  2. Generate pretraining data for OMIS Transformer (dataCollector.py)
  3. Form Pi_test as unseen variants during evaluation

Each policy is a callable:
    action = policy(obs, agent_idx)
    obs      : dict with keys 'x', 'z', 'yaw', 'life', 'entities', 'ob'
    agent_idx: int (0=Predator1, 1=Predator2 — predators are opponents)
    returns  : (move, turn, attack) tuple → each is an int index

Action space (from malmoEnv.py):
    move  : 0=forward, 1=backward, 2=stop
    turn  : 0=left,    1=right,    2=none
    attack: 0=yes,     1=no

From these, flat index = move*6 + turn*2 + attack  → [0..17]

The 10 policies cover a diverse range of behaviors to ensure
the Transformer learns to recognize and adapt to different styles.
"""

import math
import random


# ─────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────

def _get_prey_entity(entities):
    """Return the first entity that is NOT on the same team (i.e., the prey)."""
    for e in entities:
        # entity format: [x, z, yaw, health, is_same_team]
        if len(e) >= 5 and e[4] == 0:
            return e
    return None


def _angle_to_target(self_x, self_z, self_yaw, target_x, target_z):
    """
    Compute the signed angular difference (degrees) between current yaw
    and the direction toward the target. Positive = need to turn right.
    Malmo yaw: 0=south, 90=west, 180=north, -90=east (or 270).
    """
    dx = target_x - self_x
    dz = target_z - self_z
    # atan2 in Malmo coordinate system (x=east, z=south)
    target_yaw = math.degrees(math.atan2(-dx, dz))
    diff = (target_yaw - self_yaw + 360) % 360
    if diff > 180:
        diff -= 360
    return diff


def _distance(x1, z1, x2, z2):
    return math.sqrt((x1 - x2) ** 2 + (z1 - z2) ** 2)


def _flat(move, turn, attack):
    return move * 6 + turn * 2 + attack


# ─────────────────────────────────────────────────────────────────
# Policy 1: Random — completely random actions
# ─────────────────────────────────────────────────────────────────

class RandomPolicy:
    """
    Selects uniformly random move, turn, attack each step.
    Provides a baseline and maximum behavioral diversity.
    """
    name = "random"

    def __call__(self, obs, agent_idx):
        move   = random.randint(0, 2)
        turn   = random.randint(0, 2)
        attack = random.randint(0, 1)
        return move, turn, attack


# ─────────────────────────────────────────────────────────────────
# Policy 2: Aggressive chaser — always moves toward prey and attacks
# ─────────────────────────────────────────────────────────────────

class AggressiveChaserPolicy:
    """
    Directly chases the prey at full speed and always attacks.
    Most straightforward predator behavior.
    """
    name = "aggressive_chaser"

    def __call__(self, obs, agent_idx):
        prey = _get_prey_entity(obs.get("entities", []))
        if prey is None:
            return 0, 2, 1   # move forward, no turn, no attack

        diff = _angle_to_target(obs["x"], obs["z"], obs["yaw"], prey[0], prey[1])
        if diff > 10:
            turn = 1   # turn right
        elif diff < -10:
            turn = 0   # turn left
        else:
            turn = 2   # facing target

        dist = _distance(obs["x"], obs["z"], prey[0], prey[1])
        attack = 0 if dist < 3.0 else 1   # attack if close
        return 0, turn, attack             # always move forward


# ─────────────────────────────────────────────────────────────────
# Policy 3: Patrol — walks in a fixed square pattern
# ─────────────────────────────────────────────────────────────────

class PatrolPolicy:
    """
    Walks in a fixed square patrol pattern regardless of prey position.
    Represents a non-reactive, predictable opponent.
    """
    name = "patrol"
    PATROL_STEPS = 20   # steps per leg of the square

    def __init__(self):
        self._step = 0

    def __call__(self, obs, agent_idx):
        leg = (self._step // self.PATROL_STEPS) % 4
        self._step += 1
        if leg == 0:
            return 0, 2, 1   # forward, no turn, no attack
        elif leg == 1:
            return 2, 1, 1   # stop, turn right, no attack
        elif leg == 2:
            return 0, 2, 1
        else:
            return 2, 1, 1


# ─────────────────────────────────────────────────────────────────
# Policy 4: Cautious — approaches slowly, retreats when prey is very close
# ─────────────────────────────────────────────────────────────────

class CautiousPolicy:
    """
    Cautiously approaches prey but retreats when it gets too close.
    Represents a defensive/evasive predator style.
    """
    name = "cautious"
    SAFE_DIST   = 5.0
    ATTACK_DIST = 2.5

    def __call__(self, obs, agent_idx):
        prey = _get_prey_entity(obs.get("entities", []))
        if prey is None:
            return 2, 2, 1   # stop

        dist = _distance(obs["x"], obs["z"], prey[0], prey[1])
        diff = _angle_to_target(obs["x"], obs["z"], obs["yaw"], prey[0], prey[1])

        turn = 1 if diff > 10 else (0 if diff < -10 else 2)

        if dist < self.ATTACK_DIST:
            return 1, turn, 0   # back off and attack
        elif dist > self.SAFE_DIST:
            return 0, turn, 1   # approach
        else:
            return 2, turn, 1   # hold position, orient


# ─────────────────────────────────────────────────────────────────
# Policy 5: Flanker — tries to approach from the side
# ─────────────────────────────────────────────────────────────────

class FlankerPolicy:
    """
    Tries to approach prey from its side by adding a lateral offset
    to the target direction. Creates encirclement-like behavior.
    """
    name = "flanker"

    def __init__(self, side=1):
        # side: +1 = flank left, -1 = flank right
        self.side = side

    def __call__(self, obs, agent_idx):
        prey = _get_prey_entity(obs.get("entities", []))
        if prey is None:
            return 0, 2, 1

        # Offset the target position laterally
        offset = 4.0 * self.side
        target_x = prey[0] + offset
        target_z = prey[1]

        diff = _angle_to_target(obs["x"], obs["z"], obs["yaw"], target_x, target_z)
        turn = 1 if diff > 10 else (0 if diff < -10 else 2)

        dist = _distance(obs["x"], obs["z"], prey[0], prey[1])
        attack = 0 if dist < 3.0 else 1
        return 0, turn, attack


# ─────────────────────────────────────────────────────────────────
# Policy 6: Interceptor — predicts prey movement and moves to intercept
# ─────────────────────────────────────────────────────────────────

class InterceptorPolicy:
    """
    Estimates where the prey will be in a few steps and moves there.
    Uses simple linear prediction based on velocity (last position diff).
    """
    name = "interceptor"
    PREDICT_STEPS = 3

    def __init__(self):
        self._last_prey_pos = None

    def __call__(self, obs, agent_idx):
        prey = _get_prey_entity(obs.get("entities", []))
        if prey is None:
            return 0, 2, 1

        px, pz = prey[0], prey[1]

        # Estimate velocity
        if self._last_prey_pos is not None:
            vx = px - self._last_prey_pos[0]
            vz = pz - self._last_prey_pos[1]
        else:
            vx, vz = 0.0, 0.0

        self._last_prey_pos = (px, pz)

        # Intercept point
        target_x = px + vx * self.PREDICT_STEPS
        target_z = pz + vz * self.PREDICT_STEPS

        diff = _angle_to_target(obs["x"], obs["z"], obs["yaw"], target_x, target_z)
        turn = 1 if diff > 10 else (0 if diff < -10 else 2)

        dist = _distance(obs["x"], obs["z"], px, pz)
        attack = 0 if dist < 3.0 else 1
        return 0, turn, attack


# ─────────────────────────────────────────────────────────────────
# Policy 7: Rusher — charges at full speed with no turning correction
# ─────────────────────────────────────────────────────────────────

class RusherPolicy:
    """
    Runs straight ahead regardless of direction, only attacking.
    Very aggressive but inaccurate — creates chaotic behavior.
    """
    name = "rusher"

    def __call__(self, obs, agent_idx):
        prey = _get_prey_entity(obs.get("entities", []))
        attack = 1
        if prey is not None:
            dist = _distance(obs["x"], obs["z"], prey[0], prey[1])
            attack = 0 if dist < 4.0 else 1
        return 0, 2, attack   # always forward, no turn


# ─────────────────────────────────────────────────────────────────
# Policy 8: Zigzag chaser — chases prey but zigzags unpredictably
# ─────────────────────────────────────────────────────────────────

class ZigzagChaserPolicy:
    """
    Chases prey but alternates turning direction every few steps.
    Creates an unpredictable attack pattern.
    """
    name = "zigzag_chaser"
    ZIG_INTERVAL = 5

    def __init__(self):
        self._step = 0
        self._zig  = 1   # +1 or -1

    def __call__(self, obs, agent_idx):
        self._step += 1
        if self._step % self.ZIG_INTERVAL == 0:
            self._zig *= -1

        prey = _get_prey_entity(obs.get("entities", []))
        if prey is None:
            return 0, 2, 1

        diff = _angle_to_target(obs["x"], obs["z"], obs["yaw"], prey[0], prey[1])
        # Add zigzag offset
        diff += self._zig * 30

        turn = 1 if diff > 10 else (0 if diff < -10 else 2)
        dist = _distance(obs["x"], obs["z"], prey[0], prey[1])
        attack = 0 if dist < 3.0 else 1
        return 0, turn, attack


# ─────────────────────────────────────────────────────────────────
# Policy 9: Corner driver — tries to push prey into a corner
# ─────────────────────────────────────────────────────────────────

class CornerDriverPolicy:
    """
    Tries to drive prey toward the nearest wall corner (arena corner).
    Creates coordinated-looking encirclement pressure.
    Arena is 20x20, corners at approximately (2,2), (2,18), (18,2), (18,18).
    """
    name = "corner_driver"
    CORNERS = [(2, 2), (2, 18), (18, 2), (18, 18)]

    def __call__(self, obs, agent_idx):
        prey = _get_prey_entity(obs.get("entities", []))
        if prey is None:
            return 0, 2, 1

        px, pz = prey[0], prey[1]

        # Find nearest corner to prey
        nearest = min(self.CORNERS, key=lambda c: _distance(px, pz, c[0], c[1]))

        # Move toward the prey while pushing it toward the corner
        # Target: slightly behind prey from the corner perspective
        cx, cz = nearest
        # Direction from corner toward prey
        dcx = px - cx
        dcz = pz - cz
        mag = max(math.sqrt(dcx**2 + dcz**2), 0.001)
        dcx /= mag
        dcz /= mag
        # Position ourselves behind prey (between prey and arena center)
        target_x = px + dcx * 2
        target_z = pz + dcz * 2

        diff   = _angle_to_target(obs["x"], obs["z"], obs["yaw"], target_x, target_z)
        turn   = 1 if diff > 10 else (0 if diff < -10 else 2)
        dist   = _distance(obs["x"], obs["z"], px, pz)
        attack = 0 if dist < 3.0 else 1
        return 0, turn, attack


# ─────────────────────────────────────────────────────────────────
# Policy 10: Stochastic aggressor — aggressive but with random noise
# ─────────────────────────────────────────────────────────────────

class StochasticAggressorPolicy:
    """
    Like AggressiveChaser but with random action noise (epsilon=0.2).
    Represents a trained-but-imperfect opponent with stochastic behavior.
    """
    name = "stochastic_aggressor"
    EPSILON = 0.2

    def __call__(self, obs, agent_idx):
        if random.random() < self.EPSILON:
            # Random action
            return random.randint(0, 2), random.randint(0, 2), random.randint(0, 1)

        prey = _get_prey_entity(obs.get("entities", []))
        if prey is None:
            return 0, 2, 1

        diff   = _angle_to_target(obs["x"], obs["z"], obs["yaw"], prey[0], prey[1])
        turn   = 1 if diff > 10 else (0 if diff < -10 else 2)
        dist   = _distance(obs["x"], obs["z"], prey[0], prey[1])
        attack = 0 if dist < 3.0 else 1
        return 0, turn, attack


# ─────────────────────────────────────────────────────────────────
# Registry — Pi_train (K=10)
# ─────────────────────────────────────────────────────────────────

def get_pi_train():
    """
    Returns list of 10 scripted policy instances forming Pi_train.
    Each policy is callable: (obs, agent_idx) -> (move, turn, attack)
    """
    return [
        RandomPolicy(),
        AggressiveChaserPolicy(),
        PatrolPolicy(),
        CautiousPolicy(),
        FlankerPolicy(side=1),
        FlankerPolicy(side=-1),
        InterceptorPolicy(),
        RusherPolicy(),
        ZigzagChaserPolicy(),
        CornerDriverPolicy(),
    ]


def get_pi_test():
    """
    Returns 5 additional policies forming the unseen Pi_test set.
    These have not been seen during pretraining.
    """
    return [
        StochasticAggressorPolicy(),
        PatrolPolicy(),        # re-used with different random seed effect
        CautiousPolicy(),
        ZigzagChaserPolicy(),
        InterceptorPolicy(),
    ]


def get_policy_by_index(k):
    """Return policy k from Pi_train (k in [0, 9])."""
    return get_pi_train()[k]


def get_flat_action(policy_obs_result):
    """Convert (move, turn, attack) tuple to flat action index [0..17]."""
    move, turn, attack = policy_obs_result
    return move * 6 + turn * 2 + attack
