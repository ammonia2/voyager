# MARLeOM Paper vs This Implementation

This document summarizes what the paper describes and how the current codebase implements it.

## 1. What The Paper Says

The paper describes MARLeOM under CTDE and emphasizes opponent modeling with these key ideas:

1. Build an opponent modeling module to predict adversary behavior.
2. Use recursive reasoning to construct hierarchical opponent models (level-0, level-1, ..., level-m).
3. Use an environment model and rollout to approximate opponent best responses.
4. Fuse multi-level opponent models with Bayesian-weighted mixing.
5. In decentralized execution, each actor uses local observation plus predicted opponent actions.

## 2. What This Repository Implements

### 2.1 CTDE structure

Implemented in [src/agents/marleom.py](src/agents/marleom.py):

1. Decentralized actors for predator agents.
2. Centralized double critics consuming global state and joint actions.
3. Off-policy replay and target critics.

### 2.2 Opponent modeling currently implemented

Implemented across [src/agents/marleom.py](src/agents/marleom.py) and [src/agents/opponentModel.py](src/agents/opponentModel.py):

1. Two opponent models are maintained:
2. Level-0 model (behavioral imitation from observed prey actions).
3. Level-1 model (critic-guided pseudo best-response labels).
4. Actor input uses a mixed prediction from level-0 and level-1 models.

### 2.3 How level-1 is approximated here

In [src/agents/marleom.py](src/agents/marleom.py), level-1 targets are built by one-step exhaustive search over prey actions:

1. Keep predator actions fixed from replay.
2. Evaluate each prey action with current critics.
3. Choose the prey action minimizing summed predator value.
4. Train level-1 opponent model to predict that action from prey observation.

This is a practical approximation of rollout-based best-response reasoning.

## 3. Differences From Full Paper Method

The current implementation is intentionally lighter than the full method in the paper:

1. No learned environment transition model is trained in this repo for simulated rollouts.
2. Recursive depth is limited to two levels (L0 and L1), not arbitrary level-m.
3. Model fusion is fixed-weight mixing (`levelMix`) rather than Bayesian adaptive weighting.
4. Opponent policy in training script is still scripted prey behavior in [experiments/trainMarleom.py](experiments/trainMarleom.py), so adaptation pressure is lower than fully learning opponents.

## 4. Where To Look In Code

1. Training entrypoint: [experiments/trainMarleom.py](experiments/trainMarleom.py)
2. Main MARLeOM agent: [src/agents/marleom.py](src/agents/marleom.py)
3. Opponent model network: [src/agents/opponentModel.py](src/agents/opponentModel.py)
4. Critic network: [src/agents/centralisedCritic.py](src/agents/centralisedCritic.py)
5. Observation shaping for opponent information: [src/utils/obsUtils.py](src/utils/obsUtils.py)

## 5. Practical Notes

1. New checkpoints now store `oppModelL0` and `oppModelL1`.
2. Backward compatibility is included: older checkpoints with only `oppModel` can still be loaded.
3. TensorBoard now logs `Loss/oppModelL0` and `Loss/oppModelL1` in addition to total opponent loss.
