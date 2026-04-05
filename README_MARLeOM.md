# MARLeOM Implementation Spec (This Branch)

This document gives a concise, code-accurate summary of the current MARLeOM branch.

## 1. Scope

1. MARLeOM training pipeline for predator-prey in Malmo (2 predators, 1 prey).
2. Persistent mission runtime with soft episode resets.
3. Opponent-modeling based CTDE agent with discrete SAC-style updates.
4. Module reorganization for cleaner model and environment structure.

## 2. Core Algorithm (Implemented)

Main implementation: [src/agents/marleom.py](src/agents/marleom.py)

1. CTDE setup:
	1. Per-predator decentralized actors.
	2. Per-predator centralized double critics.
	3. Replay-buffer off-policy training with target critics.
2. Opponent modeling:
	1. Level-0 model learns from observed prey actions.
	2. Level-1 model learns critic-guided pseudo best-response labels.
	3. Bayesian posterior-style weights are maintained for L0/L1 mixing.
3. Level-1 target generation:
	1. One-step exhaustive search over prey action combinations.
	2. Select action minimizing summed predator value under current critics.
4. Policy/entropy details:
	1. Auto-tuned alpha (SAC-v2 style).
	2. Target-entropy schedule with stability gating.
	3. Additional entropy regularization for stability.

## 3. Environment Runtime (Implemented)

Environment module: [src/envs/marleomEnv.py](src/envs/marleomEnv.py)

1. Mission is kept alive across episodes when possible.
2. Episode boundaries use soft resets (teleport/reset commands + health checks).
3. Restart path is fallback-only and triggered when mission is confirmed dead.
4. Start/restart paths include retry handling and cooldowns.

Mission config: [configs/missionPredatorPrey.xml](configs/missionPredatorPrey.xml)

1. Config is aligned for long-running sessions (forced-quit handlers disabled).
2. Agent layout is 2 predators + 1 prey.

## 4. Training Script Behavior

Training entrypoint: [experiments/trainMarleom.py](experiments/trainMarleom.py)

1. Warmup with random actions then MARLeOM policy updates.
2. Prey uses scripted policy during early episodes, then trainable prey actor.
3. Episode diagnostics include:
	1. critic/actor/OM losses,
	2. alpha and entropy metrics,
	3. level-mix and Bayesian mix weights.
4. Win reporting:
	1. exact cumulative win%,
	2. rolling-100 win% for smoother local trend.

## 5. Code Organization (Updated)

1. Model definitions moved to [src/models](src/models):
	1. [src/models/actorNetwork.py](src/models/actorNetwork.py)
	2. [src/models/centralisedCritic.py](src/models/centralisedCritic.py)
	3. [src/models/opponentModel.py](src/models/opponentModel.py)
2. MARLeOM environment renamed from old malmoEnv2 to [src/envs/marleomEnv.py](src/envs/marleomEnv.py).
3. Imports in agent/training code were updated to match these paths.

## 6. Known Simplifications vs Full Paper

1. No learned world model for long rollout-based opponent planning.
2. Opponent hierarchy depth is limited to L0/L1.
3. Level-1 best response is one-step critic-guided approximation (not full rollout planning).
