# MAPPO — Predator/Prey in Project Malmo

Multi-Agent PPO for a 2-predator vs 1-prey pursuit-evasion task running inside a 20×20 Malmo voxel arena.

---

## Environment

**Arena:** 20×20 voxel grid, agents spawn with minimum 8-unit separation.

**Agents:** 2 predators + 1 prey (3 total).

**Episode termination:** fixed at 500 steps, or when the Malmo mission dies.

**Step:**
- Predators send `move`, a continuous `turn` float in `[-1, 1]`, and `attack`.
- Prey sends `move` and a discrete `turn` (3 commands). Prey never attacks.

**Rewards:**

| Agent | Condition | Reward |
|---|---|---|
| Predator | per step | −0.3 |
| Predator | tag (within 3.5 units) | +10 |
| Prey | per step survived | +0.1 |
| Prey | tagged | −10 |

Tag credit is only given to a predator if it is within melee range when the prey takes damage.

---

## Observation Space

Each agent gets a flat vector of **147 dims** (`OBS_DIM`):

```
[self (4)] + [opponent × 2 (9 each)] + [voxel grid (125)]
```

- **Self (4):** normalised `pos_x`, `pos_z`, `yaw`, `life`
- **Opponent (9 each):** relative position (2), life (1), last-action one-hot (6)
  - Predator action encoding: `move_oh(3) + turn_cont(1) + attack_oh(2)`
  - Prey action encoding: `move_oh(3) + turn_oh(3)`
- **Voxel grid (125):** 5×5×5 block IDs around the agent, normalised to `[0,1]`

**Global state** (used by the centralised critic): all 3 obs concatenated → **441 dims**.

---

## Architecture

### VoxelEncoder (`voxelEncoder.py`)
Shared encoder used by all actor networks. Outputs a **128-dim** embedding.

Three parallel paths fused at the end:

**Voxel CNN path**
Embeds 125 block IDs (embedding dim 8) → reshape to `(B, 8, 5, 5, 5)` → two Conv3d layers → AdaptiveAvgPool to `(2,2,2)` → linear project to 64-dim.

**Opponent attention path**
Two opponent feature vectors `(B, 2, 9)` → linear embed to 32-dim each → dot-product attention using CNN features as query → attended 32-dim vector.

**Stats path**
Self features `(4,)` → linear → 16-dim.

Fusion: `cat(64, 32, 16)` → linear → ReLU → **128-dim output**.

---

### Predator Actor (`actorNetwork.py`)
One shared network across both predators.

```
flatObs (147) → VoxelEncoder (128) → MLP [128→128→128]
                                           ├── moveHead       → Categorical (3)
                                           ├── turnMeanHead   → tanh → Normal mean ∈ (−1,1)
                                           ├── turnLogStdHead → clamped → Normal std
                                           └── attackHead     → Categorical (2)
```

Log-prob at update time: `log π(move) + log π(turn) + log π(attack)`.

---

### OM Head (`omHead.py`)
Attached to the predator actor. Predicts the **prey's** next (move, turn) from encoder features, providing an auxiliary training signal that sharpens the encoder's opponent representations.

```
VoxelEncoder features (128) → Linear → ReLU (64)
                                          ├── moveHead → CE loss vs true prey move
                                          └── turnHead → CE loss vs true prey turn
```

OM loss is added to the predator actor loss with coefficient `omCoeff = 0.5`. Gradients flow back into `VoxelEncoder`.

---

### Prey Actor (`preyActorNetwork.py`)
Independent network, same encoder architecture as the predator but no OM head and no attack.

```
flatObs (147) → VoxelEncoder (128) → MLP [128→128→128]
                                           ├── moveHead → Categorical (3)
                                           └── turnHead → Categorical (3)
```

---

### Centralised Critic (`centralisedCritic.py`)
Single shared critic for **predator** advantage estimation.

```
globalState (441) → Linear → ReLU → Linear → ReLU → Linear → scalar V(s)
                    [256]            [256]
```

Targets are normalised online using a running mean/variance (`ValueNorm`). The critic is used for GAE; prey advantage uses a simpler GAE over its own per-step rewards with no separate critic.

---

## Training (`mappo.py`)

- **Algorithm:** PPO-Clip with GAE
- **Optimisers:** separate Adam for predator actor, critic, and prey actor (lr `3e-4`)
- **PPO epochs:** 4 per rollout, random mini-batch shuffling (batch size 64)
- **GAE:** γ = 0.99, λ = 0.95
- **Clip ε:** 0.2
- **Entropy bonus:** 0.01 (both predator and prey)
- **Value loss coefficient:** 0.5
- **OM loss coefficient:** 0.5
- **Gradient clipping:** max norm 0.5 on all networks

Update order per mini-batch: critic backward → predator actor + OM backward → prey actor backward → (optional distributed reduce) → clip + step all three optimisers.

Distributed training is supported: pass a `reduceFn` to `agent.update()` to all-reduce gradients before stepping.

---

## File Structure

```
src/
├── models/
│   ├── actorNetwork.py       # predator actor + OM loss wrapper
│   ├── preyActorNetwork.py   # prey actor (discrete move+turn)
│   ├── centralisedCritic.py  # global-state V(s)
│   ├── voxelEncoder.py       # shared CNN+attention encoder
│   ├── omHead.py             # opponent modelling auxiliary head
│   └── opponentModel.py      # standalone OM (used in QMIX baseline)
├── utils/
│   ├── obsUtils.py           # obs flattening, action one-hot helpers, constants
│   └── logs.py
├── mappo.py                  # MAPPO agent (update, action selection, save/load)
├── mappoEnv.py               # Malmo environment wrapper
├── rolloutBuffer.py          # on-policy rollout storage
├── replayBuffer.py           # off-policy buffer (QMIX baseline)
├── dataCollector.py
└── trainMappo.py             # training entry point
```

---

## Running

First, launch the Malmo client instances:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run\launch_6_clients.ps1
```

Then start training:

```bash
python trainMappo.py
```

Checkpoints are saved via `agent.save(path)` / loaded via `agent.load(path)`. The trainer handles rollout collection, buffer filling, and calling `agent.update(rollout)`.