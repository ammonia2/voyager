# Project Structure

```
voyager/
├── src/
│   ├── agents/
│   │   └── randomAgent.py        # random action sampler
│   ├── envs/
│   │   └── malmoEnv.py           # 4-agent Malmo env wrapper
│   ├── models/
│   │   ├── voxelEncoder.py       # CNN + entity attention encoder
│   │   └── opponentModelingHead.py  # predicts opponent actions
│   └── utils/
├── configs/
│   └── missionPredatorPrey.xml   # Malmo mission definition
└── experiments/
    └── testRandomRollout.py      # random episode test
```

---

# Running

### 1. Set env variable (each new terminal)
```cmd
set MALMO_XSD_PATH=C:\Malmo\Schemas
```

### 2. Launch 4 Minecraft clients (4 separate terminals)
```cmd
conda activate marl-malmo
cd /d C:\Malmo\Minecraft
gradlew.bat runClient -Pport=10000
```
Repeat for ports `10001`, `10002`, `10003`. Wait for all 4 to reach the main menu.

### 3. Run test
```cmd
conda activate marl-malmo
cd /d D:\projects\voyager
python experiments/testRandomRollout.py
```

---

# Agents & Roles

| Index | Name       | Role     |
|-------|------------|----------|
| 0     | Predator1  | Server host (role 0, start first) |
| 1     | Predator2  | Predator |
| 2     | Prey1      | Prey     |
| 3     | Prey2      | Prey     |

---

# Rewards (computed in wrapper, not XML)

| Agent    | Reward |
|----------|--------|
| Predator | +5 per prey HP damaged, -5 friendly fire, -0.1 per step |
| Prey     | +0.1 per step survival, -5 per HP damaged |

---

# Action Space (multidiscrete per agent)

| Head   | Actions |
|--------|---------|
| Move   | forward / backward / stop |
| Turn   | left / right / none |
| Attack | yes / no |

---

# Notes
- Role 0 (Predator1) hosts the mission server — always starts first with a 30s delay
- Observation: 7x7 voxel grid (partial) + nearby entities within 7x7 range
- Arena: 20x20 walled, flat stone floor