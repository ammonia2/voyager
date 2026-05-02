# Multi-Agent Soft Actor-Critic (MASAC) with Parallel Distributed Computing (PDC)

This directory contains the implementation for **Algorithm 3: MASAC + PDC**, a novel approach combining Multi-Agent Soft Actor-Critic with Asynchronous Off-Policy Actor-Learner architecture in a 3D Minecraft Predator-Prey environment.

## Overview

Traditional MARL training in Minecraft (Project Malmo) suffers from extreme GPU idle time due to synchronous environment stepping. This implementation resolves the bottleneck using **Parallel Distributed Computing (PDC)**:
- **Workers (CPU):** N independent worker processes, each managing its own Malmo arena. Workers interact with the environment using a CPU-only inference copy of the policy and push joint transitions to a shared queue.
- **Learner (GPU):** A single dedicated learner process that constantly drains the queue into a Prioritized Replay Buffer and performs continuous backpropagation.
- **Broadcast:** The learner periodically broadcasts updated network weights back to the workers.

### Algorithm Details (MASAC)
- **CTDE Architecture:** Each predator has an independent `ActorNetwork` (using a shared `VoxelEncoder`), but training uses a `TwinQNetwork` (Centralised Critic) that observes the global concatenated state and joint actions.
- **Opponent Modeling:** The `VoxelEncoder` is co-trained with an auxiliary `OMHead` that predicts the scripted prey's next action, enriching the shared feature representation.
- **Maximum Entropy:** Temperature ($\alpha$) is automatically tuned per-agent to balance exploration and exploitation.

---

## Hardware Recommendations

Each worker controls **3 Minecraft clients** (2 Predators, 1 Prey). Running Minecraft instances is very RAM-intensive.

| RAM | Recommended Workers (`--numWorkers`) | Total Minecraft Clients |
|-----|---------------------------------------|-------------------------|
| 16 GB | `2` | 6 (ports 10000 - 10005) |
| 32 GB | `4` | 12 (ports 10000 - 10011) |

> **Note for 16GB RAM:** Ensure your client launch scripts restrict the JVM heap (e.g., `-Xmx512m -Xms256m`) otherwise Windows will run out of memory.

---

## Guide for Running the Algorithm

### 0. Prerequisites
Before running, ensure you have your Malmo Environment built and your Python environment activated.
```bash
# Example conda environment activation
conda activate marl-malmo
```
Make sure Project Malmo is installed at `C:\Malmo\Minecraft` (or update the path in `launch_6_clients.ps1`).

### 1. Launch the Minecraft Clients
Depending on your chosen number of workers, launch the corresponding PowerShell script. For **2 workers**, you need 6 clients.

Open a PowerShell window and run:
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run\launch_6_clients.ps1
```
Wait for all 6 Minecraft windows to load and reach the Main Menu.

### 2. Start the MASAC PDC Training
Open a new terminal, activate your Python environment, and start the script:

```bash
python experiments/trainMASAC.py --numWorkers 2
```

**Training Logs:**
Metrics are printed to the console and saved continuously to `checkpoints/masac/masac_metrics.jsonl`.
Checkpoints are saved in `checkpoints/masac/`.

### 3. Resuming Training
If training was interrupted, you can resume from the latest checkpoint (`masac_latest.pt`):
```bash
python experiments/trainMASAC.py --numWorkers 2 --resume
```

---

## How to Evaluate

Evaluation uses a single environment (3 clients) and runs the trained policies greedily against seen and unseen opponent (prey) behaviors to measure the generalization gap.

1. Ensure at least **3 Minecraft clients** are running (ports 10000, 10001, 10002).
2. Run the evaluation flag:
```bash
python experiments/trainMASAC.py --evalOnly
```

This will output a summary detailing Average Episode Return, Win Rate, and Opponent Action Prediction Accuracy across both seen and unseen prey policies.
