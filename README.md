<p align="right">
  <strong>English</strong> | <a href="README.zh-CN.md">简体中文</a>
</p>

<p align="center">
  <a href="https://github.com/Lloomverse/DanKS-Train/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/Lloomverse/DanKS-Train/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://github.com/Lloomverse/DanKS-Train/releases/latest"><img alt="Release" src="https://img.shields.io/github/v/release/Lloomverse/DanKS-Train"></a>
  <a href="https://www.python.org/"><img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white"></a>
  <a href="LICENSE"><img alt="Apache-2.0 license" src="https://img.shields.io/badge/license-Apache--2.0-D22128"></a>
</p>

<h1 align="center">DanKS-Train: Open-Source GuanDan AI & Online Self-Play Framework</h1>

<p align="center">
  <strong>A fully open-source, self-trainable reinforcement learning framework for GuanDan (掼蛋).</strong>
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#benchmarks--performance">Benchmarks</a> ·
  <a href="#evaluation">Evaluation</a> ·
  <a href="#online-self-play-training">Online training</a> ·
  <a href="#repository-layout">Repository layout</a> ·
  <a href="docs/technical-report.md">Technical report</a>
</p>

---

## Overview

**DanKS-Train** provides an end-to-end, reproducible training and inference stack for **GuanDan (掼蛋)**, the popular four-player partnership card game:

- **Complete official rules engine** (`guandan/`): 108 cards, partner collaboration, tribute/return, promotion matches, and legal action generation in pure Python.
- **Combinatorial candidate retrieval** (`versions/v3/DanKS/retrieval`): Solves GuanDan's massive action space through structured decomposition into a compact set of Top-10 strategic actions.
- **High-throughput online PPO** (`versions/v3/DanKS/training`): In-memory colocated multi-table simulation, batched GPU inference, and PPO updates with **no disk I/O shuttling**.
- **Pretrained weights**: Download verified model checkpoints (`v0.1.0`) directly via GitHub Releases.

---

## Benchmarks & Performance

| Metric | Measurement | Notes |
| --- | ---: | --- |
| **Online Training Throughput** | **~4,340** env step / s | Warm median on a single GPU (~19× vs 225 step/s naive table baseline) |
| **Playing Strength** | **57.5%** win rate | 200-game head-to-head self-play vs init checkpoint (+0.1117 avg score) |
| **Rules Engine Speed** | **>10,000** steps / s | Standalone Python implementation with full promotion mechanics |

> For training curves (PPO loss, policy entropy, explained variance) and measurement methodology, see the [Technical Report](docs/technical-report.md).

---

## Quick Start

### 1. Installation

The framework supports Linux, macOS, and Windows with Python 3.10+.

```bash
git clone https://github.com/Lloomverse/DanKS-Train.git
cd DanKS-Train

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install -e versions/v3

# Install PyTorch (select CPU or your CUDA version)
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
# For CPU only:
# python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
```

### 2. Download Pretrained Weights

Download the official `v0.1.0` weights:

```bash
mkdir -p checkpoints
curl -L -o checkpoints/top10_selector_v3_online.pt \
  https://github.com/Lloomverse/DanKS-Train/releases/download/v0.1.0/top10_selector_v3_online.pt
```

Verify your environment with quick sanity checks:

```bash
python examples/engine_quickstart.py
python examples/v3_model_smoke.py
```

### 3. Checkpoint Evaluation (Head-to-Head)

Play two checkpoints against each other in seat-swapping self-play:

```bash
python -m DanKS.training.eval_vs_ckpt \
  --checkpoint-a checkpoints/init.pt \
  --checkpoint-b checkpoints/top10_selector_v3_online.pt \
  --games 200 \
  --device cuda:0
```

### 4. Online Self-Play Training

Run the single-machine colocated online training loop:

```bash
python -m DanKS.training.train_online \
  --output ./checkpoints/my_run \
  --batch-size 8192 \
  --device cuda:0
```

---

## Architectural Progression

DanKS includes the complete progression across three generations:

| Version | Focus | Core Technique | Key Entry Point |
| --- | --- | --- | --- |
| **V1** | Structural retrieval | Heuristic decomposition + NumPy selector | [`ranker.py`](versions/v1/DanKS/retrieval/ranker.py) |
| **V2** | Learned selection | Extended action generation + ONNX selector | [`action_generator.py`](versions/v2/DanKS/retrieval/action_generator.py) |
| **V3** | Neural policy & PPO | Memory-aware candidate scoring + Online PPO | [`train_online.py`](versions/v3/DanKS/training/train_online.py) |

---

## Standalone Rules Engine

The official GuanDan engine can be used independently without reinforcement learning dependencies:

```python
from guandan import Environment

game = Environment(first_player=0)
for seat in range(4):
    game.add_player(f"player-{seat}", seat)

messages = game.start()
assert all(len(player.hand_cards) == 27 for player in game.players)
```

---

## Repository Layout

```text
DanKS-Train/
├── guandan/               # Standalone Python rules engine (108 cards, official rules)
├── versions/
│   ├── v1/DanKS/          # Generation 1: Structural retrieval + NumPy selector
│   ├── v2/DanKS/          # Generation 2: Learned selection + ONNX runtime
│   └── v3/DanKS/          # Generation 3: Neural policy, retrieval, and online PPO
├── examples/              # Runnable smoke tests and quickstart scripts
├── docs/                  # In-depth technical report and training dynamics
├── assets/evidence/       # Raw metrics CSVs and self-play evaluation logs
├── tests/                 # Unit and repository integration test suite
└── scripts/               # Metric extraction and visualization scripts
```

---

## Contributing & License

Contributions, bug reports, and optimizations are welcome! Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md).

DanKS-Train is released under the [Apache 2.0 License](LICENSE). Third-party components retain their respective licenses; see [NOTICE](NOTICE).
