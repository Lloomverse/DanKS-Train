<p align="right">
  <a href="README.md">English</a> | <strong>简体中文</strong>
</p>

<p align="center">
  <a href="https://github.com/Lloomverse/DanKS-Train/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/Lloomverse/DanKS-Train/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://github.com/Lloomverse/DanKS-Train/releases/latest"><img alt="Release" src="https://img.shields.io/github/v/release/Lloomverse/DanKS-Train"></a>
  <a href="https://www.python.org/"><img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white"></a>
  <a href="LICENSE"><img alt="Apache-2.0 许可证" src="https://img.shields.io/badge/license-Apache--2.0-D22128"></a>
</p>

<h1 align="center">DanKS-Train：开源掼蛋 AI 与在线自博弈训练框架</h1>

<p align="center">
  <strong>完整开源、可自主训练的端到端掼蛋强化学习系统。</strong>
</p>

<p align="center">
  <a href="#快速上手">快速上手</a> ·
  <a href="#性能与基准">性能与基准</a> ·
  <a href="#模型对局评估">模型评估</a> ·
  <a href="#在线自博弈训练">在线训练</a> ·
  <a href="#仓库结构">仓库结构</a> ·
  <a href="docs/technical-report.zh-CN.md">技术报告</a>
</p>

---

## 概述

**DanKS-Train** 提供了一套可完全复现的掼蛋智能体训练与推理全链路栈：

- **完整官方规则引擎**（`guandan/`）：纯 Python 实现 108 张牌、组队协作、进贡/还贡、抗贡、级牌升级与合法动作生成。
- **组合动作结构化检索**（`versions/v3/DanKS/retrieval`）：针对掼蛋庞大的动作空间，通过手牌分解与结构评估筛选高质量 Top-10 战术动作子集。
- **高吞吐在线自博弈 PPO**（`versions/v3/DanKS/training`）：内存内牌桌模拟、批量 GPU 推理与 PPO 策略更新同机共置，**无需落盘中间数据集**。
- **开箱即用预训练权重**：直接通过 GitHub Releases 获取已验证的模型权重（`v0.1.0`）。

---

## 性能与基准

| 指标 | 测量值 | 说明 |
| --- | ---: | --- |
| **在线训练吞吐** | **~4,340** env step / s | 单卡 GPU 热身中位数（相比单进程朴素对局 225 step/s 提升约 19 倍） |
| **实战对局胜率** | **57.5%** 胜率 | 200 局对抗自博弈相比初始基线胜率（57.5% vs 42.5%，平均局分 +0.1117） |
| **规则引擎速度** | **>10,000** steps / s | 纯 Python 规则计算，包含完整升级赛制支持 |

> 详细训练动态（PPO loss、策略熵收敛、价值解释方差曲线）及测速方法，请参阅[技术报告](docs/technical-report.zh-CN.md)。

---

## 快速上手

### 1. 安装环境

支持 Linux、macOS 与 Windows（Python 3.10+）：

```bash
git clone https://github.com/Lloomverse/DanKS-Train.git
cd DanKS-Train

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install -e versions/v3

# 安装 PyTorch（选择对应平台的 CUDA 或 CPU 版本）
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
# 仅 CPU:
# python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
```

### 2. 下载预训练权重

获取官方 `v0.1.0` 权重文件：

```bash
mkdir -p checkpoints
curl -L -o checkpoints/top10_selector_v3_online.pt \
  https://github.com/Lloomverse/DanKS-Train/releases/download/v0.1.0/top10_selector_v3_online.pt
```

运行环境自检：

```bash
python examples/engine_quickstart.py
python examples/v3_model_smoke.py
```

### 3. 模型对局评估（互换座位自博弈）

对两个模型 Checkpoint 进行严格对战测试（每局轮换队伍以消除座位偏差）：

```bash
python -m DanKS.training.eval_vs_ckpt \
  --checkpoint-a checkpoints/init.pt \
  --checkpoint-b checkpoints/top10_selector_v3_online.pt \
  --games 200 \
  --device cuda:0
```

### 4. 运行单机在线自博弈训练

直接启动内存内共置的在线训练循环：

```bash
python -m DanKS.training.train_online \
  --output ./checkpoints/my_run \
  --batch-size 8192 \
  --device cuda:0
```

---

## 技术演进路线

仓库保留了三代核心技术实现的完整代码：

| 版本 | 重点 | 核心机制 | 关键入口 |
| --- | --- | --- | --- |
| **V1** | 结构化检索 | 启发式牌型分解 + NumPy 选择器 | [`ranker.py`](versions/v1/DanKS/retrieval/ranker.py) |
| **V2** | 学习型选择 | 扩展动作生成 + ONNX 选择器 | [`action_generator.py`](versions/v2/DanKS/retrieval/action_generator.py) |
| **V3** | 神经网络与 PPO | 记忆感知候选打分 + 在线自博弈 PPO | [`train_online.py`](versions/v3/DanKS/training/train_online.py) |

---

## 独立规则引擎使用

规则引擎可脱离强化学习环境独立运行：

```python
from guandan import Environment

game = Environment(first_player=0)
for seat in range(4):
    game.add_player(f"player-{seat}", seat)

messages = game.start()
assert all(len(player.hand_cards) == 27 for player in game.players)
```

---

## 仓库结构

```text
DanKS-Train/
├── guandan/               # 独立纯 Python 掼蛋规则引擎（108 张牌、官方规则）
├── versions/
│   ├── v1/DanKS/          # 第一代：结构化检索 + NumPy 选择器
│   ├── v2/DanKS/          # 第二代：学习型动作生成 + ONNX Runtime
│   └── v3/DanKS/          # 第三代：神经网络策略网络、检索与在线 PPO
├── examples/              # 可直接执行的环境自检与快速入门示例
├── docs/                  # 深度技术报告、架构分析与训练收敛曲线
├── assets/evidence/       # 原始指标 CSV 与对局评估结果数据
├── tests/                 # 单元测试与仓库集成测试
└── scripts/               # 训练指标导出与证据绘图脚本
```

---

## 贡献与开源许可

欢迎提交 Bug 报告、优化建议与 Pull Request！详情请参考 [CONTRIBUTING.md](.github/CONTRIBUTING.md)。

DanKS-Train 代码及其发布的模型权重均采用 [Apache 2.0 许可证](LICENSE)（允许商业与研究用途）。第三方依赖遵循其自身许可，详见 [NOTICE](NOTICE)。
