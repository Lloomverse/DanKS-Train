<p align="right">
  <a href="technical-report.md">English</a> | <strong>简体中文</strong>
</p>

# DanKS-Train 技术报告：系统架构与高吞吐在线自博弈训练

本报告详细说明 **DanKS-Train** 的架构设计、在线自博弈强化学习流程及实验训练动态。

---

## 1. 系统架构

掼蛋是非完美信息、双人协作对抗的棋牌博弈，早期回合的合法动作空间常超过 $10^6$。传统的离散策略网络无法直接对如此庞大的空间进行打分。

DanKS 采用四阶段解耦架构：

<p align="center">
  <img
    src="../assets/danks-overall-architecture.png"
    alt="DanKS 总体处理流水线：从信息状态到结构化动作检索与 PPO 自博弈"
    width="920"
  />
</p>

1. **信息状态编码**：编码玩家私有手牌、公开出牌历史（出牌人、牌型、出牌次序）、当前升级级牌及对局上下文。
2. **带预算结构化检索**：将手牌分解为合法牌型，从中抽取出兼顾多样性与进攻性的 Top-10 战术动作子集。
3. **候选动作打分**：共享神经网络将信息状态与候选动作联合评估，输出动作对数几率（logits）及状态价值估计。
4. **轨迹优化**：由自博弈产生的轨迹计算广义优势估计（GAE），执行带截断的 PPO 策略更新。

### 延迟结构结果

掼蛋的关键难点在于：当前看起来很小或贪心的出牌，可能会破坏后续的关键牌型（如拆散同花顺、炸弹）；而提早打出大牌，有时反而能保护残牌结构并创造更顺畅的走牌路线。

<p align="center">
  <img
    src="../assets/structure-aware-delayed-outcomes.png"
    alt="同一局势下三个候选动作导致的不同延迟结构结果"
    width="680"
  />
</p>

DanKS 为每个候选动作显式提取出牌后剩余手牌的结构特征，由 Critic 结合整局结局进行长期信用分配。

---

## 2. 高吞吐共置在线自博弈 PPO

传统的复杂棋牌强化学习系统通常将模拟采样（Actor）与模型训练（Learner）通过磁盘（如生成 `.npz` 文件）或分布式消息中间件解耦。在需要频繁合法性检验与特征提取的棋牌环境中，磁盘 I/O 与进程通信往往成为速度瓶颈。

DanKS-Train 采用内存内**共置训练架构**：
- 牌桌模拟、结构化动作检索与批量 GPU 推理在同一节点内协同调度，共享内存与显卡资源。
- 采样轨迹保存在环形缓冲区中，直接提供给 PPO 小批量训练消费，无需中间落盘。

### 吞吐对比

<p align="center">
  <img
    src="../assets/online-training-speedup.png"
    alt="朴素在线对局与共置在线训练流水线吞吐对比"
    width="720"
  />
</p>

| 架构 | env step / s | 相对吞吐 |
| --- | ---: | ---: |
| 朴素多桌在线循环 | 225.19 | 1.0× |
| **共置在线 PPO 流水线** | **~4,340**（热身中位数） | **~19.3×** |

*注：基线为未批量化的单进程对局采样；共置数字为单 GPU 长跑的热身中位数。*

---

## 3. 训练动态与收敛稳定性

### 长期运行的吞吐稳定性

共置在线训练流水线在数万次 PPO 更新中维持高吞吐，无内存泄漏或工作进程衰减。

<p align="center">
  <img
    src="../assets/online-training-throughput.png"
    alt="长跑在线训练中吞吐指标随 PPO 更新的变化"
    width="860"
  />
</p>

### 训练动态曲线

下图展示了在线自博弈长跑过程中核心强化学习指标的变化趋势：

<p align="center">
  <img
    src="../assets/online-training-convergence.png"
    alt="在线训练动态：PPO loss、策略熵与解释方差"
    width="860"
  />
</p>

- **PPO Loss（上图）**：持续在 0.35 附近平稳振荡，无发散或梯度爆炸。
- **策略熵（中图）**：从 ~0.20 逐步收敛到 ~0.08，表明模型由早期的广泛探索逐步过渡到确定的战术决策。
- **解释方差（下图）**：稳定保持在 0.58 ~ 0.61 区间，验证价值网络在整个训练期间均保持良好的拟合能力。

---

## 4. 棋力评估（200 局自博弈）

为脱离训练 Loss 真实评估策略提升，我们组织了初始 Baseline 权重与 `v0.1.0` 发布的续训权重之间的 200 局对抗比赛。

为消除座位先手偏差，两方模型在每局结束后交换队伍座位（0/2 与 1/3）。

<p align="center">
  <img
    src="../assets/online-training-strength.png"
    alt="初始 checkpoint 与续训在线权重 200 局自博弈胜率对比"
    width="640"
  />
</p>

| Checkpoint | 胜率 | 平均局分 |
| --- | ---: | ---: |
| 初始基线权重 | 42.5% | -0.1117 |
| **续训在线权重（`v0.1.0`）** | **57.5%** | **+0.1117** |

*评测命令：`python -m DanKS.training.eval_vs_ckpt`（固定随机种子 2026）。*

---

## 5. 证据复现

原始指标与绘图脚本均包含在仓库中：

- 指标日志：`assets/evidence/online_training_metrics.csv`
- 评测结果：`assets/evidence/eval_init_vs_latest_200.json`
- 速度对比：`assets/evidence/speed_comparison.csv`

复现图表：

```bash
# 从训练日志导出采样指标
python scripts/export_online_metrics.py \
  --log /path/to/online.log \
  --out-csv assets/evidence/online_training_metrics.csv \
  --stride 20

# 重新生成吞吐、加速比与收敛曲线图
python scripts/plot_open_source_evidence.py \
  --metrics-csv assets/evidence/online_training_metrics.csv \
  --out-dir assets \
  --baseline-steps-per-sec 225.19 \
  --baseline-label $'Naive online\n(tables)' \
  --current-label $'Current online\n(colocated+PPO)'
```
