# 5G Network Slicing Paper Draft

This file is the current writing base for the paper. It focuses on the parts that are already supported by the repository files and verified outputs, namely the modeling backbone and the current result section. Literature polishing, final figure layout, and later-section language refinement can be added afterward.

## Working Title

- Chinese: 面向干扰耦合 5G 微基站网络的双时间尺度切片与功率控制分层优化方法
- English: A Two-Time-Scale Hierarchical Optimization Framework for Network Slicing and Power Control in Interference-Coupled 5G Micro-Base-Station Networks

## Scope And Data

- Scope: Questions 1 to 3 only.
- Background and modeling source:
  - `背景信息/5G环境开发设计文档 (1).md`
  - `背景信息/附录.pdf`
  - `背景信息/Reinforcement-Learning-Based_Network_Slicing_and_Resource_Allocation_for_Multi-Access_Edge_Computing_Networks.pdf`
- Question 1 data:
  - `channel_data等2个文件/channel_data.xlsx`
- Question 2 data:
  - `channel_data等2个文件(1)/channel_data.xlsx`
- Question 3 data:
  - `BS2等5个文件/BS1.xlsx`
  - `BS2等5个文件/BS2.xlsx`
  - `BS2等5个文件/BS3.xlsx`
  - `BS2等5个文件/taskflow.xlsx`

## Draft Abstract

针对第三次训练赛前 3 问，本文构建了一套面向论文写作的统一求解框架。对于问题 1，采用单微基站、单时刻静态切片划分的精确整数优化建模；对于问题 2，采用单微基站、有限时域、动态队列与信道演化下的滚动优化框架，并在 `q2_mpc.py` 中实现有限时域 MPC 基线；对于问题 3，针对多微基站频率复用、同频干扰耦合以及切片划分与功率控制联合决策问题，在 `q3_hierarchical_rl.py` 与 `q3_sb3.py` 中构建双时间尺度分层控制框架。模型遵循赛题给定的 `100 ms` 资源重配置周期与 `1 ms` 业务执行周期，并严格使用仓库中的原始 Excel 数据。现有结果表明，问题 2 的 `lookahead=2` MPC 方案在目标值和运行时间之间取得最佳折中，问题 3 的分层 PPO 在多 `seed` 复现实验中稳定优于轻量级基线，其中 `seed=17` 的联合评估目标值达到 `0.4994806473699616`，而 `seed=7 / 17 / 27` 三组结果的目标值均值为 `0.4959624942033084`，说明当前框架已经具备进入论文整理阶段的稳定性基础。

## 1. Problem Reframing

本文不是把赛题简单视为“资源块怎么分”，而是将前三问统一重述为一个在异构切片 SLA、随机到达、移动信道和多站干扰耦合约束下的优化问题。对应关系如下：

- 问题 1：单微基站、单时刻、静态资源切片，适合精确整数优化。
- 问题 2：单微基站、10 个决策窗口、动态队列与信道演化，适合有限时域滚动优化。
- 问题 3：多微基站频率复用与同频干扰耦合，需联合考虑切片 RB 分配和发射功率控制，适合采用分层强化学习框架。

## 2. Core Modeling Assumptions

以下假设均来自 `背景信息/5G环境开发设计文档 (1).md` 与代码实现的一致口径：

- 资源重配置周期为 `100 ms`，业务执行时间粒度为 `1 ms`。
- 每个微基站总 RB 数为 `50`。
- 第 1 问与第 2 问使用固定发射功率 `30 dBm`。
- 第 3 问允许发射功率在 `10 dBm` 到 `30 dBm` 之间连续调节。
- 当前问题 3 环境中，任务到达后的服务基站按最近微基站分配。这是显式建模假设，接入优化不在当前范围内。

## 3. Method Overview

### 3.1 Question 1

问题 1 是静态小规模资源分配问题，重点应是给出逻辑闭合、与切片 SLA 一致的整数优化表述，而不是用 RL 替代精确解。

### 3.2 Question 2

问题 2 在 `q2_mpc.py` 中实现为滚动 DP/MPC 框架。核心特征是：

- 每个 `100 ms` 决策窗口做一次资源切片动作。
- 在窗口内部依据业务到达和信道条件推进任务服务。
- 对窗口末仍未完成且超出约束的任务施加终端惩罚。

### 3.3 Question 3

问题 3 在 `q3_sb3.py` 中被分解为两个层次：

- 切片层：以 `MultiDiscrete` PPO 策略决定各基站切片 RB 预算。
- 功率层：在切片结果已知的前提下，以连续动作 PPO 策略决定基站-切片功率。

训练使用代理奖励，但最终对外报告以环境总结出的 `objective`、完成数、丢弃数、平均时延、平均功率和平均干扰比为准。

## 4. Current Verifiable Results

### 4.1 Question 2

基于以下结果文件：

- `outputs/q2_mpc/lookahead1.json`
- `outputs/q2_mpc/lookahead2.json`
- `outputs/q2_mpc/lookahead3.json`

当前可以确认：

- `lookahead=1`：`objective = 0.8492915076842188`
- `lookahead=2`：`objective = 0.8986439771627552`
- `lookahead=3`：`objective = 0.8877485078261866`

同时，从输出日志可见：

- `lookahead=2` 的运行时间约为 `36 s`
- `lookahead=3` 的运行时间约为 `645 s`

因此，问题 2 可以在论文中明确写成：`lookahead=2` 是当前模型下的主结果，因为它在目标值和计算代价之间给出了最佳折中，而继续增加前瞻深度并未带来有效收益。

### 4.2 Question 3

基于以下结果文件：

- `outputs/q3_rl/short_run_metrics.json`
- `outputs/q3_sb3/q3_combined_eval_fresh_10k.json`
- `outputs/q3_sb3/q3_combined_eval_seed17_10k.json`
- `outputs/q3_sb3/q3_combined_eval_seed27_10k.json`

当前可直接写入正文的结论为：

- 轻量级 numpy 基线的 `best_eval_objective = 0.4552520100347934`
- 分层 PPO 的三组 `10k` 结果分别为：
  - `seed=7`：`objective = 0.49188382075400083`
  - `seed=17`：`objective = 0.4994806473699616`
  - `seed=27`：`objective = 0.4965230144859628`
- 三组结果的目标值均值为 `0.4959624942033084`，标准差为 `0.0031266148786072943`

这说明：

- 分层 PPO 相比轻量级基线有稳定提升。
- `seed=17` 可作为主展示结果。
- 当前结果已具备从“继续大规模调参”转向“整理论文和展示材料”的条件。

## 5. Result Interpretation

现阶段最适合的论文表述不是“模型已经完美”，而是：

- URLLC 指标表现稳定，三组复现实验均实现 `0` 丢弃。
- eMBB 指标接近 `100 ms` 时延边界，说明模型在保障高吞吐的同时仍然处于紧约束状态。
- mMTC 仍然是最难服务的切片，但其平均时延在多 `seed` 结果中较旧版追踪结果已有改善。
- 当前最需要补的不是更多训练，而是将已获得的稳定结果组织成论文图表和论证结构。

## 6. What Can Be Written Now

现在就可以稳定推进的部分：

- 摘要初稿
- 问题重述与研究定位
- 系统模型与双时间尺度设定
- Q2 的 MPC 建模与结果分析
- Q3 的分层 RL 建模与结果分析
- 实验数据来源说明
- 当前结果的讨论与局限性

## 7. What Can Be Added Later

后续再补充的部分：

- 文献综述的最终压缩与润色
- Q1 的数值结果表格与图形排版
- 更完整的图题、表题和论文格式统一
- 结论与展望的最终措辞
- 若需要更强展示效果，可补一轮基于最佳 checkpoint 的图像美化，但这不是当前阻塞项
