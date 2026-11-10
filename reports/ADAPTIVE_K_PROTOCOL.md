# 成本感知 K 选择协议（v2，含实测数字）

投机解码的候选长度 K 不能只按接受率选择。本协议由 `specforge/calibrate_k.py` 与 `adaptive_k_benchmark.py` 实现，并已用真实运行数据填齐（`run_research_pipeline.py` 2026-11-09 复现）。

## 协议内容

1. **分片隔离：** prompt 池前一半为 calibration、后一半为 held-out；`calibrate_k.py` 强制校验输入 benchmark 的 `prompt_split` 字段，从非 calibration 数据选 K 直接报错。
2. **选择规则：** 在 calibration prompt 上测量每个 K 的墙钟 speedup 及其配对自助 95% CI，取 argmax 写入 `artifacts/k_policy_calibration.json`。
3. **成本模型与最优停时：** `adaptive_k_benchmark.py` 另测逐轮成本 C(K)（线性拟合 R²=0.986，calibration hazard=0.9918），以 `K*=argmin C(K)/(E[A(K)]+1)` 选固定 K，并用有限时域 Bellman 方程 `V(n)=min_K[C(K)+E[V(n-A-1)]]`（A 为截断几何接受数）求逐轮策略；固定 K 是该策略空间的子集。
4. **计时协议：** 所有策略（含 baseline）在同一时间窗内按 repeat 轮转交错测量，避免 CPU 频率漂移混入比较。

## 实测结果（校准/评估分离，greedy 模式，6 条 prompt × 64 token）

- 校准集选出 **K=32：2.351×，95% CI [2.11, 2.65]**，风险集接受率 98.66%。
- **held-out 复测 K=32：2.358×，95% CI [2.02, 2.78]**，接受率 98.40%，贪婪输出 100% 等价；相对 held-out oracle 的 regret 为 0。
- K=28：校准 2.337×，held-out 2.225×——历史 2.340× 落在该量级，但必须带 CI 报告。
- target token ratio 在 K=32 时约 114.6%，即草稿与 bonus 带来的额外 token 成本不可忽略。

## 随机模式实测（sample，T=1.0，同一分片协议）

- 校准选 K=32：**2.656×，CI [2.48, 2.81]**，风险集接受率 99.73%。
- **held-out 复测 K=32：2.723×，CI [2.64, 2.79]**；正确性判据是输出分布等价（TV=0.030，p=1.0），不是逐 token exact match——sample 模式下与贪婪 baseline 的文本差异属预期，exact 列不可引用。
- held-out K=28 单点 CI [1.39, 2.24]、censored 23.4%、回滚率 52%：随机模式单 K 波动更大，结论以校准选中点与 CI 为准。
- 两种模式的角色：**贪婪模式是正确性锚点**（唯一可硬断言序列等价），**随机模式是第二正式基准**（分布不变性 + 更高实测加速）。校准协议、CI 口径、风险集接受率在两种模式下完全一致。

## 分布偏移下的表现（held-out 三组，交错计时）

| 组 | hazard | 最优固定 K | Bellman 自适应 | 结论 |
|---|---:|---|---|---|
| easy | 0.983 | fixed_k32 = 2.226× | 2.279×（平均 K=20） | CI 重叠，差异不显著 |
| hard | 0.876 | fixed_k12 = 0.975× | 0.927×（平均 K=10.5） | 固定 K=28 跌至 **0.760× 负加速** |
| mixed | 0.934 | fixed_k12 = 1.481× | 1.430×（平均 K=18） | CI 重叠，差异不显著 |

## 面试表述

- 正确说法是“**硬件和模型相关的成本感知 K 选择 + Bellman 最优停时**”，并报告 calibration 与 held-out 两个集合的数字与 CI。
- 平稳 hazard 下固定 K 已接近最优（与 Bellman 最优性一致，固定 K 是策略子集）；自适应的可复现收益是**分布偏移时避免长 K 的负加速**，不是凭空提高峰值。
- 不应宣称“自适应 K 显著优于固定 K”——三组 CI 均重叠，实测结论只支持“不低于最优固定 K 且在偏移时更稳”。
- 旧版基于 draft softmax 置信度早停的策略（`adaptive_k_benchmark_v1_legacy.py`）已被取代：其动机来自一个被 v2 理论分析证伪的“hazard 随深度衰减”结论。
