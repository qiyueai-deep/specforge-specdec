# SpecForge 代码学习手册

## 1. 项目全景

普通自回归生成每个 token 都调用一次昂贵的 target。SpecForge 让廉价的 draft 先猜 K 个 token，再让 target 用一次分块前向验证该块；连续接受的 token 一次提交，首次拒绝时用 target 修正并回滚缓存。

```text
data.py / tokenizer.py → training.py → artifacts
                                      ↓
decoding.py ← model.py + cache.py ← correctness.py / benchmark.py / plots.py
```

最终入口 [pycharm_fast_train.py](pycharm_fast_train.py) 的 target 是 `d=256,L=6,H=8`（4,828,928 参数），draft 是 `d=80,L=2,H=4`（159,440 参数）。它们共享字符级词表，才能比较同一 token 的 p 与 q。

## 2. 训练：`specforge/training.py`

训练顺序是：

1. target 用下一个 token 交叉熵训练；
2. draft 用同一标签 SFT；
3. 固定混合蒸馏更新 draft。

最终损失为：

`L = αT² KL(softmax(z_target/T) || softmax(z_draft/T)) + (1-α) CE(z_draft,y)`

其中 `T=2.0`，`α=0.75`。target 在蒸馏阶段处于 `no_grad`，因此只更新 draft。最终预设不使用 CE-only 第二阶段：`two_stage_distillation=False`。图上的 loss 来自固定验证 batch；这样横向测量点可比较，不会把随机训练 batch 波动误判为趋势。

## 3. 模型与缓存：`model.py`、`cache.py`

`model.py` 实现 token/位置嵌入、多头自注意力和 MLP。增量前向时每层 cache 保存 key/value，形状为 `[batch, heads, length, head_dim]`。一次验证 K 个候选时，因果掩码确保块内第 i 个位置看不到未来候选。

`cache.truncate(new_length)` 是关键：若原缓存长 L、只接受 a 个候选，就截断至 `L+a`。target 和 draft 都必须回滚；否则错误 token 留在任一缓存，下一轮注意力上下文就与真实输出不一致。

## 4. 解码：`specforge/decoding.py`

草稿阶段逐 token 采样/贪婪提出候选，记录其分布 q。target 将整个候选块一次输入，得到验证分布。对齐关系容易 off-by-one：候选 0 用 prefill 的 `target_next` 验证；候选 i 用 `block_logits[i-1]` 验证，因为 `block_logits[i]` 已经看过候选 i。

贪婪模式接受条件是 `proposal == argmax(p)`；首次失败输出目标 argmax，因此其最终文本严格等价于 target-only。若全部接受，还能使用 target 已计算出的 bonus token。

随机模式接受概率是 `min(1,p(x)/q(x))`。拒绝时从 `normalize(max(p-q,0))` 采样。原因是接受贡献为 `min(p,q)`，未覆盖的目标质量正是 `max(p-q,0)`；两部分相加恢复 p。

## 5. 正确性与性能：`correctness.py`、`benchmark.py`

正确性检查分别验证：贪婪文本严格相同；随机接受+残差分布满足概率恒等式。性能不能只看接受率，还记录 target 调用减少、tokens/s 和墙钟 speedup。2026-08-27 固定单线程 CPU、64-token 离线基准中，K=28 是本轮最优点（2.340×）；K=32 因额外草稿与调度开销回落到 2.095×。

## 6. 图与调参：`plots.py`、`pycharm_fast_train.py`

`plots.py` 只读取本轮 `manifest.json` 和 `benchmark.json`，不手填数据。训练后 PNG 位于本轮 `experiments/.../figures/`。修改 `distill_steps`、`TEMPERATURE`、`DISTILL_ALPHA` 或模型尺寸都会改变结果，必须重新引用本轮结果，不能沿用旧数字。

推荐第一次实验仅改 `CANDIDATE_LENGTHS`，因为它不重新训练模型。`K` 取正整数；`NEW_TOKENS=64` 时不建议取大于 64，且应始终满足生成长度不超训练上下文范围。

## 7. 可观测性：`observability.py`

基准不再只输出一个 speedup。每次解码都通过 `GenerationStats` 记录 prefill 时间、TTFT、端到端时间、提议轮数、回滚次数和两个 KV Cache 的峰值 token 数；`observability.py` 将它们聚合为请求级 P50/P95 延迟、TTFT、回滚率、接受率和缓存峰值。指标写入 `benchmark.json` 的 `request_telemetry`，同时显示在 PyCharm 控制台。

当前是单进程 CPU 离线基线，所以 cache 以 token positions 统计，而不是虚构 GPU 显存数值。未来服务化时可把同一 schema 导出到 Prometheus / OpenTelemetry，并补充队列等待、并发数、GPU 显存和错误码。

## 8. 面试时的边界

这套实现验证的是推测解码算法、缓存状态和 CPU 工程闭环。程序化语料与微型模型不代表通用语言质量，也不能外推为 7B/70B 的 GPU 或线上服务收益。生产升级方向是接入真实同词表模型、Paged KV Cache、融合 kernel、量化和连续批处理。

## 9. v3 新增模块（2026-11-09）

**`specforge/stats.py`（纯标准库，无 SciPy 依赖）：** `RoundRecord` 记录每轮 `proposed/accepted/rejected/verified/target_calls/draft_steps`，`censored = proposed - verified`。核心函数：`verified_acceptance_rate`（风险集口径）、`risk_set_counts` + `kaplan_meier`（逐位置 hazard 与存活函数）、`constant_hazard_lrt`（似然比检验，渐近 χ² + 参数自助 p 值）、`wilson_interval`、`bootstrap_ci`、`pinsker_acceptance_lower_bound`、`total_variation`、`permutation_test`、`power_curve`。`tests/test_stats.py` 用真 i.i.d. 数据证明旧估计器会伪造 hazard 衰减而 KM 估计保持平稳。

**`specforge/costs.py`：** `CostModel` 存储逐 K 的 `round_seconds` 与 `single_token_cost`；`microbenchmark_components` 分离 C_single/C_draft/C_block(K)；`profile_round_costs` 从解码 trace 逐样本归一逐轮成本；`optimal_fixed_k` 用 `tau(K)=C(K)/(E[A(K)]+1)` 选固定 K；`DPKPlanner` 解有限时域 Bellman 方程 `V(n)=min_K[C(K)+E[V(n−A−1)]]`；`AdaptiveKController` 在线 Beta 平滑估计 hazard 并逐轮调 K。`tests/test_costs.py` 验证 DP 值不超过任何固定 K。

**`specforge/decoding.py` 的 `trace` 与 `k_policy`：** `speculative_generate` 新增两个参数，每个 proposal 轮写一条 `RoundRecord`（含 terminal round），`k_policy(remaining, generated)` 支持逐轮动态 K（compiled path 固定 K，因 graph 按 S=K padding）。所有分析与自适应策略共用这一份解码循环，消除重复实现与记账漂移。

**`specforge/distill_tv.py`：** `overlap_and_tv` 计算 `sum min(p,q)` 与 TV；`build_distill_loss` 支持 `ce_only/kl_ce/tv_ce/kl_tv_ce` 四种目标，默认 `kl_ce` 与原训练损失逐位等价（loss 与梯度 max diff 均为 0.0）；`alignment_metrics_extended` 同时报告 teacher-forced 对齐与自由生成接受率，分开暴露 exposure bias。

**`specforge/equivalence.py`：** `sample_sequences` 从两种解码器各采 N 条序列；`ngram_counter` 统计 unigram/bigram；`parametric_bootstrap_tv_pvalue` 在 H0（同分布）下从 pooled 分布重采样计算 TV 的零分布与 p 值；`tv_bootstrap_ci` 与 `power_at_effect` 给出 CI 和检验功效；`run_equivalence_suite` 串成一套。

**`specforge/data.py` rich 模式：** `TokenCorpus.create(mode="rich")` 用稀疏 Markov chain 生成高熵语料（successor 分布 (0.34,0.56,0.72,0.84,0.93)），`corpus_entropy_report` 输出 H1/H2/predictability，供 `entropy_stress_test.py` 做压力测试。

**`specforge/benchmark.py` 与 `specforge/calibrate_k.py`：** benchmark 增加 `--split calibration/held_out`、`--mode sample`、温度/top-k、`speedup_ci95`（配对 prompt 自助）、`target_token_ratio` 与风险集口径字段；`calibrate_k.py` 强制校验 `prompt_split`，从 calibration 选 K 后在 held-out 单独报告，oracle K 仅作诊断字段。

**研究脚本与一键复现：** `analyze_acceptance_theory.py`（生存分析 v2）、`adaptive_k_benchmark.py`（成本感知自适应 K，交错计时）、`run_distill_ablation.py`（4 目标 × 2 seed）、`tv_kl_report.py`（跨温度 TV/KL，区分贪婪/随机）、`run_equivalence_test.py`、`entropy_stress_test.py`、`run_attribution_control.py`（dispatch 归因）、`run_research_pipeline.py`（一键串行全部步骤与单元测试）。旧版错误口径脚本保留为 `*_v1_legacy.py` 快照，旧 v1 分析产物已被 v2 产物取代。
