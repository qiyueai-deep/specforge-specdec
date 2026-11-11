# SpecForge：知识蒸馏与推测解码 CPU 推理优化系统

一个可在普通 Windows CPU 上从零训练、验证和出图的微型 Transformer 推测解码项目。它用于复现算法与工程闭环，不是生产大模型服务。

## 作品集定位（大模型算法岗）

本项目面向大模型算法/推理优化岗位的求职作品集场景，考察点是三件事：

1. **机制理解与工程闭环：** 从零实现推测解码（贪婪严格等价 + 随机拒绝采样 + KV Cache truncate 回滚）、温度蒸馏与请求级遥测，全部代码可读、可跑、可复现（3 分钟从零训练，一键流水线复现全部结论）。
2. **统计与建模能力：** 修正接受率统计中的右删失错误（风险集口径 + Kaplan–Meier + 似然比检验），用实测成本 C(K) 与有限时域 Bellman 最优停时建模自适应 K，并配 calibration/held-out 分离协议——这是与"调参跑分"拉开差距的核心。
3. **诚实评测素养：** 高熵语料压力测试、随机模式分布等价检验、加速来源归因、全程置信区间。每个数字都带边界，不把低熵语料上的收益包装成普适结论。

**5 分钟审阅路径：** 本节 → [PROJECT_BRIEF.md](PROJECT_BRIEF.md)（简历口径）→ `analysis/adaptive_k_v2.md` 与 `analysis/entropy_stress.md`（核心创新证据）→ `specforge/stats.py` 与 `specforge/costs.py`（核心代码）。完整验证证据见 [VALIDATION_REPORT.md](VALIDATION_REPORT.md)。

## 最终版本事实源

最终推荐入口是 [pycharm_fast_train.py](pycharm_fast_train.py)：在 PyCharm 中直接点击绿色 Run 即可从零训练、完成正确性检查和 K 消融，并自动打开三张图。最终预设为：

| 项目 | 最终配置 / 固定环境实测 |
|---|---|
| Target / Draft | 4,828,928（d=256,L=6,H=8）/ 159,440（d=80,L=2,H=4）参数 |
| 参数压缩比 | 30.29× |
| 蒸馏 | SFT → 固定混合蒸馏，T=2.0，α=0.75 |
| 训练配置 | 220 target steps、120 draft SFT steps、220 distill steps，seq_len=96 |
| 运行时间 | 103.8 秒（2026-08-27 本机完整复测；目标小于 3 分钟） |
| 对齐 | Top-1 agreement 98.35%，KL(T\|\|S)=0.0480 |
| 正确性 | 贪婪输出与 target-only 100% 等价 |
| 最优基准 | K=28：接受率 90.63%，目标调用减少 90.10%，2.340× 加速 |

性能结果是在固定**单线程 CPU**、6 条 prompt、prompt 长度 18、每条生成 64 token、warmup=1、repeats=2 的自建离线基准上记录。CPU、PyTorch 版本和频率策略不同，实际速度会变化，应在本机重测。

> **口径说明（v3）：** 上表 2.340× 与 90.63% 为历史实测，保留作为复现记录。其中 90.63% 使用旧口径 `accepted/proposed`，把右删失 token 计入拒绝；正式口径已改为风险集接受率 `accepted/verified`（约 98.6%–99.2%）。正式性能数字请使用 [PROJECT_BRIEF.md](PROJECT_BRIEF.md) 与下文 v2 分析产物。

## v3 研究改造（2026-11-09）

在原实现之上完成六个方向的改造，全部结论由 `analysis/` 下 v2 产物与 `run_research_pipeline.py` 一键复现（单元测试 46 个用例）：

1. **接受过程生存分析（`specforge/stats.py`，`analyze_acceptance_theory.py`）：** 右删失修正、风险集口径、Kaplan–Meier hazard、似然比检验、Wilson CI。结论：旧「hazard 随深度衰减」是删失伪影；实测常数 hazard 不被拒绝，风险集接受率 98.6%–99.2%。产物 `analysis/theory_check_v2.*`；旧 `theory_check.md`、`distill_metrics.md`、`adaptive_k.json` 均为 legacy，已被 v2 取代。
2. **成本感知自适应 K（`specforge/costs.py`，`adaptive_k_benchmark.py`）：** 逐轮实测 C(K)、`K*=argmin C(K)/(E[A]+1)`、有限时域 Bellman DP、calibration/held-out 分离、交错计时与配对自助 CI。结论：平稳 hazard 下固定 K 已接近最优；可复现收益是 hazard 偏移时避免长 K 的负加速（hard 组固定 K=28 为 0.760×，自适应 0.927×）。产物 `analysis/adaptive_k_v2.*`、`artifacts/k_policy.json`。
3. **TV-aware 蒸馏（`specforge/distill_tv.py`，`run_distill_ablation.py`）：** 可切换的 TV/KL 混合目标，区分随机模式（`sum min(p,q)`）与贪婪模式（top-1 一致率）两个范畴。4 目标 × 2 seed 消融：Pinsker 下界全部成立，低熵语料上各目标差异不显著。产物 `analysis/distill_ablation.*`、`analysis/distill_metrics_v2.*`。
4. **高熵语料压力测试（`specforge/data.py` rich 模式，`entropy_stress_test.py`）：** rich 语料 H2 0.637→1.668，接受率 98%→70%，最优加速 2.297×→**0.707×（负加速）**，证明原加速依赖低熵语料。产物 `analysis/entropy_stress.*`。
5. **随机模式分布等价检验（`specforge/equivalence.py`，`run_equivalence_test.py`）：** 解码器级 TV 距离 + 参数自助检验 + power curve。两组配置均不拒绝分布等价（TV=0.030/0.020，p=1.0），随机模式自身加速 1.45×/1.50×。产物 `analysis/equivalence.*`。
6. **加速来源归因（`run_attribution_control.py`）：** 对照 chunked forward 与 microbenchmark，dispatch 开销占比 92.3%，并给出 target token ratio（K=32 时 114.6%）。产物 `analysis/attribution.*`。

benchmark 与校准协议同步重构：`specforge/benchmark.py` 增加 `--split calibration/held_out`、`--mode sample`、风险集口径与 `target_token_ratio`；`specforge/calibrate_k.py` 强制校验数据来源分片，拒绝用测试集选 K。正式口径数字：校准选 K=32（2.351×，CI [2.11,2.65]），held-out **2.358×，CI [2.02,2.78]**，贪婪输出 100% 等价（`artifacts/benchmark_calibration.json`、`artifacts/benchmark_held_out.json`、`artifacts/k_policy_calibration.json`）。

**随机模式正式基准（2026-11-09 补测，T=1.0）**：校准选 K=32（2.656×，CI [2.48,2.81]），held-out **2.723×，CI [2.64,2.79]**，风险集接受率 99.7%（`artifacts/benchmark_calibration_sample.json`、`artifacts/benchmark_held_out_sample.json`、`artifacts/k_policy_sample.json`）。正确性判据在随机模式下不是逐 token exact match（与贪婪 baseline 的文本差异属预期，该列不可引用），而是输出分布等价检验：TV=0.030，参数自助 p=1.0 不拒绝等价。注意 held-out K=28 出现单点波动（CI [1.39,2.24]、censored 23.4%），这正是必须报 CI 而不是单点数的原因。

## 一键运行（PyCharm）

1. **完整解压**压缩包，在 PyCharm 用 `File → Open` 打开项目根目录；不要直接运行 ZIP 临时目录中的 `.py` 文件。
2. 为项目选择 64 位 Python 3.10–3.14 解释器；当前用户环境已验证 Python 3.14.3。安装依赖：`python -m pip install -r requirements.txt`，然后 `python -m pip install -e . --no-deps`。仅 `requirements.txt` 即可完成从零训练、正确性检查、基准和出图；若还要跑完整测试套件或 Hugging Face 真实模型入口，再装 `python -m pip install -r requirements-hf.txt`。
3. 先运行 `check_environment.py`，再运行 `pycharm_fast_train.py`。

该入口会把每次结果写到 `experiments/fast_train_T2.0_a0.75_speed15_verified[_runN]/`，不会覆盖旧结果。图像位于该目录的 `figures/`；若图片正被 Windows 查看器占用，程序会自动另存新文件。设置环境变量 `SPECFORGE_NO_PLOT_POPUP=1` 可禁止自动弹图——所有 PyCharm 入口（`pycharm_fast_train.py`、`pycharm_run.py`、`hf_pycharm_run.py`、`pycharm_3min.py`、`pycharm_experiment*.py`、`run_all.py`、`sweep.py`）都通过同一个查看器封装打开图片，开关在它们上面行为一致。

更多 IDE 操作、调参边界和排错见 [PYCHARM_GUIDE.md](PYCHARM_GUIDE.md)。

## 项目能力

- 手写 Transformer、多头注意力、分块因果掩码和逐层 KV Cache；
- `truncate` 回滚，避免被拒绝候选污染缓存；
- 贪婪推测解码与严格输出等价检查；
- 随机推测解码：`min(1,p/q)` 接受及残差分布 `normalize(max(p-q,0))` 修正；
- SFT、温度缩放 KL+CE 知识蒸馏、自动训练/对齐/性能图；
- **生产化可观测性基线：** 每次离线请求记录 TTFT（首 token 延迟）、端到端延迟 P50/P95、吞吐、接受率、回滚率、目标调用数与 target/draft KV Cache 峰值 token；随 `benchmark.json` 自动保存；
- 46 个单元与集成测试：`python -m unittest discover -s tests -q`，覆盖缓存回滚等价性、贪婪严格等价、拒绝采样恒等式、删失统计与 Kaplan–Meier、成本模型与 Bellman DP 最优性、绘图刻度自适应、弹窗开关一致等；其中包含本地随机初始化 Hugging Face 模型的 assisted-generation 端到端接口测试（需要 `requirements-hf.txt`；未安装时该用例自动跳过，套件仍 `OK (skipped=2)`，其余均通过）；

## K 候选长度消融

正式入口默认测试 `K=(4,8,12,16,20,24,28,32)`。`K` 必须是正整数；单次生成设置为 64 token 时，实验上不建议高于 64，因为剩余生成长度会截断提议块。理论上还应保证 `prompt_length + max_new_tokens < max_seq_len`，当前约为 `18+64<96`。

2026-08-27 本机完整复测结果：K=4 为 1.314×，K=8 为 1.680×，K=12 为 1.999×，K=16 为 2.104×，K=20 为 2.183×，K=24 为 2.200×，K=28 为 2.340×，K=32 为 2.095×。这说明接受率和目标调用减少都不能单独代表最终墙钟收益。

> **v2 正式口径**见「v3 研究改造」第 2 条与 `analysis/adaptive_k_v2.md`：同一 K 网格在 calibration/held-out 分片协议下重测（greedy held-out K=32 为 2.358×，CI [2.02,2.78]；sample held-out K=32 为 2.723×，CI [2.64,2.79]），并增加分布偏移三组与自适应策略对比。上行为历史记录，用于对照。

## 其他入口与历史记录

- `pycharm_run.py`：加载已有 artifact 的快速演示，不是最终从零复现入口；
- `pycharm_experiment*.py`、`pycharm_3min.py`：历史或备选调参预设，参数和结果不代表本 README 的最终数据；
- `run_all.py`：通用验证流水线；
- `run_research_pipeline.py`：v3 研究改造一键复现入口（理论分析、自适应 K、蒸馏消融、等价检验、熵压力、归因、greedy+sample 双模式 calibration/held-out 基准、K 校准与全量单元测试，支持 `--only` 选择性运行）；
- `hf_download_models.py`：从 ModelScope 断点下载 Qwen3-1.7B/0.6B 到项目内 `.model_cache`；
- `hf_validate_models.py`：不加载两套完整网络的快速离线权重、配置和 tokenizer 完整性检查；
- `hf_pycharm_run.py`：可选的 Hugging Face 真实模型推测解码入口，使用 `enable_thinking=False` 的确定性基准；运行后自动弹出吞吐/加速/延迟总览、P50/P95 延迟对比、模型规模与输出一致性三张图，详见 `HUGGINGFACE_GUIDE.md`；
- `experiments/`：历史调参记录，仅最终入口新生成目录可作为新一轮实测的来源。

面试简历请使用 [PROJECT_BRIEF.md](PROJECT_BRIEF.md) 的措辞；代码学习见 [CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md)，问答见 [INTERVIEW_GUIDE.md](INTERVIEW_GUIDE.md)，验证证据见 [VALIDATION_REPORT.md](VALIDATION_REPORT.md)。

LaTeX 简历源文件为 `latex/specforge_resume_project.tex`；请使用已更新的 `latex/specforge_resume_project_final.pdf`。同目录未带 `_final` 后缀的旧 PDF 因 Windows 查看器占用未覆盖，不应作为最终投递文件。

## 边界

项目使用程序化中文技术语料与微型 Transformer，旨在展示推测解码、蒸馏和缓存回滚的正确实现。不得将其描述为 7B/70B 模型、GPU 服务或生产线上加速收益。
