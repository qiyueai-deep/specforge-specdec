# 最终版本验证报告

## 验证对象

最终默认入口：`pycharm_fast_train.py`。配置为 Target `d=256,L=6,H=8,d_ff=1024`、Draft `d=80,L=2,H=4,d_ff=240`；训练采用 SFT → 固定混合蒸馏（`T=2.0, α=0.75`），不启用 mix → CE-only 两阶段细化。

## 记录的完整实测

2026-08-27 使用固定种子 `20260820`、Python 3.14.3、PyTorch 2.12.0+cpu 完整执行用时 **103.8 s**，事实源为 `experiments/fast_train_T2.0_a0.75_speed15_verified_run22/`。参数量为 Target **4,828,928**、Draft **159,440**，压缩比 **30.29×**。验证 loss 分别为 Target `5.5814 → 0.1115`、Draft `5.5157 → 0.1130`；Top-1 agreement `93.20% → 98.35%`，KL(T||S) `0.9145 → 0.0480`。

CPU 基准固定为 1 个 Torch 线程、6 条 prompt、prompt 长度 18、`NEW_TOKENS=64`、warmup=1、repeats=2。K=28 为本轮最佳点：接受率 **90.63%**、目标调用减少 **90.10%**、吞吐 `809.96 vs 346.08 token/s`、端到端 **2.340×**。K=8、12、32 的速度分别为 `1.680×、1.999×、2.095×`。

## 正确性与工程检查

- 贪婪推测解码与 target-only 逐 token 输出 **100% 等价**；
- 随机 rejection sampling 概率恒等式数值误差 `<1e-7`；
- `python -m unittest discover -s tests -q`：**20 个用例，19 通过 + 1 跳过**（跳过项需 `requirements-hf.txt`；缺少该依赖时套件仍 `OK (skipped=1)`，不报错）；新增测试使用本地随机初始化的 Hugging Face Target/Draft，验证官方 `assistant_model` 流水线、贪婪输出等价和 TTFT 采集，无需下载权重；
- 图表由每轮 artifact 与 benchmark JSON 自动生成；训练曲线使用固定验证批，避免随机训练批造成视觉伪波动；
- 每轮 benchmark 同时持久化请求级 TTFT、端到端延迟 P50/P95、prefill 延迟、回滚率、接受率、目标调用数和 KV Cache 峰值 token；CPU 离线版本不虚构 GPU 显存指标；
- PNG 被 Windows 图片查看器占用时会自动用新文件名保存，避免 `PermissionError`。
- 已修复 Windows 目录联接场景下 `Path.resolve()` 将桌面项目重定向到旧 `WorkBuddy` 路径的问题；所有 PyCharm 入口现在保留用户实际打开的绝对路径，可正常创建新实验目录。

## 结论与边界

本入口在记录机器上满足三分钟内完成、从零训练、出图和校验的要求。性能数字仅对该 CPU、线程数、prompt 和生成长度有效；项目是程序化中文语料上的微型 Transformer 离线实验，不能外推为生产模型收益。

可选入口 `hf_pycharm_run.py` 已切换为 Qwen3-1.7B/0.6B，并使用 `enable_thinking=False` 的聊天模板进行贪婪等价评测。2026-08-27 已完成约 5.2GB 权重下载、离线完整性验证、两模型实际加载与 assisted-generation 端到端运行；最新 CPU 快速配置（`experiments/huggingface_qwen3_real_models_run4/`）耗时 35.0s，K=4/K=8 分别为 2.517x/1.893x，贪婪输出一致率均为 100%。该入口还会从实测 JSON 自动生成吞吐/加速/延迟总览、P50/P95 延迟对比、模型规模与输出一致性三张图。单提示词短基准仍存在运行波动，因此这些数据只证明真实模型链路可运行，不能直接作为正式简历性能指标。首次模型下载时间不计入微型模型三分钟约束。

## v3 研究改造验证（2026-11-09）

以上 2026-08-27 的记录保留为历史复现证据。v3 改造后，正式统计口径与结论由以下实测支撑（事实源 `analysis/pipeline.log`，全部步骤 EXIT=0，单元测试 **46 个用例，45 通过 + 1 跳过**；跳过项为 `test_compiled_greedy_matches_eager`，本机无 C++ 编译器故 `torch.compile` CPU 后端不可用）：

**统计口径修正：**

- 旧口径 `accepted/proposed`（90.63%）把右删失 token 计入拒绝，已被 Monte Carlo 证明会在真 i.i.d. 数据上伪造 hazard 衰减（0.9705 → 0.4263）；新口径 `accepted/verified` 在同数据上保持平稳（`tests/test_stats.py`）。
- v2 理论分析（`analysis/theory_check_v2.*`）：各 K 风险集接受率 98.6%–99.2%，似然比检验不拒绝常数 hazard，旧「i.i.d. 破缺」结论撤回；有限长度 Monte Carlo 与实测目标调用平均绝对误差 0.172 轮。

**性能正式口径（calibration/held-out 分离）：**

- 校准集（6 prompt × 64 token，repeats=5）选出 K=32：**2.351×，CI [2.11, 2.65]**，风险集接受率 98.66%；
- held-out 复测：**2.358×，CI [2.02, 2.78]**，接受率 98.40%，贪婪输出 100% 等价，regret=0（`artifacts/benchmark_calibration.json`、`artifacts/benchmark_held_out.json`、`artifacts/k_policy_calibration.json`）；
- 历史数字 2.340× / 90.63% 保留为旧口径记录，不再作为创新论证依据。

**压力测试与等价性：**

- 高熵语料（`analysis/entropy_stress.*`）：H2 0.637→1.668，接受率 98%→70%，最优加速 2.297×→0.707×（负加速），证明原加速依赖低熵语料；
- 分布偏移（`analysis/adaptive_k_v2.*`）：hard 组 hazard 0.876，固定 K=28 为 0.760× 负加速，Bellman 自适应修正为 0.927×；easy/mixed 组自适应与最优固定 K 的 CI 均重叠，报告不宣称显著超越；
- 随机模式分布等价（`analysis/equivalence.*`）：TV=0.030/0.020，参数自助 p=1.0，不拒绝等价，随机模式自身加速 1.45×/1.50×；
- 蒸馏目标消融（`analysis/distill_ablation.*`）：4 目标 × 2 seed，Pinsker 下界全部成立；贪婪接受率对应 top-1 一致率而非 `sum min(p,q)`，v1 范畴错误已修正（旧脚本快照 `tv_kl_report_v1_legacy.py`）；
- 加速归因（`analysis/attribution.*`）：dispatch 开销占比 92.3%，target token ratio K=32 时 114.6%。

**复现方式：** `python run_research_pipeline.py` 依次运行理论分析、自适应 K、蒸馏消融、TV/KL v2、等价检验、熵压力、归因、calibration/held-out 基准、calibrate_k 与全量单元测试。

**随机模式正式基准补测（2026-11-09）：**

- `--mode sample --temperature 1.0`，同一分片协议：校准选 K=32，**2.656×，CI [2.48, 2.81]**，风险集接受率 99.73%；held-out 复测 **2.723×，CI [2.64, 2.79]**（`artifacts/benchmark_calibration_sample.json`、`artifacts/benchmark_held_out_sample.json`、`artifacts/k_policy_sample.json`）；
- 随机模式正确性判据为输出分布等价（TV=0.030，p=1.0，见 `analysis/equivalence.*`）；benchmark 输出中的 exact 列在 sample 模式下是与贪婪 baseline 的文本对比，属预期不同，**不作为正确性指标引用**；
- held-out K=28 单点 CI [1.39, 2.24]、censored 23.4%，说明随机模式下单 K 波动更大，结论必须以 CI 与校准选中点为准。
