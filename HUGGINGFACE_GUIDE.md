# Hugging Face 真实模型扩展指南

该扩展使用 `Qwen3-1.7B` 作为 Target、`Qwen3-0.6B` 作为 Draft，通过 Hugging Face 官方 `assistant_model` 接口执行真实预训练模型推测解码。两者属于同一 Qwen3 家族；代码还会逐项检查 tokenizer 映射和模型词表。它与 `pycharm_fast_train.py` 的三分钟微型模型实验相互独立。

## 安装与运行

在项目根目录安装可选依赖：

```powershell
python -m pip install -r requirements-hf.txt
python -m pip install -e . --no-deps
```

Qwen3 需要 `transformers>=4.51`；当前已验证环境为 Transformers 5.12.1。

然后在 PyCharm 运行 `hf_download_models.py`，默认从 ModelScope 的 Qwen 官方镜像下载两套模型权重；完成后运行 `hf_validate_models.py`，最后运行 `hf_pycharm_run.py`。建议准备稳定网络、至少约 10 GB 可用磁盘空间和 12 GB 系统内存。NVIDIA GPU 推荐至少 8 GB 显存；CPU 可以运行。基准结束后会自动打开三张真实模型图；不希望弹图时将文件顶部 `SHOW_FIGURE=False`。

若下载曾停在 `Fetching 2 files: 0%`，先停止旧运行，再单独运行 `hf_download_models.py`。当前网络直连 Hugging Face 出现 WinError 10060，因此默认通过 ModelScope 的 Qwen 官方镜像断点下载权重，再交给 Transformers 从本地目录加载；模型名称、权重和推测解码逻辑不变，也不需要 HF_TOKEN。缓存固定在项目的 `.model_cache`，避免 Windows 用户目录权限问题，并已通过 `.gitignore` 排除，不应放入项目压缩包。下载后可先运行 `hf_validate_models.py` 做快速离线完整性检查，再运行 `hf_pycharm_run.py`；已缓存权重不会重复下载。若所在网络能稳定访问 Hugging Face，可把 `download_provider` 改回 `"huggingface"`。

Qwen3 默认支持 thinking mode，但其模型卡不建议在思考模式使用贪婪解码。本项目为了验证 target-only 与 assisted generation 的逐 token 严格等价，固定使用聊天模板并设置 `enable_thinking=False`。不要在引用当前贪婪基准时打开 thinking mode。

`hf_pycharm_run.py` 默认采用 CPU 快速配置（1 个提示词、8 个新 token、K=4/8、无 warmup）。最新本机可复测记录 `huggingface_qwen3_real_models_run4` 耗时 35.0 秒；短基准会受 CPU 频率与后台负载影响而波动。正式性能实验应增加 `max_new_tokens`、`prompt_count`、`repeats` 和 `warmups`，耗时会明显增加。真实模型首次约 5.2GB 权重下载时间不计入运行基准。

公开模型无需 Token 也能下载。为获得更高下载限额，可在 PyCharm Terminal 执行 `hf auth login` 并使用只读 Token；程序会自动使用 `HF_TOKEN` 或本机登录凭据，不会打印 Token。Windows 未开启开发者模式时缓存仍可正常工作，项目已关闭重复的 symlink 警告，但磁盘占用可能略高。

建议先运行 `check_hf_environment.py`，检查 Transformers、Accelerate、设备和可用磁盘空间。

模型下载完成后，可把 `local_files_only` 改成 `True`，完全使用本地缓存。

### 复用已下载的权重（不重复下载 5.2GB）

模型缓存默认固定在项目内的 `.model_cache`，但可以通过环境变量 `SPECFORGE_MODEL_CACHE` 指向任意已有缓存目录。`check_hf_environment.py`、`hf_download_models.py`、`hf_validate_models.py`、`hf_pycharm_run.py` 都读这个变量。例如把权重放在另一块磁盘后，运行前设置：

```powershell
$env:SPECFORGE_MODEL_CACHE = "D:\models\specforge_cache"
python hf_pycharm_run.py
```

`check_hf_environment.py` 现在也会按这个变量报告缓存位置和分片数量，而不是硬编码 `.model_cache`。

## 自动输出

每次运行写入独立目录：

```text
experiments/huggingface_qwen3_real_models[_runN]/
├── hf_benchmark.json
├── hf_benchmark_report.md
├── hf_speculative_benchmark.png
├── hf_latency_comparison.png
└── hf_model_validation.png
```

运行后会自动弹出三张图：吞吐/加速/延迟总览、baseline 与各 K 的 P50/P95 延迟对比、真实模型规模与贪婪输出一致性验证。图表全部从本轮 `hf_benchmark.json` 生成；报告包含 Target/Draft 真实参数量、吞吐、端到端 P50/P95、TTFT P50/P95、GPU 峰值显存、不同 K 的加速比和贪婪输出一致率。

Hugging Face 官方 assisted generation API 不公开内部候选接受与回滚计数，因此 HF 报告中的这两个字段为 `null`。项目自研微型解码器仍提供完整接受率、回滚率及 KV Cache 状态验证。不得根据输出文本反推或伪造接受率。

## 推荐调参顺序

1. 首次冒烟：`prompt_count=1`、`repeats=1`、`max_new_tokens=16`、`candidate_lengths=(4,)`；
2. 确认 `greedy_exact_match_rate=100%`；
3. 再恢复 `(4,8,12,16)` 和多个 prompt；
4. 只有完整基准结束后，才能把本机真实结果写入简历。

如出现内存不足，先减小 `prompt_count` 和 `max_new_tokens`；它们减少测试工作量，但不会减少模型权重本身占用。仍不足时应换用更大内存机器或 GPU，不建议把 Target/Draft 改成同一个模型来制造“可运行”结果。
