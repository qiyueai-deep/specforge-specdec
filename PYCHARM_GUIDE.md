# PyCharm 最终运行与调参指南（Windows）

## 首次配置

1. 将 ZIP **完整解压**到普通目录，例如 `D:\Projects\SpecForge_AI_Interview_Project`。不能直接在 `.zip.49a` 临时目录运行文件，否则会出现 `ModuleNotFoundError: specforge`。
2. PyCharm 选择 `File → Open`，打开包含 `pycharm_fast_train.py`、`requirements.txt` 和 `specforge/` 的项目根目录。
3. 用 64 位 Python 3.10–3.14 创建项目专用 virtualenv（当前环境已验证 3.14.3），然后在 PyCharm Terminal 执行：

```powershell
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

4. 先点运行 `check_environment.py`。最后显示 `[OK]` 后，运行最终入口 `pycharm_fast_train.py`。

## 最终入口

`pycharm_fast_train.py` 不需要任何命令行参数。点击绿色 Run 后它会从零训练、运行正确性校验、执行 K 消融、保存图像并自动打开图片。2026-08-27 使用 Python 3.14.3、PyTorch 2.12.0+cpu 的完整复测耗时 103.8 秒，满足小于三分钟的设计目标。

控制台末尾还会显示本轮最佳 K 的 request telemetry：端到端延迟和 TTFT 的 P50/P95、回滚率以及 target/draft KV Cache 峰值 token。完整机器可读数据在本轮 `artifacts/benchmark.json` 的 `request_telemetry` 字段中。

每次结果都写入 `experiments/fast_train_T2.0_a0.75_speed15_verified[_runN]/`，因此请从新生成的目录读取本轮 JSON、PNG 和实际指标。图像若被图片查看器占用，程序会自动新建文件名保存；设置环境变量 `SPECFORGE_NO_PLOT_POPUP=1` 可关闭自动弹图。

## 可安全调的参数

只修改 `pycharm_fast_train.py` 顶部配置区，再点击 Run：

- `TEMPERATURE=2.0`：软标签平滑温度。建议 `1.5–3.0`；过高会使分布过平。
- `DISTILL_ALPHA=0.75`：KL 权重。建议 `0.6–0.85`；越高越强调对齐 target，越低越强调真实标签 CE。
- `distill_steps=220`：蒸馏更新次数；增大通常改善对齐但增加运行时间。
- `TARGET_MODEL` / `DRAFT_MODEL`：模型大小；改动后参数量、速度和指标都必须重新报告。
- `CANDIDATE_LENGTHS=(4,8,12,16,20,24,28,32)`：K 消融集合。`K` 为正整数；当前 `NEW_TOKENS=64` 下不建议超过 64，且理论上应让 `prompt_length + NEW_TOKENS < seq_len`。
- `NEW_TOKENS=64`：生成长度。改动后必须重新做性能比较，不能继续引用本文档的 2.340×。

不要为“好看”修改 JSON 或图表数据。曲线由固定验证批的真实测量点生成，首次运行后应以本轮 `manifest.json`、`benchmark.json` 为准。

## 其他入口的定位

`pycharm_run.py` 只加载已有模型做快速演示；`pycharm_experiment*.py` 和 `pycharm_3min.py` 是历史/备选预设。它们可用于学习，但不代表最终简历版本的模型、参数或性能。

## 并发服务模式

项目的 API 支持请求级并发。每个请求拥有独立的 Target/Draft KV Cache，服务端用有界信号量限制同时生成的请求数，避免 CPU 线程过度竞争。可以直接在 PyCharm 中运行 `pycharm_api_server.py`，也可以在 Terminal 执行下面的命令。建议 CPU 推理保持每个请求 1 个 PyTorch 算子线程：

```powershell
python -m specforge.api --artifact-dir artifacts --port 8000 `
  --max-concurrency 2 --queue-timeout 30 --torch-threads 1
```

`/health` 会返回当前活跃请求数、累计请求数、拒绝数和峰值并发数。`/generate` 的 `stats` 会额外返回 `service_queue_seconds` 和 `service_wall_seconds`。当并发槽位在指定时间内没有空闲时，服务返回 HTTP 429；这部分请求不会进入模型推理。

请求级并发与单请求内部线程数是两种不同的配置。正式 speedup 基准仍固定 `torch_threads=1`，多请求服务可通过 `max-concurrency` 调整吞吐；调整后应重新记录 P50/P95 延迟和 requests/s。
