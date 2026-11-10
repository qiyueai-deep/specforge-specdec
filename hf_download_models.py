"""PyCharm one-click Qwen3 pre-download with serial HTTPS and resume support."""

from specforge.hf_models import HFModelConfig, download_hf_models


MODEL_CONFIG = HFModelConfig(
    target_name="Qwen/Qwen3-1.7B",
    draft_name="Qwen/Qwen3-0.6B",
    local_files_only=False,
    download_provider="modelscope",
)


if __name__ == "__main__":
    print("Qwen3 model pre-download started.")
    download_hf_models(MODEL_CONFIG)
    print("[OK] Both Qwen3 models are fully cached. You can run hf_pycharm_run.py.")
