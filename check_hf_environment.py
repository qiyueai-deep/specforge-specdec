"""Preflight check for the optional Hugging Face real-model experiment."""

from __future__ import annotations

import importlib.util
import shutil
import sys

import torch

from specforge.hf_models import model_cache_root


def main() -> None:
    missing = [name for name in ("transformers", "accelerate", "safetensors", "sentencepiece", "modelscope")
               if importlib.util.find_spec(name) is None]
    if missing:
        print(f"[FAIL] Missing optional packages: {', '.join(missing)}")
        print("Run: python -m pip install -r requirements-hf.txt")
        raise SystemExit(1)
    import transformers
    from huggingface_hub import get_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    free_gb = shutil.disk_usage(__file__).free / (1024 ** 3)
    cache_root = model_cache_root()
    cached_weights = sorted(
        path for path in cache_root.rglob("*.safetensors") if path.stat().st_size >= 1_000_000
    )
    cached_gb = sum(path.stat().st_size for path in cached_weights) / (1024 ** 3)
    print(f"Python:       {sys.version.split()[0]}")
    print(f"PyTorch:      {torch.__version__}")
    print(f"Transformers: {transformers.__version__}")
    print(f"HF login:     {'authenticated' if get_token() else 'not authenticated (not required for ModelScope)'}")
    print(f"Cache root:   {cache_root}")
    print(f"Qwen3 cache:  {'ready' if cached_weights else 'not complete'} "
          f"({len(cached_weights)} shard(s), {cached_gb:.2f} GB)")
    if not cached_weights:
        print("  Set SPECFORGE_MODEL_CACHE to reuse an existing cache, or run hf_download_models.py.")
    print(f"Device:       {device}")
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU:          {props.name}")
        print(f"GPU memory:   {props.total_memory / (1024 ** 3):.1f} GB")
    else:
        print("[WARNING] CPU mode is supported but the Qwen3 1.7B/0.6B benchmark may be slow.")
    print(f"Free disk:    {free_gb:.1f} GB")
    if free_gb < 10:
        print("[WARNING] At least 10 GB free disk space is recommended for model caches and reports.")
    print("[OK] Real-model software environment is ready. Run hf_download_models.py to verify weights.")


if __name__ == "__main__":
    main()
