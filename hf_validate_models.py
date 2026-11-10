"""Fast offline integrity check for the cached Qwen3 target/draft pair.

Run this file directly in PyCharm after ``hf_download_models.py``.  It checks
the local files, Transformers configs, tokenizers, and shared token mapping
without allocating memory for both neural networks.
"""

from __future__ import annotations

from pathlib import Path

from transformers import AutoConfig, AutoTokenizer

from specforge.hf_models import HFModelConfig, download_hf_models, model_cache_root


MODEL_CONFIG = HFModelConfig(
    target_name="Qwen/Qwen3-1.7B",
    draft_name="Qwen/Qwen3-0.6B",
    download_provider="modelscope",
)


def _weight_summary(model_path: str) -> tuple[int, int]:
    files = sorted(Path(model_path).glob("*.safetensors"))
    if not files:
        raise RuntimeError(f"No safetensors weights found: {model_path}")
    sizes = [path.stat().st_size for path in files]
    if any(size < 1_000_000 for size in sizes):
        raise RuntimeError(f"Incomplete safetensors weight found: {model_path}")
    return len(files), sum(sizes)


def main() -> None:
    target_path, draft_path = download_hf_models(MODEL_CONFIG)

    target_cfg = AutoConfig.from_pretrained(target_path, local_files_only=True)
    draft_cfg = AutoConfig.from_pretrained(draft_path, local_files_only=True)
    target_tokenizer = AutoTokenizer.from_pretrained(target_path, local_files_only=True)
    draft_tokenizer = AutoTokenizer.from_pretrained(draft_path, local_files_only=True)

    if target_tokenizer.get_vocab() != draft_tokenizer.get_vocab():
        raise RuntimeError("Target and Draft tokenizer mappings differ.")
    if target_cfg.vocab_size != draft_cfg.vocab_size:
        raise RuntimeError("Target and Draft model vocabularies differ.")

    target_files, target_bytes = _weight_summary(target_path)
    draft_files, draft_bytes = _weight_summary(draft_path)
    print("=" * 72)
    print("Qwen3 offline cache validation succeeded")
    print("=" * 72)
    print(f"Target: {target_path}")
    print(f"  architecture: {target_cfg.architectures[0]}")
    print(f"  weights:      {target_files} file(s), {target_bytes / 1e9:.3f} GB")
    print(f"Draft:  {draft_path}")
    print(f"  architecture: {draft_cfg.architectures[0]}")
    print(f"  weights:      {draft_files} file(s), {draft_bytes / 1e9:.3f} GB")
    print(f"Shared tokenizer vocabulary: {len(target_tokenizer):,} tokens")
    print(f"Cache root: {model_cache_root()}")
    print("[OK] You can now run hf_pycharm_run.py in PyCharm.")


if __name__ == "__main__":
    main()
