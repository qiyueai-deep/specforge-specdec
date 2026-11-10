"""Preflight checks for local/PyCharm execution with actionable errors."""

from __future__ import annotations

import platform
import struct
import sys
import tempfile
from pathlib import Path


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    raise SystemExit(1)


def main() -> None:
    root = Path(__file__).absolute().parent
    print(f"Project root: {root}")
    print(f"Python: {sys.version.split()[0]} ({struct.calcsize('P') * 8}-bit)")
    print(f"Platform: {platform.platform()}")
    if not ((3, 10) <= sys.version_info[:2] < (3, 15)):
        fail("Use 64-bit Python 3.10 through 3.14")
    if struct.calcsize("P") * 8 != 64:
        fail("A 64-bit Python interpreter is required")
    try:
        import torch
    except ImportError:
        fail("PyTorch is missing. Run: python -m pip install -r requirements.txt")
    try:
        import matplotlib
    except ImportError:
        fail("Matplotlib is missing. Run: python -m pip install -r requirements.txt")
    print(f"PyTorch: {torch.__version__}; CUDA required: no")
    print(f"Matplotlib: {matplotlib.__version__}")
    required = (
        root / "artifacts" / "target.pt",
        root / "artifacts" / "draft.pt",
        root / "artifacts" / "tokenizer.json",
        root / "artifacts" / "manifest.json",
    )
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    if missing:
        fail("Missing project files: " + ", ".join(missing))
    # benchmark.json is canonical; a cached run_all.py writes benchmark_latest.json
    # instead, so accept either one as proof a benchmark was produced.
    benchmark_candidates = (
        root / "artifacts" / "benchmark.json",
        root / "artifacts" / "benchmark_latest.json",
    )
    if not any(path.is_file() for path in benchmark_candidates):
        fail("Missing project files: artifacts/benchmark.json (or artifacts/benchmark_latest.json)")
    # Prove the directory is writable by creating AND writing a temp dir.  The
    # sandbox intercepts the cleanup rmdir and routes it to an unavailable recycle
    # bin, raising on teardown even though creation/write already succeeded, so the
    # cleanup failure must not be reported as "not writable".
    probe = tempfile.TemporaryDirectory(dir=root)
    try:
        Path(probe.name, "write_check.tmp").write_text("ok", encoding="utf-8")
    except OSError as error:
        fail(f"Project directory is not writable: {error}")
    finally:
        try:
            probe.cleanup()
        except OSError:
            pass
    from specforge.training import load_artifacts

    target, draft, tokenizer, _ = load_artifacts(root / "artifacts")
    sample = torch.tensor([[tokenizer.bos_id, tokenizer.encode("推")[0]]], dtype=torch.long)
    with torch.inference_mode():
        target_logits = target(sample)
        draft_logits = draft(sample)
    if target_logits.shape[:2] != sample.shape or draft_logits.shape[:2] != sample.shape:
        fail("Model forward output shape is invalid")
    print(f"Target parameters: {target.parameter_count:,}")
    print(f"Draft parameters: {draft.parameter_count:,}")
    print("[OK] Interpreter, dependencies, artifacts, write access and model forward are ready.")


if __name__ == "__main__":
    main()
