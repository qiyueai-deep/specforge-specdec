"""Three-minute PyCharm presentation runner.

This entry point is intended for local interviews/demo runs.  It reuses the
already trained, verified compact12x artifact instead of retraining a 5M-
parameter CPU model, so the whole run normally finishes well below 3 minutes.
It copies artifacts into a new experiment folder, regenerates all figures and
opens them with the default Windows viewer.

The reported Loss/KL/CE values are copied from the real artifact manifest.
Nothing is clipped or fabricated; the 0.08 reference line is shown on the
distillation chart so the small remaining CE gap is visible and explainable.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from specforge.benchmark import format_benchmark
from specforge.plots import generate_plots, show_plot_files
from specforge.report import build_summary, format_summary
from specforge.utils import load_json, save_json


# =============================== CONFIGURATION ==============================
EXPERIMENT_NAME = "three_minute_T1.2_alpha0.85"
SOURCE_EXPERIMENT = "T1.2_a85_d400"
SHOW_FIGURES = True
# ============================================================================


def main() -> None:
    root = Path(__file__).absolute().parent
    source = root / "experiments" / SOURCE_EXPERIMENT / "artifacts"
    if not (source / "manifest.json").exists():
        raise FileNotFoundError(f"已验证缓存不存在：{source}")

    out = root / "experiments" / EXPERIMENT_NAME
    artifact_dir = out / "artifacts"
    figure_dir = out / "figures"
    started = time.perf_counter()
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(source, artifact_dir)

    manifest = load_json(artifact_dir / "manifest.json")
    manifest.setdefault("demo", {})
    manifest["demo"].update({
        "mode": "cached_verified_artifact",
        "source_experiment": SOURCE_EXPERIMENT,
        "runtime_target_seconds": 180,
        "loss_reference": 0.08,
    })
    save_json(artifact_dir / "manifest.json", manifest)

    benchmark = load_json(artifact_dir / "benchmark.json")
    print("=" * 72)
    print("SpecForge 3-minute PyCharm presentation runner")
    print(f"T={manifest['train_config']['temperature']}  "
          f"alpha={manifest['train_config']['distill_alpha']}")
    print(f"Source: {SOURCE_EXPERIMENT}")
    print("=" * 72)
    print("\n[1/3] Loading verified CPU artifacts (no retraining)")
    print("[2/3] Benchmark summary")
    print(format_benchmark(benchmark))
    print("[3/3] Regenerating figures")
    figures = generate_plots(artifact_dir, figure_dir, artifact_dir / "benchmark.json")
    for path in figures:
        print(path.absolute())
    print("\n" + format_summary(build_summary(artifact_dir)))

    metrics = manifest["metrics"]
    print("\nLoss threshold audit (real values; reference=0.08)")
    print(f"target validation CE: {metrics['target_validation_loss_after']:.6f}")
    print(f"draft validation CE:  {metrics['draft_after_distillation']['draft_validation_loss']:.6f}")
    print(f"distillation KL:      {metrics['draft_after_distillation']['kl_target_to_draft']:.6f}")
    print("说明：KL 已低于 0.08；CE 接近但略高于 0.08，图中保留真实值与阈值线。")
    elapsed = time.perf_counter() - started
    print(f"\n总耗时：{elapsed:.1f}s（目标上限 180s）")

    if SHOW_FIGURES:
        print(f"display mode: {show_plot_files(figures, block=False)}")


if __name__ == "__main__":
    main()
