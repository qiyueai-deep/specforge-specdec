"""PyCharm-friendly no-argument entry point.

Open this file in PyCharm and click the green Run button. The working directory
must be the project root; PYCHARM_GUIDE.md explains the exact setup.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from specforge.decoding import autoregressive_generate, speculative_generate
from specforge.plots import generate_plots, show_plot_files
from specforge.report import build_summary, format_summary
from specforge.training import load_artifacts


def main() -> None:
    root = Path(__file__).absolute().parent
    artifacts = root / "artifacts"
    target, draft, tokenizer, manifest = load_artifacts(artifacts)
    torch.set_num_threads(1)
    prompt_text = "推测解码通过"
    prompt = tokenizer.encode(prompt_text, add_bos=True)
    baseline, _ = autoregressive_generate(target, prompt, 36, mode="greedy")
    generated, stats = speculative_generate(target, draft, prompt, 36, draft_k=8, mode="greedy")
    figures = generate_plots(artifacts, root / "figures")
    print("=" * 72)
    print("SpecForge PyCharm run succeeded")
    print("=" * 72)
    print(format_summary(build_summary(artifacts)))
    print("\nGenerated text:")
    print(tokenizer.decode(generated))
    print("\nGreedy output exact match:", generated == baseline)
    print("Generation stats:")
    print(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2))
    print("\nFigures:")
    for path in figures:
        print(path.absolute())
    print(f"\nPyTorch: {torch.__version__}; threads=1; training threads={manifest['train_config']['torch_threads']}")
    if os.environ.get("SPECFORGE_NO_PLOT_POPUP") == "1":
        print("\nPlot popup suppressed by SPECFORGE_NO_PLOT_POPUP=1.")
    else:
        print("\nOpening plots. Close the plot window(s) to finish the program.")
    print("Display mode:", show_plot_files(figures))


if __name__ == "__main__":
    main()
