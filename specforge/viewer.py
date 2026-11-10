from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _windows_fallback(paths: list[Path], reason: str) -> int:
    print(f"Interactive Matplotlib display is unavailable ({reason}).")
    if sys.platform == "win32":
        print("Opening PNG files with the Windows default image viewer.")
        for path in paths:
            os.startfile(path)  # type: ignore[attr-defined]
        return 0
    print("Plots remain available at:")
    for path in paths:
        print(path)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive viewer for generated SpecForge plots")
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    paths = [Path(value).absolute() for value in args.paths]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit("Missing plot files: " + ", ".join(missing))
    if os.environ.get("SPECFORGE_NO_PLOT_POPUP") == "1":
        print("Plot popup suppressed by SPECFORGE_NO_PLOT_POPUP=1")
        return
    try:
        import matplotlib
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt

        backend = str(matplotlib.get_backend()).lower()
        non_interactive = {
            "agg", "pdf", "pgf", "ps", "svg", "cairo", "template",
            "module://matplotlib_inline.backend_inline",
        }
        if backend in non_interactive:
            raise RuntimeError(f"non-interactive backend {backend}")
        for path in paths:
            image = mpimg.imread(path)
            figure = plt.figure(figsize=(12.5, 4.8), num=f"SpecForge - {path.stem}")
            axis = figure.add_subplot(111)
            axis.imshow(image)
            axis.axis("off")
            figure.tight_layout(pad=0.2)
        print(f"Showing {len(paths)} plot(s) with backend: {backend}")
        print("Close the plot window(s) to finish.")
        plt.show()
    except Exception as error:
        raise SystemExit(_windows_fallback(paths, str(error)))


if __name__ == "__main__":
    main()
