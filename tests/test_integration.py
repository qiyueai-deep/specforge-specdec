from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import torch

from specforge.plots import generate_plots, show_plot_files
from specforge.report import build_summary
from specforge.training import load_artifacts


class ProjectIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.artifacts = cls.root / "artifacts"
        cls.target, cls.draft, cls.tokenizer, cls.manifest = load_artifacts(cls.artifacts)

    def test_packaged_artifacts_load_and_forward(self) -> None:
        ids = torch.tensor([[self.tokenizer.bos_id, self.tokenizer.encode("推")[0]]], dtype=torch.long)
        with torch.inference_mode():
            target_logits = self.target(ids)
            draft_logits = self.draft(ids)
        self.assertEqual(tuple(target_logits.shape[:2]), tuple(ids.shape))
        self.assertEqual(tuple(draft_logits.shape[:2]), tuple(ids.shape))
        self.assertEqual(self.target.parameter_count, self.manifest["target_parameters"])

    def test_figures_are_generated_from_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = generate_plots(self.artifacts, directory)
            self.assertEqual(len(paths), 3)
            for path in paths:
                self.assertGreater(path.stat().st_size, 20_000)
                self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_official_summary_is_internally_consistent(self) -> None:
        summary = build_summary(self.artifacts)
        best = summary["best_benchmark"]
        self.assertEqual(best["greedy_exact_match_rate"], 1.0)
        self.assertGreater(best["wall_clock_speedup"], 1.0)
        self.assertGreater(best["target_call_reduction"], 0.5)

    def test_plot_popup_can_be_disabled_for_automation(self) -> None:
        paths = sorted((self.root / "figures").glob("*.png"))
        # Set/restore only the one variable by hand. patch.dict would restore the
        # entire os.environ on teardown, and the sandbox can inject a variable whose
        # value exceeds the 32767-char Windows os.environ limit, raising on cleanup.
        key = "SPECFORGE_NO_PLOT_POPUP"
        previous = os.environ.get(key)
        os.environ[key] = "1"
        try:
            self.assertEqual(show_plot_files(paths), "suppressed")
        finally:
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


if __name__ == "__main__":
    unittest.main(verbosity=2)
