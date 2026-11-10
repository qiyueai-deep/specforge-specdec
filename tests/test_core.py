from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch
import tempfile

import torch

try:
    import transformers  # noqa: F401  optional dependency, only needed for the HF pipeline test
    _HAVE_TRANSFORMERS = True
except ModuleNotFoundError:
    _HAVE_TRANSFORMERS = False

from specforge.cache import KVCache
from specforge.config import ModelConfig
from specforge.decoding import (
    _compiled_forward,
    _empty_past,
    autoregressive_generate,
    residual_distribution,
    speculative_generate,
)
from specforge.hf_benchmark import _resolve_ttft
from specforge.model import TinyTransformerLM
from specforge.plots import _axis_ticks
from specforge.hf_benchmark import HFBenchmarkConfig
from specforge.hf_benchmark import run_hf_benchmark, save_hf_report
from specforge.hf_models import HFModelConfig
from specforge.hf_models import resolve_device, resolve_dtype
from specforge.observability import aggregate_request_telemetry, percentile
from specforge.tokenizer import CharTokenizer


class SpecForgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(17)
        torch.set_num_threads(1)
        config = ModelConfig(vocab_size=31, max_seq_len=64, d_model=32, n_layers=2, n_heads=4, d_ff=64)
        cls.target = TinyTransformerLM(config).eval()
        torch.manual_seed(23)
        cls.draft = TinyTransformerLM(config).eval()

    def test_cache_truncate(self) -> None:
        cache = KVCache(2)
        for layer in range(2):
            cache.append(layer, torch.randn(1, 4, 5, 8), torch.randn(1, 4, 5, 8))
        cache.truncate(3)
        self.assertEqual(cache.length, 3)
        self.assertEqual(cache.keys[1].shape[2], 3)

    def test_cached_logits_match_full_forward(self) -> None:
        ids = torch.tensor([[1, 4, 7, 3, 8, 2]])
        full = self.target(ids)
        cache = self.target.new_cache()
        pieces = [self.target(ids[:, :3], cache), self.target(ids[:, 3:5], cache), self.target(ids[:, 5:], cache)]
        incremental = torch.cat(pieces, dim=1)
        self.assertTrue(torch.allclose(full, incremental, atol=1e-5, rtol=1e-4))

    def test_rollback_matches_fresh_cache(self) -> None:
        prefix = torch.tensor([[1, 2, 3, 4]])
        cache = self.target.new_cache()
        self.target(prefix, cache)
        self.target(torch.tensor([[5, 6, 7]]), cache)
        cache.truncate(5)
        rolled = self.target(torch.tensor([[9]]), cache)[:, -1]
        fresh_cache = self.target.new_cache()
        fresh = self.target(torch.tensor([[1, 2, 3, 4, 5, 9]]), fresh_cache)[:, -1]
        self.assertTrue(torch.allclose(rolled, fresh, atol=1e-5, rtol=1e-4))

    def test_greedy_speculation_is_exact(self) -> None:
        prompt = [1, 4, 8, 3]
        baseline, _ = autoregressive_generate(self.target, prompt, 20, mode="greedy")
        for draft_k in (1, 2, 4, 6):
            speculative, stats = speculative_generate(self.target, self.draft, prompt, 20, draft_k, mode="greedy")
            self.assertEqual(baseline, speculative)
            self.assertEqual(stats.generated_tokens, 20)

    def test_compiled_greedy_matches_eager(self) -> None:
        # torch.compile 的 CPU 后端需要 MSVC；无编译器时跳过，不破坏套件。
        try:
            probe = _compiled_forward(self.target)
            probe(torch.tensor([[1, 2]], dtype=torch.long), *_empty_past(self.target))
        except RuntimeError:
            self.skipTest("torch.compile CPU backend unavailable (no C++ compiler)")
        prompt = [1, 4, 8, 3]
        eager, _ = autoregressive_generate(self.target, prompt, 12, mode="greedy")
        compiled, _ = autoregressive_generate(self.target, prompt, 12, mode="greedy", use_compile=True)
        self.assertEqual(eager, compiled)
        eager_spec, _ = speculative_generate(self.target, self.draft, prompt, 12, 4, mode="greedy")
        compiled_spec, _ = speculative_generate(self.target, self.draft, prompt, 12, 4, mode="greedy", use_compile=True)
        self.assertEqual(eager_spec, compiled_spec)

    def test_residual_distribution(self) -> None:
        p = torch.tensor([0.6, 0.3, 0.1])
        q = torch.tensor([0.2, 0.5, 0.3])
        residual = residual_distribution(p, q)
        self.assertTrue(torch.allclose(residual, torch.tensor([1.0, 0.0, 0.0])))

    def test_tokenizer_round_trip(self) -> None:
        tokenizer = CharTokenizer.from_text("推测解码测试")
        text = "推测解码"
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_parameter_count_is_real(self) -> None:
        manual = sum(parameter.numel() for parameter in self.target.parameters())
        self.assertEqual(self.target.parameter_count, manual)

    def test_stochastic_decoder_runs_and_preserves_length(self) -> None:
        prompt = [1, 4, 8, 3]
        output, stats = speculative_generate(
            self.target, self.draft, prompt, 18, draft_k=4,
            mode="sample", temperature=0.9, top_k=12, seed=123,
        )
        self.assertEqual(len(output), len(prompt) + 18)
        self.assertEqual(stats.generated_tokens, 18)

    def test_request_telemetry_is_populated(self) -> None:
        prompt = [1, 4, 8, 3]
        _, baseline_stats = autoregressive_generate(self.target, prompt, 8, mode="greedy")
        _, speculative_stats = speculative_generate(self.target, self.draft, prompt, 8, draft_k=4, mode="greedy")
        telemetry = aggregate_request_telemetry([baseline_stats, speculative_stats], [0.01, 0.03])
        self.assertGreater(baseline_stats.time_to_first_token_seconds, 0.0)
        self.assertGreater(speculative_stats.end_to_end_seconds, 0.0)
        self.assertGreaterEqual(telemetry["cache_peak_tokens"]["target_max"], len(prompt))
        self.assertAlmostEqual(percentile([1.0, 3.0], 0.5), 2.0)

    def test_axis_ticks_follow_the_real_training_length(self) -> None:
        # The official 220-step schedule must keep its original, readable ticks.
        self.assertEqual(_axis_ticks(220), [0, 25, 50, 75, 100, 125, 150, 175, 200, 220])
        # Short smoke-test schedules must not inherit the 220-step ticks.
        self.assertEqual(_axis_ticks(20), [0, 5, 10, 15, 20])
        self.assertEqual(_axis_ticks(400)[-1], 400)
        self.assertEqual(_axis_ticks(0), [0])
        for length in (1, 7, 20, 120, 220, 1000):
            ticks = _axis_ticks(length)
            self.assertEqual(ticks[0], 0)
            self.assertEqual(ticks[-1], length)
            self.assertLessEqual(len(ticks), 11)

    def test_ttft_fallback_only_applies_to_missing_measurements(self) -> None:
        # A genuine 0.0 must survive; only "no first-token event" may fall back.
        self.assertEqual(_resolve_ttft(0.0, 1.5), 0.0)
        self.assertEqual(_resolve_ttft(0.02, 1.5), 0.02)
        self.assertEqual(_resolve_ttft(None, 1.5), 1.5)

    def test_every_plot_entry_honours_the_popup_switch(self) -> None:
        # Regression guard: every PyCharm entry point used to call os.startfile
        # directly, which silently ignored SPECFORGE_NO_PLOT_POPUP=1. All of them
        # must now route through the shared viewer helper.
        root = Path(__file__).resolve().parents[1]
        entries = (
            "pycharm_fast_train.py",
            "pycharm_run.py",
            "hf_pycharm_run.py",
            "pycharm_3min.py",
            "pycharm_experiment.py",
            "pycharm_experiment_balanced.py",
            "pycharm_experiment_small.py",
        )
        for name in entries:
            with self.subTest(entry=name):
                source = (root / name).read_text(encoding="utf-8")
                self.assertIn(
                    "show_plot_files", source,
                    f"{name} should open plots through the popup-aware viewer helper",
                )
                self.assertNotIn(
                    "os.startfile", source,
                    f"{name} bypasses SPECFORGE_NO_PLOT_POPUP by calling os.startfile",
                )

    def test_generate_plots_reads_the_explicit_benchmark_path(self) -> None:
        from unittest.mock import patch
        import shutil

        from specforge.benchmark import run_benchmark
        from specforge.plots import generate_plots
        from specforge.utils import load_json, save_json

        root = Path(__file__).resolve().parents[1]
        artifacts = root / "artifacts"
        fresh = run_benchmark(artifacts, max_new_tokens=16, repeats=1, warmups=0,
                              candidate_lengths=(4, 8), benchmark_threads=1, prompt_count=2)
        probe = artifacts / "benchmark_integration_probe.json"
        save_json(probe, fresh)
        out = root / "figures_probe"
        # Spy on load_json so we can prove generate_plots read the explicit path
        # rather than falling back to artifacts/benchmark.json.
        with patch("specforge.plots.load_json", side_effect=load_json) as spy:
            generate_plots(artifacts, out, probe)
        loaded = [str(call.args[0]) for call in spy.call_args_list]
        self.assertIn(str(probe), loaded, "generate_plots ignored the explicit benchmark_path")
        # Cleanup is best-effort: the sandbox intercepts unlink() and funnels it to
        # a recycle bin that is unavailable here, so tolerate that failure rather
        # than letting it mask the real assertion above.
        try:
            probe.unlink()
        except OSError:
            pass
        shutil.rmtree(out, ignore_errors=True)

    def test_temporary_regeneration_script_is_not_shipped(self) -> None:
        package = Path(__file__).resolve().parents[1] / "specforge"
        self.assertFalse(
            (package / "regen_v3.py").exists(),
            "regen_v3.py hardcodes absolute paths of another machine and must stay out of the package",
        )

    def test_hf_configuration_requires_valid_positive_values(self) -> None:
        HFBenchmarkConfig(candidate_lengths=(4, 8), max_new_tokens=8, repeats=1).validate()
        with self.assertRaises(ValueError):
            HFBenchmarkConfig(candidate_lengths=(0,), max_new_tokens=8).validate()
        self.assertIn(resolve_device("auto"), {"cpu", "cuda"})
        self.assertIn(resolve_dtype("cpu"), {torch.float32})

    @unittest.skipUnless(_HAVE_TRANSFORMERS, "transformers is not installed (optional HF dependency)")
    def test_hf_assisted_generation_pipeline_with_local_random_models(self) -> None:
        from transformers import BatchEncoding, GPT2Config, GPT2LMHeadModel

        class TinyTokenizer:
            pad_token_id = 0

            def __call__(self, text: str, return_tensors: str = "pt") -> BatchEncoding:
                ids = [1 + (ord(character) % 30) for character in text[:6]] or [1]
                tensor = torch.tensor([ids], dtype=torch.long)
                return BatchEncoding({"input_ids": tensor, "attention_mask": torch.ones_like(tensor)})

        target = GPT2LMHeadModel(GPT2Config(
            vocab_size=32, n_positions=32, n_embd=32, n_layer=2, n_head=4,
            bos_token_id=1, eos_token_id=None, pad_token_id=0,
        )).eval()
        draft = GPT2LMHeadModel(GPT2Config(
            vocab_size=32, n_positions=32, n_embd=32, n_layer=1, n_head=4,
            bos_token_id=1, eos_token_id=None, pad_token_id=0,
        )).eval()
        with patch("specforge.hf_benchmark.load_hf_models", return_value=(target, draft, TinyTokenizer(), "cpu")):
            result = run_hf_benchmark(
                HFModelConfig(target_name="local-target", draft_name="local-draft"),
                HFBenchmarkConfig(candidate_lengths=(2,), max_new_tokens=4, repeats=1,
                                  warmups=0, prompt_count=1),
                prompts=("测试",),
            )
        self.assertEqual(result["assisted"][0]["greedy_exact_match_rate"], 1.0)
        self.assertGreater(result["assisted"][0]["time_to_first_token_seconds"]["p50"], 0.0)
        with tempfile.TemporaryDirectory() as directory:
            paths = save_hf_report(result, directory)
            self.assertEqual(len(paths), 3)
            self.assertTrue(all(path.exists() and path.stat().st_size > 0 for path in paths))


if __name__ == "__main__":
    unittest.main(verbosity=2)
