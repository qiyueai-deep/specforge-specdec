# SpecForge: Knowledge Distillation + Speculative Decoding for CPU Inference

[中文](README.md) | English

A miniature Transformer speculative-decoding system that trains from scratch, validates itself and
produces figures on an ordinary Windows CPU. It is built to demonstrate a reproducible
algorithm-and-engineering loop, not to serve as a production LLM service.

## What the project covers

1. **Mechanism and engineering loop:** speculative decoding implemented from scratch (greedy exact
   equivalence, stochastic rejection sampling, KV Cache truncate-and-rollback), temperature
   distillation and per-request telemetry. All code is readable, runnable and reproducible —
   from-scratch training takes about 3 minutes and one pipeline reproduces every conclusion.
2. **Statistics and modelling:** correcting right-censoring in acceptance-rate statistics (risk-set
   denominator, Kaplan–Meier, likelihood-ratio test), modelling adaptive K with measured cost C(K)
   and finite-horizon Bellman optimal stopping, under a calibration / held-out separation protocol.
3. **Evaluation discipline:** high-entropy corpus stress tests, distributional equivalence tests for
   the stochastic mode, attribution of speedups, and confidence intervals throughout. Every number
   carries its boundary; gains on low-entropy corpora are not generalised.

**Five-minute review path:** this section → [PROJECT_BRIEF.md](PROJECT_BRIEF.md) →
`analysis/adaptive_k_v2.md` and `analysis/entropy_stress.md` → `specforge/stats.py` and
`specforge/costs.py`. Full validation evidence is in [VALIDATION_REPORT.md](VALIDATION_REPORT.md).

## Final configuration and measured facts

The recommended entry point is [pycharm_fast_train.py](pycharm_fast_train.py).

| Item | Final configuration / measured on a fixed machine |
|---|---|
| Target / Draft | 4,828,928 parameters (d=256, L=6, H=8) / 159,440 (d=80, L=2, H=4) |
| Parameter compression | 30.29× |
| Distillation | SFT → fixed mixed distillation, $T=2.0$, $\alpha=0.75$ |
| Training | 220 target steps, 120 draft SFT steps, 220 distillation steps, seq_len=96 |
| Runtime | 103.8 s (full local rerun, 2026-08-27; target below 3 minutes) |
| Alignment | Top-1 agreement 98.35%, $\mathrm{KL}(T\Vert S)=0.0480$ |
| Correctness | Greedy output 100% equivalent to target-only decoding |
| Best benchmark | K=28: acceptance 90.63%, target calls reduced by 90.10%, 2.340× speedup |

Performance figures were recorded on a self-built offline benchmark with a fixed **single-thread CPU**,
6 prompts, prompt length 18, 64 generated tokens per sequence, warmup=1 and repeats=2. Absolute speed
depends on CPU, PyTorch build and frequency policy, and should be re-measured locally.

> **Calibration note:** the 2.340× / 90.63% pair was measured on this machine with the
> `accepted/proposed` denominator, which counts right-censored tokens as rejections. Under the
> risk-set `accepted/verified` denominator the acceptance rate is approximately 98.6%–99.2%. The
> two denominators are not interchangeable; use [PROJECT_BRIEF.md](PROJECT_BRIEF.md) for performance
> conclusions.

## Six research directions

Around this implementation the project validates six directions; every conclusion is reproduced by
`run_research_pipeline.py` and the analysis artefacts under `analysis/` (46 test cases):

1. **Survival analysis of the acceptance process** (`specforge/stats.py`,
   `analyze_acceptance_theory.py`): right-censoring correction, risk-set denominator, Kaplan–Meier
   hazard, likelihood-ratio test, Wilson CI. Finding: "hazard decays with depth" is a censoring
   artefact; a constant hazard is not rejected and the risk-set acceptance rate is 98.6%–99.2%.
   Artefacts: `analysis/theory_check_v2.*`.
2. **Cost-aware adaptive K** (`specforge/costs.py`, `adaptive_k_benchmark.py`): per-round measured
   C(K), $K^{*} = \arg\min_K C(K)/(\mathbb{E}[A]+1)$, finite-horizon Bellman DP, calibration / held-out separation,
   interleaved timing and paired bootstrap CI. Finding: under a stationary hazard a fixed K is already
   near-optimal; the reproducible benefit is avoiding negative speedup from long K when the hazard
   shifts (hard group: fixed K=28 gives 0.760×, adaptive gives 0.927×).
3. **TV-aware distillation** (`specforge/distill_tv.py`, `run_distill_ablation.py`): switchable TV/KL
   mixed objective, separating the stochastic-mode criterion ($\sum \min(p_i, q_i)$) from the greedy-mode
   criterion (top-1 agreement). 4 objectives × 2 seeds: the Pinsker lower bound holds in all runs and
   differences between objectives are not significant on low-entropy corpora.
4. **High-entropy corpus stress test** (`specforge/data.py` rich mode, `entropy_stress_test.py`): with
   H2 rising 0.637 → 1.668, acceptance falls 98% → 70% and the best speedup goes 2.297× →
   **0.707× (negative)**, showing the speedup depends on a low-entropy corpus.
5. **Distributional equivalence test for the stochastic mode** (`specforge/equivalence.py`,
   `run_equivalence_test.py`): decoder-level TV distance, parametric bootstrap, power curve. Both
   configurations fail to reject equivalence (TV=0.030 / 0.020, p=1.0); the stochastic mode itself
   gives 1.45× / 1.50× speedup.
6. **Attribution of the speedup** (`run_attribution_control.py`): comparing chunked forward against a
   microbenchmark shows dispatch overhead at 92.3%, with a target token ratio of 114.6% at K=32.

Measured figures under the benchmark and calibration protocol: calibration selects K=32
(2.351×, CI [2.11, 2.65]), held-out **2.358×, CI [2.02, 2.78]**, greedy output 100% equivalent.
Stochastic mode (T=1.0): calibration K=32 (2.656×, CI [2.48, 2.81]), held-out **2.723×,
CI [2.64, 2.79]** at a 99.7% risk-set acceptance rate. Correctness for the stochastic mode is not
token-exact match but the distributional equivalence test above.

## Running the project

1. Unpack the archive fully and open the project root with `File → Open` in PyCharm.
2. Select a 64-bit Python 3.10–3.14 interpreter and install dependencies:
   `python -m pip install -r requirements.txt`, then `python -m pip install -e . --no-deps`.
   `requirements-hf.txt` is only needed for the full Hugging Face entry points.
3. Run `check_environment.py`, then `pycharm_fast_train.py`.

Each run writes to `experiments/fast_train_T2.0_a0.75_speed15_verified[_runN]/` without overwriting
earlier results; figures are placed in that directory's `figures/`. Setting
`SPECFORGE_NO_PLOT_POPUP=1` disables automatic figure pop-ups across all entry points.

## Capabilities

- Hand-written Transformer, multi-head attention, block causal masking and layer-wise KV Cache
- `truncate` rollback so rejected candidates never pollute the cache
- Greedy speculative decoding with strict output-equivalence checking
- Stochastic speculative decoding: acceptance probability plus residual-distribution correction

  $$
  p_{\text{acc}}=\min\left(1,\frac{p}{q}\right),\qquad
  p_{\text{res}}=\mathrm{normalize}\left(\max(p-q,0)\right)
  $$
- SFT, temperature-scaled KL+CE distillation, and automatic training / alignment / performance figures
- **Observability baseline:** per offline request it records TTFT, end-to-end P50/P95 latency,
  throughput, acceptance rate, rollback rate, number of target calls and peak target/draft KV Cache
  tokens, saved alongside `benchmark.json`
- 46 unit and integration tests: `python -m unittest discover -s tests -q`, covering cache-rollback
  equivalence, greedy exactness, rejection-sampling identities, censoring statistics and Kaplan–Meier,
  cost model and Bellman DP optimality, plotting tick adaptation and pop-up switch consistency
  (45 pass, 1 skipped when the optional Hugging Face dependencies are absent)

## K ablation

The default entry point tests `K = (4, 8, 12, 16, 20, 24, 28, 32)` with 64 generated tokens. A full
local rerun on 2026-08-27 gave: K=4 1.314×, K=8 1.680×, K=12 1.999×, K=16 2.104×, K=20 2.183×,
K=24 2.200×, K=28 2.340×, K=32 2.095× — acceptance rate and target-call reduction alone do not
determine wall-clock gain.

## Scope

The project uses a programmatic Chinese technical corpus and a miniature Transformer to demonstrate a
correct implementation of speculative decoding, distillation and cache rollback. It must not be
described as a 7B/70B model, a GPU service, or a production speedup.
