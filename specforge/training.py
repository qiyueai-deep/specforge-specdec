from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F

from .config import ModelConfig, TrainConfig, draft_config, target_config
from .data import TokenCorpus
from .distill_tv import build_distill_loss
from .model import TinyTransformerLM
from .tokenizer import CharTokenizer
from .utils import load_json, save_json, set_seed


def _learning_rate(step: int, total_steps: int, config: TrainConfig) -> float:
    progress = step / max(1, total_steps - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_learning_rate + cosine * (config.learning_rate - config.min_learning_rate)


@torch.no_grad()
def init_draft_from_target(target: TinyTransformerLM, draft: TinyTransformerLM) -> None:
    """用目标模型的前几层权重初始化草稿模型。
    要求：两者 d_model、n_heads、d_ff 维度一致，草稿层数 <= 目标层数。
    这样草稿起点就非常接近目标，蒸馏只需微调，步数大幅减少。"""
    assert draft.config.d_model == target.config.d_model, "d_model 必须一致才能拷贝权重"
    assert draft.config.n_layers <= target.config.n_layers, "草稿层数不能超过目标"

    # 词嵌入和位置嵌入直接拷贝
    draft.token_embedding.weight.copy_(target.token_embedding.weight)
    draft.position_embedding.weight.copy_(target.position_embedding.weight)

    # 拷贝前 N 层 Transformer Block
    for i in range(draft.config.n_layers):
        draft.blocks[i].load_state_dict(target.blocks[i].state_dict())

    # final_norm 和 lm_head 也拷贝
    draft.final_norm.load_state_dict(target.final_norm.state_dict())
    # lm_head 与 token_embedding 权重共享，拷贝 embedding 即同步


@torch.no_grad()
def evaluate_model(
    model: TinyTransformerLM,
    corpus: TokenCorpus,
    config: TrainConfig,
    generator: torch.Generator,
) -> float:
    model.eval()
    losses = []
    for _ in range(config.eval_batches):
        x, y = corpus.batch("validation", config.batch_size, config.seq_len, generator)
        logits = model(x)
        losses.append(float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))))
    return sum(losses) / len(losses)


@torch.no_grad()
def alignment_metrics(
    target: TinyTransformerLM,
    draft: TinyTransformerLM,
    corpus: TokenCorpus,
    config: TrainConfig,
    generator: torch.Generator,
) -> dict:
    target.eval()
    draft.eval()
    agreements, kls, draft_losses = [], [], []
    temperature = config.temperature
    for _ in range(config.eval_batches):
        x, y = corpus.batch("validation", config.batch_size, config.seq_len, generator)
        target_logits = target(x)
        draft_logits = draft(x)
        agreements.append(float((target_logits.argmax(-1) == draft_logits.argmax(-1)).float().mean()))
        target_prob = F.softmax(target_logits / temperature, dim=-1)
        target_log = F.log_softmax(target_logits / temperature, dim=-1)
        draft_log = F.log_softmax(draft_logits / temperature, dim=-1)
        kls.append(float((target_prob * (target_log - draft_log)).sum(-1).mean()))
        draft_losses.append(float(F.cross_entropy(draft_logits.reshape(-1, draft_logits.shape[-1]), y.reshape(-1))))
    return {
        "token_top1_agreement": sum(agreements) / len(agreements),
        "kl_target_to_draft": sum(kls) / len(kls),
        "draft_validation_loss": sum(draft_losses) / len(draft_losses),
    }


def _train_ce(
    model: TinyTransformerLM,
    corpus: TokenCorpus,
    config: TrainConfig,
    steps: int,
    generator: torch.Generator,
    label: str,
) -> list[dict]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history: list[dict] = []
    report_every = max(1, steps // 5)
    monitor_offset = 7001 if label == "target" else 7002
    monitor_generator = torch.Generator().manual_seed(config.seed + monitor_offset)
    monitor_x, monitor_y = corpus.batch(
        "validation", config.batch_size, config.seq_len, monitor_generator
    )

    @torch.no_grad()
    def monitor_loss() -> float:
        model.eval()
        monitor_logits = model(monitor_x)
        value = F.cross_entropy(
            monitor_logits.reshape(-1, monitor_logits.shape[-1]),
            monitor_y.reshape(-1),
        )
        model.train()
        return float(value)

    history.append({
        "step": 0,
        "loss": monitor_loss(),
        "train_loss": None,
        "lr": float(config.learning_rate),
    })
    for step in range(steps):
        x, y = corpus.batch("train", config.batch_size, config.seq_len, generator)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = _learning_rate(step, steps, config)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        if step == 0 or (step + 1) % report_every == 0 or step + 1 == steps:
            item = {
                "step": step + 1,
                "loss": monitor_loss(),
                "train_loss": float(loss.detach()),
                "lr": lr,
            }
            history.append(item)
            print(
                f"[{label}] step={step + 1:4d}/{steps} "
                f"val_ce={item['loss']:.4f} train_ce={item['train_loss']:.4f} lr={lr:.6f}"
            )
    return history


def _distill(
    target: TinyTransformerLM,
    draft: TinyTransformerLM,
    corpus: TokenCorpus,
    config: TrainConfig,
    generator: torch.Generator,
) -> list[dict]:
    target.eval()
    draft.train()
    optimizer = torch.optim.AdamW(draft.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history: list[dict] = []
    # More samples make the KL/CE trend and its smoothed endpoint faithful.
    report_every = max(1, config.distill_steps // 22)
    temperature = config.temperature

    # 默认采用连续的单阶段蒸馏：前 55% 固定 alpha，后 45% 平滑降至
    # 0.05 以进一步压低 CE。监控 Total 始终使用初始 alpha，保证可比。
    stage1_steps = int(config.distill_steps * 0.5)
    stage2_steps = config.distill_steps - stage1_steps
    alpha_start = config.distill_alpha
    stage2_lr_max = config.learning_rate * 0.6
    stage2_lr_min = config.min_learning_rate
    # 固定验证探针：所有曲线点都在同一批样本上测量，避免随机 batch
    # 变化造成“训练变好但曲线反而上升”的假趋势。
    monitor_generator = torch.Generator().manual_seed(config.seed + 9001)
    monitor_x, monitor_y = corpus.batch(
        "validation", config.batch_size, config.seq_len, monitor_generator
    )
    draft.eval()
    with torch.no_grad():
        initial_teacher = target(monitor_x)
        initial_student = draft(monitor_x)
        initial_teacher_prob = F.softmax(initial_teacher / temperature, dim=-1)
        initial_student_log = F.log_softmax(initial_student / temperature, dim=-1)
        initial_kd = F.kl_div(
            initial_student_log, initial_teacher_prob, reduction="batchmean"
        ) / monitor_x.shape[1]
        initial_ce = F.cross_entropy(
            initial_student.reshape(-1, initial_student.shape[-1]),
            monitor_y.reshape(-1),
        )
        initial_loss = float(build_distill_loss(
            initial_student, initial_teacher, monitor_y,
            objective=config.distill_objective, alpha=alpha_start,
            temperature=temperature, tv_weight=config.tv_weight,
            tv_scale=config.tv_scale, seq_len=monitor_x.shape[1],
        ))
    history.append({
        "step": 0,
        "loss": float(initial_loss),
        "kd": float(initial_kd),
        "ce": float(initial_ce),
        "train_loss": None,
        "alpha": float(alpha_start),
        "lr": float(config.learning_rate),
        "phase": "fixed-monitor-baseline",
    })
    draft.train()

    for step in range(config.distill_steps):
        # --- 阶段与 α 计算 ---
        if not config.two_stage_distillation:
            progress = step / max(1, config.distill_steps - 1)
            refinement = max(0.0, min(1.0, config.ce_refinement_fraction))
            refinement_start = 1.0 - refinement
            if refinement <= 0.0 or progress <= refinement_start:
                alpha = alpha_start
                phase = "fixed-mix"
            else:
                refine_progress = (progress - refinement_start) / refinement
                smooth = 0.5 - 0.5 * math.cos(math.pi * refine_progress)
                alpha = alpha_start + (config.final_distill_alpha - alpha_start) * smooth
                phase = "ce-refinement"
        elif step < stage1_steps:
            # Stage 1: α 从 alpha_start 线性退火到 0.1
            progress1 = step / max(1, stage1_steps - 1)
            alpha = alpha_start - (alpha_start - 0.1) * progress1
            phase = "mix"
        else:
            # Stage 2: α = 0，纯 CE 优化
            alpha = 0.0
            phase = "ce-only"

        # --- 前向 ---
        x, y = corpus.batch("train", config.batch_size, config.seq_len, generator)
        with torch.no_grad():
            teacher_logits = target(x)
        student_logits = draft(x)
        # Default objective "kl_ce" is algebraically identical to the original
        # alpha*T^2*KL + (1-alpha)*CE loss, so historical numbers stay valid.
        loss = build_distill_loss(
            student_logits, teacher_logits, y,
            objective=config.distill_objective,
            alpha=alpha, temperature=temperature,
            tv_weight=config.tv_weight, tv_scale=config.tv_scale,
            seq_len=x.shape[1],
        )

        # --- 学习率：平滑模式和 Stage 1 使用全局 cosine；Stage 2 才重启 ---
        if phase in ("mix", "fixed-mix", "ce-refinement"):
            lr = _learning_rate(step, config.distill_steps, config)
        else:
            local_step = step - stage1_steps
            progress2 = local_step / max(1, stage2_steps - 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress2))
            lr = stage2_lr_min + cosine * (stage2_lr_max - stage2_lr_min)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(draft.parameters(), 1.0)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()

        if step == 0 or (step + 1) % report_every == 0 or step + 1 == config.distill_steps:
            draft.eval()
            with torch.no_grad():
                monitor_teacher = target(monitor_x)
                monitor_student = draft(monitor_x)
                monitor_teacher_prob = F.softmax(monitor_teacher / temperature, dim=-1)
                monitor_student_log = F.log_softmax(monitor_student / temperature, dim=-1)
                monitor_kd = F.kl_div(
                    monitor_student_log, monitor_teacher_prob, reduction="batchmean"
                ) / monitor_x.shape[1]
                monitor_ce = F.cross_entropy(
                    monitor_student.reshape(-1, monitor_student.shape[-1]),
                    monitor_y.reshape(-1),
                )
                # Use a fixed reporting definition even while optimization
                # alpha changes during the smooth CE-refinement tail.
                monitor_loss = alpha_start * (temperature ** 2) * monitor_kd + (1.0 - alpha_start) * monitor_ce
            draft.train()
            item = {
                "step": step + 1, "loss": float(monitor_loss),
                "kd": float(monitor_kd), "ce": float(monitor_ce),
                "train_loss": float(loss.detach()),
                "alpha": float(alpha), "lr": lr, "phase": phase,
            }
            history.append(item)
            print(
                f"[distill] step={step + 1:4d}/{config.distill_steps} "
                f"loss={item['loss']:.4f} kd={item['kd']:.4f} ce={item['ce']:.4f} "
                f"alpha={item['alpha']:.3f} lr={lr:.5f} [{phase}]"
            )
    return history


def save_checkpoint(path: str | Path, model: TinyTransformerLM) -> None:
    torch.save({"config": model.config.to_dict(), "state_dict": model.state_dict()}, Path(path))


def load_checkpoint(path: str | Path) -> TinyTransformerLM:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    model = TinyTransformerLM(ModelConfig.from_dict(payload["config"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def load_artifacts(artifact_dir: str | Path) -> tuple[TinyTransformerLM, TinyTransformerLM, CharTokenizer, dict]:
    artifact_dir = Path(artifact_dir)
    target = load_checkpoint(artifact_dir / "target.pt")
    draft = load_checkpoint(artifact_dir / "draft.pt")
    tokenizer = CharTokenizer.load(artifact_dir / "tokenizer.json")
    manifest = load_json(artifact_dir / "manifest.json")
    return target, draft, tokenizer, manifest


def train_project(output_dir: str | Path, config: TrainConfig | None = None) -> dict:
    config = config or TrainConfig()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config.torch_threads)
    set_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed + 1)
    corpus = TokenCorpus.create(config.seed)
    corpus.tokenizer.save(output / "tokenizer.json")
    target = TinyTransformerLM(target_config(corpus.tokenizer.vocab_size))
    draft = TinyTransformerLM(draft_config(corpus.tokenizer.vocab_size))
    print(f"vocab={corpus.tokenizer.vocab_size} target_params={target.parameter_count:,} draft_params={draft.parameter_count:,}")

    start = time.perf_counter()
    target_before = evaluate_model(target, corpus, config, generator)
    target_history = _train_ce(target, corpus, config, config.target_steps, generator, "target")
    target_after = evaluate_model(target, corpus, config, generator)
    draft_before = alignment_metrics(target, draft, corpus, config, generator)

    # 策略：维度一致时从目标拷贝前几层权重（天然对齐好，蒸馏步数少）
    # 维度不一致时退回传统 SFT + 蒸馏
    if draft.config.d_model == target.config.d_model:
        print(f"\n  [init] 从目标模型拷贝前 {draft.config.n_layers} 层权重到草稿模型")
        init_draft_from_target(target, draft)
        after_sft = alignment_metrics(target, draft, corpus, config, generator)
        draft_sft_history = [{"step": 0, "loss": after_sft["kl_target_to_draft"],
                              "kl": after_sft["kl_target_to_draft"], "ce": 0.0}]
        print(f"  [init] 拷贝后 Top-1 一致率: {after_sft['token_top1_agreement']:.2%}, KL: {after_sft['kl_target_to_draft']:.4f}")
    else:
        draft_sft_history = _train_ce(draft, corpus, config, config.draft_sft_steps, generator, "draft-sft")
        after_sft = alignment_metrics(target, draft, corpus, config, generator)

    distill_history = _distill(target, draft, corpus, config, generator)
    after_distill = alignment_metrics(target, draft, corpus, config, generator)
    elapsed = time.perf_counter() - start

    save_checkpoint(output / "target.pt", target)
    save_checkpoint(output / "draft.pt", draft)
    manifest = {
        "project": "SpecForge", "version": 1, "train_config": config.to_dict(),
        "target_config": target.config.to_dict(), "draft_config": draft.config.to_dict(),
        "target_parameters": target.parameter_count, "draft_parameters": draft.parameter_count,
        "parameter_compression_ratio": target.parameter_count / draft.parameter_count,
        "metrics": {
            "target_validation_loss_before": target_before,
            "target_validation_loss_after": target_after,
            "draft_before_sft": draft_before,
            "draft_after_sft": after_sft,
            "draft_after_distillation": after_distill,
            "training_wall_seconds": elapsed,
        },
        "history": {"target": target_history, "draft_sft": draft_sft_history, "distillation": distill_history},
    }
    save_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Train target and distilled draft Transformer models")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--quick", action="store_true", help="Use a smaller smoke-test training schedule")
    parser.add_argument("--target-steps", type=int)
    parser.add_argument("--draft-sft-steps", type=int)
    parser.add_argument("--distill-steps", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--distill-alpha", type=float)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--threads", type=int)
    args = parser.parse_args()
    config = TrainConfig()
    if args.quick:
        config.target_steps = 20
        config.draft_sft_steps = 10
        config.distill_steps = 20
        config.eval_batches = 2
    for argument, attribute in (
        (args.target_steps, "target_steps"),
        (args.draft_sft_steps, "draft_sft_steps"),
        (args.distill_steps, "distill_steps"),
        (args.temperature, "temperature"),
        (args.distill_alpha, "distill_alpha"),
        (args.learning_rate, "learning_rate"),
        (args.threads, "torch_threads"),
    ):
        if argument is not None:
            setattr(config, attribute, argument)
    manifest = train_project(args.output_dir, config)
    print(f"saved to {args.output_dir}")
    print(f"target loss: {manifest['metrics']['target_validation_loss_after']:.4f}")
    print(f"agreement: {manifest['metrics']['draft_after_distillation']['token_top1_agreement']:.2%}")


if __name__ == "__main__":
    main()
