from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from .tokenizer import CharTokenizer


SUBJECTS = ("推测解码", "知识蒸馏", "注意力缓存", "语言模型", "并行验证", "拒绝采样", "草稿模型", "目标模型")
ACTIONS = ("减少串行调用", "提高生成吞吐", "保持采样分布", "复用历史状态", "预测候选词元", "校准接受概率")
CONDITIONS = ("候选分布接近时", "批量长度合适时", "缓存正确回滚后", "训练与生成分布一致时", "线程数固定时")
RESULTS = ("系统能够稳定运行", "端到端延迟明显下降", "输出仍与目标模型一致", "验证成本得到摊薄", "错误不会沿上下文传播")
NAMES = ("小明", "小林", "研究员", "工程师", "算法团队", "测试程序")
OBJECTS = ("蓝色机器人", "城市列车", "智能仓库", "天气传感器", "搜索助手", "代码生成器")
PLACES = ("实验室", "数据中心", "图书馆", "车站", "云端平台", "安静办公室")
NUMBERS = ("二", "三", "四", "五", "六", "八")


def build_corpus(seed: int = 20260820, train_sentences: int = 3200, validation_sentences: int = 500) -> tuple[str, str]:
    rng = random.Random(seed)

    def sentence(index: int) -> str:
        mode = index % 8
        subject = rng.choice(SUBJECTS)
        action = rng.choice(ACTIONS)
        condition = rng.choice(CONDITIONS)
        result = rng.choice(RESULTS)
        if mode == 0:
            return f"{subject}通过{action}，在{condition}，{result}。"
        if mode == 1:
            return f"问题：为什么需要{subject}？回答：因为它可以{action}，并且{result}。"
        if mode == 2:
            return f"步骤一生成{rng.choice(NUMBERS)}个候选，步骤二并行验证，步骤三回滚缓存并继续生成。"
        if mode == 3:
            return f"{rng.choice(NAMES)}在{rng.choice(PLACES)}测试{rng.choice(OBJECTS)}，记录速度、接受率和输出一致性。"
        if mode == 4:
            return f"当草稿词元被接受时保留缓存；当词元被拒绝时截断缓存，再从修正分布采样。"
        if mode == 5:
            return f"实验编号{rng.choice(NUMBERS)}：固定线程和序列长度，对比自回归解码与推测解码的每秒词元数。"
        if mode == 6:
            return f"温度升高会平滑概率分布，蒸馏损失同时包含交叉熵与相对熵。"
        return f"可靠的性能结论必须报告硬件、批大小、候选长度、预热次数和重复实验。"

    train = "\n".join(sentence(i) for i in range(train_sentences))
    validation = "\n".join(sentence(i + train_sentences) for i in range(validation_sentences))
    return train, validation


@dataclass
class TokenCorpus:
    train_ids: torch.Tensor
    validation_ids: torch.Tensor
    tokenizer: CharTokenizer
    mode: str = "template"

    @classmethod
    def create(cls, seed: int = 20260820, mode: str = "template") -> "TokenCorpus":
        """mode="template" reproduces the original low-entropy corpus exactly.

        mode="rich" generates a sparse Markov-chain corpus with a much higher
        conditional entropy, used as a stress test for the decoding algorithm.
        mode="file" reads a real text file from data/corpus.txt.
        """
        if mode == "template":
            train_text, validation_text = build_corpus(seed)
        elif mode == "rich":
            train_text, validation_text = build_rich_corpus(seed)
        elif mode == "file":
            train_text, validation_text = build_corpus_from_file(seed)
        else:
            raise ValueError(f"unknown corpus mode: {mode}")
        tokenizer = CharTokenizer.from_text(train_text + validation_text)
        train_ids = torch.tensor(tokenizer.encode(train_text, add_bos=True, add_eos=True), dtype=torch.long)
        validation_ids = torch.tensor(tokenizer.encode(validation_text, add_bos=True, add_eos=True), dtype=torch.long)
        return cls(train_ids, validation_ids, tokenizer, mode)

    def batch(self, split: str, batch_size: int, seq_len: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.train_ids if split == "train" else self.validation_ids
        starts = torch.randint(0, len(data) - seq_len - 1, (batch_size,), generator=generator)
        x = torch.stack([data[start:start + seq_len] for start in starts.tolist()])
        y = torch.stack([data[start + 1:start + seq_len + 1] for start in starts.tolist()])
        return x, y

    def validation_prompts(self, count: int = 8, prompt_len: int = 18) -> list[list[int]]:
        step = max(1, (len(self.validation_ids) - prompt_len - 1) // count)
        return [self.validation_ids[i * step:i * step + prompt_len].tolist() for i in range(count)]


# ---------------------------------------------------------------------------
# High-entropy stress-test corpus
# ---------------------------------------------------------------------------
# The original corpus is eight fixed sentence templates over 247 characters.
# A target model reaches validation CE 0.11 on it, so the next token is nearly
# deterministic and the 90%+ acceptance rate says little about the algorithm.
# The rich corpus is a sparse random Markov chain over a larger alphabet: its
# conditional entropy is much higher, which brings the acceptance rate into the
# range where speculative decoding is actually challenging.

RICH_ALPHABET = (
    "的一是在不了有人我他这为之大来以个中上们到说国和地也子时道出而要于就下得可你年生自会那后能对着事其里所去行过家十用发天如然作方成者多日都三小军二无同么经法当起与好看学进种将还分此心前面又定见只主没公从"
    "数据流模型训练推理缓存验证采样分布概率统计随机过程马尔可夫链注意力蒸馏解码候选长度吞吐延迟线程批量权重梯度损失熵"
    "实验报告基线对照消融置信区间显著水平自助重采样生存分析风险集右删失检验功效总变差不等式"
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "，。；：？！、（）【】《》"
)


def build_rich_corpus(
    seed: int = 20260820,
    train_chars: int = 180000,
    validation_chars: int = 30000,
) -> tuple[str, str]:
    """A reproducible, high conditional-entropy character stream."""
    rng = random.Random(seed + 7919)
    alphabet = list(dict.fromkeys(RICH_ALPHABET))
    m = len(alphabet)
    successors = {c: [alphabet[rng.randrange(m)] for _ in range(6)] for c in alphabet}

    def stream(length: int) -> str:
        c = alphabet[rng.randrange(m)]
        chars = []
        for _ in range(length):
            chars.append(c)
            r = rng.random()
            if r < 0.34:
                c = successors[c][0]
            elif r < 0.56:
                c = successors[c][1]
            elif r < 0.72:
                c = successors[c][2]
            elif r < 0.84:
                c = successors[c][3]
            elif r < 0.93:
                c = successors[c][4]
            else:
                c = successors[c][5]
            if rng.random() < 0.01:
                c = alphabet[rng.randrange(m)]
        return "".join(chars)

    return stream(train_chars), stream(validation_chars)


def build_corpus_from_file(
    seed: int = 20260820,
    path: str = "data/corpus.txt",
    validation_fraction: float = 0.1,
) -> tuple[str, str]:
    """Load real text when available. The split is deterministic, not shuffled."""
    from pathlib import Path

    source = Path(__file__).resolve().parent.parent / path
    if not source.exists():
        raise FileNotFoundError(
            f"{source} not found. Place a UTF-8 text corpus there to use mode='file'."
        )
    text = source.read_text(encoding="utf-8")
    cut = int(len(text) * (1.0 - validation_fraction))
    return text[:cut], text[cut:]


def corpus_entropy_report(ids, tokenizer=None) -> dict:
    """Shannon entropies (nats) of the token stream.

    H1 is the unigram entropy, H2 is H(X_t | X_{t-1}) estimated from bigram
    counts. predictability = 1 - H2/H1 quantifies how much of the uncertainty
    the previous token removes. This is the quantity that makes the original
    template corpus an easy benchmark.
    """
    import math
    from collections import Counter

    seq = ids.tolist() if hasattr(ids, "tolist") else list(ids)
    n = len(seq)
    if n < 2:
        raise ValueError("corpus too short")
    counts = Counter(seq)
    h1 = -sum((c / n) * math.log(c / n) for c in counts.values())
    pairs = Counter(zip(seq[:-1], seq[1:]))
    previous = Counter(seq[:-1])
    h2 = 0.0
    for (a, _b), c in pairs.items():
        p = c / previous[a]
        h2 -= (previous[a] / (n - 1)) * p * math.log(p)
    return {
        "tokens": n,
        "vocab_size": len(counts) if tokenizer is None else tokenizer.vocab_size,
        "unigram_entropy_nats": h1,
        "bigram_conditional_entropy_nats": h2,
        "predictability": 1.0 - h2 / max(1e-12, h1),
    }
