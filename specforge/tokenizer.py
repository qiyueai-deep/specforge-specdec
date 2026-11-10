from __future__ import annotations

import json
from pathlib import Path


class CharTokenizer:
    PAD = "<pad>"
    BOS = "<bos>"
    EOS = "<eos>"
    UNK = "<unk>"

    def __init__(self, chars: list[str]):
        special = [self.PAD, self.BOS, self.EOS, self.UNK]
        clean = sorted(set(chars) - set(special))
        self.tokens = special + clean
        self.token_to_id = {token: index for index, token in enumerate(self.tokens)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(list(text))

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.PAD]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[self.BOS]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.EOS]

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = [self.token_to_id.get(char, self.token_to_id[self.UNK]) for char in text]
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        pieces: list[str] = []
        special = {self.PAD, self.BOS, self.EOS, self.UNK}
        for index in ids:
            token = self.tokens[int(index)]
            if skip_special and token in special:
                continue
            pieces.append(token if token != self.UNK else "�")
        return "".join(pieces)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"tokens": self.tokens}, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls([])
        obj.tokens = data["tokens"]
        obj.token_to_id = {token: index for index, token in enumerate(obj.tokens)}
        return obj
