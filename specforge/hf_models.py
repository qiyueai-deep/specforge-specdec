"""Hugging Face target/draft model loading and compatibility checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
import json
import logging
import os
from pathlib import Path

import torch


_PROJECT_ROOT = Path(__file__).absolute().parents[1]
_MODEL_CACHE_ROOT = Path(
    os.environ.get("SPECFORGE_MODEL_CACHE", _PROJECT_ROOT / ".model_cache")
).expanduser().absolute()

# Keep ModelScope away from locked Windows user-profile directories.  The
# project-local cache is writable from PyCharm, resumable, and excluded from
# source archives via .gitignore.  Advanced users can override it with
# SPECFORGE_MODEL_CACHE before starting Python.
os.environ.setdefault("MODELSCOPE_HOME", str(_MODEL_CACHE_ROOT / "config"))
os.environ.setdefault("MODELSCOPE_CACHE", str(_MODEL_CACHE_ROOT / "models"))


def model_cache_root() -> Path:
    """Effective model cache root, honouring the SPECFORGE_MODEL_CACHE override.

    Prefer this over touching ``.model_cache`` directly so tools agree on where
    the weights live when the cache is relocated.
    """
    return _MODEL_CACHE_ROOT


# Windows without Developer Mode cannot create Hugging Face cache symlinks.
# Downloads still work; suppress only the repetitive warning, not real errors.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# hf-xet can remain at 0 bytes on some Windows/network combinations.  Direct
# HTTPS is slower in theory but substantially more reliable and resumable here.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")


@dataclass(frozen=True)
class HFModelConfig:
    target_name: str = "Qwen/Qwen3-1.7B"
    draft_name: str = "Qwen/Qwen3-0.6B"
    device: str = "auto"
    trust_remote_code: bool = False
    local_files_only: bool = False
    download_provider: str = "modelscope"

    def to_dict(self) -> dict:
        return asdict(self)


def require_transformers() -> None:
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError(
            "Hugging Face dependencies are missing. Run: "
            "python -m pip install -r requirements-hf.txt"
        )


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        return torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def parameter_count(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


class _UnauthenticatedWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "unauthenticated requests to the HF Hub" not in record.getMessage()


def _hub_token() -> str | None:
    """Use HF_TOKEN or a token saved by `hf auth login`, without printing it."""
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:
        return None


def _complete_modelscope_snapshot(repo_id: str) -> Path | None:
    """Return a verified local ModelScope snapshot without touching network."""
    repo_dir = repo_id.replace("/", "--")
    snapshot = _MODEL_CACHE_ROOT / "models" / "models" / repo_dir / "snapshots" / "master"
    if not snapshot.is_dir():
        return None
    required_metadata = ("config.json", "tokenizer_config.json", "tokenizer.json")
    if any(not (snapshot / name).is_file() for name in required_metadata):
        return None
    weight_files = list(snapshot.glob("*.safetensors"))
    if not weight_files or any(path.stat().st_size < 1_000_000 for path in weight_files):
        return None
    index_path = snapshot / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            declared = set(index["weight_map"].values())
        except (KeyError, TypeError, ValueError, OSError):
            return None
        if not declared or any(
            not (snapshot / filename).is_file()
            or (snapshot / filename).stat().st_size < 1_000_000
            for filename in declared
        ):
            return None
    return snapshot.absolute()


def download_hf_models(config: HFModelConfig) -> tuple[str, str]:
    """Serial, resumable pre-download that avoids Windows Xet stalls."""
    require_transformers()
    if config.download_provider == "modelscope":
        if importlib.util.find_spec("modelscope") is None:
            raise RuntimeError(
                "ModelScope is required for the reliable China download path. "
                "Run: python -m pip install modelscope"
            )
        print(f"ModelScope cache: {_MODEL_CACHE_ROOT}")

        paths = []
        for role, repo_id in (("Target", config.target_name), ("Draft", config.draft_name)):
            cached = _complete_modelscope_snapshot(repo_id)
            if cached is not None:
                paths.append(str(cached))
                print(f"{role} verified local cache: {cached}")
                continue
            # Import only when a network download is actually required.  This
            # keeps fully cached runs independent of ModelScope API changes.
            try:
                from modelscope import snapshot_download as modelscope_snapshot_download
            except ImportError:
                try:
                    from modelscope.hub.snapshot_download import (
                        snapshot_download as modelscope_snapshot_download,
                    )
                except ImportError:
                    from modelscope_hub.compat.snapshot_download import (
                        snapshot_download as modelscope_snapshot_download,
                    )
            print(f"Downloading {role} from ModelScope: {repo_id} (resumable)")
            path = modelscope_snapshot_download(repo_id, revision="master")
            weight_files = list(Path(path).glob("*.safetensors"))
            if not weight_files or any(file.stat().st_size < 1_000_000 for file in weight_files):
                raise RuntimeError(f"ModelScope returned incomplete weights for {repo_id}: {path}")
            paths.append(str(Path(path).absolute()))
            print(f"{role} cache ready: {paths[-1]}")
        return paths[0], paths[1]
    if config.download_provider != "huggingface":
        raise ValueError("download_provider must be 'modelscope' or 'huggingface'")

    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    token = _hub_token()
    if token is None:
        print("[INFO] Downloading public models without HF_TOKEN (lower rate limits).")
        print("       Optional login: hf auth login")
        _suppress_unauthenticated_notice()
    allow_patterns = [
        "*.json", "*.safetensors", "*.model", "*.txt",
        "tokenizer*", "vocab*", "merges*",
    ]
    paths = []
    for role, repo_id in (("Target", config.target_name), ("Draft", config.draft_name)):
        print(f"Downloading {role}: {repo_id} (serial HTTPS, resumable)")
        api = HfApi(token=token)
        repo_files = api.list_repo_files(repo_id=repo_id)
        weight_files = sorted(
            name for name in repo_files
            if name.endswith(".safetensors") and "/" not in name
        )
        if not weight_files:
            raise RuntimeError(f"No safetensors weights found in {repo_id}")
        path = snapshot_download(
            repo_id=repo_id,
            token=token,
            local_files_only=config.local_files_only,
            allow_patterns=allow_patterns,
            max_workers=1,
        )
        # On some Windows installations snapshot_download may return while a
        # large Xet-backed shard is still only an .incomplete file.  Download
        # every declared weight explicitly and verify the resolved local file.
        for index, filename in enumerate(weight_files, start=1):
            print(f"  weight {index}/{len(weight_files)}: {filename}")
            resolved = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                token=token,
                local_files_only=config.local_files_only,
            )
            resolved_path = os.path.abspath(resolved)
            if not os.path.isfile(resolved_path) or os.path.getsize(resolved_path) < 1_000_000:
                raise RuntimeError(f"Incomplete model weight: {resolved_path}")
        paths.append(path)
        print(f"{role} cache ready: {path}")
    return paths[0], paths[1]


def _suppress_unauthenticated_notice() -> None:
    notice_filter = _UnauthenticatedWarningFilter()
    for logger_name in ("huggingface_hub", "huggingface_hub.utils._http"):
        logging.getLogger(logger_name).addFilter(notice_filter)
    for handler in logging.getLogger().handlers:
        handler.addFilter(notice_filter)


def validate_model_pair(target, draft, tokenizer) -> None:
    target_vocab = int(target.config.vocab_size)
    draft_vocab = int(draft.config.vocab_size)
    tokenizer_vocab = len(tokenizer)
    if target_vocab != draft_vocab:
        raise ValueError(
            f"Target/Draft vocab mismatch: {target_vocab} != {draft_vocab}. "
            "Use models from the same family with a shared token space."
        )
    if tokenizer_vocab > target_vocab:
        raise ValueError(
            f"Tokenizer contains {tokenizer_vocab} tokens but model vocab has {target_vocab}."
        )


def load_hf_models(config: HFModelConfig):
    """Load a deterministic target/draft pair without altering the tiny-model path."""
    require_transformers()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = resolve_device(config.device)
    dtype = resolve_dtype(device)
    token = _hub_token()
    if token is None and not config.local_files_only:
        print("[INFO] HF_TOKEN not found; public downloads will work with lower rate limits.")
        print("       Optional login: hf auth login")
        _suppress_unauthenticated_notice()
    tokenizer = AutoTokenizer.from_pretrained(
        config.target_name,
        trust_remote_code=config.trust_remote_code,
        local_files_only=config.local_files_only,
        token=token,
    )
    draft_tokenizer = AutoTokenizer.from_pretrained(
        config.draft_name,
        trust_remote_code=config.trust_remote_code,
        local_files_only=config.local_files_only,
        token=token,
    )
    if tokenizer.get_vocab() != draft_tokenizer.get_vocab():
        raise ValueError(
            "Target and Draft tokenizer mappings differ. Assisted generation requires "
            "the same token IDs for the same text."
        )
    common = {
        "dtype": dtype,
        "trust_remote_code": config.trust_remote_code,
        "local_files_only": config.local_files_only,
        "token": token,
    }
    # low_cpu_mem_usage requires accelerate; use it when installed.
    if importlib.util.find_spec("accelerate") is not None:
        common["low_cpu_mem_usage"] = True
    target = AutoModelForCausalLM.from_pretrained(config.target_name, **common).to(device).eval()
    draft = AutoModelForCausalLM.from_pretrained(config.draft_name, **common).to(device).eval()
    validate_model_pair(target, draft, tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return target, draft, tokenizer, device
