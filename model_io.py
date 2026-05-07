from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig


DEFAULT_CHECKPOINT_DIR = Path("checkpoints")
DEFAULT_FINAL_CHECKPOINT = DEFAULT_CHECKPOINT_DIR / "model_state_dict.pth"


@dataclass(frozen=True)
class LocalRunConfig:
    mod: int = 59
    vocab: int = 74
    seq_len: int = 16
    plus_id: int = 59
    equal_id: int = 60
    pad_id: int = 61
    a_token_id: int = 62
    n_vars: int = 12
    n_layers: int = 2
    n_heads: int = 1
    d_model: int = 128
    seed: int = 42

    @property
    def var_ids(self) -> tuple[int, ...]:
        return tuple(range(self.a_token_id, self.a_token_id + self.n_vars))

    @property
    def token_strings(self) -> tuple[str, ...]:
        return tuple(
            [str(i) for i in range(self.mod)]
            + ["+", "=", "PAD"]
            + [chr(ord("a") + i) for i in range(self.n_vars)]
        )


def default_config() -> dict:
    run_cfg = LocalRunConfig()
    return {
        "SEQ_LEN": run_cfg.seq_len,
        "BATCH_SIZE": 512,
        "D_MODEL": run_cfg.d_model,
        "N_HEADS": run_cfg.n_heads,
        "N_LAYERS": run_cfg.n_layers,
        "LR": 1e-3,
        "WD": 0.02,
        "MAX_STEPS": 40_000,
        "EVAL_EVERY": 100,
        "MOD": run_cfg.mod,
        "VOCAB": run_cfg.vocab,
        "PLUS_ID": run_cfg.plus_id,
        "EQUAL_ID": run_cfg.equal_id,
        "PAD_ID": run_cfg.pad_id,
        "A_TOKEN_ID": run_cfg.a_token_id,
        "VAR_LEN": run_cfg.n_vars,
        "VARS": list(run_cfg.var_ids),
        "TOKEN_STRINGS": list(run_cfg.token_strings),
        "NUMS_TRAIN_PAIRS": int(0.7 * run_cfg.mod * run_cfg.mod),
        "TRAIN_PAIRS_FRAC": 0.7,
        "TWO_VAR_FREQUENCY": 1.0,
        "SEED": run_cfg.seed,
        "RESTRICT_LEFT_HALF_VARS": 2,
        "RESTRICT_RIGHT_HALF_VARS": 2,
        "TRAIN_FRAC": 0.8,
        "USE_SIMPLE_16": False,
    }


def normalize_config(config: dict | None = None) -> dict:
    merged = default_config()
    if config:
        merged.update(config)
    return merged


def build_run_config(config: dict | None = None) -> LocalRunConfig:
    cfg = normalize_config(config)
    vars_ = [int(v) for v in cfg.get("VARS", [])]
    n_vars = int(cfg.get("VAR_LEN", len(vars_) or 12))
    return LocalRunConfig(
        mod=int(cfg["MOD"]),
        vocab=int(cfg["VOCAB"]),
        seq_len=int(cfg["SEQ_LEN"]),
        plus_id=int(cfg["PLUS_ID"]),
        equal_id=int(cfg["EQUAL_ID"]),
        pad_id=int(cfg["PAD_ID"]),
        a_token_id=int(cfg["A_TOKEN_ID"]),
        n_vars=n_vars,
        n_layers=int(cfg["N_LAYERS"]),
        n_heads=int(cfg["N_HEADS"]),
        d_model=int(cfg["D_MODEL"]),
        seed=int(cfg["SEED"]),
    )


def build_model(
    run_cfg: LocalRunConfig | None = None,
    device: torch.device | str = "cpu",
) -> HookedTransformer:
    run_cfg = run_cfg or LocalRunConfig()
    cfg = HookedTransformerConfig(
        n_layers=run_cfg.n_layers,
        n_heads=run_cfg.n_heads,
        d_model=run_cfg.d_model,
        d_head=run_cfg.d_model // run_cfg.n_heads,
        d_mlp=4 * run_cfg.d_model,
        n_ctx=run_cfg.seq_len,
        d_vocab=run_cfg.vocab,
        d_vocab_out=run_cfg.mod,
        act_fn="relu",
        device=device,
        normalization_type=None,
        seed=run_cfg.seed,
    )
    model = HookedTransformer(cfg).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


_CHECKPOINT_RE = re.compile(r"checkpoint_step_(\d+)\.pth$")


def discover_checkpoints(checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR) -> list[int]:
    checkpoint_dir = Path(checkpoint_dir)
    steps: list[int] = []
    for path in checkpoint_dir.glob("checkpoint_step_*.pth"):
        match = _CHECKPOINT_RE.fullmatch(path.name)
        if match is not None and path.stat().st_size > 0:
            steps.append(int(match.group(1)))
    return sorted(steps)


def checkpoint_path(step: int, checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR) -> Path:
    path = Path(checkpoint_dir) / f"checkpoint_step_{step}.pth"
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing checkpoint for step {step}: {path}")
    return path


def select_checkpoint_steps(
    checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
    *,
    step_start: int | None = None,
    step_end: int | None = None,
    step_stride: int | None = None,
    testing_mode: bool = False,
) -> list[int]:
    steps = discover_checkpoints(checkpoint_dir)
    if step_start is not None:
        steps = [step for step in steps if step >= step_start]
    if step_end is not None:
        steps = [step for step in steps if step <= step_end]
    if step_stride is not None and step_start is not None:
        steps = [step for step in steps if (step - step_start) % step_stride == 0]
    if testing_mode and len(steps) > 2:
        steps = [steps[0], steps[-1]]
    return steps


def safe_torch_load(path: str | Path, map_location: torch.device | str = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


torch_load = safe_torch_load


def load_checkpoint(
    model: HookedTransformer,
    step: int,
    checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
    device: torch.device | str | None = None,
) -> None:
    load_device = device if device is not None else model.cfg.device
    state_dict = torch_load(checkpoint_path(step, checkpoint_dir), load_device)
    model.load_state_dict(state_dict)
    model.eval()


def load_model_from_checkpoint(
    checkpoint: str | Path = DEFAULT_FINAL_CHECKPOINT,
    *,
    config: dict | None = None,
    device: torch.device | str = "cpu",
) -> tuple[HookedTransformer, dict, LocalRunConfig]:
    cfg = normalize_config(config)
    run_cfg = build_run_config(cfg)
    model = build_model(run_cfg, device)
    state_dict = torch_load(checkpoint, device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, cfg, run_cfg


def checkpoint_pairs(steps: Iterable[int]) -> list[tuple[int, Path]]:
    return [(step, checkpoint_path(step)) for step in steps]
