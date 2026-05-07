"""
Early checkpoint diagnostics for the first generalization spike.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from model_io import safe_torch_load


DEFAULT_OUTPUT_DIR = "checkpoint_add_restricted_early_analysis"
DEFAULT_CHECKPOINT_DIR = "checkpoints"
DEFAULT_STEPS = (5301, 5401)
DEFAULT_EXAMPLE = "PAD PAD a 40 PAD j 18 PAD PAD PAD k 9 + a 27 ="
ATTENTION_LAYER_IDX = 0


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write early-spike QK/attention arrays."
    )
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--step",
        type=int,
        action="append",
        dest="steps",
        default=None,
        help="Checkpoint step to include. Defaults to 5301 and 5401.",
    )
    parser.add_argument("--example", default=DEFAULT_EXAMPLE)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    return parser.parse_args()


def build_model(run_cfg: LocalRunConfig, device: torch.device) -> HookedTransformer:
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


def load_checkpoint(model: HookedTransformer, checkpoint_dir: Path, step: int) -> None:
    path = checkpoint_dir / f"checkpoint_step_{step}.pth"
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    state_dict = safe_torch_load(path, map_location=model.cfg.device)
    model.load_state_dict(state_dict)
    model.eval()


def symbol_to_token_map(run_cfg: LocalRunConfig) -> dict[str, int]:
    mapping = {str(i): i for i in range(run_cfg.mod)}
    mapping.update(
        {
            "+": run_cfg.plus_id,
            "=": run_cfg.equal_id,
            "PAD": run_cfg.pad_id,
        }
    )
    mapping.update(
        {chr(ord("a") + idx): token_id for idx, token_id in enumerate(run_cfg.var_ids)}
    )
    return mapping


def parse_example(example: str, run_cfg: LocalRunConfig) -> torch.Tensor:
    mapping = symbol_to_token_map(run_cfg)
    token_ids = [mapping[symbol] for symbol in example.split()]
    if len(token_ids) != run_cfg.seq_len:
        raise ValueError(
            f"Expected {run_cfg.seq_len} tokens, got {len(token_ids)} in {example!r}"
        )
    return torch.tensor(token_ids, dtype=torch.long)


def token_labels(tokens: torch.Tensor, run_cfg: LocalRunConfig) -> list[str]:
    return [run_cfg.token_strings[int(token)] for token in tokens.tolist()]


def target_value_positions(tokens: torch.Tensor, run_cfg: LocalRunConfig) -> list[int]:
    positions = []
    for operand_pos in (run_cfg.seq_len - 3, run_cfg.seq_len - 2):
        token_id = int(tokens[operand_pos])
        if token_id < run_cfg.mod:
            positions.append(operand_pos)
            continue

        matches = [
            pos + 1
            for pos in range(run_cfg.seq_len - 1)
            if int(tokens[pos]) == token_id and int(tokens[pos + 1]) < run_cfg.mod
        ]
        if not matches:
            raise ValueError(f"No assignment found for operand token {token_id}")
        positions.append(matches[0])
    return positions


def qk1_var_identity(model: HookedTransformer, run_cfg: LocalRunConfig) -> torch.Tensor:
    qk1 = (model.blocks[1].attn.W_Q[0] @ model.blocks[1].attn.W_K[0].T).detach()
    ov0 = (model.blocks[0].attn.W_V[0] @ model.blocks[0].attn.W_O[0]).detach()
    var_emb = model.embed.W_E[list(run_cfg.var_ids)].detach()
    var_through_ov = var_emb @ ov0
    return (var_through_ov @ qk1 @ var_through_ov.T).detach().cpu()


def attention_pattern(
    model: HookedTransformer,
    tokens: torch.Tensor,
    layer: int,
) -> torch.Tensor:
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens.unsqueeze(0), return_type="logits")
    return cache["attn", layer][0, 0].detach().cpu()


def diag_margin(matrix: torch.Tensor) -> float:
    off_diag = matrix.clone()
    idx = torch.arange(matrix.shape[0])
    off_diag[idx, idx] = -math.inf
    return float((matrix.diag() - off_diag.max(dim=-1).values).mean())


def main() -> None:
    args = parse_args()
    steps = list(args.steps or DEFAULT_STEPS)
    run_cfg = LocalRunConfig()
    device = torch.device(args.device)
    model = build_model(run_cfg, device)
    tokens = parse_example(args.example, run_cfg).to(device)
    labels = token_labels(tokens.cpu(), run_cfg)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    attention_by_step = {}
    var_identity_by_step = {}
    margins = {}
    for step in steps:
        load_checkpoint(model, Path(args.checkpoint_dir), step)
        attention_by_step[str(step)] = attention_pattern(
            model,
            tokens,
            layer=ATTENTION_LAYER_IDX,
        ).tolist()
        var_identity = qk1_var_identity(model, run_cfg)
        var_identity_by_step[str(step)] = var_identity.tolist()
        margins[str(step)] = diag_margin(var_identity)

    data = {
        "steps": steps,
        "attention_layer": ATTENTION_LAYER_IDX,
        "example": args.example,
        "token_labels": labels,
        "target_value_positions": target_value_positions(tokens.cpu(), run_cfg),
        "var_labels": [run_cfg.token_strings[var_id] for var_id in run_cfg.var_ids],
        "attention_patterns": attention_by_step,
        "var_identity_qk1": var_identity_by_step,
        "var_identity_diag_margin": margins,
    }
    path = output_dir / "early_add_restricted_analysis.json"
    path.write_text(json.dumps(data, indent=2))
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
