"""
Late-stage checkpoint analysis for 2-var var-restricted generalization.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/cache")

import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable

try:
    from tueplots import bundles
except ImportError:
    bundles = None

from eval_pools import _pool_from_cfg, get_restriction_vars
from model_io import build_model, build_run_config, default_config, load_checkpoint


DEFAULT_CACHE_DIR = "checkpoints"
DEFAULT_OUTPUT_DIR = "checkpoint_var_restricted_late_analysis"
DEFAULT_STEP_START = 14001
DEFAULT_STEP_END = 16101
DEFAULT_STEP_STRIDE = 100
DEFAULT_FOCUS_STEPS = (14501, 14901, 15401, 15501, 15601, 16001)
DEFAULT_NUMBER_PAIRS = ((16, 34), (18, 36), (20, 40), (7, 48))

_FONT_SANS_SERIF = ["Helvetica", "Arial", "DejaVu Sans"]
if bundles is not None:
    plt.rcParams.update(bundles.neurips2024(usetex=False))
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = _FONT_SANS_SERIF

FONT_SIZE_BODY = 9
FONT_SIZE_TICK = 7
FONT_SIZE_LEGEND = 7

FIGSIZE_WIDE = (8.0, 5.5)
FIGSIZE_TERMS = (8.0, 7.0)
HEATMAP_FIGSIZE = (8.0, 5.6)
SMALL_HEATMAP_FIGSIZE = (8.0, 5.4)
COLORBAR_SIZE = "5%"
COLORBAR_PAD = 0.08

LHS_VAR_POS = 13
RHS_VAR_POS = 14
LHS_VALUE_POS = 5
RHS_VALUE_POS = 9
EQUAL_POS = 15
ATTENTION_HEAD_IDX = 0


@dataclass(frozen=True)
class FixedExampleGroup:
    name: str
    lhs_vars: tuple[int, ...]
    rhs_vars: tuple[int, ...]
    number_pairs: tuple[tuple[int, int], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the late checkpoint window for fixed-layout 2-var train "
            "vs var-restricted examples."
        )
    )
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--step-start", type=int, default=DEFAULT_STEP_START)
    parser.add_argument("--step-end", type=int, default=DEFAULT_STEP_END)
    parser.add_argument("--step-stride", type=int, default=DEFAULT_STEP_STRIDE)
    parser.add_argument(
        "--focus-step",
        action="append",
        dest="focus_steps",
        type=int,
        default=None,
        help="Checkpoint step to include in the variable-submatrix heatmap grid.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda", "mps"),
        help="Device for model loading and forward passes.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also open matplotlib windows in addition to saving PNGs.",
    )
    return parser.parse_args()


def style_axes(ax) -> None:
    ax.tick_params(axis="both", labelsize=FONT_SIZE_TICK)


def add_colorbar(fig, ax, im):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=COLORBAR_SIZE, pad=COLORBAR_PAD)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=FONT_SIZE_TICK)
    return cbar


def save_figure(fig, path: Path, *, show: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved: {path}")


def build_fixed_layout_tokens(
    run_cfg,
    lhs_var: int,
    rhs_var: int,
    lhs_value: int,
    rhs_value: int,
) -> torch.Tensor:
    if run_cfg.seq_len != 16:
        raise ValueError(
            f"This script expects seq_len=16 for the fixed layout, got {run_cfg.seq_len}."
        )

    tokens = [run_cfg.pad_id] * run_cfg.seq_len
    tokens[LHS_VAR_POS] = lhs_var
    tokens[LHS_VALUE_POS] = lhs_value
    tokens[RHS_VAR_POS] = rhs_var
    tokens[RHS_VALUE_POS] = rhs_value
    tokens[12] = run_cfg.plus_id
    tokens[13] = lhs_var
    tokens[14] = rhs_var
    tokens[15] = run_cfg.equal_id
    return torch.tensor(tokens, dtype=torch.long)


def pair_name(lhs_var: int, rhs_var: int, run_cfg) -> str:
    return f"{run_cfg.token_strings[lhs_var]}/{run_cfg.token_strings[rhs_var]}"


def build_group_tokens(
    group: FixedExampleGroup, run_cfg
) -> tuple[torch.Tensor, list[str]]:
    rows = []
    names = []
    for lhs_var in group.lhs_vars:
        for rhs_var in group.rhs_vars:
            if lhs_var == rhs_var:
                continue
            for lhs_value, rhs_value in group.number_pairs:
                rows.append(
                    build_fixed_layout_tokens(
                        run_cfg, lhs_var, rhs_var, lhs_value, rhs_value
                    )
                )
                names.append(pair_name(lhs_var, rhs_var, run_cfg))
    return torch.stack(rows), names


def score_term(
    q: torch.Tensor,
    qk: torch.Tensor,
    k: torch.Tensor,
) -> torch.Tensor:
    return torch.einsum("bd,df,bf->b", q, qk, k)


def group_examples(
    run_cfg, config: dict
) -> tuple[FixedExampleGroup, FixedExampleGroup]:
    left_restrict, right_restrict = get_restriction_vars(config, list(run_cfg.var_ids))
    train_lhs = tuple(v for v in run_cfg.var_ids if v not in set(right_restrict))
    train_rhs = tuple(v for v in run_cfg.var_ids if v not in set(left_restrict))

    return (
        FixedExampleGroup(
            name="train_fixed",
            lhs_vars=train_lhs,
            rhs_vars=train_rhs,
            number_pairs=DEFAULT_NUMBER_PAIRS,
        ),
        FixedExampleGroup(
            name="var_restricted_fixed",
            lhs_vars=tuple(right_restrict),
            rhs_vars=tuple(left_restrict),
            number_pairs=DEFAULT_NUMBER_PAIRS,
        ),
    )


def best_wrong_score(
    q_components: dict[str, torch.Tensor],
    tokens: torch.Tensor,
    cache,
    qk_scaled: torch.Tensor,
    model,
) -> torch.Tensor:
    scores = []
    device = qk_scaled.device
    for pos in range(tokens.shape[1]):
        if pos in (LHS_VALUE_POS, RHS_VALUE_POS):
            continue
        k_components = {
            "embed": model.embed.W_E[tokens[:, pos]].detach(),
            "pos": model.pos_embed.W_pos[pos]
            .detach()
            .unsqueeze(0)
            .expand(tokens.shape[0], -1),
            "attn0": cache["attn_out", 0][:, pos, :].detach(),
            "mlp0": cache["mlp_out", 0][:, pos, :].detach(),
        }
        total = torch.zeros(tokens.shape[0], device=device)
        for qv in q_components.values():
            for kv in k_components.values():
                total = total + score_term(qv, qk_scaled, kv)
        scores.append(total.unsqueeze(1))
    return torch.cat(scores, dim=1).max(dim=1).values


def analyze_group(
    model,
    tokens: torch.Tensor,
    qk_scaled: torch.Tensor,
) -> dict[str, float]:
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, return_type="logits")

    attn0 = cache["attn", 0][:, ATTENTION_HEAD_IDX]
    attn1 = cache["attn", 1][:, ATTENTION_HEAD_IDX]

    q_components = {
        "embed": model.embed.W_E[tokens[:, EQUAL_POS]].detach(),
        "pos": model.pos_embed.W_pos[EQUAL_POS]
        .detach()
        .unsqueeze(0)
        .expand(tokens.shape[0], -1),
        "attn0": cache["attn_out", 0][:, EQUAL_POS, :].detach(),
        "mlp0": cache["mlp_out", 0][:, EQUAL_POS, :].detach(),
    }

    results: dict[str, float] = {
        "l0_lhs_operand_attn": float(attn0[:, EQUAL_POS, LHS_VAR_POS].mean()),
        "l0_rhs_operand_attn": float(attn0[:, EQUAL_POS, RHS_VAR_POS].mean()),
        "l1_lhs_value_attn": float(attn1[:, EQUAL_POS, LHS_VALUE_POS].mean()),
        "l1_rhs_value_attn": float(attn1[:, EQUAL_POS, RHS_VALUE_POS].mean()),
    }

    term_totals: dict[str, list[float]] = defaultdict(list)
    total_scores: dict[str, list[float]] = defaultdict(list)
    device = qk_scaled.device

    for side, value_pos in (("lhs", LHS_VALUE_POS), ("rhs", RHS_VALUE_POS)):
        k_components = {
            "embed": model.embed.W_E[tokens[:, value_pos]].detach(),
            "pos": model.pos_embed.W_pos[value_pos]
            .detach()
            .unsqueeze(0)
            .expand(tokens.shape[0], -1),
            "attn0": cache["attn_out", 0][:, value_pos, :].detach(),
            "mlp0": cache["mlp_out", 0][:, value_pos, :].detach(),
        }

        total = torch.zeros(tokens.shape[0], device=device)
        for q_name, q_val in q_components.items():
            for k_name, k_val in k_components.items():
                contribution = score_term(q_val, qk_scaled, k_val)
                total = total + contribution
                term_totals[f"{side}:{q_name}->{k_name}"].append(
                    float(contribution.mean())
                )
        total_scores[f"{side}:total"].append(float(total.mean()))

    wrong = best_wrong_score(q_components, tokens, cache, qk_scaled, model)
    rhs_total = sum(
        term_totals[f"rhs:{q}->{k}"][0] for q in q_components for k in q_components
    )
    lhs_total = sum(
        term_totals[f"lhs:{q}->{k}"][0] for q in q_components for k in q_components
    )

    results["rhs_score_total"] = rhs_total
    results["lhs_score_total"] = lhs_total
    results["best_wrong_score"] = float(wrong.mean())
    results["rhs_score_margin"] = rhs_total - float(wrong.mean())

    for key, values in term_totals.items():
        results[key] = values[0]
    for key, values in total_scores.items():
        results[key] = values[0]

    return results


def selected_var_scores(model, run_cfg, selected_vars: list[int]) -> torch.Tensor:
    qk_raw = (
        model.blocks[1].attn.W_Q[ATTENTION_HEAD_IDX]
        @ model.blocks[1].attn.W_K[ATTENTION_HEAD_IDX].T
    )
    ov0 = (
        model.blocks[0].attn.W_V[ATTENTION_HEAD_IDX]
        @ model.blocks[0].attn.W_O[ATTENTION_HEAD_IDX]
    )
    var_ov = model.embed.W_E[selected_vars].detach() @ ov0.detach()
    return (var_ov @ qk_raw.detach() @ var_ov.T).cpu()


def qk0_matrix_scaled(model) -> torch.Tensor:
    q = model.blocks[0].attn.W_Q[ATTENTION_HEAD_IDX]
    k = model.blocks[0].attn.W_K[ATTENTION_HEAD_IDX]
    return (q @ k.T).detach() / math.sqrt(model.cfg.d_head)


def qk0_equal_query_basis_terms(
    model,
    run_cfg,
    selected_vars: list[int],
) -> torch.Tensor:
    qk0 = qk0_matrix_scaled(model)
    query_basis = torch.stack(
        [
            model.embed.W_E[run_cfg.equal_id].detach(),
            model.pos_embed.W_pos[EQUAL_POS].detach(),
        ]
    )
    key_basis = model.embed.W_E[selected_vars].detach()
    return (query_basis @ qk0 @ key_basis.T).detach().cpu()


def qk0_operand_role_scores(
    model,
    run_cfg,
    selected_vars: list[int],
) -> dict[str, list[float] | float]:
    qk0 = qk0_matrix_scaled(model)
    query = (
        model.embed.W_E[run_cfg.equal_id].detach()
        + model.pos_embed.W_pos[EQUAL_POS].detach()
    )
    var_scores = (
        (query @ qk0 @ model.embed.W_E[selected_vars].detach().T).detach().cpu()
    )
    pos_lhs = float(query @ qk0 @ model.pos_embed.W_pos[LHS_VAR_POS].detach())
    pos_rhs = float(query @ qk0 @ model.pos_embed.W_pos[RHS_VAR_POS].detach())
    lhs_scores = var_scores + pos_lhs
    rhs_scores = var_scores + pos_rhs
    return {
        "var_scores": [float(x) for x in var_scores],
        "lhs_scores": [float(x) for x in lhs_scores],
        "rhs_scores": [float(x) for x in rhs_scores],
        "lhs_position_score": pos_lhs,
        "rhs_position_score": pos_rhs,
        "rhs_minus_lhs_position_score": pos_rhs - pos_lhs,
    }


def progress_window(progress_path: Path, steps: list[int]) -> dict[str, list[float]]:
    if not progress_path.exists():
        progress_path = Path("progress_measures.json")

    payload = json.loads(progress_path.read_text())
    all_steps = payload["steps"]
    indices = [all_steps.index(step) for step in steps]

    if "accuracy_overlays" in payload:
        overlays = payload["accuracy_overlays"]
        return {
            "2var_train": [overlays["two_var_train_acc"][i] for i in indices],
            "2var_var_restricted_1": [
                overlays["two_var_variable_restricted_1_acc"][i] for i in indices
            ],
            "2var_var_restricted_2": [
                overlays["two_var_variable_restricted_2_acc"][i] for i in indices
            ],
        }

    key = "attn_l1_equal_to_values"
    return {
        "2var_train": [
            payload["results"][key]["2var_valid_pair_valid_vars"][i] for i in indices
        ],
        "2var_var_restricted_1": [
            payload["results"][key]["2var_valid_pair_1_invalid_var"][i] for i in indices
        ],
        "2var_var_restricted_2": [
            payload["results"][key]["2var_valid_pair_2_invalid_vars"][i]
            for i in indices
        ],
    }


def build_b_operand_role_pool(
    config: dict,
    run_cfg,
    *,
    pool_size: int = 4096,
    seed: int = 123,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    tokens, labels = _pool_from_cfg(
        "2var_b_operand",
        config,
        run_cfg.vocab,
        run_cfg.mod,
        pool_size=pool_size,
        seed=seed,
    )
    b_token = run_cfg.a_token_id + 1
    masks = {
        "two_var_b_lhs": tokens[:, LHS_VAR_POS] == b_token,
        "two_var_b_rhs": tokens[:, RHS_VAR_POS] == b_token,
    }
    return tokens, labels, masks


def score_b_operand_roles(
    model,
    run_cfg,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    masks: dict[str, torch.Tensor],
) -> dict[str, float]:
    with torch.no_grad():
        logits = model(tokens.to(model.cfg.device))
        preds = logits[:, -1, : run_cfg.mod].argmax(dim=-1).detach().cpu()

    role_scores = {}
    for name, mask in masks.items():
        if not bool(mask.any()):
            role_scores[name] = float("nan")
            continue
        role_scores[name] = float((preds[mask] == labels[mask]).float().mean())
    return role_scores


def plot_overview(
    steps: list[int],
    progress: dict[str, list[float]],
    results: dict[str, dict[int, dict[str, float]]],
    focus_steps: list[int],
    output_path: Path,
    *,
    show: bool,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_WIDE, layout="constrained")
    ax_a, ax_b, ax_c, ax_d = axes.flat

    ax_a.plot(
        steps, progress["2var_train"], label="2-var train", color="#2ca02c", linewidth=2
    )
    ax_a.plot(
        steps,
        progress["2var_var_restricted_1"],
        label="2-var var-restricted (1)",
        color="#2ca02c",
        linestyle=":",
        linewidth=2,
    )
    ax_a.plot(
        steps,
        progress["2var_var_restricted_2"],
        label="2-var var-restricted (2)",
        color="#2ca02c",
        linestyle="-.",
        linewidth=2,
    )
    ax_a.set_title("(a)", loc="left", fontsize=FONT_SIZE_BODY, fontweight="bold")
    ax_a.set_xlabel("Training Step", fontsize=FONT_SIZE_BODY)
    ax_a.set_ylabel("L1 Value Accuracy", fontsize=FONT_SIZE_BODY)
    ax_a.legend(fontsize=FONT_SIZE_LEGEND)
    style_axes(ax_a)

    group_styles = {
        "train_fixed": {"color": "#1f77b4", "label": "Train-style"},
        "var_restricted_fixed": {"color": "#d62728", "label": "Var-restricted"},
    }
    for name, style in group_styles.items():
        ax_b.plot(
            steps,
            [results[name][step]["l0_lhs_operand_attn"] for step in steps],
            color=style["color"],
            linewidth=2,
            label=f"{style['label']} lhs operand",
        )
        ax_b.plot(
            steps,
            [results[name][step]["l0_rhs_operand_attn"] for step in steps],
            color=style["color"],
            linewidth=2,
            linestyle="--",
            label=f"{style['label']} rhs operand",
        )
    ax_b.set_title("(b)", loc="left", fontsize=FONT_SIZE_BODY, fontweight="bold")
    ax_b.set_xlabel("Training Step", fontsize=FONT_SIZE_BODY)
    ax_b.set_ylabel("L0 = Attention", fontsize=FONT_SIZE_BODY)
    ax_b.legend(fontsize=FONT_SIZE_LEGEND, ncol=1)
    style_axes(ax_b)

    for name, style in group_styles.items():
        ax_c.plot(
            steps,
            [results[name][step]["l1_lhs_value_attn"] for step in steps],
            color=style["color"],
            linewidth=2,
            label=f"{style['label']} lhs value",
        )
        ax_c.plot(
            steps,
            [results[name][step]["l1_rhs_value_attn"] for step in steps],
            color=style["color"],
            linewidth=2,
            linestyle="--",
            label=f"{style['label']} rhs value",
        )
    ax_c.set_title("(c)", loc="left", fontsize=FONT_SIZE_BODY, fontweight="bold")
    ax_c.set_xlabel("Training Step", fontsize=FONT_SIZE_BODY)
    ax_c.set_ylabel("L1 = Attention", fontsize=FONT_SIZE_BODY)
    ax_c.legend(fontsize=FONT_SIZE_LEGEND, ncol=1)
    style_axes(ax_c)

    for name, style in group_styles.items():
        ax_d.plot(
            steps,
            [results[name][step]["rhs_score_margin"] for step in steps],
            color=style["color"],
            linewidth=2,
            label=f"{style['label']} rhs margin",
        )
    ax_d.set_title("(d)", loc="left", fontsize=FONT_SIZE_BODY, fontweight="bold")
    ax_d.set_xlabel("Training Step", fontsize=FONT_SIZE_BODY)
    ax_d.set_ylabel("RHS Score Margin", fontsize=FONT_SIZE_BODY)
    ax_d.legend(fontsize=FONT_SIZE_LEGEND)
    style_axes(ax_d)

    for ax in axes.flat:
        for step in focus_steps:
            ax.axvline(step, color="0.8", linewidth=0.8, zorder=0)

    save_figure(fig, output_path, show=show)


def plot_rhs_terms(
    steps: list[int],
    results: dict[str, dict[int, dict[str, float]]],
    output_path: Path,
    *,
    show: bool,
) -> None:
    terms = [
        "rhs:attn0->attn0",
        "rhs:attn0->pos",
        "rhs:attn0->embed",
        "rhs:pos->attn0",
        "rhs:pos->embed",
        "rhs:embed->embed",
    ]
    titles = {
        "rhs:attn0->attn0": r"$\mathrm{attn0}\rightarrow\mathrm{attn0}$",
        "rhs:attn0->pos": r"$\mathrm{attn0}\rightarrow\mathrm{pos}$",
        "rhs:attn0->embed": r"$\mathrm{attn0}\rightarrow\mathrm{embed}$",
        "rhs:pos->attn0": r"$\mathrm{pos}\rightarrow\mathrm{attn0}$",
        "rhs:pos->embed": r"$\mathrm{pos}\rightarrow\mathrm{embed}$",
        "rhs:embed->embed": r"$\mathrm{embed}\rightarrow\mathrm{embed}$",
    }

    fig, axes = plt.subplots(2, 3, figsize=FIGSIZE_TERMS, layout="constrained")
    group_styles = {
        "train_fixed": {"color": "#1f77b4", "label": "Train-style"},
        "var_restricted_fixed": {"color": "#d62728", "label": "Var-restricted"},
    }

    for ax, term in zip(axes.flat, terms):
        for name, style in group_styles.items():
            ax.plot(
                steps,
                [results[name][step][term] for step in steps],
                color=style["color"],
                linewidth=2,
                label=style["label"],
            )
        ax.set_title(titles[term], fontsize=FONT_SIZE_BODY)
        ax.set_xlabel("Training Step", fontsize=FONT_SIZE_BODY)
        ax.set_ylabel("RHS Score Term", fontsize=FONT_SIZE_BODY)
        style_axes(ax)
    axes[0, 0].legend(fontsize=FONT_SIZE_LEGEND)
    save_figure(fig, output_path, show=show)


def plot_selected_var_grid(
    matrices: dict[int, torch.Tensor],
    labels: list[str],
    output_path: Path,
    *,
    show: bool,
) -> None:
    steps = list(matrices)
    n_cols = 3
    n_rows = math.ceil(len(steps) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=HEATMAP_FIGSIZE,
        layout="constrained",
        squeeze=False,
    )
    vmax = max(float(m.abs().max()) for m in matrices.values())
    vmin = -vmax

    for ax in axes.flat[len(steps) :]:
        ax.axis("off")

    for ax, step in zip(axes.flat, steps):
        im = ax.imshow(
            matrices[step].numpy(), cmap="bwr", vmin=vmin, vmax=vmax, aspect="equal"
        )
        ax.set_title(f"Step {step}", fontsize=FONT_SIZE_BODY)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=FONT_SIZE_TICK)
        ax.set_yticklabels(labels, fontsize=FONT_SIZE_TICK)
        ax.set_xlabel("Key variable", fontsize=FONT_SIZE_BODY)
        ax.set_ylabel("Query variable", fontsize=FONT_SIZE_BODY)
        add_colorbar(fig, ax, im)

    save_figure(fig, output_path, show=show)


def plot_qk0_equal_query_basis_grid(
    matrices: dict[int, torch.Tensor],
    key_labels: list[str],
    output_path: Path,
    *,
    show: bool,
) -> None:
    steps = list(matrices)
    n_cols = 3
    n_rows = math.ceil(len(steps) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=SMALL_HEATMAP_FIGSIZE,
        layout="constrained",
        squeeze=False,
    )
    vmax = max(float(m.abs().max()) for m in matrices.values())
    vmin = -vmax
    query_labels = ["E_=", "P15"]

    for ax in axes.flat[len(steps) :]:
        ax.axis("off")

    for ax, step in zip(axes.flat, steps):
        im = ax.imshow(matrices[step].numpy(), cmap="bwr", vmin=vmin, vmax=vmax)
        ax.set_title(f"Step {step}", fontsize=FONT_SIZE_BODY)
        ax.set_xticks(range(len(key_labels)))
        ax.set_yticks(range(len(query_labels)))
        ax.set_xticklabels(key_labels, rotation=90, fontsize=FONT_SIZE_TICK)
        ax.set_yticklabels(query_labels, fontsize=FONT_SIZE_TICK)
        ax.set_xlabel("Key basis", fontsize=FONT_SIZE_BODY)
        ax.set_ylabel("Query basis", fontsize=FONT_SIZE_BODY)
        add_colorbar(fig, ax, im)

    save_figure(fig, output_path, show=show)


def plot_qk0_operand_score_components(
    steps: list[int],
    role_scores: dict[int, dict[str, list[float] | float]],
    selected_labels: list[str],
    output_path: Path,
    *,
    show: bool,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_WIDE, layout="constrained")
    ax_a, ax_b, ax_c, ax_d = axes.flat
    label_to_idx = {label: idx for idx, label in enumerate(selected_labels)}
    colors = {
        "a": "#d62728",
        "b": "#d62728",
        "c": "#1f77b4",
        "j": "#1f77b4",
        "k": "#9467bd",
        "l": "#9467bd",
    }
    styles = {"a": "-", "b": "--", "c": "-", "j": "--", "k": "-", "l": "--"}

    for label in selected_labels:
        idx = label_to_idx[label]
        ax_a.plot(
            steps,
            [role_scores[step]["var_scores"][idx] for step in steps],
            color=colors.get(label, "0.3"),
            linestyle=styles.get(label, "-"),
            linewidth=2,
            label=f"{label}",
        )
    ax_a.set_title("(a)", loc="left", fontsize=FONT_SIZE_BODY, fontweight="bold")
    ax_a.set_xlabel("Training Step", fontsize=FONT_SIZE_BODY)
    ax_a.set_ylabel(
        r"QK0 term: $(E_=+P15)\to E_{\mathrm{var}}$", fontsize=FONT_SIZE_BODY
    )
    ax_a.legend(fontsize=FONT_SIZE_LEGEND, ncol=3)
    style_axes(ax_a)

    ax_b.plot(
        steps,
        [role_scores[step]["lhs_position_score"] for step in steps],
        color="#2ca02c",
        linewidth=2,
        label="P13 operand slot",
    )
    ax_b.plot(
        steps,
        [role_scores[step]["rhs_position_score"] for step in steps],
        color="#2ca02c",
        linestyle="--",
        linewidth=2,
        label="P14 operand slot",
    )
    ax_b.set_title("(b)", loc="left", fontsize=FONT_SIZE_BODY, fontweight="bold")
    ax_b.set_xlabel("Training Step", fontsize=FONT_SIZE_BODY)
    ax_b.set_ylabel(
        r"QK0 term: $(E_=+P15)\to P_{\mathrm{slot}}$", fontsize=FONT_SIZE_BODY
    )
    ax_b.legend(fontsize=FONT_SIZE_LEGEND)
    style_axes(ax_b)

    for label in ("a", "b"):
        idx = label_to_idx[label]
        ax_c.plot(
            steps,
            [role_scores[step]["lhs_scores"][idx] for step in steps],
            color=colors[label],
            linestyle=styles[label],
            linewidth=2,
            alpha=0.55,
            label=f"{label} as LHS",
        )
        ax_c.plot(
            steps,
            [role_scores[step]["rhs_scores"][idx] for step in steps],
            color=colors[label],
            linestyle=styles[label],
            linewidth=2.5,
            label=f"{label} as RHS",
        )
    ax_c.set_title("(c)", loc="left", fontsize=FONT_SIZE_BODY, fontweight="bold")
    ax_c.set_xlabel("Training Step", fontsize=FONT_SIZE_BODY)
    ax_c.set_ylabel("QK0 operand logit", fontsize=FONT_SIZE_BODY)
    ax_c.legend(fontsize=FONT_SIZE_LEGEND, ncol=2)
    style_axes(ax_c)

    for label in ("k", "l"):
        idx = label_to_idx[label]
        ax_d.plot(
            steps,
            [role_scores[step]["lhs_scores"][idx] for step in steps],
            color=colors[label],
            linestyle=styles[label],
            linewidth=2.5,
            label=f"{label} as LHS",
        )
        ax_d.plot(
            steps,
            [role_scores[step]["rhs_scores"][idx] for step in steps],
            color=colors[label],
            linestyle=styles[label],
            linewidth=2,
            alpha=0.55,
            label=f"{label} as RHS",
        )
    ax_d.set_title("(d)", loc="left", fontsize=FONT_SIZE_BODY, fontweight="bold")
    ax_d.set_xlabel("Training Step", fontsize=FONT_SIZE_BODY)
    ax_d.set_ylabel("QK0 operand logit", fontsize=FONT_SIZE_BODY)
    ax_d.legend(fontsize=FONT_SIZE_LEGEND, ncol=2)
    style_axes(ax_d)

    save_figure(fig, output_path, show=show)


def main() -> None:
    args = parse_args()
    steps = list(range(args.step_start, args.step_end + 1, args.step_stride))
    focus_steps = list(args.focus_steps or DEFAULT_FOCUS_STEPS)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = default_config()
    run_cfg = build_run_config(config)
    model = build_model(run_cfg, torch.device(args.device))

    train_group, restricted_group = group_examples(run_cfg, config)
    group_tokens = {
        train_group.name: build_group_tokens(train_group, run_cfg)[0],
        restricted_group.name: build_group_tokens(restricted_group, run_cfg)[0],
    }
    b_role_tokens, b_role_labels, b_role_masks = build_b_operand_role_pool(
        config, run_cfg
    )

    results: dict[str, dict[int, dict[str, float]]] = {
        train_group.name: {},
        restricted_group.name: {},
    }
    b_role_accuracy = {"two_var_b_lhs": [], "two_var_b_rhs": []}
    selected_vars = [run_cfg.a_token_id + idx for idx in (0, 1, 2, 9, 10, 11)]
    selected_labels = [run_cfg.token_strings[var_id] for var_id in selected_vars]
    submatrices: dict[int, torch.Tensor] = {}
    qk0_basis_terms: dict[int, torch.Tensor] = {}
    qk0_role_scores: dict[int, dict[str, list[float] | float]] = {}

    for step in steps:
        load_checkpoint(
            model, step, Path(args.checkpoint_dir), torch.device(args.device)
        )
        qk_scaled = (
            model.blocks[1].attn.W_Q[ATTENTION_HEAD_IDX]
            @ model.blocks[1].attn.W_K[ATTENTION_HEAD_IDX].T
        ).detach() / math.sqrt(model.cfg.d_head)

        for group_name, tokens in group_tokens.items():
            results[group_name][step] = analyze_group(
                model, tokens.to(model.cfg.device), qk_scaled
            )

        b_role_scores = score_b_operand_roles(
            model,
            run_cfg,
            b_role_tokens,
            b_role_labels,
            b_role_masks,
        )
        for name, value in b_role_scores.items():
            b_role_accuracy[name].append(value)

        if step in focus_steps:
            submatrices[step] = selected_var_scores(model, run_cfg, selected_vars)
            qk0_basis_terms[step] = qk0_equal_query_basis_terms(
                model, run_cfg, selected_vars
            )
        qk0_role_scores[step] = qk0_operand_role_scores(model, run_cfg, selected_vars)

    progress = progress_window(Path("progress_measures_pools.json"), steps)

    plot_overview(
        steps,
        progress,
        results,
        focus_steps,
        output_dir / "late_var_restricted_overview.png",
        show=args.show,
    )
    plot_rhs_terms(
        steps,
        results,
        output_dir / "late_var_restricted_rhs_terms.png",
        show=args.show,
    )
    plot_selected_var_grid(
        submatrices,
        selected_labels,
        output_dir / "late_var_restricted_var_qk1_grid.png",
        show=args.show,
    )
    plot_qk0_equal_query_basis_grid(
        qk0_basis_terms,
        selected_labels,
        output_dir / "late_var_restricted_qk0_equal_query_basis.png",
        show=args.show,
    )
    plot_qk0_operand_score_components(
        steps,
        qk0_role_scores,
        selected_labels,
        output_dir / "late_var_restricted_qk0_operand_components.png",
        show=args.show,
    )

    serializable = {
        "steps": steps,
        "focus_steps": focus_steps,
        "selected_var_labels": selected_labels,
        "results": results,
        "progress": progress,
        "selected_var_matrices": {
            str(step): submatrices[step].tolist() for step in sorted(submatrices)
        },
        "b_role_accuracy": b_role_accuracy,
        "qk0_equal_query_key_labels": selected_labels,
        "qk0_equal_query_matrices": {
            str(step): qk0_basis_terms[step].tolist()
            for step in sorted(qk0_basis_terms)
        },
        "qk0_operand_role_scores": qk0_role_scores,
    }
    json_path = output_dir / "late_var_restricted_analysis.json"
    json_path.write_text(json.dumps(serializable, indent=2))
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
