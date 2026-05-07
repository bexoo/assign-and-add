"""
Generate PNG summary figures from saved analysis JSON files.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/cache")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

try:
    from tueplots import bundles
except ImportError:  # pragma: no cover - optional plotting style dependency
    bundles = None


DATA_DIR = Path(".")
FIGURE_DIR = Path("figures_png")
DEFAULT_WIDTH = 5.5
FONT_SIZE_BODY = 9
FONT_SIZE_TICK = 8
FONT_SIZE_LEGEND = 8
COLORBAR_SIZE = "5%"
COLORBAR_PAD = 0.08
MAX_ACCURACY_STEP = 20_000

if bundles is not None:
    plt.rcParams.update(bundles.neurips2024(usetex=False))
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]


def load_json(relative_path: str) -> dict[str, Any] | None:
    path = DATA_DIR / relative_path
    if not path.exists():
        print(f"[skip] {path} not found")
        return None
    with path.open() as handle:
        return json.load(handle)


def save_figure(fig: plt.Figure, filename: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def add_colorbar(fig: plt.Figure, ax: plt.Axes, image) -> None:
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=COLORBAR_SIZE, pad=COLORBAR_PAD)
    cbar = fig.colorbar(image, cax=cax)
    cbar.ax.tick_params(labelsize=FONT_SIZE_TICK)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.set_title(label, loc="left", fontsize=FONT_SIZE_BODY, fontweight="bold")


def plot_accuracy_curves(progress_data: dict[str, Any]) -> None:
    overlays = progress_data.get("accuracy_overlays", {})
    steps = progress_data.get("steps", [])
    if not overlays or not steps:
        print("[skip] progress_measures.json has no accuracy overlay data")
        return

    step_mask = [step <= MAX_ACCURACY_STEP for step in steps]
    filtered_steps = [step for step, keep in zip(steps, step_mask) if keep]
    styles = {
        "zero_var_train_acc": ("#1f77b4", "-", "0-var train"),
        "zero_var_addition_restricted_acc": (
            "#1f77b4",
            "--",
            "0-var add-restricted",
        ),
        "one_var_train_acc": ("#ff7f0e", "-", "1-var train"),
        "one_var_addition_restricted_acc": (
            "#ff7f0e",
            "--",
            "1-var add-restricted",
        ),
        "one_var_variable_restricted_acc": (
            "#ff7f0e",
            ":",
            "1-var var-restricted",
        ),
        "two_var_train_acc": ("#2ca02c", "-", "2-var train"),
        "two_var_addition_restricted_acc": (
            "#2ca02c",
            "--",
            "2-var add-restricted",
        ),
        "two_var_variable_restricted_1_acc": (
            "#2ca02c",
            ":",
            "2-var var-restricted (1)",
        ),
        "two_var_variable_restricted_2_acc": (
            "#2ca02c",
            "-.",
            "2-var var-restricted (2)",
        ),
    }

    fig, ax = plt.subplots(figsize=(DEFAULT_WIDTH, DEFAULT_WIDTH * 0.55))
    plotted = False
    for name, (color, linestyle, label) in styles.items():
        values = overlays.get(name)
        if values is None:
            continue
        filtered_values = [value for value, keep in zip(values, step_mask) if keep]
        ax.plot(
            filtered_steps,
            filtered_values,
            color=color,
            linestyle=linestyle,
            linewidth=1.3,
            label=label,
        )
        plotted = True

    if not plotted:
        plt.close(fig)
        print("[skip] no known accuracy overlay keys found")
        return

    ax.set_xlabel("Training Step", fontsize=FONT_SIZE_BODY)
    ax.set_ylabel("Accuracy", fontsize=FONT_SIZE_BODY)
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    ax.legend(fontsize=FONT_SIZE_LEGEND, loc="lower right")
    save_figure(fig, "accuracy_by_var_count.png")


PROGRESS_MEASURE_DISPLAY_NAMES = {
    "ov1_mlp1_accuracy": "ov2_mlp2_accuracy",
    "qk0_num_to_prev_var": "qk1_num_to_prev_var",
    "qk1_ov0_var_identity": "qk2_ov1_var_identity",
    "attn_l0_equal_to_operands": "attn_l1_equal_to_operands",
    "attn_l0_num_to_prev": "attn_l1_num_to_prev",
    "attn_l1_equal_to_values": "attn_l2_equal_to_values",
    "probe_l0_mid_var_from_num": "probe_l1_mid_var_from_num",
}


def plot_progress_measures(progress_data: dict[str, Any]) -> None:
    measures = progress_data.get("measures", {})
    steps = progress_data.get("steps", [])
    if not measures or not steps:
        print("[skip] progress_measures.json has no progress measure data")
        return

    prefix_colors = {
        "ov": "#1f77b4",
        "qk": "#ff7f0e",
        "attn": "#2ca02c",
        "probe": "#d62728",
    }
    linestyles = ["-", "--", ":", "-."]
    prefix_counts: dict[str, int] = {}

    fig, ax = plt.subplots(figsize=(DEFAULT_WIDTH, DEFAULT_WIDTH * 0.55))
    for raw_name, values in measures.items():
        prefix = next(
            (
                candidate
                for candidate in prefix_colors
                if raw_name.startswith(candidate)
            ),
            "other",
        )
        count = prefix_counts.get(prefix, 0)
        prefix_counts[prefix] = count + 1
        name = PROGRESS_MEASURE_DISPLAY_NAMES.get(raw_name, raw_name)
        ax.plot(
            steps,
            values,
            color=prefix_colors.get(prefix, "0.35"),
            linestyle=linestyles[count % len(linestyles)],
            linewidth=1.2,
            label=name,
        )

    for step, label, color in zip(
        [4500, 6000, 14000],
        ["(a)", "(b)", "(c)"],
        ["#9467bd", "#8c564b", "#e377c2"],
    ):
        ax.axvline(step, color=color, linewidth=1.0, alpha=0.45)
        ax.text(step, 1.07, label, ha="center", fontsize=FONT_SIZE_TICK, color=color)

    ax.set_xlabel("Training Step", fontsize=FONT_SIZE_BODY)
    ax.set_ylabel("Measure Value", fontsize=FONT_SIZE_BODY)
    ax.set_xlim(0, 21_000)
    ax.set_ylim(0, 1.1)
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    ax.legend(fontsize=FONT_SIZE_LEGEND, loc="lower right")
    save_figure(fig, "progress_measures.png")


def plot_qk_heatmaps(weights_data: dict[str, Any]) -> None:
    if "qk0_pos_masked" not in weights_data or "var_var_qk1" not in weights_data:
        print("[skip] weights_plot_data.json has no QK heatmap data")
        return

    qk0 = np.array(weights_data["qk0_pos_masked"], dtype=float)
    var_var = np.array(weights_data["var_var_qk1"], dtype=float)
    var_labels = weights_data.get(
        "var_labels", [str(i) for i in range(var_var.shape[0])]
    )

    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(DEFAULT_WIDTH, DEFAULT_WIDTH * 0.52),
        layout="constrained",
    )

    im_a = ax_a.imshow(qk0, cmap="viridis", aspect="equal")
    panel_label(ax_a, "(a)")
    ax_a.set_xlabel("Key position", fontsize=FONT_SIZE_BODY)
    ax_a.set_ylabel("Query position", fontsize=FONT_SIZE_BODY)
    pos_labels = [str(i) for i in range(qk0.shape[0])]
    ax_a.set_xticks(range(qk0.shape[0]), labels=pos_labels, fontsize=FONT_SIZE_TICK)
    ax_a.set_yticks(range(qk0.shape[0]), labels=pos_labels, fontsize=FONT_SIZE_TICK)
    for row in range(qk0.shape[0]):
        values = qk0[row]
        if np.all(np.isnan(values)):
            continue
        col = int(np.nanargmax(values))
        ax_a.add_patch(
            Rectangle(
                (col - 0.5, row - 0.5), 1, 1, lw=1.2, edgecolor="red", facecolor="none"
            )
        )
    add_colorbar(fig, ax_a, im_a)

    im_b = ax_b.imshow(var_var, cmap="viridis", aspect="equal")
    panel_label(ax_b, "(b)")
    ax_b.set_xlabel(r"Key: $OV_1 \cdot e_{\mathrm{var}}$", fontsize=FONT_SIZE_BODY)
    ax_b.set_ylabel(r"Query: $OV_1 \cdot e_{\mathrm{var}}$", fontsize=FONT_SIZE_BODY)
    ax_b.set_xticks(range(len(var_labels)), labels=var_labels, fontsize=FONT_SIZE_TICK)
    ax_b.set_yticks(range(len(var_labels)), labels=var_labels, fontsize=FONT_SIZE_TICK)
    add_colorbar(fig, ax_b, im_b)

    save_figure(fig, "qk_heatmap_combined.png")


def plot_attention_patterns(weights_data: dict[str, Any]) -> None:
    if "attention_patterns" not in weights_data:
        print("[skip] weights_plot_data.json has no attention pattern data")
        return

    attention = np.array(weights_data["attention_patterns"], dtype=float)
    labels = weights_data.get("attention_token_labels")
    if labels is None:
        labels = [str(i) for i in range(attention.shape[-1])]

    n_layers, n_heads = attention.shape[:2]
    n_cols = n_layers * n_heads
    fig, axes = plt.subplots(
        1,
        n_cols,
        figsize=(max(DEFAULT_WIDTH, 2.6 * n_cols), 3.0),
        squeeze=False,
        sharey=True,
        layout="constrained",
    )

    image = None
    col = 0
    reversed_labels = list(reversed(labels))
    for layer in range(n_layers):
        for head in range(n_heads):
            ax = axes[0, col]
            image = ax.imshow(attention[layer, head, ::-1, :], cmap="viridis")
            ax.set_title(f"Layer {layer + 1}", fontsize=FONT_SIZE_BODY)
            ax.set_xticks(
                range(len(labels)),
                labels=labels,
                rotation=50,
                ha="right",
                fontsize=FONT_SIZE_TICK,
            )
            ax.set_yticks(
                range(len(labels)),
                labels=reversed_labels if col == 0 else [],
                fontsize=FONT_SIZE_TICK,
            )
            col += 1

    if image is not None:
        colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8, pad=0.02)
        colorbar.ax.tick_params(labelsize=FONT_SIZE_TICK)
    save_figure(fig, "attention_patterns.png")


def plot_early_checkpoint_diagnostics(early_data: dict[str, Any]) -> None:
    steps = [str(step) for step in early_data.get("steps", [])]
    if len(steps) != 2:
        print("[skip] early checkpoint diagnostics require exactly two steps")
        return

    before_step, after_step = steps
    token_labels = early_data["token_labels"]
    var_labels = early_data["var_labels"]
    seq_len = len(token_labels)
    operand_positions = (seq_len - 3, seq_len - 2)
    before_attn = np.array(early_data["attention_patterns"][before_step], dtype=float)
    after_attn = np.array(early_data["attention_patterns"][after_step], dtype=float)
    before_qk = np.array(early_data["var_identity_qk1"][before_step], dtype=float)
    after_qk = np.array(early_data["var_identity_qk1"][after_step], dtype=float)
    qk_lim = float(np.nanmax(np.abs([before_qk, after_qk])))

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(DEFAULT_WIDTH, DEFAULT_WIDTH * 0.74),
        layout="constrained",
    )

    reversed_labels = list(reversed(token_labels))
    for ax, matrix, step, label in (
        (axes[0, 0], before_attn, before_step, "(a)"),
        (axes[0, 1], after_attn, after_step, "(b)"),
    ):
        image = ax.imshow(matrix[::-1, :], cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title(
            f"{label} Step {int(step) - 1}",
            loc="left",
            fontsize=FONT_SIZE_BODY,
            fontweight="bold",
        )
        ax.set_xlabel("Key position", fontsize=FONT_SIZE_BODY)
        ax.set_xticks(
            range(seq_len),
            labels=token_labels,
            rotation=50,
            ha="right",
            fontsize=FONT_SIZE_TICK,
        )
        ax.set_yticks(range(seq_len), labels=reversed_labels, fontsize=FONT_SIZE_TICK)
        final_query_row = 0
        for key_pos in operand_positions:
            ax.add_patch(
                Rectangle(
                    (key_pos - 0.5, final_query_row - 0.5),
                    1,
                    1,
                    lw=1.2,
                    edgecolor="#d62728",
                    facecolor="none",
                )
            )
        add_colorbar(fig, ax, image)

    for ax, matrix, step, label in (
        (axes[1, 0], before_qk, before_step, "(c)"),
        (axes[1, 1], after_qk, after_step, "(d)"),
    ):
        image = ax.imshow(matrix, cmap="bwr", vmin=-qk_lim, vmax=qk_lim)
        ax.set_title(
            f"{label} Step {int(step) - 1}",
            loc="left",
            fontsize=FONT_SIZE_BODY,
            fontweight="bold",
        )
        ax.set_xlabel(r"Key: $OV_1(e_{\mathrm{var}})$", fontsize=FONT_SIZE_BODY)
        ax.set_xticks(
            range(len(var_labels)), labels=var_labels, fontsize=FONT_SIZE_TICK
        )
        ax.set_yticks(
            range(len(var_labels)), labels=var_labels, fontsize=FONT_SIZE_TICK
        )
        for idx in range(len(var_labels)):
            ax.add_patch(
                Rectangle(
                    (idx - 0.5, idx - 0.5),
                    1,
                    1,
                    lw=0.6,
                    edgecolor="black",
                    facecolor="none",
                )
            )
        add_colorbar(fig, ax, image)

    axes[0, 0].set_ylabel("Query position", fontsize=FONT_SIZE_BODY)
    axes[1, 0].set_ylabel(r"Query: $OV_1(e_{\mathrm{var}})$", fontsize=FONT_SIZE_BODY)
    save_figure(fig, "early_add_restricted_qk1_combined.png")


def plot_late_checkpoint_diagnostics(late_data: dict[str, Any]) -> None:
    steps = late_data.get("steps", [])
    labels = late_data.get("selected_var_labels", [])
    role_scores = late_data.get("qk0_operand_role_scores")
    matrices = late_data.get("qk0_equal_query_matrices")
    key_labels = late_data.get("qk0_equal_query_key_labels", labels)
    if not steps or not labels or role_scores is None or matrices is None:
        print("[skip] late checkpoint diagnostics missing required arrays")
        return

    before_step = "14501" if "14501" in matrices else str(steps[0])
    after_step = "16001" if "16001" in matrices else str(steps[-1])
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    colors = {
        "a": "#d62728",
        "b": "#d62728",
        "c": "#1f77b4",
        "j": "#1f77b4",
        "k": "#9467bd",
        "l": "#9467bd",
    }
    linestyles = {"a": "-", "b": "--", "c": "-", "j": "--", "k": "-", "l": "--"}

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(DEFAULT_WIDTH, DEFAULT_WIDTH * 0.36),
        layout="constrained",
    )

    for label in labels:
        idx = label_to_idx[label]
        axes[0].plot(
            steps,
            [role_scores[str(step)]["var_scores"][idx] for step in steps],
            color=colors.get(label, "0.35"),
            linestyle=linestyles.get(label, "-"),
            linewidth=1.4,
            label=label,
        )
    panel_label(axes[0], "(a)")
    axes[0].set_xlabel("Training Step", fontsize=FONT_SIZE_BODY)
    axes[0].set_ylabel(r"QK variable score", fontsize=FONT_SIZE_BODY)
    axes[0].tick_params(labelsize=FONT_SIZE_TICK)
    axes[0].legend(fontsize=FONT_SIZE_LEGEND, ncol=2)

    for ax, step, label in (
        (axes[1], before_step, "(b)"),
        (axes[2], after_step, "(c)"),
    ):
        matrix = np.array(matrices[step], dtype=float)
        lim = float(np.nanmax(np.abs(matrix)))
        image = ax.imshow(matrix, cmap="bwr", vmin=-lim, vmax=lim, aspect="auto")
        ax.set_title(
            f"{label} Step {int(step) - 1}",
            loc="left",
            fontsize=FONT_SIZE_BODY,
            fontweight="bold",
        )
        ax.set_xlabel("Variable key", fontsize=FONT_SIZE_BODY)
        ax.set_xticks(
            range(len(key_labels)), labels=key_labels, fontsize=FONT_SIZE_TICK
        )
        ax.set_yticks(
            range(matrix.shape[0]),
            labels=[r"$E_=$", r"$P_{15}$"],
            fontsize=FONT_SIZE_TICK,
        )
        add_colorbar(fig, ax, image)

    save_figure(fig, "late_var_restricted_qk0_combined.png")


def plot_cosine_histograms(exploration_data: dict[str, Any]) -> None:
    required = (
        "mlp_cos_matched",
        "mlp_cos_shuffled",
        "ov1_cos_matched",
        "ov1_cos_shuffled",
    )
    if any(key not in exploration_data for key in required):
        print("[skip] exploration_composition_plot_data.json missing cosine arrays")
        return

    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(DEFAULT_WIDTH, DEFAULT_WIDTH * 0.42),
        layout="constrained",
    )
    viridis = plt.cm.viridis

    ax_a.hist(
        exploration_data["mlp_cos_matched"],
        bins=40,
        label="Matched",
        color=viridis(0.2),
        alpha=0.55,
    )
    ax_a.hist(
        exploration_data["mlp_cos_shuffled"],
        bins=80,
        label="Mismatched",
        color=viridis(0.75),
        alpha=0.55,
    )
    panel_label(ax_a, "(a)")
    ax_a.set_xlabel("Cosine Similarity", fontsize=FONT_SIZE_BODY)
    ax_a.set_ylabel("Count", fontsize=FONT_SIZE_BODY)
    ax_a.tick_params(labelsize=FONT_SIZE_TICK)
    ax_a.legend(fontsize=FONT_SIZE_LEGEND)

    ax_b.hist(
        exploration_data["ov1_cos_matched"],
        bins=50,
        label="Matched",
        color=viridis(0.2),
        alpha=0.6,
    )
    ax_b.hist(
        exploration_data["ov1_cos_shuffled"],
        bins=80,
        label="Shuffled",
        color=viridis(0.75),
        alpha=0.6,
    )
    panel_label(ax_b, "(b)")
    ax_b.set_xlabel("Cosine Similarity", fontsize=FONT_SIZE_BODY)
    ax_b.tick_params(labelsize=FONT_SIZE_TICK)
    ax_b.legend(fontsize=FONT_SIZE_LEGEND)

    save_figure(fig, "cosine_sim_combined.png")


def plot_fourier_comparison(fourier_data: dict[str, Any]) -> None:
    required = (
        "actual_normalized",
        "theoretical_normalized",
        "best_neuron",
        "target_freq",
    )
    if any(key not in fourier_data for key in required):
        print("[skip] fourier_plot_data.json missing comparison arrays")
        return

    actual = np.array(fourier_data["actual_normalized"], dtype=float)
    theoretical = np.array(fourier_data["theoretical_normalized"], dtype=float)
    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(DEFAULT_WIDTH, DEFAULT_WIDTH * 0.45),
        layout="constrained",
    )

    image = ax_a.imshow(actual, cmap="viridis", vmin=-1, vmax=1)
    ax_a.set_title(f"Neuron {fourier_data['best_neuron']}", fontsize=FONT_SIZE_BODY)
    ax_a.set_xlabel("m", fontsize=FONT_SIZE_BODY)
    ax_a.set_ylabel("n", fontsize=FONT_SIZE_BODY)
    ax_a.tick_params(labelsize=FONT_SIZE_TICK)

    ax_b.imshow(theoretical, cmap="viridis", vmin=-1, vmax=1)
    ax_b.set_title(
        f"Theoretical k={fourier_data['target_freq']}", fontsize=FONT_SIZE_BODY
    )
    ax_b.set_xlabel("m", fontsize=FONT_SIZE_BODY)
    ax_b.set_ylabel("n", fontsize=FONT_SIZE_BODY)
    ax_b.tick_params(labelsize=FONT_SIZE_TICK)

    colorbar = fig.colorbar(image, ax=[ax_a, ax_b], shrink=0.8, pad=0.02)
    colorbar.ax.tick_params(labelsize=FONT_SIZE_TICK)
    save_figure(fig, "fourier_normalized_comparison.png")


def main() -> None:
    print("=== Generating PNG figures ===")

    progress_data = load_json("progress_measures.json")
    if progress_data is not None:
        plot_accuracy_curves(progress_data)
        plot_progress_measures(progress_data)

    weights_data = load_json("weights_plot_data.json")
    if weights_data is not None:
        plot_qk_heatmaps(weights_data)
        plot_attention_patterns(weights_data)

    early_data = load_json(
        "checkpoint_add_restricted_early_analysis/early_add_restricted_analysis.json"
    )
    if early_data is not None:
        plot_early_checkpoint_diagnostics(early_data)

    late_data = load_json(
        "checkpoint_var_restricted_late_analysis/late_var_restricted_analysis.json"
    )
    if late_data is not None:
        plot_late_checkpoint_diagnostics(late_data)

    exploration_data = load_json("exploration_composition_plot_data.json")
    if exploration_data is not None:
        plot_cosine_histograms(exploration_data)

    fourier_data = load_json("fourier_plot_data.json")
    if fourier_data is not None:
        plot_fourier_comparison(fourier_data)

    print("=== Done ===")


if __name__ == "__main__":
    main()
