"""
Diagnose the b-as-RHS routing asymmetry across late checkpoints.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib.pyplot as plt
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

try:
    from tueplots import bundles
except ImportError:
    bundles = None

from eval_pools import _pool_from_cfg


MOD = 59
VOCAB = MOD + 3 + 12
SEQ_LEN = 16
B_TOKEN = MOD + 4
A_TOKEN = MOD + 3
LHS_OPERAND_POS = 13
RHS_OPERAND_POS = 14
EQUAL_POS = 15
ASSIGN_VAR_POS = 3
ASSIGN_VALUE_POS = 4
ASSIGNED_VALUE = 7
DIRECT_CONSTANT = 9
STEP_START = 14001
STEP_END = 16101
STEP_STRIDE = 100
FIXED_COEFFICIENT_STEP = 14501
POOL_SIZE = 4096

CHECKPOINT_DIR = Path("checkpoints")
OUTPUT_DIR = Path("checkpoint_b_rhs_asymmetry")
FIGURE_DIR = Path("figures")
QK2_COMPONENT_JSON = "qk2_operand_embedding_component.json"
B_ROLE_QK2_JSON = "b_rhs_actual_qk2_breakdown.json"
B_ROLE_QK2_TABLE = "b_rhs_actual_source_contributions.md"
B_ROLE_QK2_PRUNING_TABLE = "b_rhs_qk2_component_pruning.md"

_FONT_SANS_SERIF = ["Helvetica", "Arial", "DejaVu Sans"]
if bundles is not None:
    plt.rcParams.update(bundles.neurips2024(usetex=False))
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = _FONT_SANS_SERIF

FONT_SIZE_BODY = 9
FONT_SIZE_TICK = 7
FONT_SIZE_LEGEND = 7
FONT_SIZE_ANNOTATION = 7


def run_config() -> dict:
    return {
        "N_LAYERS": 2,
        "N_HEADS": 1,
        "D_MODEL": 128,
        "SEQ_LEN": SEQ_LEN,
        "SEED": 42,
        "MOD": MOD,
        "VOCAB": VOCAB,
        "NUMS_TRAIN_PAIRS": int(0.7 * MOD * MOD),
        "RESTRICT_LEFT_HALF_VARS": 2,
        "RESTRICT_RIGHT_HALF_VARS": 2,
    }


def build_model() -> HookedTransformer:
    cfg = HookedTransformerConfig(
        n_layers=2,
        n_heads=1,
        d_model=128,
        d_head=128,
        d_mlp=512,
        n_ctx=SEQ_LEN,
        d_vocab=VOCAB,
        d_vocab_out=MOD,
        act_fn="relu",
        normalization_type=None,
        seed=42,
        device="cpu",
    )
    return HookedTransformer(cfg)


def checkpoint_steps() -> list[int]:
    pattern = re.compile(r"checkpoint_step_(\d+)\.pth")
    steps = []
    for path in CHECKPOINT_DIR.glob("checkpoint_step_*.pth"):
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        step = int(match.group(1))
        if STEP_START <= step <= STEP_END and (step - STEP_START) % STEP_STRIDE == 0:
            steps.append(step)
    return sorted(steps)


def load_checkpoint(model: HookedTransformer, step: int) -> None:
    checkpoint_path = CHECKPOINT_DIR / f"checkpoint_step_{step}.pth"
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()


def b_value_positions(tokens: torch.Tensor) -> torch.Tensor:
    positions = []
    prefix_end = SEQ_LEN - 4
    for row in tokens:
        found = None
        for pos in range(prefix_end - 1):
            if int(row[pos]) == B_TOKEN and int(row[pos + 1]) < MOD:
                found = pos + 1
                break
        if found is None:
            raise ValueError("Could not find b assignment value position")
        positions.append(found)
    return torch.tensor(positions, dtype=torch.long)


def qk2_component_examples() -> list[dict[str, int | str]]:
    """Return the four fixed 1-var examples for the QK2 component diagnostic."""
    return [
        {
            "label": "b @ P13",
            "var_label": "b",
            "var_token": B_TOKEN,
            "operand_pos": LHS_OPERAND_POS,
        },
        {
            "label": "b @ P14",
            "var_label": "b",
            "var_token": B_TOKEN,
            "operand_pos": RHS_OPERAND_POS,
        },
        {
            "label": "a @ P13",
            "var_label": "a",
            "var_token": A_TOKEN,
            "operand_pos": LHS_OPERAND_POS,
        },
        {
            "label": "a @ P14",
            "var_label": "a",
            "var_token": A_TOKEN,
            "operand_pos": RHS_OPERAND_POS,
        },
    ]


def build_qk2_component_tokens(examples: list[dict[str, int | str]]) -> torch.Tensor:
    rows = []
    for example in examples:
        var_token = int(example["var_token"])
        operand_pos = int(example["operand_pos"])
        tokens = [MOD + 2] * SEQ_LEN
        tokens[ASSIGN_VAR_POS] = var_token
        tokens[ASSIGN_VALUE_POS] = ASSIGNED_VALUE
        tokens[12] = MOD
        tokens[LHS_OPERAND_POS] = DIRECT_CONSTANT
        tokens[RHS_OPERAND_POS] = DIRECT_CONSTANT
        tokens[operand_pos] = var_token
        tokens[EQUAL_POS] = MOD + 1
        rows.append(tokens)
    return torch.tensor(rows, dtype=torch.long)


def b_role_qk2_examples() -> list[dict[str, int | str]]:
    """Return matched fixed examples that differ only in b's operand role."""
    return [
        {
            "label": "b LHS",
            "role": "lhs",
            "var_token": B_TOKEN,
            "operand_pos": LHS_OPERAND_POS,
        },
        {
            "label": "b RHS",
            "role": "rhs",
            "var_token": B_TOKEN,
            "operand_pos": RHS_OPERAND_POS,
        },
    ]


def build_b_role_qk2_tokens(examples: list[dict[str, int | str]]) -> torch.Tensor:
    """Build fixed one-variable b examples with a matched direct operand."""
    rows = []
    for example in examples:
        tokens = [MOD + 2] * SEQ_LEN
        tokens[ASSIGN_VAR_POS] = int(example["var_token"])
        tokens[ASSIGN_VALUE_POS] = ASSIGNED_VALUE
        tokens[12] = MOD
        tokens[LHS_OPERAND_POS] = DIRECT_CONSTANT
        tokens[RHS_OPERAND_POS] = DIRECT_CONSTANT
        tokens[int(example["operand_pos"])] = int(example["var_token"])
        tokens[EQUAL_POS] = MOD + 1
        rows.append(tokens)
    return torch.tensor(rows, dtype=torch.long)


def token_label(token: int) -> str:
    """Return the compact display label for a token in the fixed examples."""
    if token < MOD:
        return str(token)
    if token == A_TOKEN:
        return "a"
    if token == B_TOKEN:
        return "b"
    if token == MOD:
        return "+"
    if token == MOD + 1:
        return "="
    return "PAD"


def qk2_component_raw_terms(
    model: HookedTransformer,
    examples: list[dict[str, int | str]],
) -> dict[str, float]:
    qk2 = (
        model.blocks[1].attn.W_Q[0] @ model.blocks[1].attn.W_K[0].T
    ).detach() / math.sqrt(model.cfg.d_head)
    ov1 = (model.blocks[0].attn.W_V[0] @ model.blocks[0].attn.W_O[0]).detach()

    terms = {}
    for example in examples:
        var_token = int(example["var_token"])
        var_through_ov1 = model.embed.W_E[var_token].detach() @ ov1
        raw = torch.einsum("d,df,f->", var_through_ov1, qk2, var_through_ov1)
        terms[str(example["label"])] = float(raw)
    return terms


def qk2_component_coefficients(
    model: HookedTransformer,
    tokens: torch.Tensor,
    examples: list[dict[str, int | str]],
) -> dict[str, dict[str, float]]:
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, return_type="logits")
    attn0 = cache["attn", 0][:, 0].detach().cpu()

    coefficients = {}
    for idx, example in enumerate(examples):
        equal_to_operand = float(attn0[idx, EQUAL_POS, int(example["operand_pos"])])
        value_to_assign_var = float(attn0[idx, ASSIGN_VALUE_POS, ASSIGN_VAR_POS])
        coefficients[str(example["label"])] = {
            "equal_to_operand": equal_to_operand,
            "value_to_assign_var": value_to_assign_var,
            "product": equal_to_operand * value_to_assign_var,
        }
    return coefficients


def collect_qk2_operand_embedding_component() -> dict:
    """Collect raw and attention-weighted QK2 variable-embedding terms."""
    examples = qk2_component_examples()
    tokens = build_qk2_component_tokens(examples)
    steps = checkpoint_steps()
    if FIXED_COEFFICIENT_STEP not in steps:
        raise ValueError(
            f"Fixed coefficient checkpoint {FIXED_COEFFICIENT_STEP} is not in "
            f"the collected step window: {steps[:3]} ... {steps[-3:]}"
        )

    model = build_model()
    load_checkpoint(model, FIXED_COEFFICIENT_STEP)
    fixed_coefficients = qk2_component_coefficients(model, tokens, examples)

    series = {
        str(example["label"]): {
            "raw": [],
            "fixed_coefficient_weighted": [],
            "actual_weighted": [],
            "actual_coefficients": {
                "equal_to_operand": [],
                "value_to_assign_var": [],
                "product": [],
            },
            "fixed_coefficients": fixed_coefficients[str(example["label"])],
        }
        for example in examples
    }

    for step in steps:
        load_checkpoint(model, step)
        raw_terms = qk2_component_raw_terms(model, examples)
        actual_coefficients = qk2_component_coefficients(model, tokens, examples)

        for example in examples:
            label = str(example["label"])
            raw = raw_terms[label]
            fixed_product = fixed_coefficients[label]["product"]
            actual_product = actual_coefficients[label]["product"]
            series[label]["raw"].append(raw)
            series[label]["fixed_coefficient_weighted"].append(fixed_product * raw)
            series[label]["actual_weighted"].append(actual_product * raw)
            for coeff_name in ("equal_to_operand", "value_to_assign_var", "product"):
                series[label]["actual_coefficients"][coeff_name].append(
                    actual_coefficients[label][coeff_name]
                )

    serializable_examples = []
    for example in examples:
        operand_pos = int(example["operand_pos"])
        operand_side = "P13" if operand_pos == LHS_OPERAND_POS else "P14"
        tokens_for_example = build_qk2_component_tokens([example])[0].tolist()
        serializable_examples.append(
            {
                "label": str(example["label"]),
                "var_label": str(example["var_label"]),
                "var_token": int(example["var_token"]),
                "operand_pos": operand_pos,
                "operand_side": operand_side,
                "tokens": tokens_for_example,
            }
        )

    return {
        "description": (
            "Variable-embedding QK2 component over fixed 1-var examples: "
            "E[var]OV_1 QK_2 (E[var]OV_1)^T, optionally multiplied by layer-1 "
            "attention coefficients from code layer 0."
        ),
        "steps": steps,
        "fixed_coefficient_step": FIXED_COEFFICIENT_STEP,
        "assigned_variable_pos": ASSIGN_VAR_POS,
        "assigned_value_pos": ASSIGN_VALUE_POS,
        "assigned_value": ASSIGNED_VALUE,
        "direct_constant": DIRECT_CONSTANT,
        "equal_pos": EQUAL_POS,
        "qk_scaled_by_sqrt_d_head": True,
        "examples": serializable_examples,
        "series": series,
    }


def qk2_role_breakdown_for_tokens(
    model: HookedTransformer,
    tokens: torch.Tensor,
    examples: list[dict[str, int | str]],
) -> dict[str, dict]:
    """Compute actual QK2 score decompositions for the matched b-role examples."""
    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens, return_type="logits")

    qk2 = (
        model.blocks[1].attn.W_Q[0] @ model.blocks[1].attn.W_K[0].T
    ).detach() / math.sqrt(model.cfg.d_head)
    ov0 = (model.blocks[0].attn.W_V[0] @ model.blocks[0].attn.W_O[0]).detach()
    attn0 = cache["attn", 0][:, 0].detach().cpu()
    attn1 = cache["attn", 1][:, 0].detach().cpu()
    probs = logits[:, -1, :MOD].softmax(dim=-1).detach().cpu()
    preds = logits[:, -1, :MOD].argmax(dim=-1).detach().cpu()

    query_names = ("E_=", "P15", "attn0_eq")
    key_names = ("E_value", "P_value", "attn0_value")
    selected_sources = {
        "P13_source": LHS_OPERAND_POS,
        "P14_source": RHS_OPERAND_POS,
        "assigned_var_source": ASSIGN_VAR_POS,
        "assigned_value_source": ASSIGN_VALUE_POS,
    }
    operand_sources = {
        "P13": LHS_OPERAND_POS,
        "P14": RHS_OPERAND_POS,
    }

    correct_label = (ASSIGNED_VALUE + DIRECT_CONSTANT) % MOD
    breakdown = {}
    for idx, example in enumerate(examples):
        row = tokens[idx]
        q_components = {
            "E_=": model.embed.W_E[row[EQUAL_POS]].detach(),
            "P15": model.pos_embed.W_pos[EQUAL_POS].detach(),
            "attn0_eq": cache["attn_out", 0][idx, EQUAL_POS].detach(),
        }
        k_components = {
            "E_value": model.embed.W_E[row[ASSIGN_VALUE_POS]].detach(),
            "P_value": model.pos_embed.W_pos[ASSIGN_VALUE_POS].detach(),
            "attn0_value": cache["attn_out", 0][idx, ASSIGN_VALUE_POS].detach(),
        }

        matrix = []
        for q_name in query_names:
            row_terms = []
            for k_name in key_names:
                score = torch.einsum(
                    "d,df,f->",
                    q_components[q_name],
                    qk2,
                    k_components[k_name],
                )
                row_terms.append(float(score))
            matrix.append(row_terms)

        attn0_value_key = k_components["attn0_value"]
        source_contributions = {}
        all_source_contributions = []
        for source_pos in range(SEQ_LEN):
            source_resid = (
                model.embed.W_E[row[source_pos]].detach()
                + model.pos_embed.W_pos[source_pos].detach()
            )
            source_query = float(attn0[idx, EQUAL_POS, source_pos]) * (
                source_resid @ ov0
            )
            contribution = torch.einsum("d,df,f->", source_query, qk2, attn0_value_key)
            all_source_contributions.append(float(contribution))

        selected_total = 0.0
        for name, source_pos in selected_sources.items():
            value = all_source_contributions[source_pos]
            source_contributions[name] = value
            selected_total += value
        source_contributions["other_sources"] = (
            float(sum(all_source_contributions)) - selected_total
        )

        b_key_source = (
            model.embed.W_E[B_TOKEN].detach()
            + model.pos_embed.W_pos[ASSIGN_VAR_POS].detach()
        ) @ ov0
        key_attention = float(attn0[idx, ASSIGN_VALUE_POS, ASSIGN_VAR_POS])
        position_path_terms = {
            "value_key_attention": key_attention,
            "position_only": {},
            "raw": {},
            "weighted": {},
            "weighted_b_key": {},
            "weighted_pos3_key": {},
        }
        for source_name, source_pos in operand_sources.items():
            position_source = model.pos_embed.W_pos[source_pos].detach() @ ov0
            b_embedding_key = model.embed.W_E[B_TOKEN].detach() @ ov0
            pos3_key = model.pos_embed.W_pos[ASSIGN_VAR_POS].detach() @ ov0
            position_only = torch.einsum(
                "d,df,f->",
                position_source,
                qk2,
                b_embedding_key,
            )
            source_resid = (
                model.embed.W_E[row[source_pos]].detach()
                + model.pos_embed.W_pos[source_pos].detach()
            )
            raw = torch.einsum("d,df,f->", source_resid @ ov0, qk2, b_key_source)
            weighted = (
                float(attn0[idx, EQUAL_POS, source_pos]) * key_attention * float(raw)
            )
            raw_b_key = torch.einsum(
                "d,df,f->",
                source_resid @ ov0,
                qk2,
                b_embedding_key,
            )
            weighted_b_key = (
                float(attn0[idx, EQUAL_POS, source_pos])
                * key_attention
                * float(raw_b_key)
            )
            raw_pos3_key = torch.einsum(
                "d,df,f->",
                source_resid @ ov0,
                qk2,
                pos3_key,
            )
            weighted_pos3_key = (
                float(attn0[idx, EQUAL_POS, source_pos])
                * key_attention
                * float(raw_pos3_key)
            )
            position_path_terms["position_only"][source_name] = float(position_only)
            position_path_terms["raw"][source_name] = float(raw)
            position_path_terms["weighted"][source_name] = weighted
            position_path_terms["weighted_b_key"][source_name] = weighted_b_key
            position_path_terms["weighted_pos3_key"][source_name] = weighted_pos3_key
        position_path_terms["weighted_sum"] = float(
            sum(position_path_terms["weighted"].values())
        )

        label = str(example["label"])
        breakdown[label] = {
            "prediction": int(preds[idx]),
            "correct_label": correct_label,
            "correct_probability": float(probs[idx, correct_label]),
            "l0_equal_to_lhs_operand": float(attn0[idx, EQUAL_POS, LHS_OPERAND_POS]),
            "l0_equal_to_rhs_operand": float(attn0[idx, EQUAL_POS, RHS_OPERAND_POS]),
            "l1_equal_to_value_b": float(attn1[idx, EQUAL_POS, ASSIGN_VALUE_POS]),
            "l1_equal_to_operand_b": float(
                attn1[idx, EQUAL_POS, int(example["operand_pos"])]
            ),
            "qk2_component_matrix": matrix,
            "qk2_component_total": float(sum(sum(row_terms) for row_terms in matrix)),
            "qk2_attn0_to_attn0": matrix[2][2],
            "source_contributions": source_contributions,
            "position_path_terms": position_path_terms,
            "all_source_contributions": all_source_contributions,
        }

    return breakdown


def collect_b_role_qk2_breakdown() -> dict:
    """Collect actual layer-1 QK2 decompositions for b on LHS vs RHS."""
    examples = b_role_qk2_examples()
    tokens = build_b_role_qk2_tokens(examples)
    steps = checkpoint_steps()
    model = build_model()

    series = {
        str(example["label"]): {
            "prediction": [],
            "correct_probability": [],
            "l0_equal_to_lhs_operand": [],
            "l0_equal_to_rhs_operand": [],
            "l1_equal_to_value_b": [],
            "l1_equal_to_operand_b": [],
            "qk2_component_matrix": [],
            "qk2_component_total": [],
            "qk2_attn0_to_attn0": [],
            "source_contributions": defaultdict(list),
            "position_path_terms": {
                "value_key_attention": [],
                "position_only": defaultdict(list),
                "raw": defaultdict(list),
                "weighted": defaultdict(list),
                "weighted_b_key": defaultdict(list),
                "weighted_pos3_key": defaultdict(list),
                "weighted_sum": [],
            },
            "all_source_contributions": [],
        }
        for example in examples
    }
    correct_label = (ASSIGNED_VALUE + DIRECT_CONSTANT) % MOD

    for step in steps:
        load_checkpoint(model, step)
        step_breakdown = qk2_role_breakdown_for_tokens(model, tokens, examples)
        for example in examples:
            label = str(example["label"])
            values = step_breakdown[label]
            for key in (
                "prediction",
                "correct_probability",
                "l0_equal_to_lhs_operand",
                "l0_equal_to_rhs_operand",
                "l1_equal_to_value_b",
                "l1_equal_to_operand_b",
                "qk2_component_matrix",
                "qk2_component_total",
                "qk2_attn0_to_attn0",
                "all_source_contributions",
            ):
                series[label][key].append(values[key])
            for source_name, value in values["source_contributions"].items():
                series[label]["source_contributions"][source_name].append(value)
            path_terms = values["position_path_terms"]
            series[label]["position_path_terms"]["value_key_attention"].append(
                path_terms["value_key_attention"]
            )
            series[label]["position_path_terms"]["weighted_sum"].append(
                path_terms["weighted_sum"]
            )
            for source_name, value in path_terms["position_only"].items():
                series[label]["position_path_terms"]["position_only"][
                    source_name
                ].append(value)
            for source_name, value in path_terms["raw"].items():
                series[label]["position_path_terms"]["raw"][source_name].append(value)
            for source_name, value in path_terms["weighted"].items():
                series[label]["position_path_terms"]["weighted"][source_name].append(
                    value
                )
            for source_name, value in path_terms["weighted_b_key"].items():
                series[label]["position_path_terms"]["weighted_b_key"][
                    source_name
                ].append(value)
            for source_name, value in path_terms["weighted_pos3_key"].items():
                series[label]["position_path_terms"]["weighted_pos3_key"][
                    source_name
                ].append(value)

    serializable_examples = []
    for example in examples:
        serializable_examples.append(
            {
                "label": str(example["label"]),
                "role": str(example["role"]),
                "var_token": int(example["var_token"]),
                "operand_pos": int(example["operand_pos"]),
                "tokens": build_b_role_qk2_tokens([example])[0].tolist(),
            }
        )

    for label_series in series.values():
        label_series["source_contributions"] = dict(
            label_series["source_contributions"]
        )
        label_series["position_path_terms"]["raw"] = dict(
            label_series["position_path_terms"]["raw"]
        )
        label_series["position_path_terms"]["position_only"] = dict(
            label_series["position_path_terms"]["position_only"]
        )
        label_series["position_path_terms"]["weighted"] = dict(
            label_series["position_path_terms"]["weighted"]
        )
        label_series["position_path_terms"]["weighted_b_key"] = dict(
            label_series["position_path_terms"]["weighted_b_key"]
        )
        label_series["position_path_terms"]["weighted_pos3_key"] = dict(
            label_series["position_path_terms"]["weighted_pos3_key"]
        )

    return {
        "description": (
            "Actual QK2 decomposition for matched b-role examples. Unlike the "
            "embedding-only component, this uses cached layer-0 attention outputs."
        ),
        "steps": steps,
        "assigned_variable_pos": ASSIGN_VAR_POS,
        "assigned_value_pos": ASSIGN_VALUE_POS,
        "assigned_value": ASSIGNED_VALUE,
        "direct_constant": DIRECT_CONSTANT,
        "equal_pos": EQUAL_POS,
        "correct_label": correct_label,
        "qk_scaled_by_sqrt_d_head": True,
        "query_components": ["E_=", "E_15", "h_="],
        "key_components": ["E_7", "E_4", "h_4"],
        "source_contribution_labels": [
            "P13_source",
            "P14_source",
            "assigned_var_source",
            "assigned_value_source",
            "other_sources",
        ],
        "position_path_expression": (
            "a^{(1)}_{=,p} a^{(1)}_{4,3} "
            "((E_{x_p}+E_p)OV_1) QK_2 ((E_b+E_3)OV_1)^T "
            "for operand source positions p in {13,14}."
        ),
        "examples": serializable_examples,
        "series": series,
    }


def b_role_qk2_summary_rows(
    breakdown_data: dict,
    selected_steps: tuple[int, ...] = (14001, 16001),
) -> list[dict[str, float | int | str]]:
    """Return table rows for the actual operand-source QK2 decomposition."""
    steps = breakdown_data["steps"]
    series = breakdown_data["series"]
    tokens_by_label = {
        str(example["label"]): [int(token) for token in example["tokens"]]
        for example in breakdown_data["examples"]
    }

    rows = []
    for step in selected_steps:
        step_idx = steps.index(step)
        for label in ("b LHS", "b RHS"):
            weighted = series[label]["position_path_terms"]["weighted"]
            p13_term = float(weighted["P13"][step_idx])
            p14_term = float(weighted["P14"][step_idx])
            tokens = tokens_by_label[label]
            rows.append(
                {
                    "step": int(step),
                    "display_step": int(step) - 1,
                    "sequence": label,
                    "p13_token": token_label(tokens[LHS_OPERAND_POS]),
                    "p14_token": token_label(tokens[RHS_OPERAND_POS]),
                    "p13_term": p13_term,
                    "p14_term": p14_term,
                    "operand_source_sum": p13_term + p14_term,
                    "all_source_qk2": float(
                        series[label]["qk2_attn0_to_attn0"][step_idx]
                    ),
                    "layer2_attention_to_value_b": float(
                        series[label]["l1_equal_to_value_b"][step_idx]
                    ),
                    "correct_probability": float(
                        series[label]["correct_probability"][step_idx]
                    ),
                }
            )
    return rows


def _signed(value: float) -> str:
    return f"{value:+.3f}"


def format_b_role_qk2_table(
    breakdown_data: dict,
    selected_steps: tuple[int, ...] = (14001, 16001),
) -> str:
    """Format the actual operand-source decomposition as a Markdown table."""
    rows = b_role_qk2_summary_rows(breakdown_data, selected_steps)
    expression = (
        "a^{(1)}_{=,p} a^{(1)}_{4,3} ((E_{x_p}+E_p)OV_1) QK_2 ((E_b+E_3)OV_1)^T"
    )
    lines = [
        "# b-role QK2 source decomposition",
        "",
        f"Operand-source term: `{expression}`",
        "",
        "| Step | Sequence | x_13 | x_14 | p=13 term | p=14 term | p=13+p=14 | all-source QK2 | attn(=, value(b)) | P(correct) |",
        "| ---: | :--- | :---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['display_step']} | "
            f"{row['sequence']} | "
            f"{row['p13_token']} | "
            f"{row['p14_token']} | "
            f"{_signed(float(row['p13_term']))} | "
            f"{_signed(float(row['p14_term']))} | "
            f"{_signed(float(row['operand_source_sum']))} | "
            f"{_signed(float(row['all_source_qk2']))} | "
            f"{float(row['layer2_attention_to_value_b']):.3f} | "
            f"{float(row['correct_probability']):.3f} |"
        )
    lines.extend(
        [
            "",
            (
                "The p=13+p=14 column is a within-sequence sum over the two "
                "actual operand sources, not a sum over counterfactual b "
                "placements."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _matrix_total(matrix: list[list[float]]) -> float:
    return float(sum(sum(float(value) for value in row) for row in matrix))


def b_role_qk2_pruning_rows(
    breakdown_data: dict,
    selected_steps: tuple[int, ...] = (14001, 16001),
) -> list[dict[str, float | int]]:
    """Summarize which full QK2 components survive the LHS/RHS subtraction."""
    steps = breakdown_data["steps"]
    series = breakdown_data["series"]
    rows = []
    for step in selected_steps:
        step_idx = steps.index(step)
        lhs_matrix = series["b LHS"]["qk2_component_matrix"][step_idx]
        rhs_matrix = series["b RHS"]["qk2_component_matrix"][step_idx]
        delta = [
            [
                float(rhs_matrix[row][col]) - float(lhs_matrix[row][col])
                for col in range(3)
            ]
            for row in range(3)
        ]
        lhs_sources = series["b LHS"]["source_contributions"]
        rhs_sources = series["b RHS"]["source_contributions"]
        source_deltas = {
            key: float(rhs_sources[key][step_idx]) - float(lhs_sources[key][step_idx])
            for key in rhs_sources
        }
        non_operand_source_delta = sum(
            value
            for key, value in source_deltas.items()
            if key not in {"P13_source", "P14_source"}
        )
        rows.append(
            {
                "step": int(step),
                "display_step": int(step) - 1,
                "full_lhs": _matrix_total(lhs_matrix),
                "full_rhs": _matrix_total(rhs_matrix),
                "full_delta": _matrix_total(delta),
                "shared_direct_query_delta": sum(sum(delta[row]) for row in (0, 1)),
                "reduced_query_delta": sum(delta[2]),
                "reduced_direct_key_delta": delta[2][0] + delta[2][1],
                "attn_to_attn_delta": delta[2][2],
                "p13_source_delta": source_deltas.get("P13_source", 0.0),
                "p14_source_delta": source_deltas.get("P14_source", 0.0),
                "non_operand_source_delta": non_operand_source_delta,
            }
        )
    return rows


def b_role_qk2_component_rows(
    breakdown_data: dict,
    selected_steps: tuple[int, ...] = (14001, 16001),
) -> list[dict[str, float | int | str]]:
    """Return the explicit 3 x 3 QK2 component table rows."""
    steps = breakdown_data["steps"]
    series = breakdown_data["series"]
    query_components = breakdown_data.get(
        "query_components",
        ["E_=", "E_15", "h_="],
    )
    key_components = breakdown_data.get(
        "key_components",
        ["E_7", "E_4", "h_4"],
    )

    rows = []
    for step in selected_steps:
        step_idx = steps.index(step)
        lhs_matrix = series["b LHS"]["qk2_component_matrix"][step_idx]
        rhs_matrix = series["b RHS"]["qk2_component_matrix"][step_idx]
        for q_idx, q_name in enumerate(query_components):
            for k_idx, k_name in enumerate(key_components):
                lhs_value = float(lhs_matrix[q_idx][k_idx])
                rhs_value = float(rhs_matrix[q_idx][k_idx])
                rows.append(
                    {
                        "step": int(step),
                        "display_step": int(step) - 1,
                        "query_component": str(q_name),
                        "key_component": str(k_name),
                        "lhs_value": lhs_value,
                        "rhs_value": rhs_value,
                        "delta": rhs_value - lhs_value,
                    }
                )
    return rows


def format_b_role_qk2_pruning_table(
    breakdown_data: dict,
    selected_steps: tuple[int, ...] = (14001, 16001),
) -> str:
    """Format the exact component pruning calculation as Markdown tables."""
    pruning_rows = b_role_qk2_pruning_rows(breakdown_data, selected_steps)
    component_rows = b_role_qk2_component_rows(breakdown_data, selected_steps)
    lines = [
        "# b-role QK2 component pruning",
        "",
        (
            "Full score: `S = (E_= + E_15 + h_=) QK_2 "
            "(E_7 + E_4 + h_4)^T`, where "
            "`h_i = sum_t a^{(1)}_{i,t} ((E_{x_t}+E_t)OV_1)`."
        ),
        "",
        (
            "For the matched examples, `E_=`, `E_15`, `E_7`, `E_4`, "
            "and `h_4` are identical. Therefore the `E_=` and `E_15` "
            "query rows can be removed exactly when comparing b RHS to b LHS."
        ),
        "",
        "## Difference After Pruning",
        "",
        "| Step | full delta | shared query delta | reduced h_= delta | h_= to direct key | h_= to h_4 | p13 source | p14 source | non-operand sources |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in pruning_rows:
        lines.append(
            "| "
            f"{row['display_step']} | "
            f"{_signed(float(row['full_delta']))} | "
            f"{_signed(float(row['shared_direct_query_delta']))} | "
            f"{_signed(float(row['reduced_query_delta']))} | "
            f"{_signed(float(row['reduced_direct_key_delta']))} | "
            f"{_signed(float(row['attn_to_attn_delta']))} | "
            f"{_signed(float(row['p13_source_delta']))} | "
            f"{_signed(float(row['p14_source_delta']))} | "
            f"{_signed(float(row['non_operand_source_delta']))} |"
        )

    lines.extend(
        [
            "",
            "## Full 3 x 3 Components",
            "",
            "| Step | query component | key component | b LHS | b RHS | RHS - LHS |",
            "| ---: | :--- | :--- | ---: | ---: | ---: |",
        ]
    )
    for row in component_rows:
        lines.append(
            "| "
            f"{row['display_step']} | "
            f"`{row['query_component']}` | "
            f"`{row['key_component']}` | "
            f"{_signed(float(row['lhs_value']))} | "
            f"{_signed(float(row['rhs_value']))} | "
            f"{_signed(float(row['delta']))} |"
        )
    return "\n".join(lines) + "\n"


def score_b_pool(
    model: HookedTransformer,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    b_value_pos: torch.Tensor,
) -> dict[str, float]:
    with torch.no_grad():
        logits = model(tokens)

    preds = logits[:, -1, :MOD].argmax(dim=-1).cpu()
    lhs_mask = tokens[:, LHS_OPERAND_POS] == B_TOKEN
    rhs_mask = tokens[:, RHS_OPERAND_POS] == B_TOKEN

    qk = (model.blocks[1].attn.W_Q[0] @ model.blocks[1].attn.W_K[0].T) / (
        model.cfg.d_head**0.5
    )
    ov0 = model.blocks[0].attn.W_V[0] @ model.blocks[0].attn.W_O[0]
    rewritten_scores = defaultdict(list)
    source_contribs = defaultdict(list)
    for i, row in enumerate(tokens):
        side = "lhs" if bool(lhs_mask[i]) else "rhs"
        b_src = LHS_OPERAND_POS if side == "lhs" else RHS_OPERAND_POS
        other_src = RHS_OPERAND_POS if side == "lhs" else LHS_OPERAND_POS
        b_var_pos = int(b_value_pos[i]) - 1
        b_var_resid = model.embed.W_E[row[b_var_pos]] + model.pos_embed.W_pos[b_var_pos]
        b_value_key = b_var_resid @ ov0
        side_total = 0.0
        for name, src in (
            ("b_operand_source", b_src),
            ("other_operand_source", other_src),
        ):
            resid_src = model.embed.W_E[row[src]] + model.pos_embed.W_pos[src]
            q_src = resid_src @ ov0
            raw = torch.einsum("d,df,f->", q_src, qk, b_value_key)
            contrib = float(0.5 * raw)
            side_total += contrib
            source_contribs[(side, name)].append(contrib)
        rewritten_scores[side].append(side_total)

    return {
        "b_lhs_acc": float((preds[lhs_mask] == labels[lhs_mask]).float().mean()),
        "b_rhs_acc": float((preds[rhs_mask] == labels[rhs_mask]).float().mean()),
        "b_lhs_rewritten_qk2_score": float(
            torch.tensor(rewritten_scores["lhs"]).mean()
        ),
        "b_rhs_rewritten_qk2_score": float(
            torch.tensor(rewritten_scores["rhs"]).mean()
        ),
        "rhs_b_operand_source": float(
            torch.tensor(source_contribs[("rhs", "b_operand_source")]).mean()
        ),
        "rhs_other_operand_source": float(
            torch.tensor(source_contribs[("rhs", "other_operand_source")]).mean()
        ),
        "lhs_b_operand_source": float(
            torch.tensor(source_contribs[("lhs", "b_operand_source")]).mean()
        ),
        "lhs_other_operand_source": float(
            torch.tensor(source_contribs[("lhs", "other_operand_source")]).mean()
        ),
    }


def accuracy_on_pool(
    model: HookedTransformer,
    tokens: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    with torch.no_grad():
        preds = model(tokens)[:, -1, :MOD].argmax(dim=-1).cpu()
    return float((preds == labels).float().mean())


def collect_data() -> dict:
    cfg = run_config()
    b_tokens, b_labels = _pool_from_cfg(
        "2var_b_operand",
        cfg,
        VOCAB,
        MOD,
        pool_size=POOL_SIZE,
        seed=123,
    )
    var2_tokens, var2_labels = _pool_from_cfg(
        "2var_valid_pair_2_invalid_vars",
        cfg,
        VOCAB,
        MOD,
        pool_size=POOL_SIZE,
        seed=123,
    )
    b_value_pos = b_value_positions(b_tokens)

    model = build_model()
    steps = checkpoint_steps()
    series = defaultdict(list)

    for step in steps:
        load_checkpoint(model, step)

        b_scores = score_b_pool(model, b_tokens, b_labels, b_value_pos)
        for name, value in b_scores.items():
            series[name].append(value)
        series["two_var_var_restricted_2_acc"].append(
            accuracy_on_pool(model, var2_tokens, var2_labels)
        )

    return {
        "steps": steps,
        "pool_size": POOL_SIZE,
        "series": dict(series),
    }


def plot_data(data: dict) -> None:
    steps = data["steps"]
    series = data["series"]

    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.75), layout="none")
    ax_a, ax_b, ax_c = axes

    definitions = (
        r"$=\mathrm{\ query}\approx\frac{1}{2}(e_{x_{13}}+P_{13})OV_1"
        r"+\frac{1}{2}(e_{x_{14}}+P_{14})OV_1,\quad "
        r"\mathrm{value}(b)\ \mathrm{key}\approx(e_b+P_{t_b-1})OV_1$"
    )
    fig.text(
        0.52,
        0.975,
        definitions,
        ha="center",
        va="top",
        fontsize=FONT_SIZE_ANNOTATION,
    )

    ax_a.plot(
        steps,
        series["b_rhs_acc"],
        color="#17becf",
        linewidth=1.7,
        label=r"$b$ as RHS",
    )
    ax_a.plot(
        steps,
        series["two_var_var_restricted_2_acc"],
        color="#2ca02c",
        linestyle="-.",
        linewidth=1.5,
        label="2-var var-restricted (2)",
    )
    ax_a.set_title("(a)", loc="left", fontsize=FONT_SIZE_BODY, fontweight="bold")
    ax_a.set_xlabel("Training Step", fontsize=FONT_SIZE_BODY)
    ax_a.set_ylabel("Accuracy", fontsize=FONT_SIZE_BODY)
    ax_a.set_ylim(-0.02, 1.05)
    ax_a.legend(fontsize=FONT_SIZE_LEGEND, loc="lower right")

    ax_b.plot(
        steps,
        series["b_lhs_rewritten_qk2_score"],
        color="#9467bd",
        linewidth=1.7,
        label=r"$b$ as LHS",
    )
    ax_b.plot(
        steps,
        series["b_rhs_rewritten_qk2_score"],
        color="#17becf",
        linewidth=1.7,
        label=r"$b$ as RHS",
    )
    ax_b.set_title("(b)", loc="left", fontsize=FONT_SIZE_BODY, fontweight="bold")
    ax_b.set_xlabel("Training Step", fontsize=FONT_SIZE_BODY)
    ax_b.set_ylabel(
        r"Approx. layer-2 logit: $=\to\mathrm{value}(b)$", fontsize=FONT_SIZE_BODY
    )
    ax_b.legend(fontsize=FONT_SIZE_LEGEND, loc="lower right")

    ax_c.plot(
        steps,
        series["rhs_b_operand_source"],
        color="#17becf",
        linewidth=1.7,
        label=(
            r"$\frac{1}{2}(e_b+P_{14})OV_1QK_2"
            r"((e_b+P_{t_b-1})OV_1)^\top$"
        ),
    )
    ax_c.plot(
        steps,
        series["rhs_other_operand_source"],
        color="#8c564b",
        linestyle="--",
        linewidth=1.7,
        label=(
            r"$\frac{1}{2}(e_{\mathrm{other}}+P_{13})OV_1QK_2"
            r"((e_b+P_{t_b-1})OV_1)^\top$"
        ),
    )
    ax_c.plot(
        steps,
        series["b_rhs_rewritten_qk2_score"],
        color="0.2",
        linestyle="-.",
        linewidth=1.4,
        label="sum",
    )
    ax_c.set_title("(c)", loc="left", fontsize=FONT_SIZE_BODY, fontweight="bold")
    ax_c.set_xlabel("Training Step", fontsize=FONT_SIZE_BODY)
    ax_c.set_ylabel(r"RHS terms for $=\to\mathrm{value}(b)$", fontsize=FONT_SIZE_BODY)
    ax_c.legend(fontsize=FONT_SIZE_LEGEND - 1, loc="lower right")

    for ax in axes:
        ax.tick_params(labelsize=FONT_SIZE_TICK)
        ax.grid(False)

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.82, wspace=0.38)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / "b_rhs_asymmetry_mechanism.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = collect_data()
    json_path = OUTPUT_DIR / "b_rhs_asymmetry_analysis.json"
    json_path.write_text(json.dumps(data, indent=2))
    print(f"Saved: {json_path}")

    qk2_data = collect_qk2_operand_embedding_component()
    qk2_json_path = OUTPUT_DIR / QK2_COMPONENT_JSON
    qk2_json_path.write_text(json.dumps(qk2_data, indent=2))
    print(f"Saved: {qk2_json_path}")

    b_role_qk2_data = collect_b_role_qk2_breakdown()
    b_role_qk2_json_path = OUTPUT_DIR / B_ROLE_QK2_JSON
    b_role_qk2_json_path.write_text(json.dumps(b_role_qk2_data, indent=2))
    print(f"Saved: {b_role_qk2_json_path}")

    b_role_qk2_table_path = OUTPUT_DIR / B_ROLE_QK2_TABLE
    b_role_qk2_table_path.write_text(format_b_role_qk2_table(b_role_qk2_data))
    print(f"Saved: {b_role_qk2_table_path}")

    b_role_qk2_pruning_path = OUTPUT_DIR / B_ROLE_QK2_PRUNING_TABLE
    b_role_qk2_pruning_path.write_text(format_b_role_qk2_pruning_table(b_role_qk2_data))
    print(f"Saved: {b_role_qk2_pruning_path}")

    plot_data(data)


if __name__ == "__main__":
    main()
