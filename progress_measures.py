# %%
import json
import os
from collections.abc import Callable
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from fancy_einsum import einsum
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer
try:
    from tueplots import bundles
except ImportError:
    bundles = None
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

from model_io import DEFAULT_CHECKPOINT_DIR, build_model, default_config, torch_load

pio.renderers.default = "notebook"
if os.getenv("GITHUB_ACTIONS") == "true":
    pio.renderers.default = "svg"

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cpu")
)
print(f"Using device: {DEVICE}")

torch.set_grad_enabled(False)

# %%
RUN_NAME = "offline-checkpoints"


def compute_ov1_mlp1_accuracy(model: HookedTransformer, mod: int) -> float:
    W_V_1 = model.blocks[1].attn.W_V
    W_O_1 = model.blocks[1].attn.W_O
    OV_1 = einsum("h d_in d_h, h d_h d_out -> d_in d_out", W_V_1, W_O_1)

    ov_1_nums = model.embed.W_E[:mod] @ OV_1

    mlp = model.blocks[1].mlp
    W_U = model.unembed.W_U

    correct = 0
    for a in range(mod):
        for b in range(mod):
            x = ov_1_nums[a] + ov_1_nums[b]
            hidden = torch.relu(x @ mlp.W_in + mlp.b_in)
            mlp_out = hidden @ mlp.W_out + mlp.b_out
            logits = mlp_out @ W_U
            if logits.argmax().item() == (a + b) % mod:
                correct += 1

    return correct / (mod ** 2)


def compute_qk0_num_to_prev_var_accuracy(model: HookedTransformer, mod: int) -> float:
    W_Q = model.blocks[0].attn.W_Q
    W_K = model.blocks[0].attn.W_K
    QK_0 = einsum(
        "heads d_model_q d_head, heads d_model_k d_head -> d_model_q d_model_k",
        W_Q, W_K
    )
    
    seq_len = model.cfg.n_ctx
    num_positions = min(12, seq_len)
    pos_emb = model.pos_embed.W_pos[:num_positions]
    
    qk_scores = pos_emb @ QK_0 @ pos_emb.T
    
    correct = 0
    total = 0
    
    for query_pos in range(1, num_positions):
        row = qk_scores[query_pos].clone()
        row[query_pos + 1:] = float('-inf')
        row[query_pos] = float('-inf')
        if query_pos >= 2:
            row[query_pos - 2] = float('-inf')
        
        target_pos = query_pos - 1
        if row.argmax().item() == target_pos:
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0.0


def compute_qk1_ov0_var_identity_accuracy(model: HookedTransformer, mod: int) -> float:
    vocab = model.cfg.d_vocab
    var_len = vocab - mod - 3
    var_start = mod + 3
    var_indices = list(range(var_start, var_start + var_len))
    
    if var_len <= 0:
        return 0.0
    
    var_emb = model.embed.W_E[var_indices]
    
    W_V_0 = model.blocks[0].attn.W_V
    W_O_0 = model.blocks[0].attn.W_O
    OV_0 = einsum("h d_in d_h, h d_h d_out -> d_in d_out", W_V_0, W_O_0)
    
    W_Q_1 = model.blocks[1].attn.W_Q
    W_K_1 = model.blocks[1].attn.W_K
    QK_1 = einsum(
        "heads d_model_q d_head, heads d_model_k d_head -> d_model_q d_model_k",
        W_Q_1, W_K_1
    )
    
    vars_through_ov0 = var_emb @ OV_0
    
    similarity = vars_through_ov0 @ QK_1 @ vars_through_ov0.T
    
    correct = 0
    for i in range(var_len):
        if similarity[i].argmax().item() == i:
            correct += 1
    
    return correct / var_len


_EVAL_BATCH_CACHE: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}


def generate_eval_batch(
    mod: int,
    vocab: int,
    seq_len: int = 16,
    batch_size: int = 512,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    import random as py_random
    
    cache_key = (mod, vocab)
    if cache_key in _EVAL_BATCH_CACHE:
        return _EVAL_BATCH_CACHE[cache_key]
    
    rng = py_random.Random(seed)
    
    plus_id = mod
    equal_id = mod + 1
    pad_id = mod + 2
    var_start = mod + 3
    var_len = vocab - mod - 3
    var_ids = list(range(var_start, var_start + var_len))
    
    all_perms = []
    for var_tok in var_ids:
        for typ in range(3):
            for num1 in range(mod):
                for num2 in range(mod):
                    all_perms.append((var_tok, typ, num1, num2))
    
    rng_sample = py_random.Random(seed)
    sampled_perms = [rng_sample.choice(all_perms) for _ in range(batch_size)]
    
    all_tokens = []
    all_target_positions = []
    
    plus_pos = seq_len - 4
    lhs_pos = seq_len - 3
    rhs_pos = seq_len - 2
    equal_pos = seq_len - 1
    
    for var_tok, typ, num1, num2 in sampled_perms:
        if typ == 2:
            var_val = num2
        elif typ == 1:
            var_val = num1
        else:
            var_val = rng.randint(0, mod - 1)
        
        if typ == 0:
            lhs_tok, rhs_tok = num1, num2
        elif typ == 1:
            lhs_tok, rhs_tok = var_tok, num2
        else:
            lhs_tok, rhs_tok = num1, var_tok
        
        core_len = 4
        assignment_len = 2
        max_assignments = min(len(var_ids), (seq_len - core_len) // assignment_len)
        n_assignments = rng.randint(1, max_assignments)
        
        assignments = [[var_tok, var_val]]

        extra_vars = [v for v in var_ids if v != var_tok]
        rng.shuffle(extra_vars)
        for _ in range(n_assignments - 1):
            v = extra_vars.pop()
            assignments.append([v, rng.randint(0, mod - 1)])
        
        rng.shuffle(assignments)
        
        var_value_pos = -1
        for idx, (v, _) in enumerate(assignments):
            if v == var_tok:
                break
        
        remaining_pads = seq_len - (assignment_len * n_assignments + core_len)
        gaps = [0] * (n_assignments + 1)
        for _ in range(remaining_pads):
            gaps[rng.randint(0, n_assignments)] += 1
        
        prefix = []
        for idx, seg in enumerate(assignments):
            prefix.extend([pad_id] * gaps[idx])
            if seg[0] == var_tok:
                var_value_pos = len(prefix) + 1
            prefix.extend(seg)
        prefix.extend([pad_id] * gaps[-1])
        
        core = [plus_id, lhs_tok, rhs_tok, equal_id]
        sequence = prefix + core
        
        if typ == 0:
            target_pos = [lhs_pos, rhs_pos]
        elif typ == 1:
            target_pos = [var_value_pos, rhs_pos]
        else:
            target_pos = [lhs_pos, var_value_pos]
        
        all_tokens.append(sequence)
        all_target_positions.append(target_pos)
    
    tokens = torch.tensor(all_tokens, dtype=torch.long)
    target_positions = torch.tensor(all_target_positions, dtype=torch.long)
    
    _EVAL_BATCH_CACHE[cache_key] = (tokens, target_positions)
    return tokens, target_positions


def compute_attn_l0_equal_to_operands(model: HookedTransformer, mod: int) -> float:
    vocab = model.cfg.d_vocab
    seq_len = model.cfg.n_ctx
    tokens, _ = generate_eval_batch(mod, vocab, seq_len)
    tokens = tokens.to(DEVICE)
    
    lhs_pos = seq_len - 3
    rhs_pos = seq_len - 2
    equal_pos = seq_len - 1
    
    _, cache = model.run_with_cache(tokens, return_type=None)
    
    attn_l0 = cache["attn", 0]
    attn_l0_summed = attn_l0.sum(dim=1)
    attn_from_equal = attn_l0_summed[:, equal_pos, :]
    
    top2_indices = attn_from_equal.topk(k=2, dim=-1).indices
    
    target_set = {lhs_pos, rhs_pos}
    correct = 0
    for i in range(tokens.shape[0]):
        actual_set = set(top2_indices[i].tolist())
        if actual_set == target_set:
            correct += 1
    
    return correct / tokens.shape[0]


def compute_attn_l0_num_to_prev(model: HookedTransformer, mod: int) -> float:
    vocab = model.cfg.d_vocab
    seq_len = model.cfg.n_ctx
    tokens, _ = generate_eval_batch(mod, vocab, seq_len)
    tokens = tokens.to(DEVICE)
    
    prefix_end = seq_len - 4
    
    _, cache = model.run_with_cache(tokens, return_type=None)
    
    attn_l0 = cache["attn", 0]
    attn_l0_summed = attn_l0.sum(dim=1)
    
    correct = 0
    total = 0
    
    for batch_idx in range(tokens.shape[0]):
        for pos in range(1, prefix_end):
            token_id = tokens[batch_idx, pos].item()
            
            if token_id < mod:
                attn_row = attn_l0_summed[batch_idx, pos, :]
                
                if attn_row.argmax().item() == pos - 1:
                    correct += 1
                total += 1
    
    return correct / total if total > 0 else 0.0


def compute_attn_l1_equal_to_values(model: HookedTransformer, mod: int) -> float:
    vocab = model.cfg.d_vocab
    seq_len = model.cfg.n_ctx
    tokens, target_positions = generate_eval_batch(mod, vocab, seq_len)
    tokens = tokens.to(DEVICE)
    target_positions = target_positions.to(DEVICE)
    
    equal_pos = seq_len - 1
    
    _, cache = model.run_with_cache(tokens, return_type=None)
    
    attn_l1 = cache["attn", 1]
    attn_l1_summed = attn_l1.sum(dim=1)
    attn_from_equal = attn_l1_summed[:, equal_pos, :]
    
    top2_indices = attn_from_equal.topk(k=2, dim=-1).indices
    
    correct = 0
    for i in range(tokens.shape[0]):
        actual_set = set(top2_indices[i].tolist())
        target_set = set(target_positions[i].tolist())
        if actual_set == target_set:
            correct += 1
    
    return correct / tokens.shape[0]


def train_linear_probe(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    in_dim: int,
    out_dim: int,
    epochs: int = 4,
    lr: float = 1e-2,
    wd: float = 1e-4,
) -> float:
    prev_grad_state = torch.is_grad_enabled()
    torch.set_grad_enabled(True)
    
    try:
        probe = nn.Linear(in_dim, out_dim, bias=True).to(DEVICE)
        optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=wd)
        loss_fn = nn.CrossEntropyLoss()
        
        probe.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            logits = probe(x_train)
            loss = loss_fn(logits, y_train)
            loss.backward()
            optimizer.step()
        
        probe.eval()
        with torch.no_grad():
            pred = probe(x_test).argmax(dim=-1)
            acc = (pred == y_test).float().mean().item()
        
        return acc
    finally:
        torch.set_grad_enabled(prev_grad_state)


_PROBE_BATCH_CACHE: dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}


def generate_probe_batch(
    mod: int,
    vocab: int,
    seq_len: int = 16,
    batch_size: int = 1024,
    seed: int = 12345,
) -> tuple[torch.Tensor, torch.Tensor]:
    import random as py_random
    
    cache_key = (mod, vocab, seed)
    if cache_key in _PROBE_BATCH_CACHE:
        return _PROBE_BATCH_CACHE[cache_key]
    
    rng = py_random.Random(seed)
    
    plus_id = mod
    equal_id = mod + 1
    pad_id = mod + 2
    var_start = mod + 3
    var_len = vocab - mod - 3
    var_ids = list(range(var_start, var_start + var_len))
    
    all_perms = []
    for var_tok in var_ids:
        for typ in range(3):
            for num1 in range(mod):
                for num2 in range(mod):
                    all_perms.append((var_tok, typ, num1, num2))
    
    rng_sample = py_random.Random(seed)
    sampled_perms = [rng_sample.choice(all_perms) for _ in range(batch_size)]
    
    all_tokens = []
    all_target_positions = []
    
    for var_tok, typ, num1, num2 in sampled_perms:
        if typ == 2:
            var_val = num2
        elif typ == 1:
            var_val = num1
        else:
            var_val = rng.randint(0, mod - 1)
        
        if typ == 0:
            lhs_tok, rhs_tok = num1, num2
        elif typ == 1:
            lhs_tok, rhs_tok = var_tok, num2
        else:
            lhs_tok, rhs_tok = num1, var_tok
        
        core_len = 4
        assignment_len = 2
        max_assignments = min(len(var_ids), (seq_len - core_len) // assignment_len)
        n_assignments = rng.randint(1, max_assignments)
        
        assignments = [[var_tok, var_val]]
        extra_vars = [v for v in var_ids if v != var_tok]
        rng.shuffle(extra_vars)
        for _ in range(n_assignments - 1):
            v = extra_vars.pop()
            assignments.append([v, rng.randint(0, mod - 1)])
        
        rng.shuffle(assignments)
        
        remaining_pads = seq_len - (assignment_len * n_assignments + core_len)
        gaps = [0] * (n_assignments + 1)
        for _ in range(remaining_pads):
            gaps[rng.randint(0, n_assignments)] += 1
        
        prefix = []
        var_value_pos = -1
        for idx, seg in enumerate(assignments):
            prefix.extend([pad_id] * gaps[idx])
            if seg[0] == var_tok:
                var_value_pos = len(prefix) + 1
            prefix.extend(seg)
        prefix.extend([pad_id] * gaps[-1])
        
        core = [plus_id, lhs_tok, rhs_tok, equal_id]
        sequence = prefix + core
        
        if typ == 0:
            target_pos = [seq_len - 3, seq_len - 2]
        elif typ == 1:
            target_pos = [var_value_pos, seq_len - 2]
        else:
            target_pos = [seq_len - 3, var_value_pos]
        
        all_tokens.append(sequence)
        all_target_positions.append(target_pos)
    
    tokens = torch.tensor(all_tokens, dtype=torch.long)
    target_positions = torch.tensor(all_target_positions, dtype=torch.long)
    
    _PROBE_BATCH_CACHE[cache_key] = (tokens, target_positions)
    return tokens, target_positions


def compute_probe_l0_mid_var_from_num(model: HookedTransformer, mod: int) -> float:
    """Train linear probe to predict variable name from resid_mid[0] at number positions.
    
    For each [var, num] pair in the prefix:
    - Extract resid_mid at layer 0 for the number position
    - Label is the variable index (which variable this number belongs to)
    
    Returns test accuracy of the trained probe.
    """
    vocab = model.cfg.d_vocab
    seq_len = model.cfg.n_ctx
    d_model = model.cfg.d_model
    
    var_start = mod + 3
    var_len = vocab - mod - 3
    
    if var_len <= 0:
        return 0.0
    
    var_to_idx = {var_start + i: i for i in range(var_len)}
    
    train_tokens, _ = generate_probe_batch(mod, vocab, seq_len, batch_size=1024, seed=11111)
    test_tokens, _ = generate_probe_batch(mod, vocab, seq_len, batch_size=256, seed=22222)
    
    train_tokens = train_tokens.to(DEVICE)
    test_tokens = test_tokens.to(DEVICE)
    
    _, train_cache = model.run_with_cache(train_tokens, return_type=None)
    _, test_cache = model.run_with_cache(test_tokens, return_type=None)
    
    train_resid_mid = train_cache["resid_pre", 0] + train_cache["attn_out", 0]
    test_resid_mid = test_cache["resid_pre", 0] + test_cache["attn_out", 0]
    
    prefix_end = seq_len - 4  # Before the core [+, lhs, rhs, =]
    
    def extract_probe_data(tokens: torch.Tensor, resid_mid: torch.Tensor):
        feats = []
        labels = []
        batch_size = tokens.shape[0]
        
        for b in range(batch_size):
            for pos in range(1, prefix_end):
                token_id = tokens[b, pos].item()
                prev_token_id = tokens[b, pos - 1].item()
                
                if token_id < mod and prev_token_id in var_to_idx:
                    feats.append(resid_mid[b, pos, :])
                    labels.append(var_to_idx[prev_token_id])
        
        if len(feats) == 0:
            return None, None
        
        x = torch.stack(feats)
        y = torch.tensor(labels, device=DEVICE, dtype=torch.long)
        return x, y
    
    x_train, y_train = extract_probe_data(train_tokens, train_resid_mid)
    x_test, y_test = extract_probe_data(test_tokens, test_resid_mid)
    
    if x_train is None or x_test is None or len(y_train) == 0 or len(y_test) == 0:
        return 0.0
    
    acc = train_linear_probe(x_train, y_train, x_test, y_test, d_model, var_len)
    
    return acc


from eval_pools import (  # noqa: E402
    SPECIAL_POOL_KINDS,
    compute_sel_pairs,
    get_restriction_vars,
    generate_special_pool,
    get_pool_0var_valid_pair,
    get_pool_0var_invalid_pair,
    get_pool_1var_valid_pair_valid_var,
    get_pool_1var_valid_pair_invalid_var,
    get_pool_1var_invalid_pair_valid_var,
    get_pool_2var_valid_pair_valid_vars,
    get_pool_2var_valid_pair_1_invalid_var,
    get_pool_2var_valid_pair_2_invalid_vars,
    get_pool_2var_invalid_pair_valid_var,
    _pool_from_cfg,
)

MEASURES: dict[str, Callable[[HookedTransformer, int], float]] = {
    "ov1_mlp1_accuracy": compute_ov1_mlp1_accuracy,
    "qk0_num_to_prev_var": compute_qk0_num_to_prev_var_accuracy,
    "qk1_ov0_var_identity": compute_qk1_ov0_var_identity_accuracy,
    "attn_l0_equal_to_operands": compute_attn_l0_equal_to_operands,
    "attn_l0_num_to_prev": compute_attn_l0_num_to_prev,
    "attn_l1_equal_to_values": compute_attn_l1_equal_to_values,
    "probe_l0_mid_var_from_num": compute_probe_l0_mid_var_from_num,
}


ACCURACY_OVERLAY_POOLS: dict[str, str] = {
    "zero_var_train_acc":                "0var_valid_pair",
    "zero_var_addition_restricted_acc":  "0var_invalid_pair",
    "one_var_train_acc":                 "1var_valid_pair_valid_var",
    "one_var_addition_restricted_acc":   "1var_invalid_pair_valid_var",
    "one_var_variable_restricted_acc":   "1var_valid_pair_invalid_var",
    "two_var_train_acc":                 "2var_valid_pair_valid_vars",
    "two_var_addition_restricted_acc":   "2var_invalid_pair_valid_var",
    "two_var_variable_restricted_1_acc": "2var_valid_pair_1_invalid_var",
    "two_var_variable_restricted_2_acc": "2var_valid_pair_2_invalid_vars",
}


def _accuracy_on_pool(
    model: HookedTransformer,
    mod: int,
    tokens: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    if tokens.shape[0] == 0:
        return float("nan")
    tokens = tokens.to(DEVICE)
    labels = labels.to(DEVICE)
    logits = model(tokens)
    preds = logits[:, -1, :mod].argmax(dim=-1)
    return (preds == labels).float().mean().item()


# %% Main loop


def run_measures(
    checkpoints: list[tuple[int, Path]],
    model: HookedTransformer,
    mod: int,
    accuracy_pools: dict[str, tuple[torch.Tensor, torch.Tensor]] = None,
) -> tuple[list[int], dict[str, list[float]], dict[str, list[float]]]:
    accuracy_pools = accuracy_pools or {}

    results = {name: [] for name in MEASURES}
    accuracy_overlays = {name: [] for name in accuracy_pools}
    steps = []

    for step, ckpt_path in tqdm(checkpoints, desc="Processing checkpoints"):
        state_dict = torch_load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(state_dict)

        steps.append(step)

        for name, fn in MEASURES.items():
            value = fn(model, mod)
            results[name].append(value)

        for name, (tokens, labels) in accuracy_pools.items():
            accuracy_overlays[name].append(_accuracy_on_pool(model, mod, tokens, labels))

    return steps, results, accuracy_overlays


# %%

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute progress measures from local checkpoints.")
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--output", default="progress_measures.json")
    parser.add_argument("--testing-mode", action="store_true")
    args = parser.parse_args()

    hyperparams = default_config()
    vocab = int(hyperparams["VOCAB"])
    mod = int(hyperparams["MOD"])
    print(f"MOD={mod}, VOCAB={vocab}")

    specialized_pools = {
        kind: _pool_from_cfg(kind, hyperparams, vocab, mod)
        for kind in SPECIAL_POOL_KINDS
    }
    print(
        "Specialized pool sizes:",
        {k: v[0].shape[0] for k, v in specialized_pools.items()},
    )

    model = build_model(device=DEVICE)
    print(f"Created model with {model.cfg.n_layers} layers, {model.cfg.n_heads} heads")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_steps = [
        int(path.stem.rsplit("_", 1)[1])
        for path in checkpoint_dir.glob("checkpoint_step_*.pth")
        if path.stat().st_size > 0
    ]
    checkpoint_steps = sorted(checkpoint_steps)
    checkpoints = [
        (step, checkpoint_dir / f"checkpoint_step_{step}.pth")
        for step in checkpoint_steps
    ]
    print(f"Found {len(checkpoints)} local checkpoints")

    if args.testing_mode and len(checkpoints) > 2:
        checkpoints = [checkpoints[0], checkpoints[-1]]
        print(
            "testing mode: using first and last checkpoints "
            f"(steps {checkpoints[0][0]}, {checkpoints[-1][0]})"
        )
    
    accuracy_pools = {
        name: specialized_pools[kind]
        for name, kind in ACCURACY_OVERLAY_POOLS.items()
        if kind in specialized_pools
    }
    print(
        "Accuracy overlay pool sizes:",
        {k: v[0].shape[0] for k, v in accuracy_pools.items()},
    )

    steps, results, accuracy_overlays = run_measures(
        checkpoints, model, mod, accuracy_pools=accuracy_pools
    )

    output_data = {
        "run_name": RUN_NAME,
        "mod": mod,
        "vocab": vocab,
        "steps": steps,
        "measures": results,
        "accuracy_overlays": accuracy_overlays,
    }
    json_output_path = args.output
    with open(json_output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved results to {json_output_path}")

    print("\nFinal values:")
    for name, values in results.items():
        print(f"  {name}: {values[-1]:.4f}")
    for name, values in accuracy_overlays.items():
        final = values[-1] if values else float("nan")
        print(f"  {name}: {final:.4f}")

# %%
