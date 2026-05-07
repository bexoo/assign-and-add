# %%
# %%
import os
import random
from collections import defaultdict
from dataclasses import asdict

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/cache")

import einops
from functools import partial
from jaxtyping import Float
import matplotlib.pyplot as plt
try:
    from tueplots import bundles
except ImportError:
    bundles = None
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio
import plotly.basedatatypes as plotly_base
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from tqdm.auto import trange
import tqdm.auto as tqdm
from fancy_einsum import einsum
from transformer_lens import HookedTransformer, HookedTransformerConfig, ActivationCache
from transformer_lens.hook_points import (
    HookPoint,
)
import transformer_lens.utils as utils

pio.renderers.default = "json"
plotly_base.BaseFigure.show = lambda self, *args, **kwargs: None

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)
print(f"Using device: {DEVICE}")

torch.set_grad_enabled(False)


def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.imshow(
        utils.to_numpy(tensor),
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        labels={"x": xaxis, "y": yaxis},
        **kwargs,
    ).show(renderer)


def line(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.line(utils.to_numpy(tensor), labels={"x": xaxis, "y": yaxis}, **kwargs).show(
        renderer
    )


def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(
        y=y, x=x, labels={"x": xaxis, "y": yaxis, "color": caxis}, **kwargs
    ).show(renderer)


# %%
from model_io import build_run_config, default_config, load_model_from_checkpoint

cfg_wb = default_config()
run_cfg = build_run_config(cfg_wb)

MOD = run_cfg.mod
print(MOD)

PLUS = run_cfg.plus_id
EQUAL = run_cfg.equal_id
PAD = run_cfg.pad_id
A_TOKEN = run_cfg.a_token_id
VAR_LEN = run_cfg.n_vars
VARS = list(run_cfg.var_ids)
FIRST_HALF, SECOND_HALF = VARS[: VAR_LEN // 2], VARS[VAR_LEN // 2 :]

VOCAB = run_cfg.vocab
TOKEN_STRINGS = list(run_cfg.token_strings)

SEQ_LEN = run_cfg.seq_len
BATCH_SIZE = 512
USE_SIMPLE_16 = bool(cfg_wb["USE_SIMPLE_16"])

# %%
V, N, TYPES = len(VARS), MOD, 3

var_g = torch.tensor(VARS).view(V, 1, 1, 1)
type_g = torch.arange(TYPES).view(1, TYPES, 1, 1)
num1_g = torch.arange(N).view(1, 1, N, 1)
num2_g = torch.arange(N).view(1, 1, 1, N)

perm_tensor = torch.stack(
    [
        var_g.repeat(1, TYPES, N, N),
        type_g.repeat(V, 1, N, N),
        num1_g.repeat(V, TYPES, 1, N),
        num2_g.repeat(V, TYPES, N, 1),
    ],
    dim=-1,
)

perm_table = einops.rearrange(perm_tensor, "v t n m f -> (v t n m) f").contiguous()
TOTAL_PERMS = perm_table.shape[0]
print(f"Total permutations: {TOTAL_PERMS}")

del perm_tensor

# %% 2. Simple random train / test split -------------------------------------
TRAIN_FRAC = float(cfg_wb.get("TRAIN_FRAC", 0.8))  # fraction of rows in train
all_idx = torch.arange(len(perm_table), dtype=torch.long)
perm = torch.randperm(len(all_idx), generator=torch.Generator().manual_seed(SEED))
split = int(len(all_idx) * TRAIN_FRAC)
train_idx = all_idx[perm[:split]]
_test_idx = all_idx[perm[split:]]
print(f"Train rows: {len(train_idx)} | Test rows: {len(_test_idx)}")


# %% 3. Sequence builder utilities ------------------------------------------
def build_sequence(row: torch.Tensor):
    """Return a length-``SEQ_LEN`` sequence and label for the given permutation row."""
    var_tok, typ, num1, num2 = row.tolist()

    if typ == 2:
        var_val = num2
    elif typ == 1:
        var_val = num1
    else:
        var_val = random.randrange(MOD)

    core_len = 4
    assignment_len = 2
    max_assignments = min(len(VARS), (SEQ_LEN - core_len) // assignment_len)
    if max_assignments < 1:
        raise ValueError("Sequence length too short to fit any assignments")
    n_assignments = random.randint(1, max_assignments)

    assignments = [[var_tok, var_val]]

    extra_vars = [v for v in VARS if v != var_tok]
    random.shuffle(extra_vars)
    for _ in range(n_assignments - 1):
        v = extra_vars.pop()
        assignments.append([v, random.randrange(MOD)])

    random.shuffle(assignments)

    remaining_pads = SEQ_LEN - (assignment_len * n_assignments + core_len)
    gaps = [0] * (n_assignments + 1)
    for _ in range(remaining_pads):
        gaps[random.randint(0, n_assignments)] += 1

    prefix = []
    for idx, seg in enumerate(assignments):
        prefix.extend([PAD] * gaps[idx])
        prefix.extend(seg)
    prefix.extend([PAD] * gaps[-1])

    if typ == 0:
        lhs_tok, rhs_tok = num1, num2
    elif typ == 1:
        lhs_tok, rhs_tok = var_tok, num2
    else:
        lhs_tok, rhs_tok = num1, var_tok

    core = [PLUS, lhs_tok, rhs_tok, EQUAL]
    tok = torch.tensor(prefix + core, dtype=torch.long)
    label = (num1 + num2) % MOD
    return tok, label


def build_sequence_simple_16(row: torch.Tensor):
    """Return a simple 16-token template sequence and its label."""
    var_tok, typ, num1, num2 = row.tolist()

    if typ == 2:
        var_val = num2
    elif typ == 1:
        var_val = num1
    else:
        var_val = random.randrange(MOD)

    correct_slot = random.randrange(3)
    assignments = []
    used_vars = {var_tok}
    for slot in range(3):
        if slot == correct_slot:
            assignments.extend([var_tok, var_val])
        else:
            avail = [v for v in VARS if v not in used_vars]
            d_var = random.choice(avail)
            used_vars.add(d_var)
            assignments.extend([d_var, random.randrange(MOD)])

    if typ == 0:
        lhs_tok, rhs_tok = num1, num2
    elif typ == 1:
        lhs_tok, rhs_tok = var_tok, num2
    else:
        lhs_tok, rhs_tok = num1, var_tok

    core = [PLUS, lhs_tok, rhs_tok, EQUAL]
    sequence = assignments + core
    padding_needed = SEQ_LEN - len(sequence)
    if padding_needed < 0:
        raise ValueError("Sequence template exceeds SEQ_LEN")
    sequence.extend([PAD] * padding_needed)

    tok = torch.tensor(sequence, dtype=torch.long)
    label = (num1 + num2) % MOD
    return tok, label


def _seq_builder(row: torch.Tensor):
    return build_sequence_simple_16(row) if USE_SIMPLE_16 else build_sequence(row)




def get_batch_from_indices(index_tensor: torch.Tensor):
    sel = index_tensor[torch.randint(0, len(index_tensor), (BATCH_SIZE,))]
    tok, lab = zip(*(_seq_builder(perm_table[i]) for i in sel))
    return torch.stack(tok).to(DEVICE), torch.tensor(lab, device=DEVICE)


# %% 4. Load local pretrained model -----------------------------------------
print("\nLoading local model from checkpoints/model_state_dict.pth...")
model, cfg_wb, _run_cfg = load_model_from_checkpoint(
    "checkpoints/model_state_dict.pth", config=cfg_wb, device=DEVICE
)
for p in model.parameters():
    p.requires_grad = False
print("Model loaded.")

# %% 5. Linear probe utilities ----------------------------------------------

PROBE_LAYERS = list(range(model.cfg.n_layers))
PROBE_TRAIN_SEQS = 4096
PROBE_TEST_SEQS = 1024
PROBE_EPOCHS = 50
PROBE_LR = 1e-2
PROBE_WD = 1e-4
D_MODEL = model.cfg.d_model

torch.set_grad_enabled(True)  # enable grads for probe training


VAR_TO_IDX = {tok: i for i, tok in enumerate(VARS)}
IDX_TO_VAR = {i: tok for tok, i in VAR_TO_IDX.items()}
PROBE_LAYER = model.cfg.n_layers - 1


def sample_sequences(
    index_tensor: torch.Tensor, count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample `count` sequences from `index_tensor` using `_seq_builder`.
    Returns (tokens [count, SEQ_LEN], gold_result_mod53 [count,]).
    """
    sel = index_tensor[torch.randint(0, len(index_tensor), (count,))]
    tok, lab = zip(*(_seq_builder(perm_table[i]) for i in sel))
    return torch.stack(tok).to(DEVICE), torch.tensor(lab, device=DEVICE)


def scan_env_up_to(tokens_1d: list[int], upto_inclusive: int) -> dict[int, int]:
    """Scan assignments of the form [VAR, VALUE] from start to ``upto_inclusive``.
    Returns a mapping var_token -> value (mod MOD). Latest assignment wins.
    """
    env = {}
    if len(tokens_1d) < 2:
        return env
    end = min(max(upto_inclusive, 0), len(tokens_1d) - 2)
    if end < 0:
        return env
    for i in range(0, end + 1):
        t0 = tokens_1d[i]
        if t0 not in VARS:
            continue
        val_idx = i + 1
        if val_idx > upto_inclusive or val_idx >= len(tokens_1d):
            continue
        t1 = tokens_1d[val_idx]
        if 0 <= t1 < MOD:
            env[t0] = int(t1)
    return env


def get_final_addition_info(
    tokens_1d: list[int],
) -> tuple[bool, int | None, int | None]:
    """Return (uses_var, var_pos, var_token) for the final addition expression.
    The final core is the last 4 tokens: [PLUS, lhs, rhs, EQUAL].
    If lhs or rhs is a VAR, returns True and its position & token; else (False, None, None).
    """
    if len(tokens_1d) < 4:
        return False, None, None
    core_start = len(tokens_1d) - 4
    plus_tok = tokens_1d[core_start]
    lhs_tok = tokens_1d[core_start + 1]
    rhs_tok = tokens_1d[core_start + 2]
    if plus_tok != PLUS:
        return False, None, None
    if lhs_tok in VARS:
        return True, core_start + 1, lhs_tok
    if rhs_tok in VARS:
        return True, core_start + 2, rhs_tok
    return False, None, None


def get_resid_feature(cache: ActivationCache, layer: int, kind: str) -> torch.Tensor:
    """Return residual features of shape [batch, seq, d_model] for the specified kind.
    kind in {"pre", "mid", "post"}. If "mid" is not cached, compute as pre + attn_out.
    """
    if kind == "pre":
        return cache["resid_pre", layer]
    if kind == "post":
        return cache["resid_post", layer]
    if kind == "mid":
        try:
            return cache["resid_mid", layer]
        except KeyError:
            return cache["resid_pre", layer] + cache["attn_out", layer]
    raise ValueError(f"Unknown resid kind: {kind}")


def train_linear_classifier(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    in_dim: int,
    out_dim: int,
    epochs: int = PROBE_EPOCHS,
    lr: float = PROBE_LR,
    wd: float = PROBE_WD,
) -> tuple[nn.Module, float]:
    """Train a single-layer linear classifier and return (model, val_acc)."""
    model_lin = nn.Linear(in_dim, out_dim, bias=True).to(DEVICE)
    opt = torch.optim.Adam(model_lin.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()
    model_lin.train()
    for _ in range(epochs):
        opt.zero_grad()
        logits = model_lin(x_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        opt.step()
    model_lin.eval()
    with torch.no_grad():
        pred = model_lin(x_val).argmax(dim=-1)
        acc = (pred == y_val).float().mean().item()
    return model_lin, acc


train_tokens, train_gold = sample_sequences(train_idx, PROBE_TRAIN_SEQS)
test_tokens, test_gold = sample_sequences(_test_idx, PROBE_TEST_SEQS)

_, train_cache = model.run_with_cache(train_tokens, return_type=None)
_, test_cache = model.run_with_cache(test_tokens, return_type=None)


def build_probe_a_dataset(tokens: torch.Tensor, cache: ActivationCache, layer: int):
    batch, seq = tokens.shape
    post = get_resid_feature(cache, layer, "post")
    feats = []
    labels = []
    rng_local = random.Random(SEED)
    for b in range(batch):
        toks_b = tokens[b].tolist()
        for t in range(seq):
            env = scan_env_up_to(toks_b, t)
            if not env:
                continue
            chosen_var_tok = rng_local.choice(list(env.keys()))
            var_idx = VAR_TO_IDX.get(chosen_var_tok, None)
            if var_idx is None:
                continue
            one_hot = torch.zeros(len(VARS), device=DEVICE)
            one_hot[var_idx] = 1.0
            feat_vec = torch.cat([post[b, t, :], one_hot], dim=-1)
            feats.append(feat_vec)
            labels.append(env[chosen_var_tok])
    if len(feats) == 0:
        return None, None
    x = torch.stack(feats)
    y = torch.tensor(labels, device=DEVICE, dtype=torch.long)
    return x, y


probe_a_accs = []
for lyr in PROBE_LAYERS:
    a_x_train, a_y_train = build_probe_a_dataset(train_tokens, train_cache, lyr)
    a_x_test, a_y_test = build_probe_a_dataset(test_tokens, test_cache, lyr)
    acc = float("nan")
    if (
        a_x_train is not None
        and a_x_test is not None
        and len(a_y_train) > 0
        and len(a_y_test) > 0
    ):
        _, acc = train_linear_classifier(
            a_x_train, a_y_train, a_x_test, a_y_test, D_MODEL + len(VARS), MOD
        )
    probe_a_accs.append(acc)
print(
    "Probe A accuracy per layer:",
    [round(x, 4) if x == x else None for x in probe_a_accs],
)


def build_probe_b_dataset(tokens: torch.Tensor, cache: ActivationCache):
    batch, seq = tokens.shape
    feats_by_comp = None
    comp_labels = None
    labels = []

    var_use_pos = [None] * batch
    label_for_seq = [None] * batch
    valid_indices = []
    for b in range(batch):
        toks_b = tokens[b].tolist()
        uses_var, var_pos, var_tok = get_final_addition_info(toks_b)
        if not uses_var or var_pos is None or var_tok is None:
            continue
        env_before = scan_env_up_to(toks_b, var_pos - 1)
        if var_tok not in env_before:
            continue
        var_use_pos[b] = var_pos
        label_for_seq[b] = env_before[var_tok]
        valid_indices.append(b)

    if len(valid_indices) == 0:
        return None, None, None

    for p in range(seq):
        per_comp0, comp_labels = (
            cache.decompose_resid(layer=-1, pos_slice=p, return_labels=True)
            if feats_by_comp is None
            else (
                cache.decompose_resid(layer=-1, pos_slice=p, return_labels=False),
                comp_labels,
            )
        )
        per_comp0_cpu = per_comp0.cpu()
        del per_comp0
        if feats_by_comp is None:
            num_components = per_comp0_cpu.shape[0]
            feats_by_comp = [[] for _ in range(num_components)]

        for b in valid_indices:
            vu = var_use_pos[b]
            lab = label_for_seq[b]
            if vu is None or lab is None or p != vu:
                continue
            for c in range(per_comp0_cpu.shape[0]):
                feats_by_comp[c].append(per_comp0_cpu[c, b, :])
            labels.append(lab)

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    if feats_by_comp is None or len(labels) == 0:
        return None, None, None
    x_by_comp = [
        torch.stack(feats) if len(feats) > 0 else None for feats in feats_by_comp
    ]
    y = torch.tensor(labels, dtype=torch.long)
    return x_by_comp, y, comp_labels


b_feats_train, b_y_train, b_comp_labels = build_probe_b_dataset(
    train_tokens, train_cache
)
b_feats_test, b_y_test, _ = build_probe_b_dataset(test_tokens, test_cache)
probe_b_accs_by_comp = []
if (
    b_feats_train is not None
    and b_feats_test is not None
    and b_y_train is not None
    and b_y_test is not None
):
    for idx, (x_tr, x_te) in enumerate(zip(b_feats_train, b_feats_test)):
        acc = float("nan")
        if (
            x_tr is not None
            and x_te is not None
            and len(b_y_train) > 0
            and len(b_y_test) > 0
        ):
            x_tr_d = x_tr.to(DEVICE)
            x_te_d = x_te.to(DEVICE)
            y_tr_d = b_y_train.to(DEVICE)
            y_te_d = b_y_test.to(DEVICE)
            _, acc = train_linear_classifier(
                x_tr_d, y_tr_d, x_te_d, y_te_d, D_MODEL, MOD
            )
            del x_tr_d, x_te_d, y_tr_d, y_te_d
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        probe_b_accs_by_comp.append(acc if acc == acc else None)
    if b_comp_labels is not None:
        print("Probe B acc per component:")
        for lab, acc in zip(b_comp_labels, probe_b_accs_by_comp):
            print(f"  {lab}: {None if acc is None else round(acc, 4)}")
    else:
        print(
            "Probe B acc per component:",
            [None if a is None else round(a, 4) for a in probe_b_accs_by_comp],
        )


def build_probe_d1_dataset(tokens: torch.Tensor, cache: ActivationCache, layer: int):
    batch, seq = tokens.shape
    post = get_resid_feature(cache, layer, "post")
    feats = []
    labels = []
    core_start = max(seq - 4, 0)
    for b in range(batch):
        toks_b = tokens[b].tolist()
        for i in range(0, core_start):
            if i + 1 >= len(toks_b):
                break
            if toks_b[i] not in VARS:
                continue
            value_tok = toks_b[i + 1]
            if not (0 <= value_tok < MOD):
                continue
            var_idx = VAR_TO_IDX.get(toks_b[i], None)
            if var_idx is None:
                continue
            feats.append(post[b, i + 1, :])
            labels.append(var_idx)
    if len(feats) == 0:
        return None, None
    x = torch.stack(feats)
    y = torch.tensor(labels, device=DEVICE, dtype=torch.long)
    return x, y


probe_d1_accs = []
for lyr in PROBE_LAYERS:
    d1_x_train, d1_y_train = build_probe_d1_dataset(train_tokens, train_cache, lyr)
    d1_x_test, d1_y_test = build_probe_d1_dataset(test_tokens, test_cache, lyr)
    acc = float("nan")
    if (
        d1_x_train is not None
        and d1_x_test is not None
        and len(d1_y_train) > 0
        and len(d1_y_test) > 0
    ):
        _, acc = train_linear_classifier(
            d1_x_train, d1_y_train, d1_x_test, d1_y_test, D_MODEL, len(VARS)
        )
    probe_d1_accs.append(acc)
print(
    "Probe D1 acc per layer:", [round(x, 4) if x == x else None for x in probe_d1_accs]
)


def build_probe_d2_dataset(tokens: torch.Tensor, cache: ActivationCache, layer: int):
    batch, seq = tokens.shape
    mid = get_resid_feature(cache, layer, "mid")
    feats = []
    labels = []
    for b in range(batch):
        toks_b = tokens[b].tolist()
        uses_var, var_pos, var_tok = get_final_addition_info(toks_b)
        if not uses_var or var_pos is None or var_tok is None:
            continue
        env_before = scan_env_up_to(toks_b, var_pos - 1)
        if var_tok not in env_before:
            continue
        feats.append(mid[b, var_pos, :])
        labels.append(env_before[var_tok])
    if len(feats) == 0:
        return None, None
    x = torch.stack(feats)
    y = torch.tensor(labels, device=DEVICE, dtype=torch.long)
    return x, y


probe_d2_accs = []
for lyr in PROBE_LAYERS:
    d2_x_train, d2_y_train = build_probe_d2_dataset(train_tokens, train_cache, lyr)
    d2_x_test, d2_y_test = build_probe_d2_dataset(test_tokens, test_cache, lyr)
    acc = float("nan")
    if (
        d2_x_train is not None
        and d2_x_test is not None
        and len(d2_y_train) > 0
        and len(d2_y_test) > 0
    ):
        _, acc = train_linear_classifier(
            d2_x_train, d2_y_train, d2_x_test, d2_y_test, D_MODEL, MOD
        )
    probe_d2_accs.append(acc)
print(
    "Probe D2 acc per layer:", [round(x, 4) if x == x else None for x in probe_d2_accs]
)


def build_probe_f_dataset(tokens: torch.Tensor, cache: ActivationCache, layer: int):
    batch, seq = tokens.shape
    post = get_resid_feature(cache, layer, "post")
    pos_m2 = seq - 2
    pos_m1 = seq - 1
    x_m2 = post[:, pos_m2, :]
    x_m1 = post[:, pos_m1, :]
    return x_m2, x_m1


f_xm2_train, f_xm1_train = build_probe_f_dataset(train_tokens, train_cache, PROBE_LAYER)
f_xm2_test, f_xm1_test = build_probe_f_dataset(test_tokens, test_cache, PROBE_LAYER)
f_acc_m2 = f_acc_m1 = float("nan")
_, f_acc_m2 = train_linear_classifier(
    f_xm2_train, train_gold, f_xm2_test, test_gold, D_MODEL, MOD
)
_, f_acc_m1 = train_linear_classifier(
    f_xm1_train, train_gold, f_xm1_test, test_gold, D_MODEL, MOD
)
print(f"Probe F accuracy — S-2: {f_acc_m2:.4f} | S-1: {f_acc_m1:.4f}")


torch.set_grad_enabled(False)  # probes trained — disable grads again


def _safe_del(name: str):
    globals().pop(name, None)


_safe_del("train_cache")
_safe_del("test_cache")
_safe_del("train_tokens")
_safe_del("test_tokens")

_safe_del("a_x_train")
_safe_del("a_y_train")
_safe_del("a_x_test")
_safe_del("a_y_test")

for _name in [
    "b_pre_train",
    "b_mid_train",
    "b_post_train",
    "b_y_train",
    "b_pre_test",
    "b_mid_test",
    "b_post_test",
    "b_y_test",
    "b_x_train",
    "b_x_test",
    "b_feats_train",
    "b_feats_test",
    "b_comp_labels",
    "probe_b_accs_by_comp",
]:
    _safe_del(_name)

_safe_del("d1_x_train")
_safe_del("d1_y_train")
_safe_del("d1_x_test")
_safe_del("d1_y_test")

_safe_del("d2_x_train")
_safe_del("d2_y_train")
_safe_del("d2_x_test")
_safe_del("d2_y_test")

_safe_del("f_xm2_train")
_safe_del("f_xm1_train")
_safe_del("f_xm2_test")
_safe_del("f_xm1_test")
import gc as _gc

_gc.collect()
if torch.backends.mps.is_available():
    torch.mps.empty_cache()

# %% 6. PCA of number embeddings -------------------------------------------
with torch.no_grad():
    number_emb = model.W_E[:MOD].detach().cpu().numpy()

coords = PCA(n_components=2, random_state=SEED).fit_transform(number_emb)
fig_pca = go.Figure(
    data=go.Scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        mode="markers+text",
        text=[str(i) for i in range(MOD)],
        textposition="middle center",
        marker=dict(size=8, color="blue"),
    )
)
fig_pca.update_layout(
    title="PCA of number token embeddings",
    xaxis_title="PC1",
    yaxis_title="PC2",
    yaxis=dict(scaleanchor="x", scaleratio=1),
)
fig_pca.show()

# %% 7. Example sequence predictions ---------------------------------------
print("\nExample predictions across evaluation subsets")
subset_indices = {
    "training": train_idx,
    "test": _test_idx,
}
for name, indices in subset_indices.items():
    if len(indices) == 0:
        continue
    rand_row_idx = int(indices[torch.randint(0, len(indices), (1,))])
    seq_tokens, target_label = _seq_builder(perm_table[rand_row_idx])
    seq_tokens = seq_tokens.to(DEVICE)
    with torch.no_grad():
        logits = model(seq_tokens.unsqueeze(0))
        pred_label = logits[0, -1, :MOD].argmax().item()
    print("\n" + "=" * 60)
    print(f"Subset: {name}")
    print("Sequence :", " ".join(TOKEN_STRINGS[t] for t in seq_tokens.tolist()))
    print(
        f"Target   : {target_label} | Prediction : {pred_label} | {'✓' if target_label == pred_label else '✗'}"
    )
# %% 8. Per-head attention heatmaps ----------------------------------------
rand_idx = int(_test_idx[torch.randint(0, len(_test_idx), (1,))])
seq_tokens, _ = _seq_builder(perm_table[rand_idx])
seq_tokens = seq_tokens.to(DEVICE)
_, cache_ex = model.run_with_cache(seq_tokens.unsqueeze(0), return_type=None)
N_LAYERS = model.cfg.n_layers
N_HEADS = model.cfg.n_heads
fig_heat = make_subplots(
    rows=N_LAYERS,
    cols=N_HEADS,
    shared_xaxes=True,
    shared_yaxes=True,
    horizontal_spacing=0.005,
    vertical_spacing=0.03,
    subplot_titles=[f"L{l}H{h}" for l in range(N_LAYERS) for h in range(N_HEADS)],
)
_token_labels = [TOKEN_STRINGS[t] for t in seq_tokens.tolist()]
_idx_vals = list(range(len(seq_tokens)))
for layer in range(N_LAYERS):
    att = cache_ex["attn", layer].squeeze(0).cpu().numpy()
    for head in range(N_HEADS):
        heat = go.Heatmap(
            z=att[head], x=_idx_vals, y=_idx_vals, colorscale="Viridis", showscale=False
        )
        fig_heat.add_trace(heat, row=layer + 1, col=head + 1)
fig_heat.update_xaxes(tickmode="array", tickvals=_idx_vals, ticktext=_token_labels)
fig_heat.update_yaxes(tickmode="array", tickvals=_idx_vals, ticktext=_token_labels)
fig_heat.update_layout(title="Per-head attention matrices (random test sequence)")
fig_heat.show()

# %% 9. Residual-stream L2-norm heatmap -------------------------------------

# %% 10. Custom string evaluation -------------------------------------------
print(f"SEQ_LEN: {SEQ_LEN}")
EXAMPLE_STRING = "; b 7 ; e 5 ; d 5 ; + b d ="

symbol_to_token = {str(i): i for i in range(MOD)}
symbol_to_token.update(
    {
        ";": PAD,
        "+": PLUS,
        "=": EQUAL,
    }
)
symbol_to_token.update({chr(ord("a") + i): A_TOKEN + i for i in range(26)})

raw_tokens = [symbol_to_token[sym] for sym in EXAMPLE_STRING.split()]

if len(raw_tokens) > SEQ_LEN:
    raw_tokens = raw_tokens[-SEQ_LEN:]
elif len(raw_tokens) < SEQ_LEN:
    raw_tokens = [PAD] * (SEQ_LEN - len(raw_tokens)) + raw_tokens

print(" ".join(TOKEN_STRINGS[t] for t in raw_tokens))

seq_tensor = torch.tensor(raw_tokens, device=DEVICE).unsqueeze(0)
with torch.no_grad():
    logits = model(seq_tensor)
    logits_vec = logits[0, -1, :MOD].detach().cpu()

pred_label = logits_vec.argmax().item()

print("\n" + "=" * 60)
print("Input sequence :", EXAMPLE_STRING)
print("Model prediction for the sum token:", pred_label)
print("Logits (descending):")
sorted_idxs = logits_vec.argsort(descending=True)
for idx in sorted_idxs:
    print(f"  {TOKEN_STRINGS[int(idx)]:>3}: {logits_vec[idx].item():.4f}")

_, cache_ex_str = model.run_with_cache(seq_tensor, return_type=None)
N_LAYERS = model.cfg.n_layers
N_HEADS = model.cfg.n_heads
fig_heat_ex = make_subplots(
    rows=N_LAYERS,
    cols=N_HEADS,
    shared_xaxes=True,
    shared_yaxes=True,
    horizontal_spacing=0.005,
    vertical_spacing=0.03,
    subplot_titles=[f"L{l}H{h}" for l in range(N_LAYERS) for h in range(N_HEADS)],
)
_token_labels_ex = [TOKEN_STRINGS[t] for t in raw_tokens]
_idx_vals_ex = list(range(len(raw_tokens)))
for layer in range(N_LAYERS):
    att = cache_ex_str["attn", layer].squeeze(0).cpu().numpy()
    for head in range(N_HEADS):
        heat = go.Heatmap(
            z=att[head],
            x=_idx_vals_ex,
            y=_idx_vals_ex,
            colorscale="Viridis",
            showscale=False,
        )
        fig_heat_ex.add_trace(heat, row=layer + 1, col=head + 1)
fig_heat_ex.update_xaxes(
    tickmode="array", tickvals=_idx_vals_ex, ticktext=_token_labels_ex
)
fig_heat_ex.update_yaxes(
    tickmode="array", tickvals=_idx_vals_ex, ticktext=_token_labels_ex
)
fig_heat_ex.update_layout(title="Per-head attention matrices (example sequence)")
fig_heat_ex.show()

# %% 11. Difference of per-head attention maps between two example strings --
EXAMPLE_STRING_A = "; b 7 ; ; d 5 ; ; ; + 12 18 ="
EXAMPLE_STRING_B = "; b 7 ; ; d 5 ; ; ; + d 18 ="



def _tokenize_custom_string(s: str) -> torch.Tensor:
    """Tokenise the arithmetic string `s` into a 1xseq tensor.
    Commas are treated as semicolon separators.
    """
    s = s.replace(",", " ;")

    tokens = []
    for sym in s.split():
        tokens.append(symbol_to_token[sym])

    if len(tokens) > SEQ_LEN:
        tokens = tokens[-SEQ_LEN:]
    elif len(tokens) < SEQ_LEN:
        tokens = [PAD] * (SEQ_LEN - len(tokens)) + tokens

    return torch.tensor(tokens, device=DEVICE).unsqueeze(0)


seq_A = _tokenize_custom_string(EXAMPLE_STRING_A)
seq_B = _tokenize_custom_string(EXAMPLE_STRING_B)

with torch.no_grad():
    _, cache_A = model.run_with_cache(seq_A, return_type=None)
    _, cache_B = model.run_with_cache(seq_B, return_type=None)

N_LAYERS = model.cfg.n_layers
N_HEADS = model.cfg.n_heads

diff_fig = make_subplots(
    rows=N_LAYERS,
    cols=N_HEADS,
    shared_xaxes=True,
    shared_yaxes=True,
    horizontal_spacing=0.005,
    vertical_spacing=0.03,
    subplot_titles=[f"L{l}H{h}" for l in range(N_LAYERS) for h in range(N_HEADS)],
)

tokens_A = seq_A.squeeze(0).tolist()
tokens_B = seq_B.squeeze(0).tolist()
token_labels = [
    TOKEN_STRINGS[a] if a == b else f"{TOKEN_STRINGS[a]}/{TOKEN_STRINGS[b]}"
    for a, b in zip(tokens_A, tokens_B)
]
idx_vals = list(range(len(token_labels)))


for layer in range(N_LAYERS):
    diff = (cache_B["attn", layer] - cache_A["attn", layer]).squeeze(0).cpu().numpy()
    for head in range(N_HEADS):
        heat = go.Heatmap(
            z=diff[head],
            x=idx_vals,
            y=idx_vals,
            colorscale="RdBu",
            zmid=0,
            zmin=-1.0,
            zmax=1.0,
            showscale=False,
        )
        diff_fig.add_trace(heat, row=layer + 1, col=head + 1)

diff_fig.update_xaxes(tickmode="array", tickvals=idx_vals, ticktext=token_labels)

diff_fig.update_yaxes(tickmode="array", tickvals=idx_vals, ticktext=token_labels)

diff_fig.update_layout(title="Per-head attention difference (d addition - b addition)")

diff_fig.show()

# %%
import transformer_lens.patching as patching

clean_prompts = [
    "; a 7 ; b 2 ; c 51 ; + b 18 =",
    "; d 1 ; e 2 ; f 3 ; + e 19 =",
]
corrupted_prompts = [
    "; a 7 ; b 3 ; c 51 ; + b 18 =",
    "; d 1 ; e 3 ; f 3 ; + e 19 =",
]
clean_answers = [20, 21]
corrupted_answers = [21, 22]


def logits_to_logit_diff(
    logits: torch.Tensor, correct_answers=clean_answers, wrong_answers=corrupted_answers
):
    """
    Compute the average logit difference (correct - wrong) across a batch.
    `logits` has shape [batch, seq_len, vocab_size].
    """
    batch_size = logits.shape[0]
    device = logits.device
    batch_indices = torch.arange(batch_size, device=device)
    correct_tensor = torch.tensor(correct_answers, device=device)
    wrong_tensor = torch.tensor(wrong_answers, device=device)
    correct_logits = logits[batch_indices, -1, correct_tensor]
    wrong_logits = logits[batch_indices, -1, wrong_tensor]
    return (correct_logits - wrong_logits).mean()


clean_tokens = torch.cat([_tokenize_custom_string(p) for p in clean_prompts], dim=0)
corrupted_tokens = torch.cat(
    [_tokenize_custom_string(p) for p in corrupted_prompts], dim=0
)

clean_logits, clean_cache = model.run_with_cache(clean_tokens)
clean_logit_diff = logits_to_logit_diff(clean_logits)
print(f"Clean logit difference (avg over batch): {clean_logit_diff.item():.3f}")

corrupted_logits = model(corrupted_tokens)
corrupted_logit_diff = logits_to_logit_diff(corrupted_logits)
print(f"Corrupted logit difference (avg over batch): {corrupted_logit_diff.item():.3f}")


# %%
def residual_stream_patching_hook(
    resid_pre: Float[torch.Tensor, "batch pos d_model"], hook: HookPoint, position: int
) -> Float[torch.Tensor, "batch pos d_model"]:
    clean_resid_pre = clean_cache[hook.name]
    resid_pre[:, position, :] = clean_resid_pre[:, position, :]
    return resid_pre


num_positions = len(clean_tokens[0])
ioi_patching_result = torch.zeros(
    (model.cfg.n_layers, num_positions), device=model.cfg.device
)

for layer in tqdm.tqdm(range(model.cfg.n_layers)):
    for position in range(num_positions):
        temp_hook_fn = partial(residual_stream_patching_hook, position=position)
        patched_logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(utils.get_act_name("resid_pre", layer), temp_hook_fn)],
        )
        patched_logit_diff = logits_to_logit_diff(patched_logits).detach()
        ioi_patching_result[layer, position] = (
            patched_logit_diff - corrupted_logit_diff
        ) / (clean_logit_diff - corrupted_logit_diff)


# %%
print(clean_tokens[0].tolist())

# %%
if isinstance(clean_tokens, torch.Tensor):
    clean_tokens_list = clean_tokens[0].tolist()

token_labels = [
    f"{TOKEN_STRINGS[token]}_{idx}" for idx, token in enumerate(clean_tokens_list)
]
layer_labels = [f"L{layer}" for layer in range(model.cfg.n_layers)]
imshow(
    ioi_patching_result,
    x=token_labels,
    y=layer_labels,
    xaxis="Position",
    yaxis="Layer",
    title="Normalized Logit Difference After Patching Residual Stream on the IOI Task",
)

# %%
logit_diff_directions = (
    model.tokens_to_residual_directions(torch.tensor(clean_answers))
    - model.tokens_to_residual_directions(torch.tensor(corrupted_answers))
).mean(0)
print(logit_diff_directions.shape)

# %%
import numpy as np


def residual_stack_to_logit_diff(
    residual_stack: Float[torch.Tensor, "components batch d_model"],
    cache: ActivationCache,
) -> torch.Tensor:
    """
    Convert a stack of residual activations into the contribution to the
    logit difference between the clean and corrupted answers.
    The returned tensor has shape (components,) where *components* indexes
    every residual stream activation (layer.0_resid_pre, layer.0_attn_out, etc.).
    """
    scaled_residual_stack = cache.apply_ln_to_stack(
        residual_stack, layer=-1, pos_slice=-1
    )
    batched_logit_diff = logit_diff_directions.expand(
        scaled_residual_stack.shape[1], -1
    )
    return (
        einsum(
            "... batch d_model, batch d_model -> ...",
            scaled_residual_stack,
            batched_logit_diff,
        )
        / scaled_residual_stack.shape[1]
    )  # average across the batch (clean & corrupted)


# %%
batched_tokens = torch.cat([clean_tokens, corrupted_tokens], dim=0).to(DEVICE)
_, batched_cache = model.run_with_cache(batched_tokens, return_type=None)

accumulated_residual, resid_labels = batched_cache.accumulated_resid(
    layer=-1, incl_mid=True, pos_slice=-1, return_labels=True
)

logit_lens_logit_diffs = residual_stack_to_logit_diff(
    accumulated_residual, batched_cache
)

px.line(
    x=np.arange(model.cfg.n_layers * 2 + 1) / 2,
    y=logit_lens_logit_diffs.tolist(),
    hover_name=resid_labels,
    title="Logit Difference From Accumulated Residual Stream (clean + corrupted average)",
).show()

# %%
print(logit_lens_logit_diffs)

# %%
per_layer_residual, per_layer_labels = batched_cache.decompose_resid(
    layer=-1, pos_slice=-1, return_labels=True
)
per_layer_logit_diffs = residual_stack_to_logit_diff(per_layer_residual, batched_cache)

line(
    per_layer_logit_diffs,
    hover_name=per_layer_labels,
    title="Logit Difference From Each Component of the Final Layer (clean + corrupted average)",
)

# %%
per_head_residual, per_head_labels = batched_cache.stack_head_results(
    layer=-1, pos_slice=-1, return_labels=True
)
per_head_logit_diffs = residual_stack_to_logit_diff(per_head_residual, batched_cache)
per_head_logit_diffs = einops.rearrange(
    per_head_logit_diffs,
    "(layer head_index) -> layer head_index",
    layer=model.cfg.n_layers,
    head_index=model.cfg.n_heads,
)
imshow(
    per_head_logit_diffs,
    xaxis="Head",
    yaxis="Layer",
    title="Logit Difference From Each Head (clean + corrupted average)",
)

# %% Trying to replicate Neel Nanda's modular addition grokking
try:
    from neel_plotly.plot import line
except ImportError:
    def line(*args, **kwargs):
        return None


def _build_core_only_sequence(lhs: int, rhs: int) -> torch.Tensor:
    seq = [PAD] * (SEQ_LEN - 4) + [PLUS, lhs, rhs, EQUAL]
    return torch.tensor(seq, device=DEVICE, dtype=torch.long)


all_pairs = [(a, b) for a in range(MOD) for b in range(MOD)]
dataset_tokens = torch.stack(
    [_build_core_only_sequence(a, b) for a, b in all_pairs], dim=0
)
dataset_labels = torch.tensor([(a + b) % MOD for a, b in all_pairs], device=DEVICE)

original_logits, cache = model.run_with_cache(dataset_tokens)
print(original_logits.shape)


# %%
L = model.cfg.n_layers - 1

W_E_numbers = model.embed.W_E[:MOD]
print("W_E_numbers shape:", W_E_numbers.shape)
W_neur = (
    W_E_numbers
    @ model.blocks[L].attn.W_V
    @ model.blocks[L].attn.W_O
    @ model.blocks[L].mlp.W_in
)
print("W_neur", W_neur.shape)
W_logit = model.blocks[L].mlp.W_out @ model.unembed.W_U
print("W_logit", W_logit.shape)
# %%
dest_pos = -1
lhs_pos = -3
rhs_pos = -2
pattern_lhs = cache["pattern", L, "attn"][:, :, dest_pos, lhs_pos]
pattern_rhs = cache["pattern", L, "attn"][:, :, dest_pos, rhs_pos]
pattern_lhs_mean = pattern_lhs.mean(dim=0)
pattern_rhs_mean = pattern_rhs.mean(dim=0)

neuron_acts = cache["post", L, "mlp"][:, dest_pos, :]
neuron_pre_acts = cache["pre", L, "mlp"][:, dest_pos, :]
neuron_acts_mean = neuron_acts.mean(dim=0)
neuron_pre_acts_mean = neuron_pre_acts.mean(dim=0)

# %%
for param_name, param in cache.items():
    print(param_name, param.shape)

# %%
imshow(
    cache["pattern", L].mean(dim=0)[:, dest_pos, :],
    title="Average Attention Pattern per Head (Last Layer)",
    xaxis="Source",
    yaxis="Head",
)

# %%
imshow(
    cache["pattern", L][5][:, dest_pos, :],
    title="Average Attention Pattern per Head (Last Layer)",
    xaxis="Source",
    yaxis="Head",
)

# %%
dataset_tokens[5].tolist()
# %%
imshow(
    cache["pattern", L][:, 0, dest_pos, lhs_pos].reshape(MOD, MOD),
    title="Attention for Head 0 from a -> =",
    xaxis="b",
    yaxis="a",
)
# %%
imshow(
    cache["pattern", L][:, 0, dest_pos, rhs_pos].reshape(MOD, MOD),
    title="Attention for Head 0 from b -> =",
    xaxis="b",
    yaxis="a",
)
# %%
imshow(
    einops.rearrange(
        cache["pattern", L][:, :, dest_pos, lhs_pos],
        "(a b) head -> head a b",
        a=MOD,
        b=MOD,
    ),
    title="Attention for Head 0 from lhs -> =",
    xaxis="rhs",
    yaxis="lhs",
    facet_col=0,
)
# %% Plotting neuron activations
print(cache["post", L, "mlp"].shape)
print(neuron_acts.shape)


# %%
imshow(
    einops.rearrange(neuron_acts[:, -5:], "(a b) neuron -> neuron a b", a=MOD, b=MOD),
    title="First 5 neuron acts",
    xaxis="b",
    yaxis="a",
    facet_col=0,
)

# %% Singular Value Decomposition
print(W_E_numbers.shape)

# %%
U, S, Vh = torch.svd(W_E_numbers)
line(S, title="Singular Values")
imshow(U, title="Principal Components on the Input")

# %%
U, S, Vh = torch.svd(torch.randn_like(W_E_numbers))
line(S, title="Singular Values Random")
imshow(U, title="Principal Components Random")

# %%
U, S, Vh = torch.svd(W_E_numbers)
print(U[:, :8].T.shape)
line(
    U[:, :8].T, title="Principal Components of the embedding", xaxis="Input Vocabulary"
)

# %%
fourier_basis = []
fourier_basis_names = []
fourier_basis.append(torch.ones(MOD))
fourier_basis_names.append("Constant")
for freq in range(1, MOD // 2 + 1):
    fourier_basis.append(torch.sin(torch.arange(MOD) * 2 * torch.pi * freq / MOD))
    fourier_basis_names.append(f"Sin {freq}")
    fourier_basis.append(torch.cos(torch.arange(MOD) * 2 * torch.pi * freq / MOD))
    fourier_basis_names.append(f"Cos {freq}")
fourier_basis = torch.stack(fourier_basis, dim=0).to(DEVICE)
fourier_basis = fourier_basis / fourier_basis.norm(dim=-1, keepdim=True)
imshow(fourier_basis, xaxis="Input", yaxis="Component", y=fourier_basis_names)

# %%
line(
    fourier_basis[:8],
    xaxis="Input",
    line_labels=fourier_basis_names[:8],
    title="First 8 Fourier Components",
)
line(
    fourier_basis[25:29],
    xaxis="Input",
    line_labels=fourier_basis_names[25:29],
    title="Middle Fourier Components",
)
# %%
imshow(
    fourier_basis @ W_E_numbers,
    yaxis="Fourier Component",
    xaxis="Residual Stream",
    y=fourier_basis_names,
    title="Embedding in Fourier Basis",
)

# %%
line(
    (fourier_basis @ W_E_numbers).norm(dim=-1),
    xaxis="Fourier Component",
    x=fourier_basis_names,
    title="Norms of Embedding in Fourier Basis",
)
# %%
print(W_E_numbers.shape)
print(fourier_basis.shape)
# %%
key_freqs = [0, 9, 17, 18, 19, 21, 25]
key_freq_indices = [0, 17, 18, 33, 34, 35, 36, 37, 38, 41, 42, 49, 50]
fourier_embed = fourier_basis @ W_E_numbers
key_fourier_embed = fourier_embed[key_freq_indices]
print(key_fourier_embed.shape)
imshow(
    key_fourier_embed @ key_fourier_embed.T,
    title="Dot Product of embedding of key Fourier Terms",
)
# %%
line(
    fourier_basis[[18, 34, 36, 38, 42, 50]],
    title="Cos of key freqs",
    line_labels=[18, 34, 36, 38, 42, 50],
)
# %%
line(fourier_basis[[18, 34, 36, 38, 42, 50]].mean(0), title="Constructive Interference")
# %% Analyse Neurons
imshow(
    einops.rearrange(neuron_acts[:, :5], "(a b) neuron -> neuron a b", a=MOD, b=MOD),
    title="First 5 neuron acts",
    xaxis="b",
    yaxis="a",
    facet_col=0,
)
# %%
imshow(
    einops.rearrange(neuron_acts[:, 5], "(a b) -> a b", a=MOD, b=MOD),
    title="5th neuron act",
    xaxis="b",
    yaxis="a",
)
# %%
imshow(
    fourier_basis[18][None, :] * fourier_basis[18][:, None], title="Cos 18a * cos 18b"
)
# %%
imshow(fourier_basis[18][None, :] * fourier_basis[0][:, None], title="Cos 18a * const")
# %%
imshow(
    fourier_basis @ neuron_acts[:, 5].reshape(MOD, MOD) @ fourier_basis.T,
    title="2D Fourier Transformer of neuron 5",
    xaxis="b",
    yaxis="a",
    x=fourier_basis_names,
    y=fourier_basis_names,
)
# %%
imshow(
    fourier_basis
    @ torch.randn_like(neuron_acts[:, 0]).reshape(MOD, MOD)
    @ fourier_basis.T,
    title="2D Fourier Transformer of RANDOM",
    xaxis="b",
    yaxis="a",
    x=fourier_basis_names,
    y=fourier_basis_names,
)
# %%
fourier_neuron_acts = (
    fourier_basis
    @ einops.rearrange(neuron_acts, "(a b) neuron -> neuron a b", a=MOD, b=MOD)
    @ fourier_basis.T
)
fourier_neuron_acts[:, 0, 0] = 0.0
print("fourier_neuron_acts", fourier_neuron_acts.shape)
# %%
neuron_freq_norm = torch.zeros(MOD // 2, model.cfg.d_mlp).to(DEVICE)
for freq in range(0, MOD // 2):
    for x in [0, 2 * (freq + 1) - 1, 2 * (freq + 1)]:
        for y in [0, 2 * (freq + 1) - 1, 2 * (freq + 1)]:
            neuron_freq_norm[freq] += fourier_neuron_acts[:, x, y] ** 2
neuron_freq_norm = (
    neuron_freq_norm / fourier_neuron_acts.pow(2).sum(dim=[-1, -2])[None, :]
)
TOPK_NEURONS = 200
max_over_freq = neuron_freq_norm.max(dim=0).values
_, top_idx = torch.topk(max_over_freq, k=min(TOPK_NEURONS, max_over_freq.numel()))
neuron_freq_norm_top = neuron_freq_norm[:, top_idx]
top_idx_list = top_idx.detach().cpu().tolist()
tick_every = max(1, len(top_idx_list) // 20)  # ~20 tick labels
tick_pos = list(range(0, len(top_idx_list), tick_every))
tick_text = [str(top_idx_list[i]) for i in tick_pos]

fig = px.imshow(
    utils.to_numpy(neuron_freq_norm_top),
    color_continuous_scale="Viridis",  # values are >= 0, so use sequential scale
    aspect="auto",
    labels={"x": "Neuron (top-K index)", "y": "Freq"},
    title=f"Neuron Frac Explained by Freq (top {len(top_idx_list)})",
)
fig.update_layout(width=1100, height=500, margin=dict(l=60, r=20, t=60, b=60))
fig.update_xaxes(tickmode="array", tickvals=tick_pos, ticktext=tick_text, tickangle=0)
fig.update_yaxes(
    tickmode="array",
    tickvals=list(range(0, MOD // 2)),
    ticktext=list(range(1, MOD // 2 + 1)),
)
fig.show()
# %%
line(
    neuron_freq_norm.max(dim=0).values.sort().values,
    xaxis="Neuron",
    title="Max Neuron Frac Explained over Freqs",
)
# %%
W_logit = model.blocks[L].mlp.W_out @ model.unembed.W_U
print("W_logit", W_logit.shape)
# %%
line(
    (W_logit @ fourier_basis.T).norm(dim=0),
    x=fourier_basis_names,
    title="W_logit in the Fourier Basis",
)
# %%
neurons_9 = neuron_freq_norm[9 - 1] > 0.85
print(neurons_9.shape)
neurons_9.sum()
line(
    (W_logit[neurons_9] @ fourier_basis.T).norm(dim=0),
    x=fourier_basis_names,
    title="W_logit for freq 9 neurons in the Fourier Basis",
)
# %%
freq = 19
W_logit_fourier = W_logit @ fourier_basis
neurons_sin_19 = W_logit_fourier[:, 2 * freq - 1]
line(neurons_sin_19)
# %%
inputs_sin_19c = neuron_acts @ neurons_sin_19
imshow(
    fourier_basis @ inputs_sin_19c.reshape(MOD, MOD) @ fourier_basis.T,
    title="Fourier Heatmap over inputs for sin19c",
    x=fourier_basis_names,
    y=fourier_basis_names,
)
# %% Print every part of the model by name
for name, param in model.named_parameters():
    print(name, param.shape)
# %% Analyzing pre-MLP embeddings


from itertools import product

a_tok, b_tok, c_tok = VARS[:3]
pad = [PAD] * (SEQ_LEN - 10)  # 6 assignment tokens + 4 core tokens

tokens_num_num = torch.tensor(
    [
        [a_tok, 7 % MOD, b_tok, 2 % MOD, c_tok, 51 % MOD, *pad, PLUS, n1, n2, EQUAL]
        for n1, n2 in product(range(MOD), repeat=2)
    ],
    dtype=torch.long,
    device=DEVICE,
)

tokens_num_var = torch.tensor(
    [
        [a_tok, 7 % MOD, b_tok, n1, c_tok, 51 % MOD, *pad, PLUS, b_tok, n2, EQUAL]
        for n1, n2 in product(range(MOD), repeat=2)
    ],
    dtype=torch.long,
    device=DEVICE,
)

print(tokens_num_num.shape, tokens_num_var.shape)

# %%


import torch.nn.functional as F

L = model.cfg.n_layers - 1
pos = -1

assert tokens_num_num.shape == tokens_num_var.shape


def _final_layer_mlp_in(tok_batch: torch.Tensor, batch_size: int = 512) -> torch.Tensor:
    """Return LN2(resid_mid) at final layer and pos=-1. Shape: [N, d_model]."""
    outs = []
    with torch.no_grad():
        for i in range(0, tok_batch.shape[0], batch_size):
            toks = tok_batch[i : i + batch_size]
            _, cache = model.run_with_cache(toks, return_type=None)
            resid_mid = cache["resid_mid", L][:, pos, :]
            mlp_in = model.blocks[L].ln2(resid_mid)
            outs.append(mlp_in.detach().cpu())
    return torch.cat(outs, dim=0)


mlp_in_num_num = _final_layer_mlp_in(tokens_num_num)
mlp_in_num_var = _final_layer_mlp_in(tokens_num_var)

cos = F.cosine_similarity(mlp_in_num_num, mlp_in_num_var, dim=-1)
print(f"cos(mlp_in) mean={cos.mean().item():.4f} std={cos.std().item():.4f}")

fig = go.Figure(go.Histogram(x=cos.numpy(), nbinsx=80))
fig.update_layout(
    title="Final-layer MLP input similarity: num+num vs num+var (fixed template)",
    xaxis_title="cosine similarity",
    yaxis_title="count",
)
fig.show()

# %%


perm = torch.randperm(mlp_in_num_var.shape[0])
cos_shuffled = F.cosine_similarity(mlp_in_num_num, mlp_in_num_var[perm], dim=-1)
print(
    f"cos(mlp_in) mismatched mean={cos_shuffled.mean().item():.4f} std={cos_shuffled.std().item():.4f}"
)

_mlp_cos_matched = cos.numpy().tolist()
_mlp_cos_shuffled = cos_shuffled.numpy().tolist()

# %%

NUM_SAMPLES = 10000
LHS_POS = SEQ_LEN - 3
RHS_POS = SEQ_LEN - 2
TARGET_LAYER = 1  # Layer to extract embeddings from

literal_mask = perm_table[:, 1] == 0
literal_indices = torch.where(literal_mask)[0]
print(f"Total literal-operand sequences available: {len(literal_indices)}")

sampled_idx = literal_indices[torch.randperm(len(literal_indices))[:NUM_SAMPLES]]
tokens_list = []
lhs_values = []
rhs_values = []

for i in sampled_idx:
    row = perm_table[i]
    tok, _ = _seq_builder(row)
    tokens_list.append(tok)
    lhs_values.append(row[2].item())  # num1
    rhs_values.append(row[3].item())  # num2

tokens = torch.stack(tokens_list)  # Keep on CPU initially
lhs_values = torch.tensor(lhs_values)
rhs_values = torch.tensor(rhs_values)

print(f"Generated {len(tokens)} sequences")
print(f"Sequence shape: {tokens.shape}")

EMBED_BATCH_SIZE = 512
resid_lhs_list = []
resid_rhs_list = []

print(f"Processing in batches of {EMBED_BATCH_SIZE}...")
with torch.no_grad():
    for start in range(0, len(tokens), EMBED_BATCH_SIZE):
        end = min(start + EMBED_BATCH_SIZE, len(tokens))
        batch_tokens = tokens[start:end].to(DEVICE)
        _, batch_cache = model.run_with_cache(batch_tokens, return_type=None)

        resid = batch_cache["resid_post", TARGET_LAYER]
        resid_lhs_list.append(resid[:, LHS_POS, :].cpu())
        resid_rhs_list.append(resid[:, RHS_POS, :].cpu())

        del batch_cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

resid_lhs = torch.cat(resid_lhs_list, dim=0).to(DEVICE)
resid_rhs = torch.cat(resid_rhs_list, dim=0).to(DEVICE)
del resid_lhs_list, resid_rhs_list

d_model = model.cfg.d_model  # Model dimension

embeddings_left = torch.zeros(MOD, d_model, device=DEVICE)
counts_left = torch.zeros(MOD, device=DEVICE)

for val in range(MOD):
    mask = lhs_values == val
    if mask.sum() > 0:
        embeddings_left[val] = resid_lhs[mask].mean(dim=0)
        counts_left[val] = mask.sum()

embeddings_right = torch.zeros(MOD, d_model, device=DEVICE)
counts_right = torch.zeros(MOD, device=DEVICE)

for val in range(MOD):
    mask = rhs_values == val
    if mask.sum() > 0:
        embeddings_right[val] = resid_rhs[mask].mean(dim=0)
        counts_right[val] = mask.sum()

print(f"\nEmbeddings computed for layer {TARGET_LAYER}:")
print(f"  Left operand embeddings shape: {embeddings_left.shape}")
print(f"  Right operand embeddings shape: {embeddings_right.shape}")
print(
    f"  Min/Max counts (left): {counts_left.min().item():.0f} / {counts_left.max().item():.0f}"
)
print(
    f"  Min/Max counts (right): {counts_right.min().item():.0f} / {counts_right.max().item():.0f}"
)

cos_sim_left = torch.nn.functional.cosine_similarity(
    embeddings_left.unsqueeze(1), embeddings_left.unsqueeze(0), dim=-1
)
cos_sim_right = torch.nn.functional.cosine_similarity(
    embeddings_right.unsqueeze(1), embeddings_right.unsqueeze(0), dim=-1
)
print(f"\nSanity check - average off-diagonal cosine similarity:")
print(
    f"  Left embeddings: {(cos_sim_left.sum() - cos_sim_left.trace()) / (MOD * (MOD - 1)):.4f}"
)
print(
    f"  Right embeddings: {(cos_sim_right.sum() - cos_sim_right.trace()) / (MOD * (MOD - 1)):.4f}"
)

# %%

ANALYSIS_LAYER = model.cfg.n_layers - 1
DEST_POS = SEQ_LEN - 1  # EQUAL position (destination for attention)

print(
    f"Model has {model.cfg.n_layers} layers, using layer {ANALYSIS_LAYER} for OV analysis"
)

W_V_3 = model.blocks[ANALYSIS_LAYER].attn.W_V
W_O_3 = model.blocks[ANALYSIS_LAYER].attn.W_O

OV_3 = einsum(
    "heads d_model_in d_head, heads d_head d_model_out -> d_model_in d_model_out",
    W_V_3,
    W_O_3,
)
print(f"OV matrix shape (layer {ANALYSIS_LAYER}): {OV_3.shape}")


def apply_ov(x: torch.Tensor) -> torch.Tensor:
    """Apply the summed OV circuit: x @ OV_3."""
    return x @ OV_3


ov_left = apply_ov(embeddings_left)
ov_right = apply_ov(embeddings_right)

print(f"\nGenerating one sequence per (num1, num2) pair...")

pair_to_idx = {}
for idx in range(len(perm_table)):
    row = perm_table[idx]
    typ = row[1].item()
    if typ == 0:  # literal operands
        num1, num2 = row[2].item(), row[3].item()
        if (num1, num2) not in pair_to_idx:
            pair_to_idx[(num1, num2)] = idx

pair_tokens_list = []
pair_labels = []  # (num1, num2) for each sequence
for num1 in range(MOD):
    for num2 in range(MOD):
        if (num1, num2) in pair_to_idx:
            idx = pair_to_idx[(num1, num2)]
            tok, _ = _seq_builder(perm_table[idx])
            pair_tokens_list.append(tok)
            pair_labels.append((num1, num2))

pair_tokens = torch.stack(pair_tokens_list)  # Keep on CPU initially
print(f"Generated {len(pair_tokens)} sequences for {MOD}^2 = {MOD**2} pairs")

PAIR_BATCH_SIZE = 512
resid_mid_3_list = []
attn_out_3_list = []
attn_pattern_lhs_list = []  # Attention weights to left operand
attn_pattern_rhs_list = []  # Attention weights to right operand

print(f"Processing in batches of {PAIR_BATCH_SIZE}...")
with torch.no_grad():
    for start in range(0, len(pair_tokens), PAIR_BATCH_SIZE):
        end = min(start + PAIR_BATCH_SIZE, len(pair_tokens))
        batch_tokens = pair_tokens[start:end].to(DEVICE)
        _, batch_cache = model.run_with_cache(batch_tokens, return_type=None)

        resid_pre_3 = batch_cache["resid_pre", ANALYSIS_LAYER][:, DEST_POS, :]
        attn_out_3 = batch_cache["attn_out", ANALYSIS_LAYER][:, DEST_POS, :]
        resid_mid_3_list.append((resid_pre_3 + attn_out_3).cpu())
        attn_out_3_list.append(attn_out_3.cpu())

        pattern = batch_cache["pattern", ANALYSIS_LAYER]
        attn_pattern_lhs_list.append(
            pattern[:, :, DEST_POS, LHS_POS].mean(dim=-1).cpu()
        )
        attn_pattern_rhs_list.append(
            pattern[:, :, DEST_POS, RHS_POS].mean(dim=-1).cpu()
        )

        del batch_cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

resid_mid_3 = torch.cat(resid_mid_3_list, dim=0).to(DEVICE)
attn_out_3 = torch.cat(attn_out_3_list, dim=0).to(DEVICE)
attn_weights_lhs = torch.cat(attn_pattern_lhs_list, dim=0).to(DEVICE)
attn_weights_rhs = torch.cat(attn_pattern_rhs_list, dim=0).to(DEVICE)
del resid_mid_3_list, attn_out_3_list, attn_pattern_lhs_list, attn_pattern_rhs_list

print(f"\nAttention weights from EQUAL to operands (averaged over heads):")
print(f"  To LHS: mean={attn_weights_lhs.mean():.4f}, std={attn_weights_lhs.std():.4f}")
print(f"  To RHS: mean={attn_weights_rhs.mean():.4f}, std={attn_weights_rhs.std():.4f}")
print(f"  Sum (LHS+RHS): mean={(attn_weights_lhs + attn_weights_rhs).mean():.4f}")

predicted_unweighted = torch.zeros_like(resid_mid_3)

for i, (num1, num2) in enumerate(pair_labels):
    predicted_unweighted[i] = ov_left[num1] + ov_right[num2]

print("\n" + "=" * 60)
print("Comparison: predicted vs ACTUAL ATTENTION OUTPUT (attn_out)")
print("=" * 60)

cos_unweighted_attn = torch.nn.functional.cosine_similarity(
    predicted_unweighted, attn_out_3, dim=-1
)
print(f"\nUnweighted OV(a) + OV(b) vs attn_out:")
print(
    f"  Cosine sim: mean={cos_unweighted_attn.mean():.4f}, std={cos_unweighted_attn.std():.4f}"
)
norm_ratio_unweighted = attn_out_3.norm(dim=-1) / predicted_unweighted.norm(dim=-1)
print(f"  Norm ratio (actual/pred): mean={norm_ratio_unweighted.mean():.4f}")


print("\n" + "=" * 60)
print("Comparison: predicted vs RESID_MID (includes resid_pre)")
print("=" * 60)

cos_unweighted_mid = torch.nn.functional.cosine_similarity(
    predicted_unweighted, resid_mid_3, dim=-1
)
print(f"\nUnweighted OV(a) + OV(b) vs resid_mid:")
print(f"  Cosine sim: mean={cos_unweighted_mid.mean():.4f}")


shuffled_indices = torch.randperm(len(pair_labels))
predicted_shuffled = torch.zeros_like(resid_mid_3)
for i, (num1, num2) in enumerate(pair_labels):
    _, num2_shuffled = pair_labels[shuffled_indices[i].item()]
    predicted_shuffled[i] = ov_left[num1] + ov_right[num2_shuffled]

cos_shuffled_attn = torch.nn.functional.cosine_similarity(
    predicted_shuffled, attn_out_3, dim=-1
)
print(f"\nControl (shuffled) vs attn_out:")
print(f"  Cosine sim: mean={cos_shuffled_attn.mean():.4f}")

fig = go.Figure()
fig.add_trace(
    go.Histogram(
        x=cos_unweighted_attn.cpu().numpy(),
        nbinsx=50,
        name="Matched",
        marker_color="steelblue",
        opacity=0.6,
    )
)
fig.add_trace(
    go.Histogram(
        x=cos_shuffled_attn.cpu().numpy(),
        nbinsx=50,
        name="Shuffled",
        marker_color="coral",
        opacity=0.6,
    )
)
fig.update_layout(
    barmode="overlay",
    title=f"Cosine Similarity: OV(a)+OV(b) vs Layer {ANALYSIS_LAYER} Attention Output (Layer-1 Emb)",
    xaxis_title="Cosine Similarity",
    yaxis_title="Count",
    showlegend=True,
)
fig.show()

# %%

print("\n" + "=" * 70)
print("OV ANALYSIS WITH RAW TOKEN EMBEDDINGS")
print("=" * 70)

token_embeddings = model.embed.W_E[:MOD].detach()
print(f"Token embeddings shape: {token_embeddings.shape}")



def apply_ov_token(x: torch.Tensor) -> torch.Tensor:
    """Apply the summed OV circuit: x @ OV_3."""
    return x @ OV_3


ov_token = apply_ov_token(token_embeddings)
print(f"OV(token_emb) shape: {ov_token.shape}")


predicted_token_unweighted = torch.zeros_like(attn_out_3)

for i, (num1, num2) in enumerate(pair_labels):
    predicted_token_unweighted[i] = ov_token[num1] + ov_token[num2]

print("\n" + "=" * 60)
print("TOKEN EMBEDDINGS: predicted vs ACTUAL ATTENTION OUTPUT")
print("=" * 60)

cos_token_unweighted = torch.nn.functional.cosine_similarity(
    predicted_token_unweighted, attn_out_3, dim=-1
)
print(f"\nUnweighted OV(tok_a) + OV(tok_b) vs attn_out:")
print(
    f"  Cosine sim: mean={cos_token_unweighted.mean():.4f}, std={cos_token_unweighted.std():.4f}"
)
norm_ratio_token_unweighted = attn_out_3.norm(dim=-1) / predicted_token_unweighted.norm(
    dim=-1
)
print(f"  Norm ratio (actual/pred): mean={norm_ratio_token_unweighted.mean():.4f}")


print("\n" + "=" * 60)
print("TOKEN EMBEDDINGS: predicted vs RESID_MID")
print("=" * 60)

cos_token_unweighted_mid = torch.nn.functional.cosine_similarity(
    predicted_token_unweighted, resid_mid_3, dim=-1
)
print(f"\nUnweighted OV(tok_a) + OV(tok_b) vs resid_mid:")
print(f"  Cosine sim: mean={cos_token_unweighted_mid.mean():.4f}")


predicted_token_shuffled = torch.zeros_like(attn_out_3)
for i, (num1, num2) in enumerate(pair_labels):
    _, num2_shuffled = pair_labels[shuffled_indices[i].item()]
    predicted_token_shuffled[i] = ov_token[num1] + ov_token[num2_shuffled]

cos_token_shuffled = torch.nn.functional.cosine_similarity(
    predicted_token_shuffled, attn_out_3, dim=-1
)
print(f"\nControl (shuffled) vs attn_out:")
print(f"  Cosine sim: mean={cos_token_shuffled.mean():.4f}")

_ov1_cos_matched = cos_token_unweighted.cpu().numpy().tolist()
_ov1_cos_shuffled = cos_token_shuffled.cpu().numpy().tolist()

import json as _json

_ec_plot_data = {
    "mlp_cos_matched": _mlp_cos_matched,
    "mlp_cos_shuffled": _mlp_cos_shuffled,
    "ov1_cos_matched": _ov1_cos_matched,
    "ov1_cos_shuffled": _ov1_cos_shuffled,
    "analysis_layer": int(ANALYSIS_LAYER),
}
with open("exploration_composition_plot_data.json", "w") as _f:
    _json.dump(_ec_plot_data, _f)
print("Saved: exploration_composition_plot_data.json")

# %%
print("\n" + "=" * 70)
print("SUMMARY: Layer-1 Embeddings vs Token Embeddings")
print("=" * 70)
print(f"\nUnweighted prediction vs attn_out (cosine similarity):")
print(
    f"  Layer-1 embeddings: mean={cos_unweighted_attn.mean():.4f}, std={cos_unweighted_attn.std():.4f}"
)
print(
    f"  Token embeddings:   mean={cos_token_unweighted.mean():.4f}, std={cos_token_unweighted.std():.4f}"
)
print(f"\nControl (shuffled):")
print(f"  Layer-1 embeddings: mean={cos_shuffled_attn.mean():.4f}")
print(f"  Token embeddings:   mean={cos_token_shuffled.mean():.4f}")

fig_compare = go.Figure()
fig_compare.add_trace(
    go.Histogram(
        x=cos_unweighted_attn.cpu().numpy(),
        nbinsx=50,
        name="Layer-1 embeddings (matched)",
        marker_color="steelblue",
        opacity=0.5,
    )
)
fig_compare.add_trace(
    go.Histogram(
        x=cos_token_unweighted.cpu().numpy(),
        nbinsx=50,
        name="Token embeddings (matched)",
        marker_color="forestgreen",
        opacity=0.5,
    )
)
fig_compare.update_layout(
    barmode="overlay",
    title=f"Comparison: Layer-1 vs Token Embeddings (OV prediction vs attn_out)",
    xaxis_title="Cosine Similarity",
    yaxis_title="Count",
    showlegend=True,
)
fig_compare.show()

# %%
# %% Analyze the relationship between unweighted prediction and actual attention output

pred_flat = predicted_unweighted.cpu()
actual_flat = attn_out_3.cpu()

print("Analyzing: unweighted OV(a)+OV(b) vs actual attention output")
print("=" * 60)

norms_pred = pred_flat.norm(dim=-1)
norms_actual = actual_flat.norm(dim=-1)
scale_ratios = norms_actual / norms_pred
print(
    f"\nNorm ratio (actual/predicted): mean={scale_ratios.mean():.4f}, std={scale_ratios.std():.4f}"
)

residual = actual_flat - pred_flat
residual_norms = residual.norm(dim=-1)
print(
    f"Residual ||actual - predicted||: mean={residual_norms.mean():.4f}, std={residual_norms.std():.4f}"
)
print(
    f"Relative error ||residual|| / ||actual||: mean={(residual_norms / norms_actual).mean():.4f}"
)

mean_residual = residual.mean(dim=0)
mean_residual_normed = mean_residual / mean_residual.norm()
proj_onto_mean = (residual @ mean_residual_normed).abs()
print(f"\nResidual structure analysis:")
print(f"  Mean residual norm: {mean_residual.norm():.4f}")
print(
    f"  Projection of residuals onto mean direction: mean={proj_onto_mean.mean():.4f}, std={proj_onto_mean.std():.4f}"
)

total_attn_to_operands = (attn_weights_lhs + attn_weights_rhs).cpu()
print(f"\nAttention coverage:")
print(f"  Fraction of attention to LHS+RHS: mean={total_attn_to_operands.mean():.4f}")
print(f"  (Missing attention goes to other positions: PAD, variables, PLUS, etc.)")

# %% Is the relationship a scaling factor + bias?

print("\n" + "=" * 70)
print("LAYER-1 EMBEDDINGS: Testing hypothesis actual = k * predicted + b")
print("=" * 70)

pred = predicted_unweighted.cpu()
actual = attn_out_3.cpu()


pred_centered = pred - pred.mean(dim=0)
actual_centered = actual - actual.mean(dim=0)

numerator = (pred_centered * actual_centered).sum()
denominator = (pred_centered * pred_centered).sum()
k_global = numerator / denominator
b_global = actual.mean(dim=0) - k_global * pred.mean(dim=0)

print(f"\nGlobal fit: actual = k * predicted + b")
print(f"  k (scaling factor): {k_global:.4f}")
print(f"  ||b|| (bias norm): {b_global.norm():.4f}")

fitted_global = k_global * pred + b_global
residual_global = actual - fitted_global

cos_fitted_global = torch.nn.functional.cosine_similarity(fitted_global, actual, dim=-1)
norm_ratio_fitted = actual.norm(dim=-1) / fitted_global.norm(dim=-1)
relative_error_global = residual_global.norm(dim=-1) / actual.norm(dim=-1)

print(f"\nAfter fitting k*x + b:")
print(
    f"  Cosine similarity: mean={cos_fitted_global.mean():.4f}, std={cos_fitted_global.std():.4f}"
)
print(
    f"  Norm ratio (actual/fitted): mean={norm_ratio_fitted.mean():.4f}, std={norm_ratio_fitted.std():.4f}"
)
print(
    f"  Relative error ||residual||/||actual||: mean={relative_error_global.mean():.4f}, std={relative_error_global.std():.4f}"
)

total_variance = (actual_centered**2).sum()
unexplained_variance_global = (residual_global**2).sum()

r2_global = 1 - unexplained_variance_global / total_variance

print(f"\nVariance explained (R²):")
print(f"  Global k*x + b: {r2_global:.4f}")

print("\n" + "=" * 60)
print("CONTROL: Fitting k*x + b on MISMATCHED (shuffled) pairs")
print("=" * 60)

pred_shuffled = predicted_shuffled.cpu()

pred_shuffled_centered = pred_shuffled - pred_shuffled.mean(dim=0)

numerator_shuffled = (pred_shuffled_centered * actual_centered).sum()
denominator_shuffled = (pred_shuffled_centered * pred_shuffled_centered).sum()
k_shuffled = numerator_shuffled / denominator_shuffled
b_shuffled = actual.mean(dim=0) - k_shuffled * pred_shuffled.mean(dim=0)

print(f"\nShuffled fit: actual = k * predicted_shuffled + b")
print(f"  k (scaling factor): {k_shuffled:.4f}")
print(f"  ||b|| (bias norm): {b_shuffled.norm():.4f}")

fitted_shuffled = k_shuffled * pred_shuffled + b_shuffled
residual_shuffled = actual - fitted_shuffled

cos_fitted_shuffled = torch.nn.functional.cosine_similarity(
    fitted_shuffled, actual, dim=-1
)
norm_ratio_shuffled = actual.norm(dim=-1) / fitted_shuffled.norm(dim=-1)
relative_error_shuffled = residual_shuffled.norm(dim=-1) / actual.norm(dim=-1)

print(f"\nAfter fitting k*x + b (shuffled):")
print(
    f"  Cosine similarity: mean={cos_fitted_shuffled.mean():.4f}, std={cos_fitted_shuffled.std():.4f}"
)
print(
    f"  Norm ratio (actual/fitted): mean={norm_ratio_shuffled.mean():.4f}, std={norm_ratio_shuffled.std():.4f}"
)
print(
    f"  Relative error ||residual||/||actual||: mean={relative_error_shuffled.mean():.4f}, std={relative_error_shuffled.std():.4f}"
)

unexplained_variance_shuffled = (residual_shuffled**2).sum()
r2_shuffled = 1 - unexplained_variance_shuffled / total_variance

print(f"\nVariance explained (R²):")
print(f"  Shuffled k*x + b: {r2_shuffled:.4f}")

print("\n" + "=" * 60)
print("LAYER-1 EMBEDDINGS: Matched vs Mismatched (Control)")
print("=" * 60)
print(f"{'Metric':<40} {'Matched':>12} {'Shuffled':>12}")
print("-" * 64)
print(f"{'k (scaling factor)':<40} {k_global:>12.4f} {k_shuffled:>12.4f}")
print(f"{'||b|| (bias norm)':<40} {b_global.norm():>12.4f} {b_shuffled.norm():>12.4f}")
print(
    f"{'Cosine sim (mean)':<40} {cos_fitted_global.mean():>12.4f} {cos_fitted_shuffled.mean():>12.4f}"
)
print(
    f"{'Relative error (mean)':<40} {relative_error_global.mean():>12.4f} {relative_error_shuffled.mean():>12.4f}"
)
print(f"{'R² (variance explained)':<40} {r2_global:>12.4f} {r2_shuffled:>12.4f}")

# %%

print("\n" + "=" * 70)
print("TOKEN EMBEDDINGS: Testing hypothesis actual = k * predicted + b")
print("=" * 70)

pred_token = predicted_token_unweighted.cpu()

pred_token_centered = pred_token - pred_token.mean(dim=0)

numerator_token = (pred_token_centered * actual_centered).sum()
denominator_token = (pred_token_centered * pred_token_centered).sum()
k_token = numerator_token / denominator_token
b_token = actual.mean(dim=0) - k_token * pred_token.mean(dim=0)

print(f"\nGlobal fit: actual = k * predicted + b")
print(f"  k (scaling factor): {k_token:.4f}")
print(f"  ||b|| (bias norm): {b_token.norm():.4f}")

fitted_token = k_token * pred_token + b_token
residual_token = actual - fitted_token

cos_fitted_token = torch.nn.functional.cosine_similarity(fitted_token, actual, dim=-1)
norm_ratio_token = actual.norm(dim=-1) / fitted_token.norm(dim=-1)
relative_error_token = residual_token.norm(dim=-1) / actual.norm(dim=-1)

print(f"\nAfter fitting k*x + b:")
print(
    f"  Cosine similarity: mean={cos_fitted_token.mean():.4f}, std={cos_fitted_token.std():.4f}"
)
print(
    f"  Norm ratio (actual/fitted): mean={norm_ratio_token.mean():.4f}, std={norm_ratio_token.std():.4f}"
)
print(
    f"  Relative error ||residual||/||actual||: mean={relative_error_token.mean():.4f}, std={relative_error_token.std():.4f}"
)

unexplained_variance_token = (residual_token**2).sum()
r2_token = 1 - unexplained_variance_token / total_variance

print(f"\nVariance explained (R²):")
print(f"  Global k*x + b: {r2_token:.4f}")

print("\n" + "=" * 60)
print("CONTROL: Fitting k*x + b on MISMATCHED (shuffled) pairs")
print("=" * 60)

pred_token_shuffled = predicted_token_shuffled.cpu()

pred_token_shuffled_centered = pred_token_shuffled - pred_token_shuffled.mean(dim=0)

numerator_token_shuffled = (pred_token_shuffled_centered * actual_centered).sum()
denominator_token_shuffled = (
    pred_token_shuffled_centered * pred_token_shuffled_centered
).sum()
k_token_shuffled = numerator_token_shuffled / denominator_token_shuffled
b_token_shuffled = actual.mean(dim=0) - k_token_shuffled * pred_token_shuffled.mean(
    dim=0
)

print(f"\nShuffled fit: actual = k * predicted_shuffled + b")
print(f"  k (scaling factor): {k_token_shuffled:.4f}")
print(f"  ||b|| (bias norm): {b_token_shuffled.norm():.4f}")

fitted_token_shuffled = k_token_shuffled * pred_token_shuffled + b_token_shuffled
residual_token_shuffled = actual - fitted_token_shuffled

cos_fitted_token_shuffled = torch.nn.functional.cosine_similarity(
    fitted_token_shuffled, actual, dim=-1
)
norm_ratio_token_shuffled = actual.norm(dim=-1) / fitted_token_shuffled.norm(dim=-1)
relative_error_token_shuffled = residual_token_shuffled.norm(dim=-1) / actual.norm(
    dim=-1
)

print(f"\nAfter fitting k*x + b (shuffled):")
print(
    f"  Cosine similarity: mean={cos_fitted_token_shuffled.mean():.4f}, std={cos_fitted_token_shuffled.std():.4f}"
)
print(
    f"  Norm ratio (actual/fitted): mean={norm_ratio_token_shuffled.mean():.4f}, std={norm_ratio_token_shuffled.std():.4f}"
)
print(
    f"  Relative error ||residual||/||actual||: mean={relative_error_token_shuffled.mean():.4f}, std={relative_error_token_shuffled.std():.4f}"
)

unexplained_variance_token_shuffled = (residual_token_shuffled**2).sum()
r2_token_shuffled = 1 - unexplained_variance_token_shuffled / total_variance

print(f"\nVariance explained (R²):")
print(f"  Shuffled k*x + b: {r2_token_shuffled:.4f}")

print("\n" + "=" * 60)
print("TOKEN EMBEDDINGS: Matched vs Mismatched (Control)")
print("=" * 60)
print(f"{'Metric':<40} {'Matched':>12} {'Shuffled':>12}")
print("-" * 64)
print(f"{'k (scaling factor)':<40} {k_token:>12.4f} {k_token_shuffled:>12.4f}")
print(
    f"{'||b|| (bias norm)':<40} {b_token.norm():>12.4f} {b_token_shuffled.norm():>12.4f}"
)
print(
    f"{'Cosine sim (mean)':<40} {cos_fitted_token.mean():>12.4f} {cos_fitted_token_shuffled.mean():>12.4f}"
)
print(
    f"{'Relative error (mean)':<40} {relative_error_token.mean():>12.4f} {relative_error_token_shuffled.mean():>12.4f}"
)
print(f"{'R² (variance explained)':<40} {r2_token:>12.4f} {r2_token_shuffled:>12.4f}")

# %%
print("\n" + "=" * 70)
print("FINAL SUMMARY: Layer-1 vs Token Embeddings (k*x + b fit)")
print("=" * 70)
print(f"\n{'Metric':<40} {'Layer-1':>12} {'Token Emb':>12}")
print("-" * 64)
print(f"{'k (scaling factor)':<40} {k_global:>12.4f} {k_token:>12.4f}")
print(f"{'||b|| (bias norm)':<40} {b_global.norm():>12.4f} {b_token.norm():>12.4f}")
print(f"{'R² (matched)':<40} {r2_global:>12.4f} {r2_token:>12.4f}")
print(f"{'R² (shuffled control)':<40} {r2_shuffled:>12.4f} {r2_token_shuffled:>12.4f}")
print(
    f"{'Cosine sim after fit (matched)':<40} {cos_fitted_global.mean():>12.4f} {cos_fitted_token.mean():>12.4f}"
)

# %% PCA of OV_3 * Number Token Embeddings
ov_token_np = ov_token.detach().cpu().numpy()

coords_ov = PCA(n_components=2, random_state=SEED).fit_transform(ov_token_np)
fig_pca_ov = go.Figure(
    data=go.Scatter(
        x=coords_ov[:, 0],
        y=coords_ov[:, 1],
        mode="markers+text",
        text=[str(i) for i in range(MOD)],
        textposition="middle center",
        marker=dict(size=8, color="teal"),
    )
)
fig_pca_ov.update_layout(
    title=f"PCA of OV_3 × Number Token Embeddings (Layer {ANALYSIS_LAYER})",
    xaxis_title="PC1",
    yaxis_title="PC2",
    yaxis=dict(scaleanchor="x", scaleratio=1),
)
fig_pca_ov.show()

# %% Circularity analysis of top 10 PCs of Number Token Embeddings
import numpy as np

N_COMPONENTS = 10

token_emb_np = model.embed.W_E[:MOD].detach().cpu().numpy()

pca_full = PCA(n_components=N_COMPONENTS, random_state=SEED)
coords_full = pca_full.fit_transform(token_emb_np)

print(f"Explained variance ratio for top {N_COMPONENTS} PCs:")
for i, var in enumerate(pca_full.explained_variance_ratio_):
    print(f"  PC{i + 1}: {var:.4f} ({var * 100:.1f}%)")
print(f"  Total: {pca_full.explained_variance_ratio_.sum():.4f}")


def circularity_score(x: np.ndarray, y: np.ndarray) -> float:
    """Compute how circular a set of 2D points is.

    Returns a score where 1.0 = perfect circle (all points equidistant from center),
    lower values = less circular. Uses 1 - coefficient_of_variation of radii.
    """
    cx, cy = x.mean(), y.mean()
    radii = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    mean_r = radii.mean()
    if mean_r < 1e-10:
        return 0.0

    std_r = radii.std()
    cv = std_r / mean_r  # coefficient of variation
    return 1.0 - cv


circularity_matrix = np.zeros((N_COMPONENTS, N_COMPONENTS))
for i in range(N_COMPONENTS):
    for j in range(N_COMPONENTS):
        if i != j:
            score = circularity_score(coords_full[:, i], coords_full[:, j])
            circularity_matrix[i, j] = score

pc_labels = [f"PC{i + 1}" for i in range(N_COMPONENTS)]
fig_circ = go.Figure(
    data=go.Heatmap(
        z=circularity_matrix,
        x=pc_labels,
        y=pc_labels,
        colorscale="Viridis",
        text=np.round(circularity_matrix, 2),
        texttemplate="%{text}",
        colorbar=dict(title="Circularity"),
    )
)
fig_circ.update_layout(
    title="Circularity Score for Each Pair of PCs (Number Token Embeddings)",
    xaxis_title="PC (x-axis)",
    yaxis_title="PC (y-axis)",
)
fig_circ.show()

pairs_scores = []
for i in range(N_COMPONENTS):
    for j in range(i + 1, N_COMPONENTS):
        pairs_scores.append((i, j, circularity_matrix[i, j]))

pairs_scores.sort(key=lambda x: x[2], reverse=True)

print("\nTop 5 most circular PC pairs:")
for rank, (i, j, score) in enumerate(pairs_scores[:5], 1):
    print(f"  {rank}. PC{i + 1} vs PC{j + 1}: circularity = {score:.4f}")

from plotly.subplots import make_subplots

fig_top = make_subplots(
    rows=1,
    cols=3,
    subplot_titles=[
        f"PC{i + 1} vs PC{j + 1} (circ={s:.3f})" for i, j, s in pairs_scores[:3]
    ],
)

colors = np.arange(MOD)  # Color by number value to see ordering

for col, (i, j, score) in enumerate(pairs_scores[:3], 1):
    fig_top.add_trace(
        go.Scatter(
            x=coords_full[:, i],
            y=coords_full[:, j],
            mode="markers+text",
            text=[str(k) for k in range(MOD)],
            textposition="middle center",
            marker=dict(size=8, color=colors, colorscale="hsv", showscale=(col == 3)),
            textfont=dict(size=8),
        ),
        row=1,
        col=col,
    )
    fig_top.update_xaxes(title_text=f"PC{i + 1}", row=1, col=col)
    fig_top.update_yaxes(
        title_text=f"PC{j + 1}",
        scaleanchor=f"x{col if col > 1 else ''}",
        scaleratio=1,
        row=1,
        col=col,
    )

fig_top.update_layout(
    title="Top 3 Most Circular PC Pairs (Number Token Embeddings)",
    showlegend=False,
    height=500,
    width=1200,
)
fig_top.show()

# %% Check accuracy of mlp(ov_1(a) + ov_1(b)) over all MOD^2 pairs

W_V_1 = model.blocks[1].attn.W_V
W_O_1 = model.blocks[1].attn.W_O
OV_1 = einsum("h d_in d_h, h d_h d_out -> d_in d_out", W_V_1, W_O_1)

ov_1_nums = model.embed.W_E[:MOD] @ OV_1

mlp = model.blocks[model.cfg.n_layers - 1].mlp
W_U = model.unembed.W_U

correct = 0
for a in range(MOD):
    for b in range(MOD):
        x = ov_1_nums[a] + ov_1_nums[b]
        hidden = torch.relu(x @ mlp.W_in + mlp.b_in)
        mlp_out = hidden @ mlp.W_out + mlp.b_out
        logits = mlp_out @ W_U
        if logits.argmax().item() == (a + b) % MOD:
            correct += 1

print(
    f"Accuracy of mlp(ov_1(a) + ov_1(b)): {correct}/{MOD**2} = {correct / MOD**2:.4f}"
)

# %%
