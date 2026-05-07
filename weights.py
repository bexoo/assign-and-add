# %%
# %%
import os
import random
from collections import defaultdict
from dataclasses import asdict

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/cache")

import numpy as np
import einops
from functools import partial
from jaxtyping import Float
import matplotlib.pyplot as plt  # for quick static sanity-checks
try:
    from tueplots import bundles
except ImportError:
    bundles = None

if bundles is not None:
    plt.rcParams.update(bundles.neurips2024(usetex=False))
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio
try:
    import kaleido  # Required for plotly image export
except ImportError:
    kaleido = None
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

pio.renderers.default = "notebook"
if os.getenv("GITHUB_ACTIONS") == "true":
    pio.renderers.default = "svg"

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cpu")
)
print(f"Using device: {DEVICE}")

torch.set_grad_enabled(False)


def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def line(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.line(utils.to_numpy(tensor), labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs).show(renderer)


# %%
from model_io import build_run_config, default_config, load_model_from_checkpoint

cfg_wb = default_config()
run_cfg = build_run_config(cfg_wb)

MOD = run_cfg.mod
PLUS = run_cfg.plus_id
EQUAL = run_cfg.equal_id
PAD = run_cfg.pad_id
A_TOKEN = run_cfg.a_token_id
VAR_LEN = run_cfg.n_vars
VARS = list(run_cfg.var_ids)
FIRST_HALF, SECOND_HALF = VARS[: VAR_LEN // 2], VARS[VAR_LEN // 2 :]

VOCAB = run_cfg.vocab
TOKEN_STRINGS = [s.replace("<PAD>", "PAD") for s in run_cfg.token_strings]

SEQ_LEN = run_cfg.seq_len
BATCH_SIZE = 512
USE_SIMPLE_16 = bool(cfg_wb["USE_SIMPLE_16"])

# %%
V, N, TYPES = len(VARS), MOD, 3

var_g  = torch.tensor(VARS).view(V, 1, 1, 1)
type_g = torch.arange(TYPES).view(1, TYPES, 1, 1)
num1_g = torch.arange(N).view(1, 1, N, 1)
num2_g = torch.arange(N).view(1, 1, 1, N)

perm_tensor = torch.stack([
    var_g.repeat(1, TYPES, N, N),
    type_g.repeat(V, 1, N, N),
    num1_g.repeat(V, TYPES, 1, N),
    num2_g.repeat(V, TYPES, N, 1)
], dim=-1)

perm_table = einops.rearrange(perm_tensor, 'v t n m f -> (v t n m) f').contiguous()
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
model, cfg_wb, _run_cfg = load_model_from_checkpoint("checkpoints/model_state_dict.pth", config=cfg_wb, device=DEVICE)
for p in model.parameters():
    p.requires_grad = False
print("Model loaded.")

# %% 5. Custom string evaluation -------------------------------------------
print(f"SEQ_LEN: {SEQ_LEN}")
EXAMPLE_STRING = "; b 7 ; e 5 ; d 2 ; + 7 23 ="

symbol_to_token = {str(i): i for i in range(MOD)}
symbol_to_token.update({
    ';': PAD,
    '+': PLUS,
    '=': EQUAL,
})
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
N_HEADS  = model.cfg.n_heads
fig_heat_ex = make_subplots(
    rows=N_LAYERS,
    cols=N_HEADS,
    shared_xaxes=True,
    shared_yaxes=True,
    horizontal_spacing=0.08,
    vertical_spacing=0.08,
    subplot_titles=[f"Layer {l}" for l in range(N_LAYERS) for h in range(N_HEADS)],
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
fig_heat_ex.update_xaxes(tickmode="array", tickvals=_idx_vals_ex, ticktext=_token_labels_ex)
fig_heat_ex.update_yaxes(tickmode="array", tickvals=_idx_vals_ex, ticktext=_token_labels_ex)

red_boxes_layer0 = [
    (4, 3),   # 7 -> b
    (7, 6),   # 5 -> e  
    (10, 9),  # 2 -> d
]

orange_box_layer0 = (15, 13, 14)  # query_pos, key_start, key_end

yellow_boxes_layer1 = [
    (15, 4),   # = -> position 4
    (15, 14),  # = -> position 14
]

shapes = []
BOX_LINEWIDTH = 1.5

for query_pos, key_pos in red_boxes_layer0:
    shapes.append(dict(
        type="rect",
        xref="x", yref="y",
        x0=key_pos - 0.5, x1=key_pos + 0.5,
        y0=query_pos - 0.5, y1=query_pos + 0.5,
        line=dict(color="red", width=BOX_LINEWIDTH),
    ))

query_pos, key_start, key_end = orange_box_layer0
shapes.append(dict(
    type="rect",
    xref="x", yref="y",
    x0=key_start - 0.5, x1=key_end + 0.5,
    y0=query_pos - 0.5, y1=query_pos + 0.5,
    line=dict(color="orange", width=BOX_LINEWIDTH),
))

for query_pos, key_pos in yellow_boxes_layer1:
    shapes.append(dict(
        type="rect",
        xref="x2", yref="y2",
        x0=key_pos - 0.5, x1=key_pos + 0.5,
        y0=query_pos - 0.5, y1=query_pos + 0.5,
        line=dict(color="yellow", width=BOX_LINEWIDTH),
    ))

fig_heat_ex.update_layout(
    shapes=shapes,
    font=dict(size=18),  # Global font size for tick labels
)
for annotation in fig_heat_ex.layout.annotations:
    annotation.font.size = 20
if kaleido is not None:
    fig_heat_ex.write_image(
        "attention_patterns.pdf", engine="kaleido", width=800, height=1000
    )
    print("Saved: attention_patterns.pdf")
else:
    print("Skipped attention_patterns.pdf: kaleido is not installed")

# %% Get all token and positional embeddings
with torch.no_grad():
    number_embeddings = model.embed.W_E[:MOD].detach()
    special_embeddings = model.embed.W_E[[PLUS, EQUAL, PAD]].detach()  # +, =, <PAD>
    var_embeddings = model.embed.W_E[VARS].detach()
    positional_embeddings = model.pos_embed.W_pos[:SEQ_LEN].detach()

token_embeddings = torch.cat([number_embeddings, special_embeddings, var_embeddings], dim=0)
print(f"Number embeddings: {number_embeddings.shape}")
print(f"Special embeddings: {special_embeddings.shape}")
print(f"Variable embeddings: {var_embeddings.shape}")
print(f"Positional embeddings: {positional_embeddings.shape}")

var_labels = [chr(ord("a") + i) for i in range(len(VARS))]
E_labels = [str(i) for i in range(MOD)] + ["+", "=", "<PAD>"] + var_labels + [f"pos_{i}" for i in range(SEQ_LEN)]
print(E_labels)

# %% Plot E W_O W_V E^T for all pairs of embeddings
E = torch.cat([token_embeddings, positional_embeddings], dim=0)
n_numbers = number_embeddings.shape[0]
n_special = special_embeddings.shape[0]
n_vars = var_embeddings.shape[0]
n_pos = positional_embeddings.shape[0]
print(f"Combined embedding matrix E shape: {E.shape}")
print(f"  - Number tokens: {n_numbers} (indices 0 to {n_numbers-1})")
print(f"  - Special tokens: {n_special} (indices {n_numbers} to {n_numbers+n_special-1})")
print(f"  - Variable tokens: {n_vars} (indices {n_numbers+n_special} to {n_numbers+n_special+n_vars-1})")
print(f"  - Positional embeddings: {n_pos} (indices {n_numbers+n_special+n_vars} to {E.shape[0]-1})")

def compute_ov_matrix(layer_idx: int) -> torch.Tensor:
    """Compute the OV circuit matrix for a given layer, summed over all heads."""
    W_V = model.blocks[layer_idx].attn.W_V
    W_O = model.blocks[layer_idx].attn.W_O
    OV = einsum(
        "heads d_model_in d_head, heads d_head d_model_out -> d_model_in d_model_out",
        W_V, W_O
    )
    return OV


def compute_e_ov_e_t(layer_idx: int) -> torch.Tensor:
    """Compute E @ W_O @ W_V @ E^T for the given layer."""
    OV = compute_ov_matrix(layer_idx)
    return E @ OV @ E.T


n_layers = model.cfg.n_layers
e_ov_e_t_matrices = [compute_e_ov_e_t(layer) for layer in range(n_layers)]

# %% Plot the matrices
n_tokens = token_embeddings.shape[0]
n_positions = positional_embeddings.shape[0]

tick_positions = [0, n_tokens, n_tokens + n_positions]
tick_labels = [f"0\n(tokens)", f"{n_tokens}\n(pos)", f"{n_tokens + n_positions}"]

fig, axes = plt.subplots(1, n_layers, figsize=(6 * n_layers, 5))
if n_layers == 1:
    axes = [axes]

for layer_idx, (ax, matrix) in enumerate(zip(axes, e_ov_e_t_matrices)):
    matrix_np = matrix.cpu().numpy()
    im = ax.imshow(matrix_np, cmap='viridis', aspect='equal')
    ax.set_title(f"$EW_OW_VE^T$ Layer {layer_idx}")
    ax.set_xlabel("Embedding index")
    ax.set_ylabel("Embedding index")
    
    ax.axvline(x=n_tokens - 0.5, color='white', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.axhline(y=n_tokens - 0.5, color='white', linestyle='--', linewidth=0.5, alpha=0.7)
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)

plt.savefig("e_ov_et_all_layers.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: e_ov_et_all_layers.png")

# %% Plot individual layers with more detail (larger figures)
for layer_idx, matrix in enumerate(e_ov_e_t_matrices):
    fig, ax = plt.subplots(figsize=(8, 7))
    matrix_np = matrix.cpu().numpy()
    
    im = ax.imshow(matrix_np, cmap='viridis', aspect='equal')
    ax.set_title(f"$EW_OW_VE^T$ (Layer {layer_idx})", fontsize=14)
    ax.set_xlabel("Embedding index", fontsize=12)
    ax.set_ylabel("Embedding index", fontsize=12)
    
    ax.axvline(x=n_tokens - 0.5, color='white', linestyle='--', linewidth=1, alpha=0.8)
    ax.axhline(y=n_tokens - 0.5, color='white', linestyle='--', linewidth=1, alpha=0.8)
    
    ax.text(n_tokens / 2, -3, "Tokens", ha='center', fontsize=10, color='gray')
    ax.text(n_tokens + n_positions / 2, -3, "Positions", ha='center', fontsize=10, color='gray')
    ax.text(-3, n_tokens / 2, "Tokens", va='center', fontsize=10, color='gray', rotation=90)
    ax.text(-3, n_tokens + n_positions / 2, "Positions", va='center', fontsize=10, color='gray', rotation=90)
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)

    plt.savefig(f"e_ov_et_layer_{layer_idx}.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: e_ov_et_layer_{layer_idx}.png")


# %% Plot layer-0 QK

def compute_qk_matrix(layer_idx: int) -> torch.Tensor:
    """Compute the QK circuit matrix for a given layer, summed over all heads."""
    W_Q = model.blocks[layer_idx].attn.W_Q
    W_K = model.blocks[layer_idx].attn.W_K
    QK = einsum(
        "heads d_model_q d_head, heads d_model_k d_head -> d_model_q d_model_k",
        W_Q, W_K
    )
    return QK


def compute_e_qk_e_t(layer_idx: int) -> torch.Tensor:
    """Compute E @ W_Q @ W_K^T @ E^T for the given layer."""
    QK = compute_qk_matrix(layer_idx)
    return E @ QK @ E.T


QK_0 = compute_qk_matrix(0)
e_qk_e_t_0 = E @ QK_0 @ E.T

import json as _json

_e_qk_et_np = e_qk_e_t_0.cpu().numpy()

# %% Plot layer-2 QK similarity over [E ; E @ OV_1]
OV_1_mat = compute_ov_matrix(0)
QK_2 = compute_qk_matrix(1)
E_ov1 = E @ OV_1_mat
E_stacked = torch.cat([E, E_ov1], dim=0)
e_qk2_stacked = E_stacked @ QK_2 @ E_stacked.T

n_E = E.shape[0]
fig, ax = plt.subplots(figsize=(16, 14))
matrix_np = e_qk2_stacked.cpu().numpy().astype(float)

_var_start_ov1 = n_E + n_numbers + n_special
_var_end_ov1 = n_E + n_numbers + n_special + n_vars
matrix_np[_var_start_ov1:_var_end_ov1, _var_start_ov1:_var_end_ov1] = np.nan

im = ax.imshow(matrix_np, cmap="viridis", aspect="equal")
ax.set_title(r"$[E\,;\,E\cdot OV_1]\cdot QK_2\cdot [E\,;\,E\cdot OV_1]^T$", fontsize=14)
ax.set_xlabel("Key embedding index", fontsize=12)
ax.set_ylabel("Query embedding index", fontsize=12)

_subsection_offsets = [
    n_numbers,
    n_numbers + n_special,
    n_numbers + n_special + n_vars,
]
for section_start in (0, n_E):
    for off in _subsection_offsets:
        x = section_start + off - 0.5
        ax.axvline(x=x, color="white", linewidth=0.5, alpha=0.6)
        ax.axhline(y=x, color="white", linewidth=0.5, alpha=0.6)

ax.axvline(x=n_E - 0.5, color="white", linewidth=1.2, alpha=0.9)
ax.axhline(y=n_E - 0.5, color="white", linewidth=1.2, alpha=0.9)

fig.colorbar(im, ax=ax, shrink=0.8)
plt.savefig("e_qk_et_layer_1.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: e_qk_et_layer_1.png")

# %% QK_0 similarity: (avg number + pos) vs (avg variable + pos)

avg_number_emb = number_embeddings.mean(dim=0)
avg_var_emb = var_embeddings.mean(dim=0)

NUM_POS = 12
pos_emb_subset = positional_embeddings[:NUM_POS]

query_embeddings = pos_emb_subset
key_embeddings = pos_emb_subset

qk0_pos_similarity = query_embeddings @ QK_0 @ key_embeddings.T

matrix_np = qk0_pos_similarity.cpu().numpy()
causal_mask = np.triu(np.ones_like(matrix_np, dtype=bool), k=1)
identity_mask = np.eye(NUM_POS, dtype=bool)
offset_mask = np.eye(NUM_POS, k=-2, dtype=bool)
combined_mask = causal_mask | identity_mask | offset_mask
matrix_np_masked = np.where(combined_mask, np.nan, matrix_np)

# %% Norm comparison: ||e|| vs ||OV_0(e)|| for all tokens in E

OV_0 = compute_ov_matrix(0)
E_through_ov = E @ OV_0

e_norms = E.norm(dim=-1)
ov_norms = E_through_ov.norm(dim=-1)
norm_diff = ov_norms - e_norms
norm_ratio = ov_norms / (e_norms + 1e-8)

print("=" * 80)
print("Norm comparison: ||e|| vs ||OV_0(e)|| for all tokens in E")
print("=" * 80)
print(f"{'Token':<12} {'||e||':>10} {'||OV_0(e)||':>12} {'Diff':>10} {'Ratio':>8}")
print("-" * 80)

for i, label in enumerate(E_labels):
    print(f"{label:<12} {e_norms[i].item():>10.4f} {ov_norms[i].item():>12.4f} {norm_diff[i].item():>+10.4f} {norm_ratio[i].item():>8.4f}")

print("\n" + "=" * 80)
print("Summary by category")
print("=" * 80)

categories = [
    ("Numbers", 0, n_numbers),
    ("Special (+,=,<PAD>)", n_numbers, n_numbers + n_special),
    ("Variables", n_numbers + n_special, n_numbers + n_special + n_vars),
    ("Positional", n_numbers + n_special + n_vars, E.shape[0]),
]

for cat_name, start, end in categories:
    cat_e_norms = e_norms[start:end]
    cat_ov_norms = ov_norms[start:end]
    cat_diff = norm_diff[start:end]
    cat_ratio = norm_ratio[start:end]
    print(f"\n{cat_name}:")
    print(f"  ||e||:       mean={cat_e_norms.mean().item():.4f}, std={cat_e_norms.std().item():.4f}")
    print(f"  ||OV_0(e)||: mean={cat_ov_norms.mean().item():.4f}, std={cat_ov_norms.std().item():.4f}")
    print(f"  Diff:        mean={cat_diff.mean().item():+.4f}, std={cat_diff.std().item():.4f}")
    print(f"  Ratio:       mean={cat_ratio.mean().item():.4f}, std={cat_ratio.std().item():.4f}")

# %% Norm comparison: ||e|| vs ||MLP_0(e)|| for all tokens in E

def compute_mlp_for_norm(layer_idx: int, X: torch.Tensor) -> torch.Tensor:
    """Compute MLP output: act(X @ W_in + b_in) @ W_out + b_out."""
    mlp = model.blocks[layer_idx].mlp
    pre = X @ mlp.W_in + mlp.b_in

    act_fn = getattr(model.cfg, "act_fn", "relu")
    if act_fn == "relu":
        post = torch.nn.functional.relu(pre)
    elif act_fn == "gelu":
        post = torch.nn.functional.gelu(pre)
    elif act_fn == "silu":
        post = torch.nn.functional.silu(pre)
    else:
        raise ValueError(f"Unsupported act_fn={act_fn!r}")

    return post @ mlp.W_out + mlp.b_out


E_through_mlp = compute_mlp_for_norm(0, E)

e_norms_mlp = E.norm(dim=-1)
mlp_norms = E_through_mlp.norm(dim=-1)
norm_diff_mlp = mlp_norms - e_norms_mlp
norm_ratio_mlp = mlp_norms / (e_norms_mlp + 1e-8)

print("=" * 80)
print("Norm comparison: ||e|| vs ||MLP_0(e)|| for all tokens in E")
print("=" * 80)
print(f"{'Token':<12} {'||e||':>10} {'||MLP_0(e)||':>12} {'Diff':>10} {'Ratio':>8}")
print("-" * 80)

for i, label in enumerate(E_labels):
    print(f"{label:<12} {e_norms_mlp[i].item():>10.4f} {mlp_norms[i].item():>12.4f} {norm_diff_mlp[i].item():>+10.4f} {norm_ratio_mlp[i].item():>8.4f}")

print("\n" + "=" * 80)
print("Summary by category")
print("=" * 80)

categories_mlp = [
    ("Numbers", 0, n_numbers),
    ("Special (+,=,<PAD>)", n_numbers, n_numbers + n_special),
    ("Variables", n_numbers + n_special, n_numbers + n_special + n_vars),
    ("Positional", n_numbers + n_special + n_vars, E.shape[0]),
]

for cat_name, start, end in categories_mlp:
    cat_e_norms = e_norms_mlp[start:end]
    cat_mlp_norms = mlp_norms[start:end]
    cat_diff = norm_diff_mlp[start:end]
    cat_ratio = norm_ratio_mlp[start:end]
    print(f"\n{cat_name}:")
    print(f"  ||e||:        mean={cat_e_norms.mean().item():.4f}, std={cat_e_norms.std().item():.4f}")
    print(f"  ||MLP_0(e)||: mean={cat_mlp_norms.mean().item():.4f}, std={cat_mlp_norms.std().item():.4f}")
    print(f"  Diff:         mean={cat_diff.mean().item():+.4f}, std={cat_diff.std().item():.4f}")
    print(f"  Ratio:        mean={cat_ratio.mean().item():.4f}, std={cat_ratio.std().item():.4f}")

# %% Cosine similarity between e and MLP_0(e) for all tokens in E

cos_sim_mlp = torch.nn.functional.cosine_similarity(E, E_through_mlp, dim=-1)

print("=" * 80)
print("Cosine similarity: cos(e, MLP_0(e)) for all tokens in E")
print("=" * 80)
print(f"{'Token':<12} {'cos(e, MLP_0(e))':>18}")
print("-" * 80)

for i, label in enumerate(E_labels):
    print(f"{label:<12} {cos_sim_mlp[i].item():>18.4f}")

print("\n" + "=" * 80)
print("Summary by category")
print("=" * 80)

categories_cos = [
    ("Numbers", 0, n_numbers),
    ("Special (+,=,<PAD>)", n_numbers, n_numbers + n_special),
    ("Variables", n_numbers + n_special, n_numbers + n_special + n_vars),
    ("Positional", n_numbers + n_special + n_vars, E.shape[0]),
]

for cat_name, start, end in categories_cos:
    cat_cos_sim = cos_sim_mlp[start:end]
    print(f"\n{cat_name}:")
    print(f"  cos(e, MLP_0(e)): mean={cat_cos_sim.mean().item():.4f}, std={cat_cos_sim.std().item():.4f}, min={cat_cos_sim.min().item():.4f}, max={cat_cos_sim.max().item():.4f}")

# %% Plot (E @ OV_0 + MLP_0(E @ OV_0)) @ QK_1 @ (E @ OV_0 + MLP_0(E @ OV_0))^T

OV_0 = compute_ov_matrix(0)
E_ov0 = E @ OV_0

mlp0_E_ov0 = E_ov0 + compute_mlp_for_norm(0, E_ov0)

QK_1 = compute_qk_matrix(1)

similarity_matrix = mlp0_E_ov0 @ QK_1 @ mlp0_E_ov0.T

fig, ax = plt.subplots(figsize=(10, 9))
matrix_np = similarity_matrix.cpu().numpy()

im = ax.imshow(matrix_np, cmap='viridis', aspect='equal')
ax.set_title(r"$(E \cdot OV_0 + \mathrm{MLP}_0(E \cdot OV_0)) \cdot QK_1 \cdot (\ldots)^T$", fontsize=14)
ax.set_xlabel("Key embedding index", fontsize=12)
ax.set_ylabel("Query embedding index", fontsize=12)

ax.axvline(x=n_tokens - 0.5, color='white', linestyle='--', linewidth=1, alpha=0.8)
ax.axhline(y=n_tokens - 0.5, color='white', linestyle='--', linewidth=1, alpha=0.8)

ax.text(n_tokens / 2, -3, "Tokens", ha='center', fontsize=10, color='gray')
ax.text(n_tokens + n_positions / 2, -3, "Positions", ha='center', fontsize=10, color='gray')
ax.text(-3, n_tokens / 2, "Tokens", va='center', fontsize=10, color='gray', rotation=90)
ax.text(-3, n_tokens + n_positions / 2, "Positions", va='center', fontsize=10, color='gray', rotation=90)

cbar = fig.colorbar(im, ax=ax, shrink=0.8)
plt.savefig("mlp0_e_ov0_qk1.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: mlp0_e_ov0_qk1.png")


# %% Plot variable-to-variable QK_1 similarity

E_equal = model.embed.W_E[EQUAL].detach()
E_numbers = model.embed.W_E[:MOD].detach()
E_vars = model.embed.W_E[VARS].detach()

OV_0 = compute_ov_matrix(0)
QK_1 = compute_qk_matrix(1)

equal_resid_per_var = []
for var_idx in range(len(VARS)):
    after_attn = E_equal + E_vars[var_idx] @ OV_0
    after_attn = E_vars[var_idx] @ OV_0
    resid = after_attn
    equal_resid_per_var.append(resid)
equal_resid_per_var = torch.stack(equal_resid_per_var)

num_resid_per_var = []
for var_idx in range(len(VARS)):
    num_resids = []
    for num_idx in range(MOD):
        after_attn = E_vars[var_idx] @ OV_0
        resid = after_attn
        num_resids.append(resid)
    avg_num_resid = torch.stack(num_resids).mean(dim=0)
    num_resid_per_var.append(avg_num_resid)
num_resid_per_var = torch.stack(num_resid_per_var)

var_var_similarity = equal_resid_per_var @ QK_1 @ num_resid_per_var.T

_var_var_np = var_var_similarity.cpu().numpy()

_attn_patterns_data = []
for layer in range(N_LAYERS):
    att = cache_ex_str["attn", layer].squeeze(0).cpu().numpy()
    _attn_patterns_data.append(att.tolist())

_weights_plot_data = {
    "e_qk_et_layer0": _e_qk_et_np.tolist(),
    "qk0_pos_masked": [[None if np.isnan(v) else float(v) for v in row] for row in matrix_np_masked],
    "var_var_qk1": _var_var_np.tolist(),
    "var_labels": var_labels,
    "n_numbers": int(n_numbers),
    "n_special": int(n_special),
    "n_vars": int(n_vars),
    "seq_len": int(SEQ_LEN),
    "attention_patterns": _attn_patterns_data,
    "attention_token_labels": _token_labels_ex,
}
with open("weights_plot_data.json", "w") as _f:
    _json.dump(_weights_plot_data, _f)
print("Saved: weights_plot_data.json")

# %%
