# %%
import os
import random
from collections import defaultdict
from dataclasses import asdict

import numpy as np
import einops
from functools import partial
from jaxtyping import Float
import matplotlib.pyplot as plt
from tueplots import bundles
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio
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
TOKEN_STRINGS = list(run_cfg.token_strings)

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

# %% 2. All permutation indices for analysis ----------------------------------
all_idx = torch.arange(len(perm_table), dtype=torch.long)
print(f"Total rows for analysis: {len(all_idx)}")

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

# %% 6. PCA of OV_0 * number embeddings ------------------------------------
with torch.no_grad():
    W_E_numbers = model.embed.W_E[:MOD]
    OV_0 = model.blocks[0].attn.W_V[0] @ model.blocks[0].attn.W_O[0]
    ov_number_emb = (W_E_numbers @ OV_0 + W_E_numbers).detach().cpu().numpy()

coords = PCA(n_components=2, random_state=SEED).fit_transform(ov_number_emb)
fig_pca = go.Figure(
    data=go.Scatter(
        x=coords[:, 0], y=coords[:, 1], mode="markers+text",
        text=[str(i) for i in range(MOD)], textposition="middle center",
        marker=dict(size=8, color="blue")
    )
)
fig_pca.update_layout(title="PCA of OV_0 * number token embeddings", xaxis_title="PC1", yaxis_title="PC2",
                      yaxis=dict(scaleanchor="x", scaleratio=1))
fig_pca.show()


# %% Trying to replicate Neel Nanda's modular addition grokking
from neel_plotly.plot import line

def _build_core_only_sequence(lhs: int, rhs: int) -> torch.Tensor:
    seq = [PAD] * (SEQ_LEN - 4) + [PLUS, lhs, rhs, EQUAL]
    return torch.tensor(seq, device=DEVICE, dtype=torch.long)

all_pairs = [(a, b) for a in range(MOD) for b in range(MOD)]
dataset_tokens = torch.stack([_build_core_only_sequence(a, b) for a, b in all_pairs], dim=0)
dataset_labels = torch.tensor([(a + b) % MOD for a, b in all_pairs], device=DEVICE)

original_logits, cache = model.run_with_cache(dataset_tokens)
print(original_logits.shape)


# %%
L = model.cfg.n_layers - 1

W_E_numbers = model.embed.W_E[:MOD]
print("W_E_numbers shape:", W_E_numbers.shape)
W_neur = W_E_numbers @ model.blocks[L].attn.W_V @ model.blocks[L].attn.W_O @ model.blocks[L].mlp.W_in
print("W_neur", W_neur.shape)
W_logit = model.blocks[L].mlp.W_out @ model.unembed.W_U
print("W_logit", W_logit.shape)
# %%
dest_pos = -1
lhs_pos  = -3
rhs_pos  = -2
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
imshow(cache["pattern", L].mean(dim=0)[:, dest_pos, :], title="Average Attention Pattern per Head (Last Layer)", xaxis="Source", yaxis="Head")

# %%
imshow(cache["pattern", L][5][:, dest_pos, :], title="Average Attention Pattern per Head (Last Layer)", xaxis="Source", yaxis="Head")

# %%
dataset_tokens[5].tolist()
# %%
imshow(cache["pattern", L][:, 0, dest_pos, lhs_pos].reshape(MOD, MOD), title="Attention for Head 0 from a -> =", xaxis="b", yaxis="a")
# %%
imshow(cache["pattern", L][:, 0, dest_pos, rhs_pos].reshape(MOD, MOD), title="Attention for Head 0 from b -> =", xaxis="b", yaxis="a")
# %%
imshow(
    einops.rearrange(cache["pattern", L][:, :, dest_pos, lhs_pos], "(a b) head -> head a b", a=MOD, b=MOD), 
    title="Attention for Head 0 from lhs -> =", xaxis="rhs", yaxis="lhs", facet_col=0)
# %% Plotting neuron activations
print(cache["post", L, "mlp"].shape)
print(neuron_acts.shape)


# %%
imshow(
    einops.rearrange(neuron_acts[:, -5:], "(a b) neuron -> neuron a b", a=MOD, b=MOD), 
    title="First 5 neuron acts", xaxis="b", yaxis="a", facet_col=0)


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
line(U[:, :8].T, title="Principal Components of the embedding", xaxis="Input Vocabulary")

# %%
fourier_basis = []
fourier_basis_names = []
fourier_basis.append(torch.ones(MOD))
fourier_basis_names.append("Constant")
for freq in range(1, MOD//2+1):
    fourier_basis.append(torch.sin(torch.arange(MOD)*2 * torch.pi * freq / MOD))
    fourier_basis_names.append(f"Sin {freq}")
    fourier_basis.append(torch.cos(torch.arange(MOD)*2 * torch.pi * freq / MOD))
    fourier_basis_names.append(f"Cos {freq}")
fourier_basis = torch.stack(fourier_basis, dim=0).to(DEVICE)
fourier_basis = fourier_basis/fourier_basis.norm(dim=-1, keepdim=True)
imshow(fourier_basis, xaxis="Input", yaxis="Component", y=fourier_basis_names)

# %%
line(fourier_basis[:8], xaxis="Input", line_labels=fourier_basis_names[:8], title="First 8 Fourier Components")
line(fourier_basis[25:29], xaxis="Input", line_labels=fourier_basis_names[25:29], title="Middle Fourier Components")
# %%
imshow(fourier_basis @ W_E_numbers, yaxis="Fourier Component", xaxis="Residual Stream", y=fourier_basis_names, title="Embedding in Fourier Basis")

# %%
line((fourier_basis @ W_E_numbers).norm(dim=-1), xaxis="Fourier Component", x=fourier_basis_names, title="Norms of Embedding in Fourier Basis")
# %%
print(W_E_numbers.shape)
print(fourier_basis.shape)
# %%
key_freqs = [0,9,17,18,19,21,25]
key_freq_indices=[0,17,18,33,34,35,36,37,38,41,42,49,50]
fourier_embed = fourier_basis @ W_E_numbers
key_fourier_embed = fourier_embed[key_freq_indices]
print(key_fourier_embed.shape)
imshow(key_fourier_embed @ key_fourier_embed.T, title="Dot Product of embedding of key Fourier Terms")
# %%
line(fourier_basis[[18,34,36,38,42,50]],title="Cos of key freqs", line_labels=[18,34,36,38,42,50])
# %%
line(fourier_basis[[18,34,36,38,42,50]].mean(0),title="Constructive Interference")
# %% Analyse Neurons
imshow(
    einops.rearrange(neuron_acts[:, :5], "(a b) neuron -> neuron a b", a=MOD, b=MOD), 
    title="First 5 neuron acts", xaxis="b", yaxis="a", facet_col=0)
# %%
imshow(
    einops.rearrange(neuron_acts[:, 5], "(a b) -> a b", a=MOD, b=MOD), 
    title="5th neuron act", xaxis="b", yaxis="a")
# %%
imshow(fourier_basis[18][None, :] * fourier_basis[18][:, None], title="Cos 18a * cos 18b")
# %%
imshow(fourier_basis[18][None, :] * fourier_basis[0][:, None], title="Cos 18a * const")
# %%
imshow(fourier_basis @ neuron_acts[:, 5].reshape(MOD, MOD) @ fourier_basis.T, title="2D Fourier Transformer of neuron 5", xaxis="b", yaxis="a", x=fourier_basis_names, y=fourier_basis_names)
# %%
imshow(fourier_basis @ torch.randn_like(neuron_acts[:, 0]).reshape(MOD, MOD) @ fourier_basis.T, title="2D Fourier Transformer of RANDOM", xaxis="b", yaxis="a", x=fourier_basis_names, y=fourier_basis_names)
# %%
fourier_neuron_acts = fourier_basis @ einops.rearrange(neuron_acts, "(a b) neuron -> neuron a b", a=MOD, b=MOD) @ fourier_basis.T
fourier_neuron_acts[:, 0, 0] = 0.
print("fourier_neuron_acts", fourier_neuron_acts.shape)
# %%
neuron_freq_norm = torch.zeros(MOD//2, model.cfg.d_mlp).to(DEVICE)
for freq in range(0, MOD//2):
    for x in [0, 2*(freq+1) - 1, 2*(freq+1)]:
        for y in [0, 2*(freq+1) - 1, 2*(freq+1)]:
            neuron_freq_norm[freq] += fourier_neuron_acts[:, x, y]**2
neuron_freq_norm = neuron_freq_norm / fourier_neuron_acts.pow(2).sum(dim=[-1, -2])[None, :]
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
fig.update_yaxes(tickmode="array", tickvals=list(range(0, MOD // 2)), ticktext=list(range(1, MOD // 2 + 1)))
fig.show()
# %%
line(neuron_freq_norm.max(dim=0).values.sort().values, xaxis="Neuron", title="Max Neuron Frac Explained over Freqs")
# %%
W_logit = model.blocks[L].mlp.W_out @ model.unembed.W_U
print("W_logit", W_logit.shape)
# %%
line((W_logit @ fourier_basis.T).norm(dim=0), x=fourier_basis_names, title="W_logit in the Fourier Basis")
# %%
neurons_9 = neuron_freq_norm[9-1]>0.85
print(neurons_9.shape)
neurons_9.sum()
line((W_logit[neurons_9] @ fourier_basis.T).norm(dim=0), x=fourier_basis_names, title="W_logit for freq 9 neurons in the Fourier Basis")
# %%
freq = 19
W_logit_fourier = W_logit @ fourier_basis
neurons_sin_19 = W_logit_fourier[:, 2*freq-1]
line(neurons_sin_19)
# %%
inputs_sin_19c = neuron_acts @ neurons_sin_19
imshow(fourier_basis @ inputs_sin_19c.reshape(MOD, MOD) @ fourier_basis.T, title="Fourier Heatmap over inputs for sin19c", x=fourier_basis_names, y=fourier_basis_names)
# %% Print every part of the model by name
for name, param in model.named_parameters():
    print(name, param.shape)

# %%
neuron_acts_2d = einops.rearrange(neuron_acts, "(a b) neuron -> neuron a b", a=MOD, b=MOD)
n_neurons = neuron_acts_2d.shape[0]

fig = go.Figure()

for i in range(n_neurons):
    fig.add_trace(go.Heatmap(
        z=utils.to_numpy(neuron_acts_2d[i]),
        colorscale="RdBu",
        zmid=0,
        visible=(i == 0),
        colorbar=dict(title="Activation")
    ))

buttons = [
    dict(
        label=f"Neuron {i}",
        method="update",
        args=[{"visible": [j == i for j in range(n_neurons)]}]
    )
    for i in range(n_neurons)
]

fig.update_layout(
    title="Neuron Activations (use dropdown to select)",
    xaxis_title="b",
    yaxis_title="a",
    updatemenus=[dict(
        active=0,
        buttons=buttons,
        x=0.0,
        xanchor="left",
        y=1.15,
        yanchor="top"
    )]
)
fig.show()

# %% Check the number of neurons that have nonzero activations
nonzero_neurons = 0
for neuron in range(512):
    if neuron_acts_2d[neuron].sum() > 0:
        nonzero_neurons += 1
print(nonzero_neurons)

# %% Find neuron best matching theoretical preactivation for frequency k

TARGET_FREQ = 11

n_grid = torch.arange(MOD, device=DEVICE).float()
m_grid = torch.arange(MOD, device=DEVICE).float()

cos_kn = torch.cos(2 * torch.pi * TARGET_FREQ * n_grid / MOD)[:, None].expand(MOD, MOD)
sin_kn = torch.sin(2 * torch.pi * TARGET_FREQ * n_grid / MOD)[:, None].expand(MOD, MOD)

cos_km = torch.cos(2 * torch.pi * TARGET_FREQ * m_grid / MOD)[None, :].expand(MOD, MOD)
sin_km = torch.sin(2 * torch.pi * TARGET_FREQ * m_grid / MOD)[None, :].expand(MOD, MOD)

basis_2d = torch.stack([cos_kn, sin_kn, cos_km, sin_km], dim=0)
basis_flat = basis_2d.reshape(4, MOD * MOD)

basis_flat = basis_flat / basis_flat.norm(dim=1, keepdim=True)

neuron_pre_acts_flat = neuron_pre_acts.T

neuron_pre_acts_centered = neuron_pre_acts_flat - neuron_pre_acts_flat.mean(dim=1, keepdim=True)

coefficients = neuron_pre_acts_centered @ basis_flat.T

projection = coefficients @ basis_flat

total_var = (neuron_pre_acts_centered ** 2).sum(dim=1)
explained_var = (projection ** 2).sum(dim=1)
explained_ratio = explained_var / (total_var + 1e-8)

best_neuron = explained_ratio.argmax().item()
print(f"\nBest match for frequency k={TARGET_FREQ}: Neuron {best_neuron}")
print(f"Explained variance ratio: {explained_ratio[best_neuron]:.4f}")

top_neurons = explained_ratio.argsort(descending=True)[:10]
print(f"\nTop 10 neurons for frequency k={TARGET_FREQ}:")
for i, neuron_idx in enumerate(top_neurons):
    print(f"  {i+1}. Neuron {neuron_idx.item()}: {explained_ratio[neuron_idx]:.4f}")

# %% Visualize the best matching neuron's preactivation pattern
best_pre_acts_2d = neuron_pre_acts[:, best_neuron].reshape(MOD, MOD)

c_cos_n, c_sin_n, c_cos_m, c_sin_m = coefficients[best_neuron]
alpha_opt = torch.atan2(-c_sin_n, c_cos_n).item()
beta_opt = torch.atan2(-c_sin_m, c_cos_m).item()
amp_n = torch.sqrt(c_cos_n**2 + c_sin_n**2).item()
amp_m = torch.sqrt(c_cos_m**2 + c_sin_m**2).item()

print(f"\nOptimal phases: α = {alpha_opt:.3f} rad, β = {beta_opt:.3f} rad")
print(f"Amplitudes: A_n = {amp_n:.3f}, A_m = {amp_m:.3f}")

theoretical_pattern = (
    amp_n * torch.cos(2 * torch.pi * TARGET_FREQ * n_grid[:, None] / MOD + alpha_opt) +
    amp_m * torch.cos(2 * torch.pi * TARGET_FREQ * m_grid[None, :] / MOD + beta_opt)
).to(DEVICE)

fig = make_subplots(rows=1, cols=2, subplot_titles=[
    f"Neuron {best_neuron} Preactivations",
    f"Theoretical k={TARGET_FREQ} Pattern"
])

fig.add_trace(go.Heatmap(
    z=utils.to_numpy(best_pre_acts_2d),
    colorscale="RdBu",
    zmid=0,
), row=1, col=1)

fig.add_trace(go.Heatmap(
    z=utils.to_numpy(theoretical_pattern),
    colorscale="RdBu", 
    zmid=0,
), row=1, col=2)

fig.update_layout(
    title=f"Best Neuron Match for Frequency k={TARGET_FREQ}",
    xaxis_title="m", yaxis_title="n",
    xaxis2_title="m", yaxis2_title="n",
)
fig.show()

# %% Normalized comparison (max abs value = 1 for both)
actual_np = utils.to_numpy(best_pre_acts_2d)
theoretical_np = utils.to_numpy(theoretical_pattern)

actual_normalized = actual_np / np.abs(actual_np).max()
theoretical_normalized = theoretical_np / np.abs(theoretical_np).max()

import json as _json
_fourier_plot_data = {
    "actual_normalized": actual_normalized.tolist(),
    "theoretical_normalized": theoretical_normalized.tolist(),
    "best_neuron": int(best_neuron),
    "target_freq": int(TARGET_FREQ),
}
with open("fourier_plot_data.json", "w") as _f:
    _json.dump(_fourier_plot_data, _f)
print("Saved: fourier_plot_data.json")
# %%
