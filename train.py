# %%
import os

DEVELOPMENT_MODE = True
IN_GITHUB = os.getenv("GITHUB_ACTIONS") == "true"
try:
    import google.colab

    IN_COLAB = True
    print("Running as a Colab notebook")
except ImportError:
    IN_COLAB = False
    print("Running as a Jupyter notebook - intended for development only!")
    from IPython import get_ipython

    ipython = get_ipython()
    if ipython is not None:
        ipython.run_line_magic("reload_ext", "autoreload")
        ipython.run_line_magic("autoreload", "2")

if IN_COLAB or IN_GITHUB:
    print("Colab/GitHub environment detected; expecting dependencies preinstalled.")

# %%
import json
import math
import multiprocessing as mp
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import einops
import numpy as np
import plotly.express as px
import plotly.io as pio
import torch
import torch.nn as nn
from tqdm.auto import trange
from transformer_lens import HookedTransformer, HookedTransformerConfig

from eval_pools import (
    enumerate_valid_2var_pairs,
    generate_exact_0var_invalid_pair_pool,
    generate_exact_2var_generalization_union_pool,
    generate_special_pool,
)

pio.renderers.default = "notebook"
if IN_COLAB or IN_GITHUB:
    pio.renderers.default = "colab"

run_deterministic = False
SEED = 42


random.seed(SEED)
torch.manual_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


if run_deterministic:
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

    torch.use_deterministic_algorithms(True, warn_only=True)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# %%
MOD = 59  # numbers 0-58
PLUS, EQUAL, PAD = range(MOD, MOD + 3)
A_TOKEN = MOD + 3
VAR_LEN = 12
VARS = list(range(A_TOKEN, A_TOKEN + VAR_LEN))
FIRST_HALF, SECOND_HALF = VARS[: VAR_LEN // 2], VARS[VAR_LEN // 2 :]

RESTRICT_LEFT_HALF_VARS = 2
RESTRICT_RIGHT_HALF_VARS = 2

LEFT_RESTRICT_VARS = VARS[:RESTRICT_LEFT_HALF_VARS]
RIGHT_RESTRICT_VARS = (
    VARS[-RESTRICT_RIGHT_HALF_VARS:] if RESTRICT_RIGHT_HALF_VARS > 0 else []
)
VALID_2VAR_PAIRS = enumerate_valid_2var_pairs(
    VARS, LEFT_RESTRICT_VARS, RIGHT_RESTRICT_VARS
)

VOCAB = A_TOKEN + VAR_LEN
TOKEN_STRINGS = (
    [str(i) for i in range(MOD)]
    + ["+", "=", "<PAD>"]
    + [chr(ord("a") + i) for i in range(26)]
)

SEQ_LEN = 16
BATCH_SIZE = 512
D_MODEL = 128
N_HEADS = 1
N_LAYERS = 2
LR = 1e-3
WD = 0.02
BETAS = (0.9, 0.999)
MAX_STEPS = 30_000  # shorten default so plots render quickly
EVAL_EVERY = 100
PRINT_EVERY = 100
TRAIN_ON_2VAR = True
TWO_VAR_FREQUENCY = 1.0

TRAIN_PAIRS_FRAC = 0.7
NUMS_TRAIN_PAIRS = math.floor(TRAIN_PAIRS_FRAC * MOD * MOD)
print(f"Using {NUMS_TRAIN_PAIRS} distinct (NUM1, NUM2) pairs for training")

TRAIN_FRAC = 1.00

DEVICE = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)
print(DEVICE)
assert DEVICE != torch.device("cpu")
if DEVICE.type == "cuda":
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.get_device_capability(0))
    print(torch.cuda.get_device_properties(0))

USE_AMP = True
AMP_DTYPE_CUDA = "bf16"
AMP_DTYPE_MPS = "fp16"
USE_COMPILE = False


# %%
V, N, TYPES = len(VARS), MOD, 4

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
print("Total permutations:", TOTAL_PERMS)

del perm_tensor

# %%
rows_by_pair: dict[tuple[int, int], list[int]] = defaultdict(list)
for idx, row in enumerate(perm_table):
    pair = (int(row[2].item()), int(row[3].item()))
    rows_by_pair[pair].append(idx)

pair_pool = list(rows_by_pair.keys())

rng = random.Random(SEED)
if NUMS_TRAIN_PAIRS > len(pair_pool):
    raise ValueError(
        f"Requested {NUMS_TRAIN_PAIRS} number-pairs but only {len(pair_pool)} exist"
    )

sel_pairs: set[tuple[int, int]] = set(rng.sample(pair_pool, NUMS_TRAIN_PAIRS))

train_mask = torch.zeros(len(perm_table), dtype=torch.bool)
for pair in sel_pairs:
    for idx in rows_by_pair[pair]:
        var_tok = perm_table[idx][0].item()
        typ = perm_table[idx][1].item()
        if var_tok in LEFT_RESTRICT_VARS and typ == 2:
            continue
        if var_tok in RIGHT_RESTRICT_VARS and typ == 1:
            continue

        if typ == 3:
            if not TRAIN_ON_2VAR:
                continue
            if var_tok in RIGHT_RESTRICT_VARS:
                continue
            if TWO_VAR_FREQUENCY < 1.0:
                if random.Random(SEED + idx).random() >= TWO_VAR_FREQUENCY:
                    continue

        train_mask[idx] = True

train_idx = train_mask.nonzero(as_tuple=False).squeeze()
test_idx = (~train_mask).nonzero(as_tuple=False).squeeze()

if TRAIN_FRAC < 1.0:
    num_keep = max(1, int(len(train_idx) * TRAIN_FRAC))
    perm = torch.randperm(len(train_idx), generator=torch.Generator().manual_seed(SEED))
    keep_idx = train_idx[perm[:num_keep]]
    drop_idx = train_idx[perm[num_keep:]]
    train_idx = keep_idx
    test_idx = torch.cat((test_idx, drop_idx))

print(f"Training set size: {len(train_idx)} | Test set size: {len(test_idx)}")

vars_in_train = set(perm_table[train_idx][:, 0].tolist())
nums_in_train = set(perm_table[train_idx][:, 2].tolist())
nums_in_train.update(perm_table[train_idx][:, 3].tolist())

missing_vars = [v for v in VARS if v not in vars_in_train]
missing_nums = [n for n in range(MOD) if n not in nums_in_train]
if missing_vars or missing_nums:
    raise ValueError(
        "Training set missing vars or nums; increase pair counts or TRAIN_FRAC"
    )

rows_unseen_pairs = [
    idx
    for idx, row in enumerate(perm_table)
    if (row[2].item(), row[3].item()) not in sel_pairs
]

rows_var_swap = []
for idx, row in enumerate(perm_table):
    var_tok, typ = row[0].item(), row[1].item()
    if (typ == 1 and var_tok in RIGHT_RESTRICT_VARS) or (
        typ == 2 and var_tok in LEFT_RESTRICT_VARS
    ):
        rows_var_swap.append(idx)

unseen_pairs_idx = torch.tensor(rows_unseen_pairs, dtype=torch.long)
var_swap_idx = torch.tensor(rows_var_swap, dtype=torch.long)
both_idx = torch.tensor(
    list(set(unseen_pairs_idx.tolist()) & set(var_swap_idx.tolist())),
    dtype=torch.long,
)

print(
    f"Unseen pairs rows: {len(unseen_pairs_idx)} | Variable swap rows: {len(var_swap_idx)}"
)
print(f"Both conditions rows (unseen pair + swapped var): {len(both_idx)}")

_train_idx_set = set(train_idx.tolist())
_test_idx_set = set(test_idx.tolist())

_zero_var_rows = [idx for idx, row in enumerate(perm_table) if row[1].item() == 0]
_one_var_rows = [idx for idx, row in enumerate(perm_table) if row[1].item() in (1, 2)]

_zero_var_train_rows = [idx for idx in _zero_var_rows if idx in _train_idx_set]
_zero_var_test_rows = [idx for idx in _zero_var_rows if idx in _test_idx_set]

_one_var_train_rows = [idx for idx in _one_var_rows if idx in _train_idx_set]

_rng = random.Random(SEED + 1)
_rng.shuffle(_zero_var_train_rows)
_rng.shuffle(_zero_var_test_rows)
_rng.shuffle(_one_var_train_rows)

zero_var_train_idx = torch.tensor(_zero_var_train_rows[:2000], dtype=torch.long)
zero_var_test_idx = torch.tensor(_zero_var_test_rows[:2000], dtype=torch.long)
one_var_train_idx = torch.tensor(_one_var_train_rows[:2000], dtype=torch.long)

print(
    f"Addition-core eval sets → 0-var train: {len(zero_var_train_idx)} | "
    f"0-var add-restricted: {len(zero_var_test_idx)} | "
    f"1-var train: {len(one_var_train_idx)}"
)
# %%


def build_sequence(row: torch.Tensor):
    var_tok, typ, num1, num2 = row.tolist()

    if typ == 2:
        var_val = num2
    elif typ == 1 or typ == 3:
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

    if typ == 3:
        var1, var2 = random.choice(VALID_2VAR_PAIRS)
        assignments = [[var1, num1], [var2, num2]]

        if n_assignments < 2:
            n_assignments = 2

        num_distractors = n_assignments - 2
        extra_vars = [v for v in VARS if v != var1 and v != var2]
    else:
        num_distractors = n_assignments - 1
        extra_vars = [v for v in VARS if v != var_tok]

    random.shuffle(extra_vars)
    for _ in range(num_distractors):
        v = extra_vars.pop()
        assignments.append([v, random.randrange(MOD)])

    random.shuffle(assignments)

    remaining_pads = SEQ_LEN - (assignment_len * n_assignments + core_len)

    gaps = [0] * (n_assignments + 1)
    for _ in range(remaining_pads):
        gaps[random.randint(0, n_assignments)] += 1

    prefix = []
    for i, seg in enumerate(assignments):
        prefix.extend([PAD] * gaps[i])
        prefix.extend(seg)
    prefix.extend([PAD] * gaps[-1])

    if typ == 0:
        lhs_tok, rhs_tok = num1, num2
    elif typ == 1:
        lhs_tok, rhs_tok = var_tok, num2
    elif typ == 2:
        lhs_tok, rhs_tok = num1, var_tok
    else:
        lhs_tok, rhs_tok = var1, var2

    core = [PLUS, lhs_tok, rhs_tok, EQUAL]
    label = (num1 + num2) % MOD

    tok = torch.tensor(prefix + core, dtype=torch.long)
    assert tok.shape[0] == SEQ_LEN, "Generated sequence has incorrect length"
    return tok, label


if __name__ == "__main__" and DEVELOPMENT_MODE:
    _sample_tokens, _sample_label = build_sequence(perm_table[0])
    _sample_str = " ".join(TOKEN_STRINGS[t] for t in _sample_tokens.tolist())
    print(f"[debug] build_sequence sample: {_sample_str} | label={_sample_label}")


def _precompute_split(idx_tensor):
    if len(idx_tensor) == 0:
        return torch.empty(0, SEQ_LEN, dtype=torch.long), torch.empty(
            0, dtype=torch.long
        )
    tok_list, lab_list = zip(*(build_sequence(perm_table[i]) for i in idx_tensor))
    return torch.stack(list(tok_list)), torch.tensor(list(lab_list))


zero_var_train_tok, zero_var_train_lab = _precompute_split(zero_var_train_idx)
zero_var_test_tok, zero_var_test_lab = _precompute_split(zero_var_test_idx)
one_var_train_tok, one_var_train_lab = _precompute_split(one_var_train_idx)


def get_batch_zero_var_train():
    sel = torch.randint(0, zero_var_train_tok.shape[0], (BATCH_SIZE,))
    return zero_var_train_tok[sel].to(DEVICE), zero_var_train_lab[sel].to(DEVICE)


def get_batch_zero_var_test():
    sel = torch.randint(0, zero_var_test_tok.shape[0], (BATCH_SIZE,))
    return zero_var_test_tok[sel].to(DEVICE), zero_var_test_lab[sel].to(DEVICE)


def get_batch_one_var_train():
    sel = torch.randint(0, one_var_train_tok.shape[0], (BATCH_SIZE,))
    return one_var_train_tok[sel].to(DEVICE), one_var_train_lab[sel].to(DEVICE)


# %%
NUM_DATA_WORKERS = 16
PREFETCH_QUEUE_SIZE = 256


class _BatchPrefetcher:
    def __init__(
        self, pool_indices: torch.Tensor, batch_size: int, seed: int, name: str
    ):
        self.pool_indices = pool_indices
        self.batch_size = batch_size
        self.seed = seed
        self.name = name
        try:
            self.ctx = mp.get_context("fork")
        except ValueError:
            self.ctx = mp.get_context()
        self.q = self.ctx.Queue(maxsize=PREFETCH_QUEUE_SIZE)
        self.procs: list[mp.Process] = []
        self._start()

    def _start(self):
        def worker_main(q, pool_np, batch_size, seed_base):
            wid = int(os.getpid()) & 0xFFFF
            rng_np = np.random.default_rng([seed_base, wid])
            random.seed((seed_base << 16) ^ wid)
            while True:
                sel = rng_np.integers(0, len(pool_np), size=batch_size)
                toks = np.empty((batch_size, SEQ_LEN), dtype=np.int64)
                labs = np.empty((batch_size,), dtype=np.int64)
                for bi, pi in enumerate(sel):
                    row_idx = int(pool_np[pi])
                    tok_t, lab = build_sequence(perm_table[row_idx])
                    toks[bi, :] = tok_t.numpy()
                    labs[bi] = int(lab)
                q.put((toks, labs), block=True)

        pool_np = self.pool_indices.cpu().numpy()
        for _ in range(NUM_DATA_WORKERS):
            p = self.ctx.Process(
                target=worker_main, args=(self.q, pool_np, self.batch_size, self.seed)
            )
            p.daemon = True
            p.start()
            self.procs.append(p)

    def get(self):
        toks_np, labs_np = self.q.get()
        toks = torch.from_numpy(toks_np).to(DEVICE)
        labs = torch.from_numpy(labs_np).to(DEVICE)
        return toks, labs

    def close(self):
        for proc in self.procs:
            if proc.is_alive():
                proc.terminate()
            proc.join(timeout=1)
        self.procs.clear()


_train_prefetcher: _BatchPrefetcher | None = None
_test_prefetcher: _BatchPrefetcher | None = None


def _ensure_prefetchers():
    global _train_prefetcher, _test_prefetcher
    if _train_prefetcher is None:
        _train_prefetcher = _BatchPrefetcher(train_idx, BATCH_SIZE, SEED, name="train")
    if _test_prefetcher is None:
        _test_prefetcher = _BatchPrefetcher(test_idx, BATCH_SIZE, SEED + 1, name="test")


def _shutdown_prefetchers():
    global _train_prefetcher, _test_prefetcher
    for prefetcher in (_train_prefetcher, _test_prefetcher):
        if prefetcher is not None:
            prefetcher.close()
    _train_prefetcher = None
    _test_prefetcher = None


def get_batch(split: str):
    _ensure_prefetchers()
    if split == "train":
        return _train_prefetcher.get()
    else:
        return _test_prefetcher.get()


def get_batch_from_indices(index_tensor: torch.Tensor):
    sel = index_tensor[torch.randint(0, len(index_tensor), (BATCH_SIZE,))]
    tok, lab = zip(*(build_sequence(perm_table[i]) for i in sel))
    return torch.stack(tok).to(DEVICE), torch.tensor(lab, device=DEVICE)


_typ3_rows = (perm_table[:, 1] == 3).nonzero(as_tuple=False).squeeze(-1).tolist()

_right_restrict_set = set(RIGHT_RESTRICT_VARS)

_two_var_train_rows = [i for i in _typ3_rows if i in _train_idx_set]
_two_var_test_rows = [
    i
    for i in _typ3_rows
    if (int(perm_table[i, 2].item()), int(perm_table[i, 3].item())) not in sel_pairs
    and int(perm_table[i, 0].item()) not in _right_restrict_set
]

_two_var_rng = random.Random(SEED + 2)
_two_var_rng.shuffle(_two_var_train_rows)
_two_var_rng.shuffle(_two_var_test_rows)

two_var_train_idx = torch.tensor(_two_var_train_rows[:2000], dtype=torch.long)
two_var_test_idx = torch.tensor(_two_var_test_rows[:2000], dtype=torch.long)

two_var_train_tok, two_var_train_lab = _precompute_split(two_var_train_idx)
two_var_test_tok, two_var_test_lab = _precompute_split(two_var_test_idx)

print(
    f"Addition-core eval set → 2-var train: {two_var_train_tok.shape[0]} | "
    f"2-var test: {two_var_test_tok.shape[0]}"
)


def _assert_two_var_pool(
    tokens: torch.Tensor,
    idx_tensor: torch.Tensor,
    expected_pair_in_sel: bool,
    sel_pairs_ref: set[tuple[int, int]],
) -> None:
    left_set = set(LEFT_RESTRICT_VARS)
    right_set = set(RIGHT_RESTRICT_VARS)
    for t in tokens:
        lhs, rhs = int(t[-3].item()), int(t[-2].item())
        assert lhs not in right_set, f"LHS var {lhs} is in RIGHT_RESTRICT_VARS"
        assert rhs not in left_set, f"RHS var {rhs} is in LEFT_RESTRICT_VARS"
        assert lhs != rhs, "LHS and RHS variables must differ"
    for i in idx_tensor.tolist():
        row = perm_table[i]
        pair = (int(row[2].item()), int(row[3].item()))
        assert (pair in sel_pairs_ref) is expected_pair_in_sel, (
            f"perm_table row {i} pair={pair} in_sel={(pair in sel_pairs_ref)} "
            f"but expected in_sel={expected_pair_in_sel}"
        )


if len(two_var_train_idx) > 0:
    _assert_two_var_pool(
        two_var_train_tok,
        two_var_train_idx,
        expected_pair_in_sel=True,
        sel_pairs_ref=sel_pairs,
    )
if len(two_var_test_idx) > 0:
    _assert_two_var_pool(
        two_var_test_tok,
        two_var_test_idx,
        expected_pair_in_sel=False,
        sel_pairs_ref=sel_pairs,
    )


def get_batch_two_var_train():
    sel = torch.randint(0, two_var_train_tok.shape[0], (BATCH_SIZE,))
    return two_var_train_tok[sel].to(DEVICE), two_var_train_lab[sel].to(DEVICE)


def get_batch_two_var_test():
    sel = torch.randint(0, two_var_test_tok.shape[0], (BATCH_SIZE,))
    return two_var_test_tok[sel].to(DEVICE), two_var_test_lab[sel].to(DEVICE)


one_var_add_restricted_tok, one_var_add_restricted_lab = generate_special_pool(
    "1var_invalid_pair_valid_var",
    MOD,
    VOCAB,
    sel_pairs,
    LEFT_RESTRICT_VARS,
    RIGHT_RESTRICT_VARS,
    seq_len=SEQ_LEN,
    pool_size=2000,
    seed=SEED + 100,
)
one_var_var_restricted_tok, one_var_var_restricted_lab = generate_special_pool(
    "1var_valid_pair_invalid_var",
    MOD,
    VOCAB,
    sel_pairs,
    LEFT_RESTRICT_VARS,
    RIGHT_RESTRICT_VARS,
    seq_len=SEQ_LEN,
    pool_size=2000,
    seed=SEED + 101,
)
two_var_var_restricted_1_tok, two_var_var_restricted_1_lab = generate_special_pool(
    "2var_valid_pair_1_invalid_var",
    MOD,
    VOCAB,
    sel_pairs,
    LEFT_RESTRICT_VARS,
    RIGHT_RESTRICT_VARS,
    seq_len=SEQ_LEN,
    pool_size=2000,
    seed=SEED + 102,
)
two_var_var_restricted_2_tok, two_var_var_restricted_2_lab = generate_special_pool(
    "2var_valid_pair_2_invalid_vars",
    MOD,
    VOCAB,
    sel_pairs,
    LEFT_RESTRICT_VARS,
    RIGHT_RESTRICT_VARS,
    seq_len=SEQ_LEN,
    pool_size=2000,
    seed=SEED + 103,
)
print(
    f"Specialized eval pools → "
    f"1-var add-restricted: {one_var_add_restricted_tok.shape[0]} | "
    f"1-var var-restricted: {one_var_var_restricted_tok.shape[0]} | "
    f"2-var var-restricted (1): {two_var_var_restricted_1_tok.shape[0]} | "
    f"2-var var-restricted (2): {two_var_var_restricted_2_tok.shape[0]}"
)


def get_batch_one_var_add_restricted():
    sel = torch.randint(0, one_var_add_restricted_tok.shape[0], (BATCH_SIZE,))
    return one_var_add_restricted_tok[sel].to(DEVICE), one_var_add_restricted_lab[
        sel
    ].to(DEVICE)


def get_batch_one_var_var_restricted():
    sel = torch.randint(0, one_var_var_restricted_tok.shape[0], (BATCH_SIZE,))
    return one_var_var_restricted_tok[sel].to(DEVICE), one_var_var_restricted_lab[
        sel
    ].to(DEVICE)


def get_batch_two_var_var_restricted_1():
    sel = torch.randint(0, two_var_var_restricted_1_tok.shape[0], (BATCH_SIZE,))
    return two_var_var_restricted_1_tok[sel].to(DEVICE), two_var_var_restricted_1_lab[
        sel
    ].to(DEVICE)


def get_batch_two_var_var_restricted_2():
    sel = torch.randint(0, two_var_var_restricted_2_tok.shape[0], (BATCH_SIZE,))
    return two_var_var_restricted_2_tok[sel].to(DEVICE), two_var_var_restricted_2_lab[
        sel
    ].to(DEVICE)


# %%
hyperparams = {
    "SEQ_LEN": SEQ_LEN,
    "BATCH_SIZE": BATCH_SIZE,
    "D_MODEL": D_MODEL,
    "N_HEADS": N_HEADS,
    "N_LAYERS": N_LAYERS,
    "LR": LR,
    "WD": WD,
    "MAX_STEPS": MAX_STEPS,
    "EVAL_EVERY": EVAL_EVERY,
    "MOD": MOD,
    "VOCAB": VOCAB,
    "PLUS_ID": PLUS,
    "EQUAL_ID": EQUAL,
    "PAD_ID": PAD,
    "A_TOKEN_ID": A_TOKEN,
    "VAR_LEN": VAR_LEN,
    "VARS": VARS,
    "TOKEN_STRINGS": TOKEN_STRINGS,
    "NUMS_TRAIN_PAIRS": NUMS_TRAIN_PAIRS,
    "TRAIN_PAIRS_FRAC": TRAIN_PAIRS_FRAC,
    "TWO_VAR_FREQUENCY": TWO_VAR_FREQUENCY,
    "SEED": SEED,
    "RESTRICT_RIGHT_HALF_VARS": RESTRICT_RIGHT_HALF_VARS,
    "run_deterministic": run_deterministic,
}


def freeze_biases(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if "b_" in name:
            param.requires_grad = False


def rebuild_data_splits(
    train_pairs_frac: float,
    two_var_frequency: float = TWO_VAR_FREQUENCY,
):
    global train_idx, test_idx, unseen_pairs_idx, var_swap_idx, both_idx
    global sel_pairs
    global zero_var_train_idx, zero_var_test_idx, one_var_train_idx
    global zero_var_train_tok, zero_var_train_lab
    global zero_var_test_tok, zero_var_test_lab
    global one_var_train_tok, one_var_train_lab
    global two_var_train_idx, two_var_test_idx
    global two_var_train_tok, two_var_train_lab
    global two_var_test_tok, two_var_test_lab
    global one_var_add_restricted_tok, one_var_add_restricted_lab
    global one_var_var_restricted_tok, one_var_var_restricted_lab
    global two_var_var_restricted_1_tok, two_var_var_restricted_1_lab
    global two_var_var_restricted_2_tok, two_var_var_restricted_2_lab

    nums_train_pairs = math.floor(train_pairs_frac * MOD * MOD)
    print(
        f"[rebuild] Using {nums_train_pairs} distinct (NUM1, NUM2) pairs for training"
    )

    local_rng = random.Random(SEED)
    local_sel_pairs: set[tuple[int, int]] = set(
        local_rng.sample(pair_pool, nums_train_pairs)
    )

    local_train_mask = torch.zeros(len(perm_table), dtype=torch.bool)
    for pair in local_sel_pairs:
        for idx in rows_by_pair[pair]:
            var_tok = perm_table[idx][0].item()
            typ = perm_table[idx][1].item()
            if var_tok in LEFT_RESTRICT_VARS and typ == 2:
                continue
            if var_tok in RIGHT_RESTRICT_VARS and typ == 1:
                continue
            if typ == 3:
                if not TRAIN_ON_2VAR:
                    continue
                if var_tok in RIGHT_RESTRICT_VARS:
                    continue
                if two_var_frequency < 1.0:
                    if random.Random(SEED + idx).random() >= two_var_frequency:
                        continue
            local_train_mask[idx] = True

    local_train_idx = local_train_mask.nonzero(as_tuple=False).squeeze()
    local_test_idx = (~local_train_mask).nonzero(as_tuple=False).squeeze()

    if TRAIN_FRAC < 1.0:
        num_keep = max(1, int(len(local_train_idx) * TRAIN_FRAC))
        perm = torch.randperm(
            len(local_train_idx), generator=torch.Generator().manual_seed(SEED)
        )
        keep_idx = local_train_idx[perm[:num_keep]]
        drop_idx = local_train_idx[perm[num_keep:]]
        local_train_idx = keep_idx
        local_test_idx = torch.cat((local_test_idx, drop_idx))

    rows_unseen = [
        idx
        for idx, row in enumerate(perm_table)
        if (row[2].item(), row[3].item()) not in local_sel_pairs
    ]
    rows_swap = []
    for idx, row in enumerate(perm_table):
        var_tok, typ = row[0].item(), row[1].item()
        if (typ == 1 and var_tok in RIGHT_RESTRICT_VARS) or (
            typ == 2 and var_tok in LEFT_RESTRICT_VARS
        ):
            rows_swap.append(idx)

    local_unseen_idx = torch.tensor(rows_unseen, dtype=torch.long)
    local_swap_idx = torch.tensor(rows_swap, dtype=torch.long)
    local_both_idx = torch.tensor(
        list(set(local_unseen_idx.tolist()) & set(local_swap_idx.tolist())),
        dtype=torch.long,
    )

    train_idx = local_train_idx
    test_idx = local_test_idx
    sel_pairs = local_sel_pairs
    unseen_pairs_idx = local_unseen_idx
    var_swap_idx = local_swap_idx
    both_idx = local_both_idx

    local_train_idx_set = set(local_train_idx.tolist())
    local_test_idx_set = set(local_test_idx.tolist())

    local_zero_var_train_rows = [i for i in _zero_var_rows if i in local_train_idx_set]
    local_zero_var_test_rows = [i for i in _zero_var_rows if i in local_test_idx_set]
    local_one_var_train_rows = [i for i in _one_var_rows if i in local_train_idx_set]
    local_rng_eval = random.Random(SEED + 1)
    local_rng_eval.shuffle(local_zero_var_train_rows)
    local_rng_eval.shuffle(local_zero_var_test_rows)
    local_rng_eval.shuffle(local_one_var_train_rows)

    zero_var_train_idx = torch.tensor(
        local_zero_var_train_rows[:2000], dtype=torch.long
    )
    zero_var_test_idx = torch.tensor(local_zero_var_test_rows[:2000], dtype=torch.long)
    one_var_train_idx = torch.tensor(local_one_var_train_rows[:2000], dtype=torch.long)
    zero_var_train_tok, zero_var_train_lab = _precompute_split(zero_var_train_idx)
    zero_var_test_tok, zero_var_test_lab = _precompute_split(zero_var_test_idx)
    one_var_train_tok, one_var_train_lab = _precompute_split(one_var_train_idx)

    local_typ3_rows = (
        (perm_table[:, 1] == 3).nonzero(as_tuple=False).squeeze(-1).tolist()
    )
    local_right_restrict_set = set(RIGHT_RESTRICT_VARS)
    local_two_var_train_rows = [i for i in local_typ3_rows if i in local_train_idx_set]
    local_two_var_test_rows = [
        i
        for i in local_typ3_rows
        if (int(perm_table[i, 2].item()), int(perm_table[i, 3].item()))
        not in local_sel_pairs
        and int(perm_table[i, 0].item()) not in local_right_restrict_set
    ]
    local_two_var_rng = random.Random(SEED + 2)
    local_two_var_rng.shuffle(local_two_var_train_rows)
    local_two_var_rng.shuffle(local_two_var_test_rows)
    two_var_train_idx = torch.tensor(local_two_var_train_rows[:2000], dtype=torch.long)
    two_var_test_idx = torch.tensor(local_two_var_test_rows[:2000], dtype=torch.long)
    two_var_train_tok, two_var_train_lab = _precompute_split(two_var_train_idx)
    two_var_test_tok, two_var_test_lab = _precompute_split(two_var_test_idx)
    if len(two_var_train_idx) > 0:
        _assert_two_var_pool(
            two_var_train_tok,
            two_var_train_idx,
            expected_pair_in_sel=True,
            sel_pairs_ref=local_sel_pairs,
        )
    if len(two_var_test_idx) > 0:
        _assert_two_var_pool(
            two_var_test_tok,
            two_var_test_idx,
            expected_pair_in_sel=False,
            sel_pairs_ref=local_sel_pairs,
        )

    one_var_add_restricted_tok, one_var_add_restricted_lab = generate_special_pool(
        "1var_invalid_pair_valid_var",
        MOD,
        VOCAB,
        local_sel_pairs,
        LEFT_RESTRICT_VARS,
        RIGHT_RESTRICT_VARS,
        seq_len=SEQ_LEN,
        pool_size=2000,
        seed=SEED + 100,
    )
    one_var_var_restricted_tok, one_var_var_restricted_lab = generate_special_pool(
        "1var_valid_pair_invalid_var",
        MOD,
        VOCAB,
        local_sel_pairs,
        LEFT_RESTRICT_VARS,
        RIGHT_RESTRICT_VARS,
        seq_len=SEQ_LEN,
        pool_size=2000,
        seed=SEED + 101,
    )
    two_var_var_restricted_1_tok, two_var_var_restricted_1_lab = generate_special_pool(
        "2var_valid_pair_1_invalid_var",
        MOD,
        VOCAB,
        local_sel_pairs,
        LEFT_RESTRICT_VARS,
        RIGHT_RESTRICT_VARS,
        seq_len=SEQ_LEN,
        pool_size=2000,
        seed=SEED + 102,
    )
    two_var_var_restricted_2_tok, two_var_var_restricted_2_lab = generate_special_pool(
        "2var_valid_pair_2_invalid_vars",
        MOD,
        VOCAB,
        local_sel_pairs,
        LEFT_RESTRICT_VARS,
        RIGHT_RESTRICT_VARS,
        seq_len=SEQ_LEN,
        pool_size=2000,
        seed=SEED + 103,
    )

    _shutdown_prefetchers()

    print(f"[rebuild] Training set: {len(train_idx)} | Test set: {len(test_idx)}")
    print(
        f"[rebuild] Unseen pairs: {len(unseen_pairs_idx)} | Var swap: {len(var_swap_idx)}"
    )
    print(
        f"[rebuild] 2-var train: {len(two_var_train_idx)} | "
        f"2-var test: {len(two_var_test_idx)}"
    )

    return nums_train_pairs


def _accuracy_on_tokens(
    model: nn.Module,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int = 4096,
) -> float:
    if len(tokens) == 0:
        return float("nan")

    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for start in range(0, len(tokens), batch_size):
            end = min(start + batch_size, len(tokens))
            src = tokens[start:end].to(DEVICE)
            tgt = labels[start:end].to(DEVICE)
            pred = model(src)[:, -1, :MOD].argmax(-1)
            correct += (pred == tgt).sum().item()
            total += len(tgt)
    if was_training:
        model.train()
    return correct / total


def evaluate_final_exact_generalization(model: nn.Module) -> dict[str, float]:
    zero_tok, zero_lab = generate_exact_0var_invalid_pair_pool(
        MOD,
        VOCAB,
        sel_pairs,
        seq_len=SEQ_LEN,
        seed=SEED + 200,
    )
    two_tok, two_lab = generate_exact_2var_generalization_union_pool(
        MOD,
        VOCAB,
        sel_pairs,
        LEFT_RESTRICT_VARS,
        RIGHT_RESTRICT_VARS,
        seq_len=SEQ_LEN,
        seed=SEED + 201,
    )

    return {
        "final_exact_0var_add_restricted_acc": _accuracy_on_tokens(
            model, zero_tok, zero_lab
        ),
        "final_exact_0var_add_restricted_n": len(zero_lab),
        "final_exact_2var_union_acc": _accuracy_on_tokens(model, two_tok, two_lab),
        "final_exact_2var_union_n": len(two_lab),
    }


# %%
class LocalSummary(dict):
    def __init__(self, path: Path):
        super().__init__()
        self.path = path

    def update(self, values, **kwargs):
        super().update(values, **kwargs)
        self.path.write_text(json.dumps(dict(self), indent=2, default=str))


class LocalRun:
    """Small run logger with the subset of methods this training loop needs."""

    def __init__(self, config_overrides: dict | None = None):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.name = str((config_overrides or {}).get("RUN_NAME", f"local-{timestamp}"))
        self.id = self.name
        self.config = dict(config_overrides or {})
        self.output_dir = Path(
            self.config.get("OUTPUT_DIR", Path("outputs") / "training_runs" / self.name)
        )
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.summary = LocalSummary(self.output_dir / "summary.json")

    def __enter__(self) -> "LocalRun":
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "config.json").write_text(
            json.dumps(self.config, indent=2, default=str)
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def watch(self, *_args, **_kwargs) -> None:
        return None

    def log(self, values: dict, step: int | None = None) -> None:
        row = dict(values)
        if step is not None:
            row["step"] = step
        with self.metrics_path.open("a") as handle:
            handle.write(json.dumps(row, default=str) + "\n")

    def checkpoint_path(self, filename: str) -> Path:
        return self.checkpoint_dir / filename


def train(config_overrides: dict | None = None):
    global model  # used by probe helpers if enabled

    with LocalRun(config_overrides) as run:
        cfg_wb = run.config

        learning_rate = float(cfg_wb.get("LR", LR))
        weight_decay = float(cfg_wb.get("WD", WD))
        max_steps = int(cfg_wb.get("MAX_STEPS", MAX_STEPS))
        train_pairs_frac = float(cfg_wb.get("TRAIN_PAIRS_FRAC", TRAIN_PAIRS_FRAC))
        two_var_frequency = float(cfg_wb.get("TWO_VAR_FREQUENCY", TWO_VAR_FREQUENCY))

        nums_train_pairs = rebuild_data_splits(train_pairs_frac, two_var_frequency)

        base_cfg = {
            k: v
            for k, v in hyperparams.items()
            if k
            not in (
                "LR",
                "WD",
                "MAX_STEPS",
                "NUMS_TRAIN_PAIRS",
                "TRAIN_PAIRS_FRAC",
                "TWO_VAR_FREQUENCY",
            )
        }
        base_cfg.update(
            {
                "LR": learning_rate,
                "WD": weight_decay,
                "MAX_STEPS": max_steps,
                "TRAIN_PAIRS_FRAC": train_pairs_frac,
                "TWO_VAR_FREQUENCY": two_var_frequency,
                "NUMS_TRAIN_PAIRS": nums_train_pairs,
            }
        )
        cfg_wb.update(base_cfg)
        (run.output_dir / "config.json").write_text(
            json.dumps(cfg_wb, indent=2, default=str)
        )

        cfg = HookedTransformerConfig(
            n_layers=N_LAYERS,
            n_heads=N_HEADS,
            d_model=D_MODEL,
            d_head=D_MODEL // N_HEADS,
            d_mlp=4 * D_MODEL,
            n_ctx=SEQ_LEN,
            init_weights=True,
            d_vocab=VOCAB,
            d_vocab_out=MOD,
            act_fn="relu",
            device=DEVICE,
            normalization_type=None,
            seed=SEED,
        )
        model = HookedTransformer(cfg).to(DEVICE)

        REMOVE_LAYER_0_MLP = True
        if REMOVE_LAYER_0_MLP:
            with torch.no_grad():
                model.blocks[0].mlp.W_in.zero_()
                model.blocks[0].mlp.W_out.zero_()
                model.blocks[0].mlp.b_in.zero_()
                model.blocks[0].mlp.b_out.zero_()
            model.blocks[0].mlp.W_in.requires_grad = False
            model.blocks[0].mlp.W_out.requires_grad = False
            model.blocks[0].mlp.b_in.requires_grad = False
            model.blocks[0].mlp.b_out.requires_grad = False

        freeze_biases(model)
        if USE_COMPILE:
            model = torch.compile(model, mode="reduce-overhead")
        model.train()
        run.watch(model, log="all", log_freq=EVAL_EVERY)

        torch.set_grad_enabled(True)
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=BETAS,
            fused=(DEVICE.type == "cuda"),
        )
        loss_fn = nn.CrossEntropyLoss()

        device_type = (
            "cuda"
            if DEVICE.type == "cuda"
            else ("mps" if DEVICE.type == "mps" else "cpu")
        )
        amp_enabled = USE_AMP and device_type == "cuda"
        if device_type == "cuda":
            amp_dtype = torch.bfloat16 if AMP_DTYPE_CUDA == "bf16" else torch.float16
            scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        else:
            amp_dtype = torch.float16
            scaler = None

        train_losses, test_accs, swap_accs, unseen_accs, both_accs, steps_list = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        zero_var_train_accs, zero_var_addition_restricted_accs = [], []
        (
            one_var_train_accs,
            one_var_addition_restricted_accs,
            one_var_variable_restricted_accs,
        ) = [], [], []
        two_var_train_accs, two_var_addition_restricted_accs = [], []
        two_var_variable_restricted_1_accs, two_var_variable_restricted_2_accs = [], []

        for step in trange(1, max_steps + 1, desc="train", leave=False):
            src, tgt = get_batch("train")

            with torch.autocast(
                device_type=device_type, dtype=amp_dtype, enabled=amp_enabled
            ):
                logits = model(src)

            logits_step = logits[:, -1, :MOD]
            loss = loss_fn(logits_step.float(), tgt)
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if scaler is not None:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            else:
                opt.step()
                opt.zero_grad(set_to_none=True)

            if (step - 1) % EVAL_EVERY == 0:
                model.eval()
                with torch.no_grad():
                    src_te, tgt_te = get_batch("test")
                    with torch.autocast(
                        device_type=device_type, dtype=amp_dtype, enabled=amp_enabled
                    ):
                        acc_test = (
                            (model(src_te)[:, -1, :MOD].argmax(-1) == tgt_te)
                            .float()
                            .mean()
                            .item()
                        )

                    acc_swap = float("nan")
                    acc_unseen = float("nan")
                    acc_both = float("nan")
                    acc_zero_var_train = float("nan")
                    acc_zero_var_addition_restricted = float("nan")
                    acc_one_var_train = float("nan")
                    acc_one_var_addition_restricted = float("nan")
                    acc_one_var_variable_restricted = float("nan")
                    acc_two_var_train = float("nan")
                    acc_two_var_addition_restricted = float("nan")
                    acc_two_var_variable_restricted_1 = float("nan")
                    acc_two_var_variable_restricted_2 = float("nan")

                    if len(var_swap_idx) > 0:
                        src_swap, tgt_swap = get_batch_from_indices(var_swap_idx)
                        with torch.autocast(
                            device_type=device_type,
                            dtype=amp_dtype,
                            enabled=amp_enabled,
                        ):
                            acc_swap = (
                                (model(src_swap)[:, -1, :MOD].argmax(-1) == tgt_swap)
                                .float()
                                .mean()
                                .item()
                            )

                    if len(unseen_pairs_idx) > 0:
                        src_unseen, tgt_unseen = get_batch_from_indices(
                            unseen_pairs_idx
                        )
                        with torch.autocast(
                            device_type=device_type,
                            dtype=amp_dtype,
                            enabled=amp_enabled,
                        ):
                            acc_unseen = (
                                (
                                    model(src_unseen)[:, -1, :MOD].argmax(-1)
                                    == tgt_unseen
                                )
                                .float()
                                .mean()
                                .item()
                            )

                    if len(both_idx) > 0:
                        src_both, tgt_both = get_batch_from_indices(both_idx)
                        with torch.autocast(
                            device_type=device_type,
                            dtype=amp_dtype,
                            enabled=amp_enabled,
                        ):
                            acc_both = (
                                (model(src_both)[:, -1, :MOD].argmax(-1) == tgt_both)
                                .float()
                                .mean()
                                .item()
                            )

                    if len(zero_var_train_tok) > 0:
                        src_b, tgt_b = get_batch_zero_var_train()
                        with torch.autocast(
                            device_type=device_type,
                            dtype=amp_dtype,
                            enabled=amp_enabled,
                        ):
                            acc_zero_var_train = (
                                (model(src_b)[:, -1, :MOD].argmax(-1) == tgt_b)
                                .float()
                                .mean()
                                .item()
                            )

                    if len(zero_var_test_tok) > 0:
                        src_b, tgt_b = get_batch_zero_var_test()
                        with torch.autocast(
                            device_type=device_type,
                            dtype=amp_dtype,
                            enabled=amp_enabled,
                        ):
                            acc_zero_var_addition_restricted = (
                                (model(src_b)[:, -1, :MOD].argmax(-1) == tgt_b)
                                .float()
                                .mean()
                                .item()
                            )

                    if len(one_var_train_tok) > 0:
                        src_b, tgt_b = get_batch_one_var_train()
                        with torch.autocast(
                            device_type=device_type,
                            dtype=amp_dtype,
                            enabled=amp_enabled,
                        ):
                            acc_one_var_train = (
                                (model(src_b)[:, -1, :MOD].argmax(-1) == tgt_b)
                                .float()
                                .mean()
                                .item()
                            )

                    if len(one_var_add_restricted_tok) > 0:
                        src_b, tgt_b = get_batch_one_var_add_restricted()
                        with torch.autocast(
                            device_type=device_type,
                            dtype=amp_dtype,
                            enabled=amp_enabled,
                        ):
                            acc_one_var_addition_restricted = (
                                (model(src_b)[:, -1, :MOD].argmax(-1) == tgt_b)
                                .float()
                                .mean()
                                .item()
                            )

                    if len(one_var_var_restricted_tok) > 0:
                        src_b, tgt_b = get_batch_one_var_var_restricted()
                        with torch.autocast(
                            device_type=device_type,
                            dtype=amp_dtype,
                            enabled=amp_enabled,
                        ):
                            acc_one_var_variable_restricted = (
                                (model(src_b)[:, -1, :MOD].argmax(-1) == tgt_b)
                                .float()
                                .mean()
                                .item()
                            )

                    if len(two_var_train_tok) > 0:
                        src_b, tgt_b = get_batch_two_var_train()
                        with torch.autocast(
                            device_type=device_type,
                            dtype=amp_dtype,
                            enabled=amp_enabled,
                        ):
                            acc_two_var_train = (
                                (model(src_b)[:, -1, :MOD].argmax(-1) == tgt_b)
                                .float()
                                .mean()
                                .item()
                            )

                    if len(two_var_test_tok) > 0:
                        src_b, tgt_b = get_batch_two_var_test()
                        with torch.autocast(
                            device_type=device_type,
                            dtype=amp_dtype,
                            enabled=amp_enabled,
                        ):
                            acc_two_var_addition_restricted = (
                                (model(src_b)[:, -1, :MOD].argmax(-1) == tgt_b)
                                .float()
                                .mean()
                                .item()
                            )

                    if len(two_var_var_restricted_1_tok) > 0:
                        src_b, tgt_b = get_batch_two_var_var_restricted_1()
                        with torch.autocast(
                            device_type=device_type,
                            dtype=amp_dtype,
                            enabled=amp_enabled,
                        ):
                            acc_two_var_variable_restricted_1 = (
                                (model(src_b)[:, -1, :MOD].argmax(-1) == tgt_b)
                                .float()
                                .mean()
                                .item()
                            )

                    if len(two_var_var_restricted_2_tok) > 0:
                        src_b, tgt_b = get_batch_two_var_var_restricted_2()
                        with torch.autocast(
                            device_type=device_type,
                            dtype=amp_dtype,
                            enabled=amp_enabled,
                        ):
                            acc_two_var_variable_restricted_2 = (
                                (model(src_b)[:, -1, :MOD].argmax(-1) == tgt_b)
                                .float()
                                .mean()
                                .item()
                            )

                train_losses.append(loss.item())
                test_accs.append(acc_test)
                swap_accs.append(acc_swap)
                unseen_accs.append(acc_unseen)
                both_accs.append(acc_both)
                zero_var_train_accs.append(acc_zero_var_train)
                zero_var_addition_restricted_accs.append(
                    acc_zero_var_addition_restricted
                )
                one_var_train_accs.append(acc_one_var_train)
                one_var_addition_restricted_accs.append(acc_one_var_addition_restricted)
                one_var_variable_restricted_accs.append(acc_one_var_variable_restricted)
                two_var_train_accs.append(acc_two_var_train)
                two_var_addition_restricted_accs.append(acc_two_var_addition_restricted)
                two_var_variable_restricted_1_accs.append(
                    acc_two_var_variable_restricted_1
                )
                two_var_variable_restricted_2_accs.append(
                    acc_two_var_variable_restricted_2
                )
                steps_list.append(step)

                log_dict = {
                    "step": step,
                    "train_loss": loss.item(),
                    "test_acc": acc_test,
                    "swap_acc": acc_swap,
                    "unseen_acc": acc_unseen,
                    "both_acc": acc_both,
                    "zero_var_train_acc": acc_zero_var_train,
                    "zero_var_addition_restricted_acc": acc_zero_var_addition_restricted,
                    "one_var_train_acc": acc_one_var_train,
                    "one_var_addition_restricted_acc": acc_one_var_addition_restricted,
                    "one_var_variable_restricted_acc": acc_one_var_variable_restricted,
                    "two_var_train_acc": acc_two_var_train,
                    "two_var_addition_restricted_acc": acc_two_var_addition_restricted,
                    "two_var_variable_restricted_1_acc": acc_two_var_variable_restricted_1,
                    "two_var_variable_restricted_2_acc": acc_two_var_variable_restricted_2,
                }

                run.log(log_dict, step=step)

                if (step - 1) % PRINT_EVERY == 0:
                    ckpt_path = run.checkpoint_path(f"checkpoint_step_{step}.pth")
                    torch.save(
                        {k: v.cpu() for k, v in model.state_dict().items()}, ckpt_path
                    )
                    print(
                        f"step {step:6d} | loss {loss.item():.2e} | test {acc_test:.3f} | "
                        f"swap {acc_swap:.3f} | unseen {acc_unseen:.3f} | both {acc_both:.3f} | "
                        f"0v_tr {acc_zero_var_train:.3f} | "
                        f"0v_addR {acc_zero_var_addition_restricted:.3f} | "
                        f"1v_tr {acc_one_var_train:.3f} | "
                        f"1v_addR {acc_one_var_addition_restricted:.3f} | "
                        f"1v_varR {acc_one_var_variable_restricted:.3f} | "
                        f"2v_tr {acc_two_var_train:.3f} | "
                        f"2v_addR {acc_two_var_addition_restricted:.3f} | "
                        f"2v_varR1 {acc_two_var_variable_restricted_1:.3f} | "
                        f"2v_varR2 {acc_two_var_variable_restricted_2:.3f}"
                    )
                model.train()

        final_metrics = evaluate_final_exact_generalization(model)
        final_metrics.update(
            {
                "final_step": max_steps,
                "LR": learning_rate,
                "WD": weight_decay,
                "TRAIN_PAIRS_FRAC": train_pairs_frac,
                "TWO_VAR_FREQUENCY": two_var_frequency,
                "NUMS_TRAIN_PAIRS": nums_train_pairs,
                "SEED": SEED,
            }
        )
        run.log(final_metrics, step=max_steps)
        final_metrics.update({"run_id": run.id, "run_name": run.name})
        run.summary.update(final_metrics)
        print(
            "Final exact eval | "
            f"0v_addR={final_metrics['final_exact_0var_add_restricted_acc']:.4f} "
            f"(n={final_metrics['final_exact_0var_add_restricted_n']}) | "
            f"2v_union={final_metrics['final_exact_2var_union_acc']:.4f} "
            f"(n={final_metrics['final_exact_2var_union_n']})"
        )

        final_ckpt_path = run.checkpoint_path("model_state_dict.pth")
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, final_ckpt_path)
        print(f"Saved final model checkpoint to {final_ckpt_path}")
        _shutdown_prefetchers()
        return final_metrics


if __name__ == "__main__":
    train()
