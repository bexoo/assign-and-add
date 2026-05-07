import torch

SPECIAL_POOL_KINDS: tuple[str, ...] = (
    "0var_valid_pair",
    "0var_invalid_pair",
    "1var_valid_pair_valid_var",
    "1var_valid_pair_invalid_var",
    "1var_invalid_pair_valid_var",
    "2var_valid_pair_valid_vars",
    "2var_valid_pair_1_invalid_var",
    "2var_valid_pair_2_invalid_vars",
    "2var_invalid_pair_valid_var",
)

B_OPERAND_POOL_KINDS: tuple[str, ...] = (
    "1var_b_operand",
    "2var_b_operand",
)

ALL_POOL_KINDS: tuple[str, ...] = SPECIAL_POOL_KINDS + B_OPERAND_POOL_KINDS

_SPECIAL_POOL_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def compute_sel_pairs(mod: int, nums_train_pairs: int, seed: int) -> set:
    """Reproduce the training set's ``sel_pairs`` exactly.

    Matches the insertion order of ``pair_pool = list(rows_by_pair.keys())``
    in train.py: perm_table is laid out (v, t, num1, num2), so
    pairs first appear in the order ``(num1, num2) for num1 in range(mod)
    for num2 in range(mod)``.
    """
    import random as py_random
    pair_pool = [(n1, n2) for n1 in range(mod) for n2 in range(mod)]
    return set(py_random.Random(seed).sample(pair_pool, nums_train_pairs))


def get_restriction_vars(cfg: dict, var_ids: list[int]) -> tuple[list[int], list[int]]:
    n_left = int(cfg.get("RESTRICT_LEFT_HALF_VARS", 2))
    n_right = int(cfg.get("RESTRICT_RIGHT_HALF_VARS", 2))
    left = var_ids[:n_left]
    right = var_ids[-n_right:] if n_right > 0 else []
    return left, right


def _resolve_nums_train_pairs(cfg: dict, mod: int) -> int:
    if "NUMS_TRAIN_PAIRS" in cfg:
        return int(cfg["NUMS_TRAIN_PAIRS"])
    import math as _math
    train_pairs_frac = float(cfg.get("TRAIN_PAIRS_FRAC", 0.7))
    return _math.floor(train_pairs_frac * mod * mod)


def _build_prefix_and_core_1var(
    var_tok: int,
    typ: int,
    num1: int,
    num2: int,
    *,
    var_ids: list[int],
    mod: int,
    seq_len: int,
    plus_id: int,
    equal_id: int,
    pad_id: int,
    rng,
) -> tuple[list[int], int]:
    if typ == 1:
        var_val = num1
    elif typ == 2:
        var_val = num2
    else:
        raise ValueError(f"_build_prefix_and_core_1var requires typ in {{1, 2}}, got {typ}")

    core_len = 4
    assignment_len = 2
    max_assignments = min(len(var_ids), (seq_len - core_len) // assignment_len)
    n_assignments = rng.randint(1, max_assignments)

    assignments = [[var_tok, var_val]]
    extra_vars = [v for v in var_ids if v != var_tok]
    rng.shuffle(extra_vars)
    for _ in range(n_assignments - 1):
        v = extra_vars.pop()
        assignments.append([v, rng.randrange(mod)])

    rng.shuffle(assignments)

    remaining_pads = seq_len - (assignment_len * n_assignments + core_len)
    gaps = [0] * (n_assignments + 1)
    for _ in range(remaining_pads):
        gaps[rng.randint(0, n_assignments)] += 1

    prefix = []
    for idx, seg in enumerate(assignments):
        prefix.extend([pad_id] * gaps[idx])
        prefix.extend(seg)
    prefix.extend([pad_id] * gaps[-1])

    if typ == 1:
        lhs_tok, rhs_tok = var_tok, num2
    else:
        lhs_tok, rhs_tok = num1, var_tok

    core = [plus_id, lhs_tok, rhs_tok, equal_id]
    tokens = prefix + core
    label = (num1 + num2) % mod
    return tokens, label


def _build_prefix_and_core_0var(
    num1: int,
    num2: int,
    *,
    var_ids: list[int],
    mod: int,
    seq_len: int,
    plus_id: int,
    equal_id: int,
    pad_id: int,
    rng,
) -> tuple[list[int], int]:
    core_len = 4
    assignment_len = 2
    max_assignments = min(len(var_ids), (seq_len - core_len) // assignment_len)
    n_assignments = rng.randint(1, max_assignments)

    var_tok = rng.choice(var_ids)
    var_val = rng.randrange(mod)

    assignments = [[var_tok, var_val]]
    extra_vars = [v for v in var_ids if v != var_tok]
    rng.shuffle(extra_vars)
    for _ in range(n_assignments - 1):
        v = extra_vars.pop()
        assignments.append([v, rng.randrange(mod)])

    rng.shuffle(assignments)

    remaining_pads = seq_len - (assignment_len * n_assignments + core_len)
    gaps = [0] * (n_assignments + 1)
    for _ in range(remaining_pads):
        gaps[rng.randint(0, n_assignments)] += 1

    prefix = []
    for idx, seg in enumerate(assignments):
        prefix.extend([pad_id] * gaps[idx])
        prefix.extend(seg)
    prefix.extend([pad_id] * gaps[-1])

    core = [plus_id, num1, num2, equal_id]
    tokens = prefix + core
    label = (num1 + num2) % mod
    return tokens, label


def _build_prefix_and_core_2var(
    var1: int,
    var2: int,
    num1: int,
    num2: int,
    *,
    var_ids: list[int],
    mod: int,
    seq_len: int,
    plus_id: int,
    equal_id: int,
    pad_id: int,
    rng,
) -> tuple[list[int], int]:
    if var1 == var2:
        raise ValueError("2-var builder requires var1 != var2")

    core_len = 4
    assignment_len = 2
    max_assignments = min(len(var_ids), (seq_len - core_len) // assignment_len)
    if max_assignments < 2:
        raise ValueError("Sequence length too short for two assignments")
    n_assignments = rng.randint(2, max_assignments)

    assignments = [[var1, num1], [var2, num2]]
    extra_needed = n_assignments - 2
    if extra_needed:
        extra_vars = [v for v in var_ids if v not in (var1, var2)]
        rng.shuffle(extra_vars)
        for _ in range(extra_needed):
            v = extra_vars.pop()
            assignments.append([v, rng.randrange(mod)])

    rng.shuffle(assignments)

    remaining_pads = seq_len - (assignment_len * n_assignments + core_len)
    gaps = [0] * (n_assignments + 1)
    for _ in range(remaining_pads):
        gaps[rng.randint(0, n_assignments)] += 1

    prefix = []
    for idx, seg in enumerate(assignments):
        prefix.extend([pad_id] * gaps[idx])
        prefix.extend(seg)
    prefix.extend([pad_id] * gaps[-1])

    core = [plus_id, var1, var2, equal_id]
    tokens = prefix + core
    label = (num1 + num2) % mod
    return tokens, label


def enumerate_valid_2var_pairs(
    var_ids: list[int],
    left_restrict: list[int],
    right_restrict: list[int],
) -> list[tuple[int, int]]:
    left_set = set(left_restrict)
    right_set = set(right_restrict)
    return [
        (var1, var2)
        for var1 in var_ids
        for var2 in var_ids
        if var1 != var2 and var1 not in right_set and var2 not in left_set
    ]


def _enumerate_2var_pairs_for_kind(
    kind: str,
    var_ids: list[int],
    left_restrict: list[int],
    right_restrict: list[int],
) -> list[tuple[int, int]]:
    left_set = set(left_restrict)
    right_set = set(right_restrict)
    if kind in ("2var_invalid_pair_valid_var", "2var_valid_pair_valid_vars"):
        return enumerate_valid_2var_pairs(var_ids, left_restrict, right_restrict)
    if kind not in ("2var_valid_pair_1_invalid_var", "2var_valid_pair_2_invalid_vars"):
        raise ValueError(
            "_enumerate_2var_pairs_for_kind called with non-2var kind: "
            f"{kind!r}"
        )

    pairs = []
    for var1 in var_ids:
        for var2 in var_ids:
            if var1 == var2:
                continue
            v1_bad = var1 in right_set
            v2_bad = var2 in left_set
            if kind == "2var_valid_pair_1_invalid_var":
                if v1_bad != v2_bad:
                    pairs.append((var1, var2))
            elif kind == "2var_valid_pair_2_invalid_vars":
                if v1_bad and v2_bad:
                    pairs.append((var1, var2))
    return pairs


def _empty_pool(seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    return (torch.empty(0, seq_len, dtype=torch.long), torch.empty(0, dtype=torch.long))


def _b_operand_token(var_ids: list[int]) -> int:
    if len(var_ids) < 2:
        raise ValueError("b-operand pools require at least two variable tokens")
    return var_ids[1]


def generate_special_pool(
    kind: str,
    mod: int,
    vocab: int,
    sel_pairs: set,
    left_restrict: list[int],
    right_restrict: list[int],
    seq_len: int = 16,
    pool_size: int = 2000,
    seed: int = 54321,
) -> tuple[torch.Tensor, torch.Tensor]:
    import random as py_random

    if kind not in ALL_POOL_KINDS:
        raise ValueError(f"Unknown kind {kind!r}; expected one of {ALL_POOL_KINDS}")

    plus_id = mod
    equal_id = mod + 1
    pad_id = mod + 2
    var_start = mod + 3
    var_len = vocab - mod - 3
    if var_len <= 0:
        print(f"[warn] generate_special_pool({kind!r}): no variable tokens in vocab")
        return _empty_pool(seq_len)
    var_ids = list(range(var_start, var_start + var_len))

    all_pairs = {(n1, n2) for n1 in range(mod) for n2 in range(mod)}
    invalid_pairs = all_pairs - sel_pairs

    all_pairs_list = sorted(all_pairs)
    valid_pairs_list = sorted(sel_pairs)
    invalid_pairs_list = sorted(invalid_pairs)

    left_set = set(left_restrict)
    right_set = set(right_restrict)

    rng = py_random.Random(seed)

    builder_kwargs = dict(
        var_ids=var_ids,
        mod=mod,
        seq_len=seq_len,
        plus_id=plus_id,
        equal_id=equal_id,
        pad_id=pad_id,
        rng=rng,
    )

    sequences = []
    labels = []

    def _warn(reason: str):
        print(f"[warn] generate_special_pool({kind!r}): {reason}; returning empty tensors")

    if kind == "1var_valid_pair_invalid_var":
        if not valid_pairs_list:
            _warn("no valid pairs (sel_pairs is empty)")
            return _empty_pool(seq_len)
        typ_choices = []
        if right_restrict:
            typ_choices.append(1)
        if left_restrict:
            typ_choices.append(2)
        if not typ_choices:
            _warn("no restricted variables configured")
            return _empty_pool(seq_len)
        for _ in range(pool_size):
            typ = rng.choice(typ_choices)
            var_tok = rng.choice(right_restrict if typ == 1 else left_restrict)
            n1, n2 = rng.choice(valid_pairs_list)
            tok, lab = _build_prefix_and_core_1var(var_tok, typ, n1, n2, **builder_kwargs)
            sequences.append(tok)
            labels.append(lab)

    elif kind == "1var_b_operand":
        b_tok = _b_operand_token(var_ids)
        for _ in range(pool_size):
            typ = rng.choice([1, 2])
            n1, n2 = rng.choice(all_pairs_list)
            tok, lab = _build_prefix_and_core_1var(b_tok, typ, n1, n2, **builder_kwargs)
            sequences.append(tok)
            labels.append(lab)

    elif kind == "1var_invalid_pair_valid_var":
        if not invalid_pairs_list:
            _warn("no invalid pairs")
            return _empty_pool(seq_len)
        valid_var_type1 = [v for v in var_ids if v not in right_set]
        valid_var_type2 = [v for v in var_ids if v not in left_set]
        typ_choices = []
        if valid_var_type1:
            typ_choices.append(1)
        if valid_var_type2:
            typ_choices.append(2)
        if not typ_choices:
            _warn("no valid-position variables available for any typ")
            return _empty_pool(seq_len)
        for _ in range(pool_size):
            typ = rng.choice(typ_choices)
            var_tok = rng.choice(valid_var_type1 if typ == 1 else valid_var_type2)
            n1, n2 = rng.choice(invalid_pairs_list)
            tok, lab = _build_prefix_and_core_1var(var_tok, typ, n1, n2, **builder_kwargs)
            sequences.append(tok)
            labels.append(lab)

    elif kind == "1var_valid_pair_valid_var":
        if not valid_pairs_list:
            _warn("no valid pairs (sel_pairs is empty)")
            return _empty_pool(seq_len)
        valid_var_type1 = [v for v in var_ids if v not in right_set]
        valid_var_type2 = [v for v in var_ids if v not in left_set]
        typ_choices = []
        if valid_var_type1:
            typ_choices.append(1)
        if valid_var_type2:
            typ_choices.append(2)
        if not typ_choices:
            _warn("no valid-position variables available for any typ")
            return _empty_pool(seq_len)
        for _ in range(pool_size):
            typ = rng.choice(typ_choices)
            var_tok = rng.choice(valid_var_type1 if typ == 1 else valid_var_type2)
            n1, n2 = rng.choice(valid_pairs_list)
            tok, lab = _build_prefix_and_core_1var(var_tok, typ, n1, n2, **builder_kwargs)
            sequences.append(tok)
            labels.append(lab)

    elif kind in (
        "2var_valid_pair_1_invalid_var",
        "2var_valid_pair_2_invalid_vars",
        "2var_valid_pair_valid_vars",
    ):
        if not valid_pairs_list:
            _warn("no valid pairs")
            return _empty_pool(seq_len)
        var_pairs = _enumerate_2var_pairs_for_kind(kind, var_ids, left_restrict, right_restrict)
        if not var_pairs:
            _warn("no (var1, var2) pairs satisfy the position rule")
            return _empty_pool(seq_len)
        for _ in range(pool_size):
            var1, var2 = rng.choice(var_pairs)
            n1, n2 = rng.choice(valid_pairs_list)
            tok, lab = _build_prefix_and_core_2var(var1, var2, n1, n2, **builder_kwargs)
            sequences.append(tok)
            labels.append(lab)

    elif kind == "2var_b_operand":
        b_tok = _b_operand_token(var_ids)
        var_pairs = [
            (var1, var2)
            for var1 in var_ids
            for var2 in var_ids
            if var1 != var2 and (var1 == b_tok or var2 == b_tok)
        ]
        if not var_pairs:
            _warn("no (var1, var2) pairs include b")
            return _empty_pool(seq_len)
        for _ in range(pool_size):
            var1, var2 = rng.choice(var_pairs)
            n1, n2 = rng.choice(all_pairs_list)
            tok, lab = _build_prefix_and_core_2var(var1, var2, n1, n2, **builder_kwargs)
            sequences.append(tok)
            labels.append(lab)

    elif kind == "2var_invalid_pair_valid_var":
        if not invalid_pairs_list:
            _warn("no invalid pairs")
            return _empty_pool(seq_len)
        var_pairs = _enumerate_2var_pairs_for_kind(kind, var_ids, left_restrict, right_restrict)
        if not var_pairs:
            _warn("no (var1, var2) pairs satisfy the position rule")
            return _empty_pool(seq_len)
        for _ in range(pool_size):
            var1, var2 = rng.choice(var_pairs)
            n1, n2 = rng.choice(invalid_pairs_list)
            tok, lab = _build_prefix_and_core_2var(var1, var2, n1, n2, **builder_kwargs)
            sequences.append(tok)
            labels.append(lab)

    elif kind == "0var_invalid_pair":
        if not invalid_pairs_list:
            _warn("no invalid pairs")
            return _empty_pool(seq_len)
        for _ in range(pool_size):
            n1, n2 = rng.choice(invalid_pairs_list)
            tok, lab = _build_prefix_and_core_0var(n1, n2, **builder_kwargs)
            sequences.append(tok)
            labels.append(lab)

    elif kind == "0var_valid_pair":
        if not valid_pairs_list:
            _warn("no valid pairs (sel_pairs is empty)")
            return _empty_pool(seq_len)
        for _ in range(pool_size):
            n1, n2 = rng.choice(valid_pairs_list)
            tok, lab = _build_prefix_and_core_0var(n1, n2, **builder_kwargs)
            sequences.append(tok)
            labels.append(lab)

    tokens_t = torch.tensor(sequences, dtype=torch.long)
    labels_t = torch.tensor(labels, dtype=torch.long)
    return tokens_t, labels_t


def generate_exact_0var_invalid_pair_pool(
    mod: int,
    vocab: int,
    sel_pairs: set,
    *,
    seq_len: int = 16,
    seed: int = 54321,
) -> tuple[torch.Tensor, torch.Tensor]:
    import random as py_random

    plus_id = mod
    equal_id = mod + 1
    pad_id = mod + 2
    var_start = mod + 3
    var_len = vocab - mod - 3
    if var_len <= 0:
        return _empty_pool(seq_len)

    var_ids = list(range(var_start, var_start + var_len))
    all_pairs = {(n1, n2) for n1 in range(mod) for n2 in range(mod)}
    invalid_pairs = sorted(all_pairs - sel_pairs)
    if not invalid_pairs:
        return _empty_pool(seq_len)

    rng = py_random.Random(seed)
    builder_kwargs = dict(
        var_ids=var_ids,
        mod=mod,
        seq_len=seq_len,
        plus_id=plus_id,
        equal_id=equal_id,
        pad_id=pad_id,
        rng=rng,
    )

    sequences = []
    labels = []
    for n1, n2 in invalid_pairs:
        tok, lab = _build_prefix_and_core_0var(n1, n2, **builder_kwargs)
        sequences.append(tok)
        labels.append(lab)

    return (
        torch.tensor(sequences, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


def generate_exact_2var_generalization_union_pool(
    mod: int,
    vocab: int,
    sel_pairs: set,
    left_restrict: list[int],
    right_restrict: list[int],
    *,
    seq_len: int = 16,
    seed: int = 54321,
) -> tuple[torch.Tensor, torch.Tensor]:
    import random as py_random

    plus_id = mod
    equal_id = mod + 1
    pad_id = mod + 2
    var_start = mod + 3
    var_len = vocab - mod - 3
    if var_len <= 0:
        return _empty_pool(seq_len)

    var_ids = list(range(var_start, var_start + var_len))
    all_pairs = {(n1, n2) for n1 in range(mod) for n2 in range(mod)}
    valid_pairs = sorted(sel_pairs)
    invalid_pairs = sorted(all_pairs - sel_pairs)

    rng = py_random.Random(seed)
    builder_kwargs = dict(
        var_ids=var_ids,
        mod=mod,
        seq_len=seq_len,
        plus_id=plus_id,
        equal_id=equal_id,
        pad_id=pad_id,
        rng=rng,
    )

    sequences = []
    labels = []

    exact_components = (
        ("2var_invalid_pair_valid_var", invalid_pairs),
        ("2var_valid_pair_1_invalid_var", valid_pairs),
        ("2var_valid_pair_2_invalid_vars", valid_pairs),
    )
    for kind, number_pairs in exact_components:
        if not number_pairs:
            continue
        var_pairs = _enumerate_2var_pairs_for_kind(
            kind, var_ids, left_restrict, right_restrict
        )
        for n1, n2 in number_pairs:
            for var1, var2 in var_pairs:
                tok, lab = _build_prefix_and_core_2var(
                    var1, var2, n1, n2, **builder_kwargs
                )
                sequences.append(tok)
                labels.append(lab)

    if not sequences:
        return _empty_pool(seq_len)
    return (
        torch.tensor(sequences, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


def _pool_from_cfg(
    kind: str,
    cfg: dict,
    vocab: int,
    mod: int,
    *,
    pool_size: int = 2000,
    seed: int = 54321,
) -> tuple[torch.Tensor, torch.Tensor]:
    var_start = mod + 3
    var_len = vocab - mod - 3
    var_ids = list(range(var_start, var_start + var_len))

    nums_train_pairs = _resolve_nums_train_pairs(cfg, mod)
    training_seed = int(cfg.get("SEED", 42))
    left_restrict, right_restrict = get_restriction_vars(cfg, var_ids)
    seq_len = int(cfg.get("SEQ_LEN", 16))

    cache_key = (
        kind, mod, vocab, nums_train_pairs, training_seed,
        tuple(left_restrict), tuple(right_restrict),
        seq_len, pool_size, seed,
    )
    if cache_key in _SPECIAL_POOL_CACHE:
        return _SPECIAL_POOL_CACHE[cache_key]

    sel_pairs = compute_sel_pairs(mod, nums_train_pairs, training_seed)
    tokens, labels = generate_special_pool(
        kind, mod, vocab, sel_pairs, left_restrict, right_restrict,
        seq_len=seq_len, pool_size=pool_size, seed=seed,
    )
    _SPECIAL_POOL_CACHE[cache_key] = (tokens, labels)
    return tokens, labels


def get_pool_0var_valid_pair(
    cfg: dict, vocab: int, mod: int, *, pool_size: int = 2000, seed: int = 54321,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _pool_from_cfg("0var_valid_pair", cfg, vocab, mod,
                          pool_size=pool_size, seed=seed)


def get_pool_0var_invalid_pair(
    cfg: dict, vocab: int, mod: int, *, pool_size: int = 2000, seed: int = 54321,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _pool_from_cfg("0var_invalid_pair", cfg, vocab, mod,
                          pool_size=pool_size, seed=seed)


def get_pool_1var_valid_pair_valid_var(
    cfg: dict, vocab: int, mod: int, *, pool_size: int = 2000, seed: int = 54321,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _pool_from_cfg("1var_valid_pair_valid_var", cfg, vocab, mod,
                          pool_size=pool_size, seed=seed)


def get_pool_1var_valid_pair_invalid_var(
    cfg: dict, vocab: int, mod: int, *, pool_size: int = 2000, seed: int = 54321,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _pool_from_cfg("1var_valid_pair_invalid_var", cfg, vocab, mod,
                          pool_size=pool_size, seed=seed)


def get_pool_1var_invalid_pair_valid_var(
    cfg: dict, vocab: int, mod: int, *, pool_size: int = 2000, seed: int = 54321,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _pool_from_cfg("1var_invalid_pair_valid_var", cfg, vocab, mod,
                          pool_size=pool_size, seed=seed)


def get_pool_2var_valid_pair_valid_vars(
    cfg: dict, vocab: int, mod: int, *, pool_size: int = 2000, seed: int = 54321,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _pool_from_cfg("2var_valid_pair_valid_vars", cfg, vocab, mod,
                          pool_size=pool_size, seed=seed)


def get_pool_2var_valid_pair_1_invalid_var(
    cfg: dict, vocab: int, mod: int, *, pool_size: int = 2000, seed: int = 54321,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _pool_from_cfg("2var_valid_pair_1_invalid_var", cfg, vocab, mod,
                          pool_size=pool_size, seed=seed)


def get_pool_2var_valid_pair_2_invalid_vars(
    cfg: dict, vocab: int, mod: int, *, pool_size: int = 2000, seed: int = 54321,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _pool_from_cfg("2var_valid_pair_2_invalid_vars", cfg, vocab, mod,
                          pool_size=pool_size, seed=seed)


def get_pool_2var_invalid_pair_valid_var(
    cfg: dict, vocab: int, mod: int, *, pool_size: int = 2000, seed: int = 54321,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _pool_from_cfg("2var_invalid_pair_valid_var", cfg, vocab, mod,
                          pool_size=pool_size, seed=seed)


def get_pool_1var_b_operand(
    cfg: dict, vocab: int, mod: int, *, pool_size: int = 2000, seed: int = 54321,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _pool_from_cfg("1var_b_operand", cfg, vocab, mod,
                          pool_size=pool_size, seed=seed)


def get_pool_2var_b_operand(
    cfg: dict, vocab: int, mod: int, *, pool_size: int = 2000, seed: int = 54321,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _pool_from_cfg("2var_b_operand", cfg, vocab, mod,
                          pool_size=pool_size, seed=seed)
