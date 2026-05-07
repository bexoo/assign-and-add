# %%
"""
Accuracy Curves Viewer
======================

Loads the 9 position-rule-aware accuracy curves recomputed offline by
``progress_measures.py`` (under the ``accuracy_overlays`` field of
``progress_measures.json``) and prints their available keys.

The 9 curves correspond to the 8 logical sets defined in
``eval_pools.SPECIAL_POOL_KINDS`` (with 2-var variable-restricted split
into 1-invalid and 2-invalid sub-pools):

    zero_var_train_acc                 (set 1)
    zero_var_addition_restricted_acc   (set 2)
    one_var_train_acc                  (set 3)
    one_var_addition_restricted_acc    (set 4)
    one_var_variable_restricted_acc    (set 5)
    two_var_train_acc                  (set 6)
    two_var_addition_restricted_acc    (set 7)
    two_var_variable_restricted_1_acc  (set 8a)
    two_var_variable_restricted_2_acc  (set 8b)
"""

# %% Imports and config
import json

import pandas as pd

MAX_STEP = 20000

# %% Load progress measures (and 9 accuracy overlays) from JSON
progress_measures_path = "progress_measures.json"
print(f"Loading progress measures from {progress_measures_path}...")

with open(progress_measures_path, "r") as f:
    progress_data = json.load(f)

steps = progress_data["steps"]
measures = progress_data["measures"]
accuracy_overlays_raw = progress_data.get("accuracy_overlays", {})
if not accuracy_overlays_raw:
    raise ValueError(
        "progress_measures.json has no 'accuracy_overlays' field; rerun "
        "progress_measures.py to materialize the 9 specialized-pool accuracy "
        "curves."
    )

steps_series = pd.Series(steps)
step_mask = steps_series <= MAX_STEP
steps = steps_series[step_mask].tolist()
measures = {
    name: pd.Series(values)[step_mask].tolist()
    for name, values in measures.items()
}
accuracy_curves = {
    name: {
        "steps": steps,
        "values": pd.Series(values)[step_mask].tolist(),
    }
    for name, values in accuracy_overlays_raw.items()
}

print(f"Found {len(measures)} progress measures")
print(f"Measures: {list(measures.keys())}")
print(f"Accuracy curves ({len(accuracy_curves)}): {list(accuracy_curves.keys())}")

# %%
