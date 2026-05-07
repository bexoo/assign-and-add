import json
from pathlib import Path


OUTPUT_PATH = Path("generalization_combined2_data.json")
TWOVAR_GRID = [
    0.01 + i * (0.99 - 0.01) / 19
    for i in range(20)
]
TRAIN_FRAC_GRID = [
    0.06157894736842105 + i * (0.99 - 0.06157894736842105) / 19
    for i in range(20)
]


def run_experiment() -> dict[str, list[dict]]:
    from train import train

    results = {"twovar": [], "train_frac": []}

    for value in TWOVAR_GRID:
        metrics = train(
            {
                "GRID_PANEL": "twovar",
                "TWO_VAR_FREQUENCY": value,
                "TRAIN_PAIRS_FRAC": 0.7,
            }
        )
        results["twovar"].append(
            {
                "two_var_frequency": value,
                "test_accuracy": metrics["final_exact_2var_union_acc"],
                "eval_examples": metrics["final_exact_2var_union_n"],
                "final_step": metrics["final_step"],
                "run_id": metrics["run_id"],
                "run_name": metrics["run_name"],
                "SEED": metrics["SEED"],
                "LR": metrics["LR"],
                "WD": metrics["WD"],
                "TRAIN_PAIRS_FRAC": metrics["TRAIN_PAIRS_FRAC"],
                "TWO_VAR_FREQUENCY": metrics["TWO_VAR_FREQUENCY"],
                "NUMS_TRAIN_PAIRS": metrics["NUMS_TRAIN_PAIRS"],
            }
        )
        OUTPUT_PATH.write_text(json.dumps(results, indent=2))

    for value in TRAIN_FRAC_GRID:
        metrics = train(
            {
                "GRID_PANEL": "train_frac",
                "TWO_VAR_FREQUENCY": 1.0,
                "TRAIN_PAIRS_FRAC": value,
            }
        )
        results["train_frac"].append(
            {
                "train_pairs_frac": value,
                "test_accuracy": metrics["final_exact_0var_add_restricted_acc"],
                "eval_examples": metrics["final_exact_0var_add_restricted_n"],
                "final_step": metrics["final_step"],
                "nums_train_pairs": metrics["NUMS_TRAIN_PAIRS"],
                "run_id": metrics["run_id"],
                "run_name": metrics["run_name"],
                "SEED": metrics["SEED"],
                "LR": metrics["LR"],
                "WD": metrics["WD"],
                "TRAIN_PAIRS_FRAC": metrics["TRAIN_PAIRS_FRAC"],
                "TWO_VAR_FREQUENCY": metrics["TWO_VAR_FREQUENCY"],
                "NUMS_TRAIN_PAIRS": metrics["NUMS_TRAIN_PAIRS"],
            }
        )
        OUTPUT_PATH.write_text(json.dumps(results, indent=2))

    return results


if __name__ == "__main__":
    run_experiment()
