"""Compute observed-window calibration and matched ablation summaries for Section IV."""

import argparse
from pathlib import Path

import bootstrap_env  # Apply the experiment's numerical runtime settings.
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
COHORTS = ["CHARLS", "ELSA", "HRS"]
ABLATIONS = ["no_observation_covariates", "delete_intermediate_history_visits_rebuild"]


def observed_window_calibration(predictions):
    """Use the frozen risk bins and resample whole households within each cohort."""
    fields = ["cohort", "household_id", "y", "PartialOrder"]
    if predictions.empty or predictions[fields].isna().any().any():
        raise ValueError("Each held-out prediction needs its cohort, household, outcome and risk.")
    if not predictions["PartialOrder"].between(0, 1).all() or not predictions["y"].isin([0, 1]).all():
        raise ValueError("Expected risks in [0, 1] and observed binary outcomes.")

    edges = np.array([0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 1.0000001])
    rng = np.random.default_rng(910031)
    rows = []
    for cohort, group in predictions.groupby("cohort", sort=True):
        households, household_index = np.unique(group["household_id"], return_inverse=True)
        risk = group["PartialOrder"].to_numpy()
        bins = np.searchsorted(edges, risk, side="right") - 1
        cell = (household_index, bins)
        counts = np.zeros((len(households), len(edges) - 1))
        events = np.zeros_like(counts)
        risk_sum = np.zeros_like(counts)
        np.add.at(counts, cell, 1)
        np.add.at(events, cell, group["y"].to_numpy())
        np.add.at(risk_sum, cell, risk)

        n_households = len(households)
        weights = rng.multinomial(n_households, np.full(n_households, 1 / n_households), size=1000)
        denominator = weights @ counts
        numerator = weights @ events
        rates = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan),
                          where=denominator > 0)
        for index, n in enumerate(counts.sum(axis=0)):
            if n == 0:
                continue
            represented = int(np.count_nonzero(counts[:, index]))
            valid = rates[np.isfinite(rates[:, index]), index]
            enough = represented >= 10
            lower, upper = np.quantile(valid, [0.025, 0.975]) if enough else (np.nan, np.nan)
            rows.append({
                "cohort": cohort, "bin": index,
                "bin_left": edges[index], "bin_right": min(edges[index + 1], 1.0),
                "n": int(n), "households": represented,
                "mean_prediction": risk_sum[:, index].sum() / n,
                "observed_fraction": events[:, index].sum() / n,
                "ci_low": lower, "ci_high": upper,
                "bootstrap_valid_repetitions": len(valid), "ci_estimable": enough,
            })
    return pd.DataFrame(rows)


def observation_ablations(contrasts):
    """Report matched Brier changes and their equal-cohort mean and sample SD."""
    selected = contrasts.loc[
        contrasts["experiment"].isin(ABLATIONS) & contrasts["cohort"].isin(COHORTS)
        & contrasts["status"].eq("completed")
    ]
    changes = selected.pivot(index="experiment", columns="cohort", values="brier_change")
    changes = changes.reindex(index=ABLATIONS, columns=COHORTS)
    if not np.isfinite(changes.to_numpy()).all():
        raise ValueError("Both ablations require a completed finite Brier contrast for each cohort.")
    summary = changes.copy()
    summary["brier_change_mean"] = changes.mean(axis=1)
    summary["brier_change_cohort_sd"] = changes.std(axis=1, ddof=1)
    return summary.reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("statistic", choices=["calibration", "ablations"])
    parser.add_argument("--input", type=Path, help="Override the input CSV (plain or gzip).")
    parser.add_argument("--output", type=Path, help="Override the aggregate output CSV.")
    args = parser.parse_args()
    if args.statistic == "calibration":
        source = args.input or ROOT / "private" / "real_predictions.csv.gz"
        target = args.output or ROOT / "results" / "real" / "observed_window_calibration.csv"
        frame = pd.read_csv(source, usecols=["cohort", "household_id", "y", "PartialOrder"],
                            dtype={"household_id": str})
        summary = observed_window_calibration(frame)
    else:
        directory = ROOT / "results" / "real_supplement"
        source = args.input or directory / "experiment_long.csv"
        target = args.output or directory / "observation_ablation_summary.csv"
        frame = pd.read_csv(source, usecols=["cohort", "experiment", "status", "brier_change"])
        summary = observation_ablations(frame)
    target.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(target, index=False)
    print(f"Wrote {len(summary)} aggregate rows to {target}")


if __name__ == "__main__":
    main()
