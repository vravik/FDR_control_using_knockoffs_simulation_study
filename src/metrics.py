"""Evaluate saved results and draw diagnostic and publication figures.

read_results loads CSV batches and their metadata. rejection_summary computes
Type I error and Monte Carlo uncertainty. paired_difference compares methods
on matched repetitions. analyze saves summary tables; make_figures plots them.

Usage: python -m src.metrics {analyze,figures} [--alpha LEVEL]
Read the study identified by results/raw/latest.json. Write summary and paired
CSVs, evaluation metadata, and two figures beneath results/.
"""

import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]


def read_results():
    """Return the latest study's raw records and saved metadata.

    Run the simulation first. Load all completed scenario CSV batches into
    one pandas DataFrame, along with the run's JSON metadata dictionary.
    """
    latest = json.loads((ROOT / "results/raw/latest.json").read_text())
    run_dir = ROOT / "results/raw" / latest["study_id"]
    records = pd.concat([
        pd.read_csv(path, float_precision="round_trip")
        for path in sorted(run_dir.glob("scenario_*.csv"))
    ], ignore_index=True)
    metadata = json.loads((run_dir / "metadata.json").read_text())
    return records, metadata


def rejection_summary(records, alpha, confidence_level):
    """Return a rejection rate, MCSE, binomial interval, and failure counts.

    Parameters
    ----------
    records : pandas.DataFrame
        Rows for a single scenario and method, including failed fits.
    alpha : float
        Rejection threshold in (0, 1); reject when p_value < alpha.
    confidence_level : float
        Monte Carlo confidence level in (0, 1).

    Returns
    -------
    dict
        Summary fields. Leave the unconditional rate and uncertainty missing
        if any fit failed or produced a missing p-value; never drop failures.
    """
    n_sim = len(records)
    failed = records["status"].ne("ok") | records["p_value"].isna()
    result = {"n_sim": n_sim, "n_failed": int(failed.sum()),
              "n_warning": int(records["warnings"].notna().sum()),
              "alpha": alpha, "error_rate": np.nan, "mcse": np.nan,
              "ci_low": np.nan, "ci_high": np.nan}
    if not failed.any():
        rejections = int((records["p_value"] < alpha).sum())
        rate = rejections / n_sim
        ci = binomtest(rejections, n_sim).proportion_ci(confidence_level, method="exact")
        result.update(error_rate=rate, mcse=np.sqrt(rate * (1 - rate) / n_sim),
                      ci_low=ci.low, ci_high=ci.high)
    return result


def paired_difference(p_values_a, p_values_b, alpha):
    """Return a paired rejection-rate difference and its Monte Carlo SE.

    Parameters
    ----------
    p_values_a, p_values_b : array-like
        P-value arrays of at least two entries aligned by repetition ID. Missing values
        represent failed fits; do not pass unpaired success-only subsets.
    alpha : float
        Rejection threshold in (0, 1).

    Returns
    -------
    dict
        Difference A minus B, paired MCSE, and the number of failed pairs.
        Missing fits make the unconditional comparison missing.
    """
    a, b = np.asarray(p_values_a), np.asarray(p_values_b)
    failed = np.isnan(a) | np.isnan(b)
    result = {"n_pairs": len(a), "n_failed_pairs": int(failed.sum()),
              "difference": np.nan, "mcse": np.nan}
    if not failed.any():
        delta = (a < alpha).astype(float) - (b < alpha).astype(float)
        result.update(difference=float(delta.mean()),
                      mcse=float(delta.std(ddof=1) / np.sqrt(len(delta))))
    return result


def analyze(alpha=None):
    """Save rejection-rate and paired-comparison tables without rerunning fits.

    alpha optionally overrides config.json's evaluation level. Return the two
    summary DataFrames. Read the latest completed raw study; write tables and
    evaluation settings/source provenance beneath results/.
    """
    records, metadata = read_results()
    evaluation = json.loads((ROOT / "config.json").read_text())["evaluation"]
    alpha = evaluation["alpha"] if alpha is None else alpha
    keys = ["scenario_id", "scenario", "n", "method_id", "method"]
    summary = [{**dict(zip(keys, key)),
                **rejection_summary(group, alpha, evaluation["confidence_level"])}
               for key, group in records.groupby(keys, sort=False)]
    paired = []
    for (sid, label, n), group in records.groupby(keys[:3], sort=False):
        p_values = group.pivot(index="replicate_id", columns="method_id", values="p_value")
        names = group.set_index("method_id")["method"].to_dict()
        for aid, bid in combinations(p_values.columns, 2):
            paired.append({"scenario_id": sid, "scenario": label, "n": n,
                           "comparison": f"{names[aid]} minus {names[bid]}",
                           **paired_difference(p_values[aid], p_values[bid], alpha)})
    summary, paired = pd.DataFrame(summary), pd.DataFrame(paired)
    summary.insert(0, "study_id", metadata["study_id"])
    paired.insert(0, "study_id", metadata["study_id"])
    paired["alpha"] = alpha
    summary.to_csv(ROOT / "results/summary.csv", index=False)
    paired.to_csv(ROOT / "results/paired.csv", index=False)
    source = Path(__file__).read_text()
    (ROOT / "results/evaluation.json").write_text(json.dumps({
        "study_id": metadata["study_id"], "alpha": alpha,
        "confidence_level": evaluation["confidence_level"],
        "metrics_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
    }, indent=2) + "\n")
    (ROOT / "results/metrics_source.py").write_text(source)
    print(summary[["scenario", "n", "method", "error_rate", "mcse", "n_failed"]].to_string(index=False))
    return summary, paired


def make_figures():
    """Save an ECDF diagnostic and a publication-quality Type I error figure.

    Run analyze first. Read its tables/settings and the saved raw p-values;
    write two PNG files and a vector SVG of the main figure. Facet in design
    order, label every condition, and display exact Monte Carlo intervals.
    """
    records, metadata = read_results()
    summary = pd.read_csv(ROOT / "results/summary.csv")
    evaluation = json.loads((ROOT / "results/evaluation.json").read_text())
    alpha, confidence = evaluation["alpha"], evaluation["confidence_level"]
    design = metadata["config"]["simulation"]
    labels = list(dict.fromkeys(item["label"] for item in design["scenarios"]))
    sizes = sorted({item["n"] for item in design["scenarios"]})
    methods = metadata["methods"]
    output = ROOT / "results/figures"
    output.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(labels), len(sizes), figsize=(9, 8),
                             sharex=True, sharey=True, squeeze=False, layout="constrained")
    for i, label in enumerate(labels):
        for j, n in enumerate(sizes):
            ax = axes[i, j]
            counts = []
            for k, method in enumerate(methods):
                group = records[(records["scenario"] == label) & (records["n"] == n)
                                & (records["method_id"] == method["method_id"])]
                p = np.sort(group.loc[group["status"] == "ok", "p_value"].dropna())
                counts.append(f"{method['name']}: {len(p)}/{len(group)} valid")
                ax.step(p, np.arange(1, len(p) + 1) / len(p), where="post",
                        label=method["name"], color=f"C{k}")
            ax.plot([0, 1], [0, 1], "--", color="0.5", label="Uniform reference")
            ax.set(title=f"{label}, n={n}", xlim=(0, 0.2), ylim=(0, 0.4))
            ax.text(0.03, 0.97, "\n".join(counts), transform=ax.transAxes,
                    va="top", fontsize=7)
            if i == len(labels) - 1:
                ax.set_xlabel("Null p-value")
            if j == 0:
                ax.set_ylabel("Empirical CDF")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Diagnostic: null p-value calibration in the lower tail")
    fig.savefig(output / "diagnostic_ecdf.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(labels), figsize=(11, 4), sharey=True,
                             squeeze=False, layout="constrained")
    for i, label in enumerate(labels):
        ax = axes[0, i]
        for k, method in enumerate(methods):
            group = summary[(summary["scenario"] == label)
                            & (summary["method_id"] == method["method_id"])].sort_values("n")
            rate = group["error_rate"]
            ax.errorbar(group["n"], rate,
                        yerr=[rate - group["ci_low"], group["ci_high"] - rate],
                        fmt="o-", color=f"C{k}", capsize=3, label=method["name"])
        ax.axhline(alpha, linestyle="--", color="0.4", label="Nominal level")
        ax.set(title=label, xlabel="Sample size n", xscale="log")
        ax.set_xticks(sizes, labels=[str(n) for n in sizes])
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].set_ylabel("Type I error probability")
    axes[0, 0].set_ylim(bottom=0)
    axes[0, -1].legend(fontsize=8)
    fig.suptitle(f"Tests of the true mean at alpha={alpha:g}; "
                 f"{design['n_sim']:,} repetitions/condition; "
                 f"{confidence:.0%} exact Monte Carlo intervals")
    fig.savefig(output / "type_i_error.png", dpi=300)
    fig.savefig(output / "type_i_error.svg")
    plt.close(fig)
    print(f"Saved diagnostic and main figures in {output}")


def main():
    """Dispatch the analyze or figures command with an optional alpha override."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["analyze", "figures"])
    parser.add_argument("--alpha", type=float, help="Override alpha for analysis.")
    args = parser.parse_args()
    if args.command == "analyze":
        analyze(args.alpha)
    else:
        make_figures()


if __name__ == "__main__":
    main()
