"""Aggregate raw results and create two scientific comparison figures."""

import argparse
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.metrics import paired_ratio
from src.simulation import save_csv, save_json


def summarize(runs):
    """Return one summary row per scenario and method.

    Keep identifying columns study_id, scenario_id, experiment, n, p,
    k, A, rho, method. Add n_total (all fits), n_ok
    (nonmissing coefficient losses), mean_FDP, mean_POWER,
    FDP_mcse, POWER_mcse. Means use successful fits; each
    MCSE is the sample SD divided by sqrt(the number of nonmissing losses).
    Report failures: n_total - n_ok. Do not average over scenarios.
    """
    keys = ["study_id", "scenario_id", "experiment", "n", "p", 
            "k", "A", "rho", "method"]
    return runs.groupby(keys, as_index=False).agg(
        n_total=("fdp", "size"),
        n_ok=("fdp", "count"),
        FDP_mean=("fdp", "mean"),
        POWER_mean=("power", "mean"),
        FDP_mcse=("fdp", lambda x: x.std(ddof=1) / np.sqrt(x.count())),
        POWER_mcse=("power", lambda x: x.std(ddof=1) / np.sqrt(x.count())),
    )


def figure_a(runs):
    """Return a matplotlib Figure for the effect of varying sparsity of 
    coefficients, using RAW results.

    Use experiment A. Plot mean FDP across methods with reference at nominal
    discovery rate. Make second panel with mean power for each method.
    Pair methods by (scenario_id, rep), keeping complete pairs. 
    
    The supplied
    paired_ratio(lad, ols) returns the ratio and its paired MCSE; show about
    +/-2 MCSE. At least two complete pairs per scenario are needed.

    Choose all filtering, reshaping, axes, labels, scales, colors, and other
    formatting. Do not average replicate ratios or save files here.
    """
    fig, axes = plt.subplots(1, 2, figsize=(6, 3.2), sharey=False)
    for ax, metric in zip(axes, ["FDP", "POWER"]):
        data = runs[(runs.experiment == "A")]
        points = []
        for sparsity, group in data.groupby("sparsity", sort=True):
            paired = group.pivot(index=["scenario_id", "rep"], columns="method",
                                    values="FDP").dropna()
            ratio, mcse = paired_ratio(paired.LAD, paired.OLS)
            points.append((gamma, ratio, mcse))
        gamma, ratio, mcse = np.array(points).T
        ax.errorbar(gamma, ratio, yerr=2 * mcse, marker="o", markersize=3.5,
                    color="#D55E00", capsize=2.5, linewidth=1.2)
        ax.axhline(1, color="0.45", linestyle="--", linewidth=1)
        ax.set_title(f"{distribution.capitalize()} errors")
        ax.set_xlabel(r"Aspect ratio $p/n$")
        ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Coefficient MSE: LAD / OLS")
    fig.text(0.5, 0.015, r"$n=200$, $\rho=0$; bars show $\pm2$ Monte Carlo SE",
                ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.065, 1, 1))
    return fig
    


def figure_a(summary):
    """Return a matplotlib Figure for the estimation/prediction question.

    Use the summary DataFrame from summarize(), restricted to experiment B.
    Plot coefficient MSE and prediction MSE against rho in adjacent panels,
    with both methods and uncertainty based on the MCSEs (about +/-2 MCSE).
    Choose all data preparation, axes, labels, legends, scales, and styling.
    Do not save files here.
    """
    data = summary[summary.experiment == "A"]
    fig, axes = plt.subplots(1, 2, figsize=(6, 3.2))
    for method, color in [("knockoff", "#0072B2"), ("knockoff+", "#D51900"), ("BHq", "black")]:
        values = data[data.method == method].sort_values("sparsity")
        for ax, metric in zip(axes, ["FDP", "POWER"]):
            ax.errorbar(values.sparsity, values[metric + "_mean"],
                        yerr=2 * values[metric + "_mcse"], marker="o", markersize=3.5,
                        color=color, label=method, capsize=2.5, linewidth=1.2)
    axes[0].set_ylabel("Coefficient MSE")
    axes[0].set_yscale("log")
    axes[1].set_ylabel("Prediction MSE")
    axes[1].set_ylim(bottom=0)
    for ax in axes:
        ax.set_xlabel(r"Correlation $\rho$")
        ax.set_xticks([0, 0.5, 1])
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
    fig.text(0.5, 0.015, r"$n=200$, $p=100$, normal errors; bars show $\pm2$ Monte Carlo SE",
                ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.065, 1, 1))
    return fig


def main():
    """Supplied driver: run stages separately without refitting methods."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["analyze", "figures", "all"], default="all")
    parser.add_argument("--input", type=Path, default=Path("results/raw/runs.csv"))
    parser.add_argument("--summary", type=Path, default=Path("results/summary.csv"))
    parser.add_argument("--figure", choices=["A", "B", "both"], default="both")
    parser.add_argument("--out", type=Path, default=Path("results/figures"))
    args = parser.parse_args()
    runs = pd.read_csv(args.input)
    failed = (runs.status == "failed").sum()
    print(f"Raw results: {len(runs):,} fits; {failed} failed fits retained.")
    provenance = {
        "raw_input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {"numpy": np.__version__, "pandas": pd.__version__,
                        "matplotlib": matplotlib.__version__},
        "uncertainty": "error bars are +/-2 Monte Carlo standard errors; paired ratio in A",
    }
    if args.stage in ["analyze", "all"]:
        summary = summarize(runs)
        save_csv(summary, args.summary)
        save_json(provenance, args.summary.with_suffix(".metadata.json"))
    if args.stage in ["figures", "all"]:
        args.out.mkdir(parents=True, exist_ok=True)
        if args.figure in ["A", "both"]:
            fig = figure_a(runs)
            fig.savefig(args.out / "figure_A.png", dpi=300)
            fig.savefig(args.out / "figure_A.pdf")
            plt.close(fig)
        if args.figure in ["B", "both"]:
            summary = pd.read_csv(args.summary)
            fig = figure_b(summary)
            fig.savefig(args.out / "figure_B.png", dpi=300)
            fig.savefig(args.out / "figure_B.pdf")
            plt.close(fig)
            provenance["summary_input_sha256"] = hashlib.sha256(args.summary.read_bytes()).hexdigest()
        save_json(provenance, args.out / "metadata.json")


if __name__ == "__main__":
    main()
