"""Helper to compute false discovery rates and power of tested methods and
their monte carlo uncertainty. Since evaluation does not consider magnitude of 
recovered coefficient, we convert true beta to a binary vector. Also provides
helpers to aggregate raw results and create three scientific comparison figures.
"""

import numpy as np
import pandas as pd
import argparse
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import *


def evaluate(beta_hat, data):
    """Realized power and false discovery rates, without averaging"""
    if np.isnan(beta_hat).any():
        return{"fdp":np.nan, "power":np.nan}
    
    rejections = beta_hat.sum()
    if(rejections == 0):
        return({'fdp' : 0, 
                'power' : 0})
    else:
        return {"fdp": np.dot(beta_hat, data['beta_true'] == 0) / rejections,             
            "power": np.dot(beta_hat, data['beta_true'] != 0) /
            (data['beta_true']!=0).sum()}

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

def figure_diagnostic(summary):
    """diagnostic figure of distribution of FDR and power across tested methods."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for method, color in [("knockoff", "#0072B2"), ("knockoff+", "#D51900"),
                        ("BHq", "black")]:
        values = summary[summary.method == method]
        for ax, metric in zip(axes, ["FDP", "POWER"]):
            ax.hist(values[metric + "_mean"], color=color, bins=15, label = method, alpha = .5)
        
    axes[0].set_xlabel("FDP")
    axes[1].set_xlabel("Power")
    axes[0].axvline(x=0.2, color='red', linestyle='--', label='Target FDP')
    for ax in axes:
        ax.legend(frameon=False)

    fig.text(0.5, 0.015, r"$n=3000$, $p=1000$, normal errors; histogram of targets", 
                ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.065, 1, 1))
    return fig

def figure_a(summary):
    """Return a matplotlib Figure for the effect of varying coefficient sparsity.

    Use the summary DataFrame from summarize(), restricted to experiment A.
    Plot FDP and power against sparsity in adjacent panels,
    with all three methods and uncertainty based on the MCSEs (about +/-2 MCSE).
    Choose all filtering, reshaping, axes, labels, scales, colors, and other
    formatting. Do not average replicate ratios or save files here.
    """
    data = summary[summary.experiment == "A"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for method, color in [("knockoff", "#0072B2"), ("knockoff+", "#D51900"),
                           ("BHq", "black")]:
        values = data[data.method == method].sort_values("k")
        for ax, metric in zip(axes, ["FDP", "POWER"]):
            ax.errorbar(values.k, values[metric + "_mean"],
                        yerr=2 * values[metric + "_mcse"], marker="o", markersize=3.5,
                        color=color, label=method, capsize=2.5, linewidth=1.2)
    axes[0].set_ylabel("FDP")
    axes[1].set_ylabel("Power")

    for ax in axes:
        ax.set_xlabel("Sparsity")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
    fig.text(0.5, 0.015, r"$n=3000$, $p=1000$, normal errors; bars show $\pm2$ " \
    "Monte Carlo SE",
                ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.065, 1, 1))
    return fig

def figure_b(summary):
    """Return a matplotlib Figure for the effect of varying signal amplitude.

    Use the summary DataFrame from summarize(), restricted to experiment B.
    Plot FDP and power against sparsity in adjacent panels,
    with all three methods and uncertainty based on the MCSEs (about +/-2 MCSE).
    Choose all filtering, reshaping, axes, labels, scales, colors, and other
    formatting. Do not average replicate ratios or save files here.
    """
    data = summary[summary.experiment == "B"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for method, color in [("knockoff", "#0072B2"), ("knockoff+", "#D51900"),
                           ("BHq", "black")]:
        values = data[data.method == method].sort_values("A")
        for ax, metric in zip(axes, ["FDP", "POWER"]):
            ax.errorbar(values.A, values[metric + "_mean"],
                        yerr=2 * values[metric + "_mcse"], marker="o", markersize=3.5,
                        color=color, label=method, capsize=2.5, linewidth=1.2)
    axes[0].set_ylabel("FDP")
    axes[1].set_ylabel("Power")

    for ax in axes:
        ax.set_xlabel("Amplitude")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
    fig.text(0.5, 0.015, r"$n=3000$, $p=1000$, normal errors; bars show $\pm2$ " \
    "Monte Carlo SE",
                ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.065, 1, 1))
    return fig

def figure_c(summary):
    """Return a matplotlib Figure for the effect of varying feature correlations.

    Use the summary DataFrame from summarize(), restricted to experiment C.
    Plot FDP and power against sparsity in adjacent panels,
    with all three methods and uncertainty based on the MCSEs (about +/-2 MCSE).
    Choose all filtering, reshaping, axes, labels, scales, colors, and other
    formatting. Do not average replicate ratios or save files here.
    """
    data = summary[summary.experiment == "C"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for method, color in [("knockoff", "#0072B2"), ("knockoff+", "#D51900"), 
                          ("BHq", "black")]:
        values = data[data.method == method].sort_values("rho")
        for ax, metric in zip(axes, ["FDP", "POWER"]):
            ax.errorbar(values.rho, values[metric + "_mean"],
                        yerr=2 * values[metric + "_mcse"], marker="o", markersize=3.5,
                        color=color, label=method, capsize=2.5, linewidth=1.2)
    axes[0].set_ylabel("FDP")
    axes[1].set_ylabel("Power")

    for ax in axes:
        ax.set_xlabel(r"$\rho")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
    fig.text(0.5, 0.015, r"$n=3000$, $p=1000$, normal errors; bars show $\pm2$ " \
    "Monte Carlo SE",
                ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.065, 1, 1))
    return fig


def main():
    """Supplied driver: run stages separately without refitting methods."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["analyze", "figures", "all"], default="all")
    parser.add_argument("--input", type=Path, default=Path("results/raw/runs.csv"))
    parser.add_argument("--summary", type=Path, default=Path("results/summary.csv"))
    parser.add_argument("--figure", choices=["diagnostic", "A", "B", "C", "all"], default="all")
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
        if args.figure in ["diagnostic", "all"]:
            summary = pd.read_csv(args.summary)
            fig = figure_diagnostic(summary)
            fig.savefig(args.out / "figure_diagnostic.png", dpi=300)
            fig.savefig(args.out / "figure_diagnostic.pdf")
            plt.close(fig)
        if args.figure in ["A", "all"]:
            summary = pd.read_csv(args.summary)
            fig = figure_a(summary)
            fig.savefig(args.out / "figure_A.png", dpi=300)
            fig.savefig(args.out / "figure_A.pdf")
            plt.close(fig)
        if args.figure in ["B", "all"]:
            summary = pd.read_csv(args.summary)
            fig = figure_b(summary)
            fig.savefig(args.out / "figure_B.png", dpi=300)
            fig.savefig(args.out / "figure_B.pdf")
            plt.close(fig)
        if args.figure in ["C", "all"]:
            summary = pd.read_csv(args.summary)
            fig = figure_c(summary)
            fig.savefig(args.out / "figure_C.png", dpi=300)
            fig.savefig(args.out / "figure_C.pdf")
            plt.close(fig)    
            provenance["summary_input_sha256"] = hashlib.sha256(args.summary.read_bytes()).hexdigest()
        save_json(provenance, args.out / "metadata.json")


if __name__ == "__main__":
    main()
