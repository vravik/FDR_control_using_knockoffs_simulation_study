"""Run and checkpoint the simulation study before computing any summaries.

run_replication generates one dataset and applies every method to it.
run_study saves per-scenario CSV batches and the configuration/provenance.

Usage: python -m src.simulation [--n-sim N]
Read config.json and write results/raw/<study_id>/ plus a latest.json pointer.
An optional budget override is recorded in the saved configuration.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
from time import perf_counter
import warnings

import matplotlib
import numpy as np
import pandas as pd
import scipy

from src.dgps import DATA, METHOD, RNG_PLAN, make_rng, make_scenarios
from src.methods import METHODS

ROOT = Path(__file__).resolve().parents[1]


def run_replication(scenario, methods, replicate_id, root_seed):
    """Generate one dataset and return a result record for each method.

    Parameters
    ----------
    scenario : Scenario
        Data-generating distribution and sample size; its mean is the null.
    methods : list of MeanTest
        Competing methods, all applied to the same dataset.
    replicate_id, root_seed : int
        Zero-based repetition ID and recorded master seed.

    Returns
    -------
    list of dict
        IDs, truth, numeric outputs, timing, stream keys, and fit status.
        Method errors become failed records with missing numeric outputs;
        warnings are retained. Data-generation errors propagate normally.
    """
    sid = scenario.scenario_id
    data_rng = make_rng(root_seed, sid, replicate_id, purpose_id=DATA)
    data = scenario.dgp.rvs(size=scenario.n, random_state=data_rng)
    records = []
    for method in methods:
        record = {
            "scenario_id": sid, "scenario": scenario.label, "n": scenario.n,
            "replicate_id": replicate_id, "method_id": method.method_id,
            "method": method.name, "truth": float(scenario.dgp.mean()),
            **dict.fromkeys(["estimate", "se", "statistic", "df", "p_value"], np.nan),
            "status": "ok", "message": "",
            "data_rng_key": json.dumps([sid, replicate_id, DATA, 0, root_seed]),
            "method_rng_key": json.dumps([sid, replicate_id, METHOD, method.method_id, root_seed]),
        }
        started = perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                method_rng = make_rng(root_seed, sid, replicate_id,
                                      purpose_id=METHOD, method_id=method.method_id)
                record.update(method.test(data.copy(), record["truth"], method_rng))
            except Exception as error:
                record.update(status="failed", message=f"{type(error).__name__}: {error}")
        record.update(elapsed_s=perf_counter() - started,
                      warnings="; ".join(str(item.message) for item in caught))
        records.append(record)
    return records


def run_study(config):
    """Save the configured study, skipping already checkpointed scenarios.

    Parameters
    ----------
    config : dict
        Simulation and evaluation settings from config.json.

    Returns
    -------
    pathlib.Path
        Directory containing raw CSVs and metadata. Write each complete
        scenario batch to a temporary CSV and rename it atomically. Save
        source snapshots and dependency versions; print scenario progress.
    """
    settings = config["simulation"]
    scenarios = make_scenarios(settings)
    methods = [METHODS[name] for name in settings["methods"]]
    sources = {path.name: path.read_text() for path in (ROOT / "src").glob("*.py")}
    versions = {"python": platform.python_version(), "numpy": np.__version__,
                "scipy": scipy.__version__, "pandas": pd.__version__,
                "matplotlib": matplotlib.__version__}
    # Metric code can change without invalidating cached simulation outputs.
    simulation_source = {name: sources[name] for name in
                         ["dgps.py", "methods.py", "simulation.py"]}
    identity = {"settings": settings, "source": simulation_source, "versions": versions}
    study_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    run_dir = ROOT / "results" / "raw" / study_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if not (run_dir / "metadata.json").exists():
        commit = None
        if shutil.which("git"):
            commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                    capture_output=True, text=True).stdout.strip() or None
        metadata = {
            "study_id": study_id, "config": config, "rng_plan": RNG_PLAN,
            "git_commit": commit, "versions": versions,
            "platform": platform.platform(),
            "source_sha256": {name: hashlib.sha256(text.encode()).hexdigest()
                              for name, text in sources.items()},
            "methods": [{"method_id": m.method_id, "name": m.name,
                         "class": type(m).__name__} for m in methods],
        }
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (run_dir / "source").mkdir()
        for name, text in sources.items():
            (run_dir / "source" / name).write_text(text)
    for i, scenario in enumerate(scenarios, start=1):
        path = run_dir / f"scenario_{scenario.scenario_id}.csv"
        if path.exists():
            print(f"[{i}/{len(scenarios)}] Cached: {scenario.label}, n={scenario.n}")
            continue
        records = []
        for rid in range(settings["n_sim"]):
            records.extend(run_replication(scenario, methods, rid, settings["root_seed"]))
        table = pd.DataFrame(records)
        table.insert(0, "study_id", study_id)
        temporary = path.with_suffix(".csv.tmp")
        table.to_csv(temporary, index=False)
        temporary.replace(path)
        print(f"[{i}/{len(scenarios)}] Saved: {scenario.label}, n={scenario.n}")
    pointer = ROOT / "results" / "raw" / "latest.json"
    pointer.write_text(json.dumps({"study_id": study_id}) + "\n")
    print(f"Raw results: {run_dir}")
    return run_dir


def main():
    """Read command-line options and execute the configured simulation study."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sim", type=int, help="Override the recorded repetition budget.")
    args = parser.parse_args()
    config = json.loads((ROOT / "config.json").read_text())
    if args.n_sim is not None:
        config["simulation"]["n_sim"] = args.n_sim
    run_study(config)


if __name__ == "__main__":
    main()
