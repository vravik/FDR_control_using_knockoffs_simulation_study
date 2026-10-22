"""Orchestration, independent streams, raw outputs, and scenario checkpoints."""

import hashlib
import json
import logging
from pathlib import Path
import platform
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
import scipy
import sklearn
from threadpoolctl import threadpool_limits

from .dgps import DGP_CLASSES
from .methods import BHq, knockoff, knockoff_plus
from .metrics import evaluate


def rng_for(seed, scenario_id, rep, task):
    """Stable stream for one task, independent of loop order or worker.
    
    Hash the UTF-8 JSON list [scenario_id, rep, task] with SHA-256. Use its
    first 16 bytes as four little-endian uint32 words after the master seed
    in a SeedSequence. Unlike Python's hash(), this mapping is stable.
    """
    label = json.dumps([scenario_id, rep, task], separators=(",", ":"))
    digest = hashlib.sha256(label.encode()).digest()[:16]
    words = np.frombuffer(digest, dtype="<u4").tolist()
    return np.random.default_rng(np.random.SeedSequence([seed, *words]))


def save_csv(frame, path):
    """Publish a complete table by renaming a temporary file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def save_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


class SimulationStudy:
    """Run the supplied scenario dictionary; choosing values is the caller's job.

    scenarios maps unique, filename-safe string IDs to dictionaries with
    EXACTLY experiment ('A'/'B'/'C'), n (int), p (int), rho (float), k (int) and
    rho (float), n_reps (int). Experiments vary in one target variable 
    (sparsity / amplitude / correlation). Assume 0 < p < n,
    0 <= rho < 1, 200 / 600 reps per experiment as in original study deisgn,
    and nonnegative integer seed. 
    Use stable IDs: changing a scenario ID changes its random streams.
    run() returns a DataFrame with len(methods)*len(scenarios)*n_reps rows.
    Each (study_id, scenario_id, rep, method) identifies one fit. rep is
    zero-based; Metrics computed are false discovery rate and power.
    Every method receives the SAME dataset per replicate, with separate
    data and method streams. Failed fits remain as rows with NaN estimates
    and losses, status='failed', and a message. Warnings are also recorded.

    % check this
    One completed scenario is a checkpoint batch: its NPZ stores estimates,
    truth, covariance, and method IDs; its CSV stores metrics and fit status.
    The CSV is published last and marks the batch complete. A rerun skips
    completed batches. An interruption during a scenario repeats that batch.
    """

    def __init__(self, scenarios, alpha = 0.2, seed=20261020,
                 out_dir="results/raw", methods=None):
        self.scenarios, self.alpha, self.seed = scenarios, alpha, seed
        self.out_dir = Path(out_dir)
        self.methods = [BHq(), knockoff(), knockoff_plus()] if methods is None else methods

        root = Path(__file__).resolve().parents[1]
        files = [*sorted((root / "src").glob("*.py")),
                 root / "scenarios.py", root / "simulate.py"]
        self.config = {
            "scenarios": scenarios, "seed": seed,
            ## update this
            "methods": [{"method": m.name, "class": type(m).__name__,
                         "settings": m.settings} for m in self.methods],
                    
            "model": {"covariance": "rho**abs(j-k)",
                      "normal_errors": "N(0,1)"},

            "rng": {"generator": "PCG64", "seed_sequence": "[seed, *SHA256_words]",
                    "label": "compact JSON [scenario_id, rep, task]",
                    "hash_words": "first 16 SHA256 bytes as 4 little-endian uint32",
                    "tasks": ["data", *["method:" + m.name for m in self.methods]]},
            "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in files},
            "environment": {"python": platform.python_version(), "numpy": np.__version__,
                            "pandas": pd.__version__, "scipy": scipy.__version__,
                            "scikit-learn": sklearn.__version__},
        }
        fingerprint = json.dumps(self.config, sort_keys=True).encode()
        self.study_id = "study_" + hashlib.sha256(fingerprint).hexdigest()[:12]
        self.study_dir = self.out_dir / self.study_id

    def run_scenario(self, scenario_id, settings):
        """Run a batch and publish numeric outputs before its result table."""
        checkpoint = self.study_dir / f"{scenario_id}.csv"
        if checkpoint.exists():
            logging.info("Reusing %s", scenario_id)
            saved = pd.read_csv(checkpoint)
            saved["message"] = saved.message.fillna("")
            return saved
        dgp = DGP_CLASSES['generator'](settings["n"], settings["p"])

        # initialize the design matrix for the dgp; get rng with rep=-1
        dgp.setDesign(rng_for(self.seed, scenario_id, -1, "data"), 
                      settings['rho'])   
        
        estimates = np.full((settings['n_reps'], len(self.methods), settings["p"]), np.nan)
        rows = []
        
        ## given the scenario id, fix the generating class -- produce data by 
        ## sampling from generative model
        for rep in range(settings['n_reps']):
            # simulate coefficients and data for given parameter settings
            data = dgp.generate(rng_for(self.seed, scenario_id, rep, "data"),
                                settings['k'], settings['A'])

            # runs each method on the generated data 
            for j, method in enumerate(self.methods):
                start = perf_counter()
                status, message = "ok", ""
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    try:
                        estimates[rep, j] = method.fit(
                            data, self.alpha, rng_for(self.seed, scenario_id, rep,
                                                       "method:" + method.name))
                    except Exception as error:
                        status = "failed"
                        message = f"{type(error).__name__}: {error}"
                message = "; ".join([message, *[str(w.message) for w in caught]]).strip("; ")
                if message:
                    logging.warning("%s rep=%s %s: %s", scenario_id, rep, method.name, message)
                rows.append({"study_id": self.study_id, "scenario_id": scenario_id,
                             **settings, "rep": rep, "method": method.name,
                             **evaluate(estimates[rep, j], data), "status": status,
                             "message": message, "elapsed_seconds": perf_counter() - start})
        numeric = checkpoint.with_suffix(".npz")
        temporary = numeric.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, estimates=estimates, beta_true=dgp.beta,
                                method_ids=np.array([m.name for m in self.methods]))
        temporary.replace(numeric)
        frame = pd.DataFrame(rows)
        save_csv(frame, checkpoint)
        logging.info("Completed %s (%s replicates)", scenario_id, self.n_reps)
        return frame

    def run(self):
        """Save configuration, resume batches, and export merged raw results."""
        metadata = {"study_id": self.study_id, **self.config}
        save_json(metadata, self.study_dir / "metadata.json")
        with threadpool_limits(limits=1):
            frames = [self.run_scenario(sid, settings)
                      for sid, settings in self.scenarios.items()]
        results = pd.concat(frames, ignore_index=True)
        save_csv(results, self.study_dir / "runs.csv")
        save_csv(results, self.out_dir / "runs.csv")
        design = pd.DataFrame([{"scenario_id": sid, **s}
                               for sid, s in self.scenarios.items()])
        save_csv(design, self.out_dir / "design.csv")
        save_json(metadata, self.out_dir / "metadata.json")
        return results
