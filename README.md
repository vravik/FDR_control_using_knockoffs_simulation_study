# Mean tests under nonnormality

An original simulation study comparing the Type I error of a one-sample
t-test and an estimated-SE normal approximation. This is a compact worked
example of the Unit 2 project, continuing the first three simulation demos.

## Setup and run

Use Python 3.11 or newer and `make`. From this project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
make test
make all PYTHON=python
```

The default budget is 1,000 repetitions in each of six conditions. The first
complete run takes roughly 10-30 seconds on a typical laptop; cached reruns
take only a few seconds. Runtime depends on the machine and library versions.
For a quick 100-repetition demonstration, use:

```bash
make all SIM_ARGS="--n-sim 100"
```

## Findings from the bundled default run

At n=20, both tests exceed 5% Type I error under exponential and lognormal
sampling, with the normal approximation rejecting more often. At n=200, the
exponential results approach the nominal level, but lognormal Type I error
remains elevated; see `ANALYSIS.md` for estimates and Monte Carlo uncertainty.

## Project organization

| Path | Purpose |
| --- | --- |
| `config.json` | Central simulation and evaluation settings |
| `src/dgps.py` | Frozen SciPy DGPs, scenarios, and task-based RNGs |
| `src/methods.py` | Common mean-test interface and two methods |
| `src/simulation.py` | Shared datasets, raw outputs, provenance, checkpoints |
| `src/metrics.py` | Evaluation, paired comparisons, and both figures |
| `tests/test_study.py` | Four focused standard-library unit tests |
| `data/simulated/` | Optional dataset cache; unnecessary for these cheap DGPs |
| `results/raw/` | Study-specific CSVs, metadata, and source snapshots |
| `results/summary.csv`, `results/paired.csv` | Derived tables |
| `results/figures/` | Diagnostic PNG and main PNG/SVG |
| `ADEMP.md`, `ANALYSIS.md` | Design and critical interpretation |

There are four source modules to show in class. Module, class, function, and
method docstrings follow PEP 257: a concise summary, a blank line before
details, and documentation of inputs, outputs, and relevant side effects.
NumPy-style parameter sections are a formatting choice, not a PEP 257 rule.
Try `help()` on `make_rng` or `MeanTest.test` to show that these are real
docstrings rather than comments.

## Makefile targets

| Command | Effect |
| --- | --- |
| `make all` | Generate raw results, summaries, and both visualizations |
| `make simulate` | Run missing scenario batches and save raw outputs |
| `make analyze` | Rebuild summaries from saved outputs; simulation batches are reused |
| `make figures` | Rebuild summaries and diagnostic/main figures |
| `make test` | Run the four tests with Python's `unittest` |
| `make clean` | Delete generated raw results, tables, metadata, and figures |

The tests use the standard library to keep setup small. The assignment prefers
pytest but does not require it; this is an intentionally compact alternative.
No parallelism or optimization is needed for this project.

## Reuse, checkpoints, and provenance

Repeating the same command reuses the six scenario CSVs. Each file is a complete
batch of repetitions for one condition, written to a temporary file and renamed
when complete. If execution stops partway through a scenario, resume with the
same command: completed scenarios are reused and the unfinished one is rerun.
Do not run two writers against the same study directory.

The simulation ID includes simulation settings, simulation source, and
dependency versions. Changed simulation code or settings creates a new raw
directory. Each raw record has the unique key
`(study_id, scenario_id, replicate_id, method_id)`. A saved configuration, Git
commit when available, source snapshots/hashes, and version information record
what produced it. Content hashes cover uncommitted edits as well as commits.

SeedSequence keys give data and each method separate generators. Changing
execution order or replaying a repetition does not advance another task's
stream. Each repetition generates one dataset; both methods receive copies
of that dataset. Although these methods are deterministic, their shared
interface can accommodate methods with internal randomness.

Evaluation reads saved p-values. For example, change only the evaluation level:

```bash
python3 -m src.metrics analyze --alpha 0.01
python3 -m src.metrics figures
```

These commands generate no datasets and run no tests of means. They overwrite
the derived tables/figures, leaving raw outputs intact. Evaluation metadata and
a snapshot of `metrics.py` record the metric code and alpha used. Run `make all`
to restore the configured evaluation level. To use cached outputs from an older
study, point `results/raw/latest.json` to its saved study ID.

Failed fits remain in raw CSVs with status/messages and missing outputs.
Warnings remain attached to their results. Summaries report failure counts
and leave unconditional estimates missing when fits failed. There is one
method-error handler; configuration and interface errors otherwise surface
normally rather than being wrapped in a validation framework.

## Submission checklist

This directory includes the implementation, all requested Makefile targets,
two visualizations, tests, ADEMP, analysis, and setup/runtime/findings information.
For an actual submission, put it in a public GitHub repository and provide that
URL; upload `ANALYSIS.md` separately as requested. Creating this example does
not publish a repository.

Documentation reference: [PEP 257](https://peps.python.org/pep-0257/).