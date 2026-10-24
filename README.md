# FDR CONTROL VIA KNOCKOFFS 

A simulation study to reproduce the simulation results from the study "Controlling 
the  false discovery rate via knockoffs" by Barber, R.F and Candes, E.J. under 
different regression scenarios

## Setup and run

Use Python 3.11 or newer and `make`. From this project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
make test
make all PYTHON=python3
```

The study has three experiments, with varying number of replications. Default 
number of replicates per experiment are 20 each, although the authors use 600, 
200, 200 replications in their study design. Number of repetitions is set in 
make_scanarios. The simulation takes a long time (about 2 hours with 10 
repetitions per scenario). Analysis is instant. The entire pipeline can be run 
by calling

```bash
make all
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
| `src/methods.py` | Common regression fitting interface and three methods |
| `src/simulation.py` | Shared datasets, raw outputs, provenance, checkpoints |
| `src/metrics.py` | Evaluation, and scripts to generate three figures |
| `tests/test_study.py` | Four focused standard-library unit tests |
| `data/simulated/` | Dataset cache |
| `results/raw/` | Study-specific CSVs, metadata, and source snapshots |
| `results/summary.csv` | Derived tables |
| `results/figures/` | main PNG/PDF |
| `ADEMP.md`, `ANALYSIS.md` | Design and critical interpretation |


## Makefile targets

| Command | Effect |
| --- | --- |
| `make all` | Generate raw results, summaries, and all visualizations |
| `make simulate` | Run missing scenario batches and save raw outputs |
| `make analyze` | Rebuild summaries from saved outputs; simulation batches are reused |
| `make figures` | Rebuild summaries and main figures |
| `make test` | Run tests with Python's `pytest` |
| `make clean` | Delete generated raw results, tables, metadata, and figures |


## Reuse, checkpoints, and provenance

Repeating the same command reuses the scenario CSVs. Each file is a complete
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

Targets are computed along with data generation. To change evaluation level, use:

```bash
python3 -m src.simulate --alpha 0.1
python3 -m src.metrics --stage analyze
```

Since the output of each scenario is the fitted coefficients, the targets are tied
to the data generation process itself. It is not possible to evaluate at a different
nominal FDR threshold without running the methods on each data, and this is the 
time-consuming task. The nominal FDR threshold is an intrinsic parameter of the 
SimulationStudy class. Run `make all` to restore the configured evaluation level.
To use cached outputs from an older study, point `results/raw/latest.json` to 
its saved study ID.

Failed fits remain in raw CSVs with status/messages and missing outputs.
Warnings remain attached to their results. Summaries report failure counts
and leave unconditional estimates missing when fits failed. There is one
method-error handler; configuration and interface errors otherwise surface
normally rather than being wrapped in a validation framework.

## Submission checklist

This directory includes the implementation, all requested Makefile targets,
three visualizations, tests, ADEMP, analysis, and setup/runtime/findings information.

Documentation reference: [PEP 257](https://peps.python.org/pep-0257/).