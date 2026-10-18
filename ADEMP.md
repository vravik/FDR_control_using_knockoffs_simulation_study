# ADEMP: FDR CONTROL VIA KNOCKOFFS 

Our goal is to reproduce the simulation results from the study "Controlling the false discovery rate via knockoffs" by Barber and Candes. 


## Aims

How accurately do a two-sided one-sample t-test and an estimated-SE normal
approximation maintain a nominal Type I error probability of 0.05? How does
calibration change with skewness and sample size? Does using a t reference
distribution solve the problem of nonnormality at small sample sizes?

## Data-generating mechanisms

Within each repetition, generate n IID observations from one frozen SciPy
distribution. Cross all three distributions with both sample sizes:

| Scenario ID | Distribution | SciPy parameters | n | True mean |
| --- | --- | --- | ---: | ---: |
| 20 | Normal | loc=0, scale=1 | 20 | 0 |
| 50 | Normal | loc=0, scale=1 | 200 | 0 |
| 10 | Exponential | scale=1 | 20 | 1 |
| 40 | Exponential | scale=1 | 200 | 1 |
| 30 | Lognormal | s=1, scale=1 | 20 | exp(1/2) |
| 60 | Lognormal | s=1, scale=1 | 200 | exp(1/2) |

The lognormal distribution has log-mean 0 and log-SD 1. Its ordinary mean is
exp(1/2), not its median 1. Use `dgp.mean()` to obtain the correct null value.
Normal data provide the exact-model sanity check; the other distributions
introduce increasing right skew. There are six conditions and two methods,
with 1,000 repetitions per condition in the default run.

## Estimands and targets

Each method tests the population **mean**, with the true DGP mean supplied as
the null. Every simulated null is true; rejection therefore represents a
Type I error. The primary target is each method's rejection probability in
each condition. A secondary target is the difference between those probabilities.

## Methods

Both methods use the sample mean and standard error s/sqrt(n), with s computed
using `ddof=1`. Both are two-sided. The t-test compares its standardized
statistic with Student's t distribution on n-1 degrees of freedom. The normal
approximation compares the same statistic with a standard normal distribution;
it does not assume a known variance. Apply both methods to the same generated
dataset in every repetition.

## Performance measures

Reject when p < alpha. Report the empirical rejection rate, its plug-in
Monte Carlo SE `sqrt(rate * (1-rate) / B)`, and an exact 95% binomial Monte
Carlo interval, where B is the number of repetitions. The intervals quantify
simulation uncertainty, not uncertainty about an individual sample's mean.

Compare methods using the within-repetition difference of rejection indicators:
`delta_b = reject_t_b - reject_normal_b`. Report `mean(delta)` and the paired
MCSE `std(delta, ddof=1) / sqrt(B)`. Join outputs by repetition ID rather than
by incidental row ordering.

Keep failed fits and warnings in the raw table. If a group contains a failed
fit or missing p-value, report the failure count and leave its unconditional
rejection-rate estimate missing. Do the same for failed pairs. The diagnostic
ECDF uses available p-values and explicitly displays valid/total counts.

## Reproduction and reporting

All statistical settings are in `config.json`. The default master seed is
1234567891. Construct PCG64 generators using a SeedSequence whose entropy is
`[scenario_id, replicate_id, purpose_id, method_id, root_seed]`. Purpose 0 is
for data with method ID 0; purpose 1 is for a method's internal randomness.
Repetition IDs start at zero. Pass RNG objects to sampling and methods. IDs
stay attached to their configurations when execution order changes.

Save one raw row per repetition and method, including the study/scenario/
repetition/method IDs, truth, estimate, SE, statistic, reference degrees of
freedom, p-value, status, warning/error messages, method time, and stream keys.
Publish a completed scenario CSV by temporary-write/rename. Resume by skipping
completed scenarios; an interrupted scenario is regenerated from its task keys.

Save the configuration, RNG plan, dependency versions, Git commit when
available, and source hashes/snapshots alongside the raw CSVs. A source/settings
digest identifies the simulation; metric code has its own evaluation provenance.
Raw datasets are cheap to regenerate, so only method outputs are cached.

`make all` runs simulation, analysis, and figures. The diagnostic figure shows
null p-value ECDFs in the lower tail against a uniform reference, faceted by
distribution and sample size. The main figure shows Type I error against n,
faceted by distribution, with exact Monte Carlo intervals. `make test` runs
four small tests of DGPs, methods, paired MCSEs, and reproducibility.
