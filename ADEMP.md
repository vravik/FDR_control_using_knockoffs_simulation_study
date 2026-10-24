# ADEMP: FDR CONTROL VIA KNOCKOFFS 

Our goal is to reproduce the simulation results from the study "Controlling the 
false discovery rate via knockoffs" by Barber, R.F and Candes, E.J.  

**Aims.** Compare knockoffs with Benjamini Hochberg procedure for effectively 
controlling false discovery rates while retaining power. Compare performance of 
both methods with varying signal sparsity, signal amplitude and correlation 
between features

**Data-generating mechanisms.** Generate independent rows of X from 
N_p(0, Sigma_{\rho}), with Sigma_{\rho}[j,k]=rho**abs(j-k). When rho = 0, fix
Sigma = I. Normalize columns of 
X. Randomly pick k coefficients to assign to $\beta_j \in \{\pm A\}$, and fix 
all other coefficients at zero. Draw y ~ N(X\beta, I). In all cases, we fix n =
3000, p = 1000, and vary k, A and rho across experiements. A fresh dataset is 
generated in each replicate.

| Experiment | Vary | Hold fixed |
| --- | --- | --- |
| A | Chosen signal sparsity level k | A = 3.5, rho = 0 |
| B | Chosen signal amplitude A | k = 30, rho = 0 |
| C | Chosen feature correlation level rho | k = 30, A = 3.5 |


Exact values for parameters are set in `scenarios.py`. The complete design 
matrix, one row per scenario, is stored in `results/raw/design.csv`. The
saved metadata includes every setting, the replicate count, seed, and stream 
rule.

**Estimands/targets.** The set of non-zero coefficients in beta. We are 
interested in the task of model selection, and target is list of nominated 
non-zero coefficients at a prescribed nominal false discovery level. 

**Methods.** knockoff, knockoff+ (with SDP construction) and the Benjamini-Hochberg
produce for FDR control at a nominal level q = 20%. All methods are fit to the 
same dataset within each replicate. 

**Performance measures.** False discovery rate for each method when controlling 
FDR at a given nominal rate, and Power for recovering true positive signals. Each
Figure has two panels, reporting the mean FDR and mean power averaged across all
trials while varying the chose parameter for the experiment.

The starter uses master seed 20261020. Number of replicates per experiment are
set in make_scenarios, as authors use different numbers for each experiment.
Identical commands resume completed scenario batches; code/configuration changes
produce a new study ID. Scenario and replicate IDs identify stable, independent
task streams. 

