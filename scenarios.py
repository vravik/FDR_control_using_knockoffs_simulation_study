"""Specify the simulation design in one place."""

import numpy as np

def make_scenarios():
    """Return {scenario_id: settings} for experiments A, B and C.

    IDs are unique, stable, filename-safe strings. Each settings dictionary
    has exactly experiment ('A'/'B'/'C'), n (int), p (int), rho (float), k (int), 
    A (float), n_reps (int). Assume 0 < p < n, 0 <= rho < 1. 
    The study supplies all generation, fitting, evaluation, and replication.

    Keeping n = 3000, p = 1000 across all experiments; 
    """
    # BEGIN TASK GRID
    scenarios = {}

    for k in range(10, 201, 10):
        scenarios[f'expt_A_sparsity_{k}'] = {
            'experiment' : "A",
            'n' : 3000,
            'p' : 1000,
            'k' : k,
            'A' : 3.5,
            'rho' : 0,
            'n_reps' : 600, 
            "error_distribution": "normal"
        }

    for amplitude in np.round(np.arange(2.8, 4.2, .1), 1):
        scenarios[f'expt_B_amplitude_{amplitude}'] = {
                'experiment' : 'B',
                'n' : 3000,
                'p' : 1000,
                'k' : 30,
                'A' : amplitude,
                'rho' : 0, 
                'n_reps' : 200, 
                "error_distribution": "normal"
            }
        

    for correlation in np.round(np.arange(0, 1, 0.1),1):
        scenarios[f'expt_C_rho_{correlation}'] = {
                'experiment' : 'C',
                'n' : 3000,
                'p' : 1000,
                'k' : 30,
                'A' : 3.5,
                'rho' : correlation,
                'n_reps' : 200, 
                "error_distribution": "normal"
            }
        

    return scenarios
    
    # END TASK GRID
