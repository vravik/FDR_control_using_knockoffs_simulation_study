"""Supplied checks for the simulation engine, not plotting choices."""

import numpy as np
import pandas as pd

from src.dgps import NormalDGP
from src.methods import BHq, knockoff, knockoff_plus, Method
from src.metrics import evaluate
from src.simulate import SimulationStudy



def test_methods_recover_noiseless_coefficients():
    data = NormalDGP(500, 50).generate(np.random.default_rng(11),rho=0,k=50,A=3.5)
    data["y"] = data["X"] @ data["beta_true"]
    for method in [BHq(), knockoff(), knockoff_plus()]:
        estimate = method.fit(data, alpha = 0.2, rng=np.random.default_rng(12))
        np.testing.assert_allclose(estimate, data["beta_true"]!=0, atol=1e-7)


def test_evaluation_metric_range_check():
    data = NormalDGP(50, 5).generate(np.random.default_rng(173),rho=0,k=5,A=1)
    for method in [BHq(), knockoff(), knockoff_plus()]:
        estimate = method.fit(data, alpha = 0.2,rng=np.random.default_rng(15))
        eval  = evaluate(estimate, data)
        assert 0 <=  eval['fdp'] <= 1
        assert 0 <= eval['power']  <= 1



def test_study_is_reproducible_alone_and_after_grid_reordering(tmp_path):
    settings = {"experiment": "A", "n": 1000, "p": 100, "rho": 0.2, "k":10,
                "A":2.5, "n_reps":3,"error_distribution": "normal"}
    scenarios = {"first": settings, "second": {**settings, "k": 20}}
    combined = SimulationStudy(scenarios,alpha = .1,seed=41, out_dir=tmp_path / "both").run()
    reversed_grid = SimulationStudy(dict(reversed(list(scenarios.items()))), alpha=.1,
                                    seed=41, out_dir=tmp_path / "reverse").run()
    alone = SimulationStudy({"first": settings}, alpha=.1, seed=41,
                            out_dir=tmp_path / "alone").run()
    columns = ["scenario_id", "rep", "method", "fdp", "power"]
    def ordered(frame):
        return frame[columns].sort_values(columns[:3]).reset_index(drop=True)
    pd.testing.assert_frame_equal(ordered(combined), ordered(reversed_grid))
    pd.testing.assert_frame_equal(ordered(combined[combined.scenario_id == "first"]), ordered(alone))
    assert len(combined) == 18


def test_failed_fits_are_saved_and_completed_batches_resume(tmp_path, monkeypatch):
    class BrokenMethod(Method):
        name, settings = "broken", {}

        def fit(self, data, alpha, rng):
            raise RuntimeError("deliberate test failure")

    scenarios = {"check": {"experiment": "A", "n": 100, "p": 20, "rho": 0.,
                           "k":10,"A":1,"n_reps":2,"error_distribution": "normal"}}
    study = SimulationStudy(scenarios,alpha=.2, out_dir=tmp_path,
                            methods=[BHq(), BrokenMethod()])
    results = study.run()
    failed = results[results.method == "broken"]
    assert len(failed) == 2
    assert (failed.status == "failed").all()
    assert failed.fdp.isna().all()
    assert failed.power.isna().all()
    assert failed.message.str.contains("deliberate test failure").all()
    with np.load(study.study_dir / "check.npz", allow_pickle=False) as saved:
        assert saved["estimates"].shape == (2, 2, 20)
        assert np.isnan(saved["estimates"][:, 1, :]).all()
    def no_generation(self, rng):
        raise AssertionError("A completed batch should not generate new data")
    monkeypatch.setattr(NormalDGP, "generate", no_generation)
    resumed = study.run()
    pd.testing.assert_frame_equal(results, resumed, check_dtype=False)
