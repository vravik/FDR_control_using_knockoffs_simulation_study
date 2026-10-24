PYTHON ?= python3
SEED ?= 20261021
ALPHA ?= 0.2
ANALYZE_ARGS ?= 

.PHONY: all simulate analyze figures clean test

all: figures
	
	
# setup environment and dependencies
setup: .setup_complete

.setup_complete: requirements.txt 
	@echo "setting up the environment"
	$(PYTHON) -m venv .venv	
	./.venv/bin/python3 -m pip install --upgrade pip wheel setuptools
	./.venv/bin/python3 -m pip install -r requirements.txt
	touch .setup_complete

simulate: .simulation_complete

.simulation_complete : .setup_complete
	.venv/bin/$(PYTHON) -m src.simulate --seed $(SEED) --alpha $(ALPHA)
	touch .simulation_complete

analyze : .analysis_complete

.analysis_complete: .simulation_complete
	$(PYTHON) -m src.metrics $(ANALYZE_ARGS)
	touch .analysis_complete

figures : .figures_complete

.figures_complete: .analysis_complete
	$(PYTHON) -m src.metrics --stage figures
	touch .figures_complete



test:
	$(PYTHON) -m unittest discover -s tests -v

clean:
	rm -rf results/raw results/figures
	rm -f results/summary.csv results/evaluation.json results/metrics_source.py
	rm -f .setup_complete .simulation_complete .analysis_complete .figures_complete