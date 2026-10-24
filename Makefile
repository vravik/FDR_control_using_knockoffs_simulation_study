PYTHON ?= python3
SEED ?= 20261021
ALPHA ?= 0.2
ANALYZE_ARGS ?=

.PHONY: all simulate analyze figures clean test

# setup environment and dependencies
setup: .setup_complete

.setup_complete: requirements.txt 
	@echo "setting up the environment"
	$(PYTHON) -m venv .venv	
	./.venv/bin/python3 -m pip install --upgrade pip wheel setuptools
	./.venv/bin/python3 -m pip install -r requirements.txt
	touch .setup_complete
	
all: figures

simulate:
	.venv/bin/$(PYTHON) simulate.py --seed $(SEED) --alpha $(ALPHA)


analyze: simulate
	$(PYTHON) -m src.metrics analyze $(ANALYZE_ARGS)

figures: analyze
	$(PYTHON) -m src.metrics figures

test:
	$(PYTHON) -m unittest discover -s tests -v

clean:
	rm -rf results/raw results/figures
	rm -f results/summary.csv results/paired.csv results/evaluation.json results/metrics_source.py
