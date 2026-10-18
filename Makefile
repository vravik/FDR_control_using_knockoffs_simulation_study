PYTHON ?= python3
SIM_ARGS ?=
ANALYZE_ARGS ?=

.PHONY: all simulate analyze figures clean test

all: figures

simulate:
	$(PYTHON) -m src.simulation $(SIM_ARGS)

analyze: simulate
	$(PYTHON) -m src.metrics analyze $(ANALYZE_ARGS)

figures: analyze
	$(PYTHON) -m src.metrics figures

test:
	$(PYTHON) -m unittest discover -s tests -v

clean:
	rm -rf results/raw results/figures
	rm -f results/summary.csv results/paired.csv results/evaluation.json results/metrics_source.py
