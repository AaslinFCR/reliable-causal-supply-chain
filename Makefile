PYTHON ?= python

setup:
	$(PYTHON) -m pip install -e ".[dev]"
data:
	$(PYTHON) run.py scrc.data.official
	$(PYTHON) run.py scrc.data.build
test:
	$(PYTHON) run.py pytest -q
experiments:
	$(PYTHON) run.py scrc.eval.replay
figures:
	$(PYTHON) run.py scrc.eval.plots
demo:
	$(PYTHON) run.py scrc.demo
reproduce:
	$(PYTHON) scripts/verify_reproducibility.py
check:
	$(PYTHON) -m ruff check scrc tests
	$(PYTHON) -m black --check scrc tests
