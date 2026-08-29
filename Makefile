.PHONY: setup serve smoke test eval chaos clean

PY ?= python
N  ?= 100

setup:
	@echo "Installing dependencies and initialising the database..."
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install pyyaml
	$(PY) -m agent.storage
	@echo "Setup complete."

# Local stand-in for the mockllm/ server the brief says is provided but was not.
# See mockllm_local/README.md. `eval` and `chaos` start their own servers on
# other ports, so this is only needed for interactive `agent run`.
serve:
	$(PY) -m mockllm_local --scenario $(or $(SCENARIO),S1)

# Checks that the mock server itself misbehaves as advertised. Needs `make serve`.
smoke:
	$(PY) -m mockllm_local.smoke_test

# Offline unit tests. No sockets, so this passes with networking off.
test:
	$(PY) -m unittest discover -s tests -v

# R7. Starts its own mock server on 8765 and uses a scratch database.
eval:
	$(PY) evals/eval_suite.py

# R2. Starts its own mock server on 8766, kills the agent mid-run N times,
# resumes it, and asserts send_email fired exactly once per logical send.
chaos:
	$(PY) harness/chaos.py -n $(N)

clean:
	rm -f agent_state.db agent_state.db-wal agent_state.db-shm
	rm -f chaos_state.db chaos_state.db-wal chaos_state.db-shm
	rm -rf traces chaos_traces
	find . -name __pycache__ -type d -exec rm -rf {} +
