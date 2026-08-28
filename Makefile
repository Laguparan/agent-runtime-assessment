.PHONY: setup test eval

setup:
	@echo "Setting up the environment..."
	python -m pip install --upgrade pip
	python agent/storage.py
	@echo "Setup complete."

test:
	@echo "Running core unit tests (simulated)..."
	python agent/tools.py
	python agent/mock_client.py
	@echo "Tests passed."

eval:
	@echo "Running the evaluation suite..."
	python evals/eval_suite.py