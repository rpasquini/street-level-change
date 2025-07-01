.PHONY: run-demo

run-demo:
	@echo "Running demo script..."
	@poetry run python -m scripts.run_demo
	@echo "Demo completed. Check data/demo directory for output files."
