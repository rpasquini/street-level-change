.PHONY: run

run:
	@echo "Running pipeline..."
	@poetry run python -m scripts.run_pipeline
	@echo "Pipeline completed. Check data/regions directory for output files."
