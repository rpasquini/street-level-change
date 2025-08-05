.PHONY: run fetch stego

run:
	@echo "Running pipeline..."
	@poetry run python -m scripts.run_pipeline
	@echo "Pipeline completed. Check data/regions directory for output files."

fetch:
	@echo "Fetching Street View images..."
	@poetry run python -m scripts.test_fetcher
	@echo "Street View images fetched. Check street_view_images directory for output files."

stego:
	@echo "Running STEGO..."
	@poetry run python -m scripts.test_stego
	@echo "STEGO completed. Check street_view_images/segmentation_results directory for output files."
