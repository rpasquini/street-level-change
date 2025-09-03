.PHONY: run fetch stego change

run:
	@echo "Running pipeline..."
	@poetry run python -m scripts.run_pipeline
	@echo "Pipeline completed. Check data/regions directory for output files."

evalclusters:
	@echo "Running cluster parameters evaluation..."
	@poetry run python -m scripts.cluster_analysis
	@echo "Pipeline completed. Check data/regions directory for output files."

fetch:
	@echo "Fetching Street View images..."
	@poetry run python -m scripts.test_fetcher
	@echo "Street View images fetched. Check street_view_images directory for output files."

stego:
	@echo "Running STEGO..."
	@poetry run python -m scripts.test_stego
	@echo "STEGO completed. Check street_view_images/segmentation_results directory for output files."

change:
	@echo "Running change detection..."
	@poetry run python -m scripts.change_detection
	@echo "Change detection completed. Check data/region_slug/segmentation_results directory for output files."