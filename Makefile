.PHONY: panos unify plots run_regions

panos:
	@echo "Running panos script..."
	@poetry run python -m scripts.panos
	@echo "Panos completed. Check data/demo directory for output files."

unify:
	@echo "Running point unification on panorama data..."
	@poetry run python -m scripts.unify
	@echo "Point unification completed. Check data/point_unification_results directory for output files."

plots:
	@echo "Running plot generation..."
	@poetry run python -m scripts.plots
	@echo "Plot generation completed. Check data/plots directory for output files."

run_regions:
	@echo "Running region processing..."
	@poetry run python -m scripts.run_regions
	@echo "Region processing completed. Check data/regions directory for output files."
