.PHONY: panos unify plots

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