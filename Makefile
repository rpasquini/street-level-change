.PHONY: run-demo unify

run-demo:
	@echo "Running demo script..."
	@poetry run python -m scripts.run_demo
	@echo "Demo completed. Check data/demo directory for output files."

unify:
	@echo "Running point unification on panorama data..."
	@poetry run python -m scripts.test_point_unification
	@echo "Point unification completed. Check data/point_unification_results directory for output files."
