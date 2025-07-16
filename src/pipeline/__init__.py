"""
Pipeline module for Street Level Change Detection.

This module provides components and workflows for processing street-level imagery data.
"""

from .workflows import run_region
from .components import (
    process_region,
    process_panos,
    process_dbscan,
    process_barrios,
    evaluate_clustering,
    evaluate_clustering_full,
    calculate_coverage_area
)

__all__ = [
    'run_region',
    'process_region',
    'process_panos',
    'process_dbscan',
    'process_barrios',
    'evaluate_clustering',
    'evaluate_clustering_full',
    'calculate_coverage_area'
]
