#!/usr/bin/env python
"""
This script shows how to run a region analysis using the pipeline module,
which provides a more modular and maintainable approach.
"""

from src.pipeline import run_region

def main():
    """Run the region analysis using the pipeline module."""
    # Region to process
    region_slug = "tresdefebrero"
    region_osm = "Partido de Tres de Febrero, Buenos Aires, Argentina"
    
    print(f"Processing region: {region_slug} ({region_osm})")
    run_region(region_slug, region_osm)
    print("Processing complete!")

if __name__ == "__main__":
    main()
