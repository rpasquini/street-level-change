# Development Notes

## STEGO Module Integration & Google Street View API Setup (2025-07-22)

### STEGO Module Integration

Added [STEGO](https://github.com/mhamilton723/STEGO) module to the project under `src/ml/stego/`.

### Problem Description
When running `make stego`, the command failed with import errors. The `scripts/test_stego.py` script could not import from the stego module due to module import issues within the STEGO codebase.

### Root Cause Analysis
The STEGO module files in `src/ml/stego/src/` were using absolute imports instead of relative imports, causing `ModuleNotFoundError` when trying to import internal dependencies like `utils`, `modules`, and `data`.

**Initial Error:**
```
ModuleNotFoundError: No module named 'utils'
```

### Solution Applied

#### 1. Fixed Relative Imports in STEGO Module Files

Updated the following files to use proper relative imports:

**`src/ml/stego/src/train_segmentation.py`:**
```python
# Before:
from utils import *
from modules import *
from data import *

# After:
from .utils import *
from .modules import *
from .data import *
```

**`src/ml/stego/src/modules.py`:**
```python
# Before:
from utils import *
import dino.vision_transformer as vits

# After:
from .utils import *
from .dino import vision_transformer as vits
```

**`src/ml/stego/src/crf.py`:**
```python
# Before:
from utils import unnorm

# After:
from .utils import unnorm
```

**`src/ml/stego/src/dino/vision_transformer.py`:**
```python
# Before:
from dino.utils import trunc_normal_

# After:
from .utils import trunc_normal_
```

**Other files fixed:**
- `src/ml/stego/src/train_crf.py`
- `src/ml/stego/src/demo_segmentation.py`
- `src/ml/stego/src/eval_segmentation.py`
- `src/ml/stego/src/crop_datasets.py`
- `src/ml/stego/src/plot_dino_correspondence.py`
- `src/ml/stego/src/download_datasets.py`

#### 2. Added STEGO Dependencies to Poetry Group

Created a dedicated Poetry dependency group for STEGO-related packages in `pyproject.toml`:

```toml
[tool.poetry.group.stego.dependencies]
torch = ">=2.7.1,<3.0.0"
wget = ">=3.2,<4.0"
torchmetrics = ">=1.7.4,<2.0.0"
torchvision = ">=0.22.1,<0.23.0"
tensorboard = ">=2.20.0,<3.0.0"
dino = {extras = ["all"], version = ">=0.0.6,<0.0.7"}
hydra-core = ">=1.3.2"
omegaconf = ">=2.3.0"
pytorch-lightning = ">=2.0.0"
pydensecrf = {git = "https://github.com/lucasb-eyer/pydensecrf.git"}
```

**Installation Commands:**
```bash
# Install STEGO dependencies when needed
poetry install --with stego

# Install without STEGO dependencies (core project only)
poetry install
```

### Verification

After applying the fixes, the import test passes successfully:

```bash
poetry run python -c "from src.ml.segment_street_view import ImageSegmenter; print('Import successful!')"
# Output: Import successful!
```

The `make stego` command now successfully imports all stego modules.

## Google Street View API Integration

### Setup
Successfully integrated Google Street View API for downloading street-level imagery. The API key is configured and tested for image retrieval.

### New Scripts Added
Added two new scripts to the `scripts/` folder for testing and running the pipeline:

**`scripts/test_fetcher.py`:**
- Tests Google Street View API integration
- Downloads street view images using the configured API key
- Validates image retrieval functionality

**`scripts/test_stego.py`:**
- Tests STEGO segmentation functionality
- Loads STEGO checkpoint and processes images
- Generates segmentation results and metrics

### Makefile Commands
The project now supports the following make commands:

```bash
# Run the full pipeline
make run

# Test Google Street View image fetching
make fetch

# Test STEGO segmentation
make stego
```

### Project Structure

The STEGO module is located at:
```
src/ml/stego/
├── __init__.py
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── utils.py
│   ├── modules.py
│   ├── train_segmentation.py
│   ├── crf.py
│   ├── data.py
│   ├── dino/
│   │   ├── utils.py
│   │   └── vision_transformer.py
│   └── ... (other files)
└── ... (other directories)
```

### Commands for Future Reference

To test STEGO imports:
```bash
poetry run python -c "from src.ml.segment_street_view import ImageSegmenter; print('Import successful!')"
```

To run STEGO processing:
```bash
make stego
```

### Status
✅ **COMPLETED**: 
- All STEGO module import issues have been resolved
- STEGO dependencies organized in dedicated Poetry group for optional installation
- Google Street View API integration tested and working
- New test scripts (`test_fetcher.py`, `test_stego.py`) added to `scripts/` folder
- Makefile commands (`make run`, `make fetch`, `make stego`) configured and functional

## Coverage Calculation Function Update (July 28, 2025)

Updated `calculate_coverage_area()` in `src/pipeline/components.py` to use OpenStreetMap (OSM) road length data for more accurate coverage calculations.

### How it works:

The function calculates coverage as the **proportion of streets near and inside a polygon that are also intersected by capture point buffers corresponding to panorama clusters**.

### Key steps in the calculation:

1. **Buffer capture points**: Creates 15-meter buffers around all Google Street View capture points (`capture_points = buffer_region(capture_points, buffer_dist=buffer_dist)`)

2. **Get total OSM roads for each polygon**: For each polygon, retrieves OSM road data that includes:
   - Streets that intersect with the capture point buffers
   - Streets that intersect with a 5-meter buffer around the polygon (to capture streets that might be right outside the polygon boundary)
   
   This approach better captures roads that are near each polygon, including those just outside the polygon that are still relevant for coverage analysis.

3. **Calculate total road length**: Sums up all road lengths from the combined set of roads (inside polygon + near polygon) (`total = roads["roadlength"].sum()`)

4. **Find covered roads**: Clips the roads to the union of all buffered capture points (`result = roads.clip(capture_points.union_all())`) and recalculates lengths in projected coordinates (EPSG:3857)

5. **Calculate coverage proportion**: Computes the ratio of covered road length to total road length (`coverage = partial / total`)

### Result:
- **Coverage = (Length of roads intersected by capture point buffers) / (Total length of roads near and inside polygon)**
- This gives a more comprehensive representation of street-level coverage by including roads that are just outside the polygon boundary but still relevant for the area's street network coverage analysis.

The function processes each polygon individually and returns coverage metrics including total road length, covered road length, and the coverage proportion for each area.

### Recent Enhancement (August 2025):
The coverage calculation has been enhanced to better capture roads near each polygon by:
- Including streets that intersect with capture point buffers
- Adding streets that intersect with a 5m buffer around the polygon to capture roads just outside the polygon boundary
- Calculating coverage as the proportion of these combined roads that are intersected by capture point buffers from panorama clusters

## Heading and Field-of-View (FOV) Processing Implementation (July 28, 2025)

Implemented a comprehensive heading and FOV calculation system for panorama data to determine viewing directions and angles from capture points.

### New Components Added:

#### 1. `src/core/heading_fov.py` - Geometric Utilities
- **`calculate_heading(p1, p2)`**: Computes heading in degrees from point p1 to p2
- **`calculate_fov(panorama, p_start, p_end)`**: Calculates field-of-view angle between two vectors
- **`get_segment_points(center, radius, start_angle, end_angle)`**: Generates points along circular arc segments
- **`split_fov(heading, fov, max_fov=120)`**: Splits large FOVs into smaller sub-FOVs when exceeding maximum
- **`get_angles(panorama_point, control_point, max_distance=10, max_fov=120)`**: Main function that calculates heading and FOV for N, E, S, W directional segments

#### 2. `process_heading_fov()` in `src/pipeline/components.py`
Processes heading and FOV data for panorama collections:
- Projects data to EPSG:3857 for accurate distance calculations
- Iterates through control points and their associated panoramas
- Generates directional views (N, E, S, W) with heading and FOV angles
- Creates unique `view_id` for each panorama-direction combination
- Exports results to `heading_fov.csv`

#### 3. Integration in `src/pipeline/workflows.py`
Added heading/FOV processing to the main `run_region()` workflow:
```python
heading_fov = process_heading_fov(
    panos=joined,
    control_points=centroids,
    data_dir=output_dir,
    max_distance=10,
    max_fov=120
)
```

### How the System Works:

1. **Directional Segmentation**: Divides the 360° view around each control point into 4 cardinal directions (N, E, S, W) with specific angle ranges
2. **Geometric Calculations**: For each direction, calculates the heading (direction from panorama to segment center) and FOV (angular width of the segment as seen from the panorama)
3. **FOV Splitting**: Automatically splits large FOVs (>120°) into multiple smaller views to comply with Google Street View API's maximum FOV limit of 120°
4. **Output Generation**: Creates comprehensive dataset with `pano_id`, `cluster_id`, `direction`, `heading`, `fov`, and unique `view_id`

### Result:
This system enables precise modeling of what areas are visible from each panorama location, supporting more accurate coverage analysis and potential integration with computer vision models that require specific viewing angles and directions.

## Panorama Database Enhancement Plan (August 8, 2025)

### Problem Identified

While reviewing DBSCAN clusters with only 1 panorama, I discovered that many of these locations actually have multiple panoramas available over time in Google Street View. This indicates that our panorama database is incomplete, particularly for temporal sequences at the same location.

### Root Cause Analysis

The current panorama collection approach in `process_panos()` uses a fixed grid pattern to query the Google Street View API:

```python
points_gdf = create_point_grid_from_gdf(renabap_buffered, dist_points_grid)
panoramas = get_panoramas_for_points(points_gdf, verbose=True)
```

This grid-based approach has limitations:
1. The grid spacing may not align perfectly with actual panorama locations
2. Temporal sequences (multiple panoramas taken at the same location over time) may be missed if the grid point isn't exactly at the panorama location
3. The Street View API may return only the most recent panorama for a given query point, missing historical data

### Implemented Enhancement

Instead of using a fixed grid pattern or even the initial panorama points, I implemented an approach that uses DBSCAN centroids to find additional panoramas:

1. **Initial Collection**: Start with the current grid-based approach to get an initial set of panorama points
2. **DBSCAN Clustering**: Apply DBSCAN clustering to group nearby panoramas and generate centroids
3. **Centroid-Based Enrichment**: Use each DBSCAN centroid's lat/lon coordinates to query for additional panoramas at those precise locations

### Implementation Details

#### 1. Created New Enrichment Function

Implemented a new function in `src/pipeline/components.py` that uses DBSCAN centroids to enrich the panorama database:

```python
def enrich_panorama_database_from_centroids(
    centroids: gpd.GeoDataFrame, 
    renabap_buffered: gpd.GeoDataFrame, 
    data_dir: str, 
    max_workers: int = 10, 
    verbose: bool = True
) -> gpd.GeoDataFrame:
    """Enrich panorama database by querying at each DBSCAN centroid location."""
    # Path for enriched panoramas
    enriched_panos_path = os.path.join(data_dir, "panos_enriched.csv")
    
    # Check if enriched panoramas already exist
    if os.path.exists(enriched_panos_path):
        print(f"Loading existing enriched panoramas from {enriched_panos_path}")
        return load_panorama_data(enriched_panos_path)
    
    # Load original panoramas for comparison
    original_panos_path = os.path.join(data_dir, "panos.csv")
    original_panoramas = None
    if os.path.exists(original_panos_path):
        original_panoramas = load_panorama_data(original_panos_path)
        print(f"Loaded {len(original_panoramas)} original panoramas for comparison")
    
    # Query panoramas at each centroid location
    print(f"Querying panoramas at {len(centroids)} centroid locations...")
    enriched_panoramas = get_panoramas_for_points(
        centroids, max_workers=max_workers, verbose=verbose
    )
    enriched_panoramas = enriched_panoramas.clean(renabap_buffered)
    
    # Combine with original panoramas
    combined_panoramas = []
    if original_panoramas is not None:
        # Add original panoramas
        for _, panorama in original_panoramas.iterrows():
            combined_panoramas.append(panorama)
        
        # Track existing IDs to avoid duplicates
        existing_ids = [panorama.pano_id for _, panorama in original_panoramas.iterrows()]
        
        # Add new panoramas from centroids
        for _, panorama in enriched_panoramas.iterrows():
            if panorama.pano_id not in existing_ids:
                combined_panoramas.append(panorama)
        
        # Report statistics
        original_count = len(original_panoramas)
        combined_count = len(combined_panoramas)
        new_count = combined_count - original_count
        
        print(f"Original panorama count: {original_count}")
        print(f"New panoramas found: {new_count}")
        print(f"Total combined panoramas: {combined_count}")
    
    # Create PanoramaCollection and save results
    combined_panoramas = PanoramaCollection(combined_panoramas)
    export_to_csv(combined_panoramas.to_dataframe(), enriched_panos_path)
    
    return combined_panoramas
```

#### 2. Updated Pipeline Workflow

Modified the pipeline in `src/pipeline/workflows.py` to include the enrichment step after DBSCAN clustering and re-run DBSCAN on the enriched panoramas:

```python
# Process initial DBSCAN clustering
dbscan_results, centroids = process_dbscan(
    panoramas, dbscan_eps, dbscan_min_samples, output_dir
)

# Enrich panorama database using DBSCAN centroids
enriched_panoramas = enrich_panorama_database_from_centroids(
    centroids=centroids,
    renabap_buffered=renabap_buffered,
    data_dir=output_dir,
    max_workers=10,
    verbose=True
)

# Re-run DBSCAN on enriched panoramas to get final centroids
enriched_dbscan_results, enriched_centroids = process_dbscan(
    enriched_panoramas, 
    eps=dbscan_eps, 
    min_samples=dbscan_min_samples, 
    data_dir=output_dir,
    output_prefix="enriched_"
)

# Use enriched results for downstream tasks
centroids = enriched_centroids
dbscan_results = enriched_dbscan_results
```

#### 3. Final DBSCAN Re-run

After enriching the panorama database with additional temporal panoramas from centroid locations, we re-run DBSCAN on the complete dataset to generate a final, comprehensive set of clusters and centroids:

1. **Why Re-run DBSCAN?**
   - The enriched panorama dataset now contains additional temporal panoramas at existing locations
   - These new panoramas may form different spatial patterns and clusters
   - Re-running DBSCAN ensures we capture the most accurate representation of panorama clusters based on the complete dataset

2. **Implementation Details**
   - Modified `process_dbscan()` to support output prefixes for different runs
   - Added output prefix parameter to save enriched DBSCAN results separately
   - Implemented comparison statistics to track improvements

3. **Benefits of Final Centroids**
   - More accurate representation of panorama clusters
   - Better spatial coverage for downstream analysis
   - Improved temporal representation at each centroid location
   - Enhanced data for coverage calculations and heading/FOV processing

#### 4. Key Features of the Implementation

- **Caching**: The function checks if enriched panoramas already exist before running
- **Deduplication**: Ensures no duplicate panorama IDs when combining original and new panoramas
- **Statistics**: Reports the number of original panoramas, new panoramas found, and total combined count
- **Filtering**: Cleans the enriched panoramas to ensure they're within the buffered region of interest
- **Two-stage DBSCAN**: Initial clustering to find centroids, then re-clustering on enriched data
- **Comparison Metrics**: Tracks improvements in panorama count and cluster count

### Expected Benefits

1. **More Complete Dataset**: Capture the full temporal history of panoramas at each location
2. **Better Temporal Analysis**: Enable analysis of changes over time at specific locations
3. **Improved Coverage Metrics**: More accurate representation of areas with Street View coverage
4. **Enhanced DBSCAN Clustering**: Better input data for the spatial clustering process
5. **Optimized Queries**: Using centroids rather than all panorama points reduces API calls while still capturing all relevant locations

### Next Steps

1. Run the updated pipeline on a test region to measure the increase in panorama count
2. Analyze the temporal distribution of the enriched panorama dataset
3. Evaluate whether re-running DBSCAN on the enriched dataset improves clustering results
4. Consider further optimizations to the panorama collection process

### Summary
This session successfully:
1. **Fixed STEGO Import Issues**: Resolved all relative import problems in the STEGO module
2. **Organized Dependencies**: Created `[tool.poetry.group.stego.dependencies]` for modular dependency management
3. **Validated API Integration**: Confirmed Google Street View API is working for image downloads
4. **Enhanced Project Structure**: Added test scripts and Makefile commands for streamlined development workflow
5. **Updated Coverage Calculation**: Improved `calculate_coverage_area()` to use OSM road length data for more precise street-level coverage metrics
6. **Implemented Heading/FOV Processing**: Added comprehensive geometric utilities and workflow integration for calculating panorama viewing directions and field-of-view angles
7. **Planned Panorama Database Enhancement**: Developed approach to improve panorama collection by using existing panorama locations to find additional temporal data
