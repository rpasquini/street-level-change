# Street Level Change

`street-level-change` is a Python-based project aimed at measuring urban change at the street level by leveraging Google Street View imagery. The primary focus is on analyzing informal housing settlements in Argentina to understand their evolution and dynamics over time.

## Research Question

How have informal housing settlement areas changed through time?

## Project Overview

This project is at an early stage, exploring how to obtain panorama points from Google Street View around informal housing areas from Argentina's ReNaBaP (Registro Nacional de Barrios Populares - National Registry of Popular Neighborhoods).

### Methodology

- Using Python and Google Street View API to obtain images in informal areas
- Employing computer vision models to quantify or analyze changes in these areas
- Indexing panorama points to consistent locations to assess change dynamics accurately
- Calculating coverage area by informal settlement polygon

### Scope

- Currently focused on ReNaBaP data
- Methodology tests are being conducted in:
  - La Plata, Buenos Aires
  - San Isidro, Buenos Aires
  - 3 de Febrero, Buenos Aires (Argentina)

### Limitations

- The Google Street View API only accepts latitude and longitude points for panorama IDs, not polygons
- Data quality is challenged by coverage; not all informal areas have Street View imagery
- Quantifying urban change in these areas is complex
- Panoramas are not always taken from the exact same location, and not all areas have temporal coverage

## Installation

This project uses Poetry for dependency and environment management:

```bash
# Install dependencies
poetry install
```

## Usage

The project provides a Makefile with commands to run different functionalities:

### Run Demo

Executes the demo script that fetches panorama data for a sample region (La Plata, Buenos Aires):

```bash
make run-demo
```

This will:
1. Define a region of interest (La Plata, Buenos Aires)
2. Create a grid of points within the region
3. Query the Street View API for panoramas at each point
4. Save the results to the `data/demo` directory

### Point Unification

Run the point unification algorithm to group panorama points based on spatial proximity:

```bash
make unify
```

This will:
1. Load panorama data from the demo
2. Apply both H3-based hexagon grouping and DBSCAN clustering methods
3. Save the results to the `data/point_unification_results` directory
4. Generate visualization plots for the results

## Workflow Overview

### Step 1: Define a Region

Provide a polygon representing the area of interest.

![Region of interest](./assets/region.png "Region of interest")

### Step 2: Create a Grid of Points

Generate evenly spaced points (e.g., every 50 meters) within the polygon using geospatial transformation to UTM coordinates for accurate spacing.

```python
points = src.utils.create_point_grid(region, 50)
```

![Grid of points](./assets/region_points.png "Grid of points")

### Step 3: Query Street View API

Each point is queried to check for nearby Google Street View panoramas, and metadata such as pano ID, location, and date are collected.

```python
panos = src.sv.get_panos(points)
```

![Panos](./assets/region_points_panos.png "Resulting panoramas")

### Step 4: Point Unification

Group panorama points based on spatial proximity using either:
- H3-based hexagon grouping
- DBSCAN clustering with haversine distance

```python
# H3 unification
h3_results = h3_unification(panos_gdf, resolution=11)

# DBSCAN unification
dbscan_results = dbscan_unification(panos_gdf, eps_meters=5, min_samples=1)
```

## Next Steps & Future Developments

- Indexing panorama points to consistent locations within the areas to assess change dynamics more accurately
- Calculating coverage area by informal settlement polygon, where panorama points have an "area of influence" that intersects with the polygon (coverage = area of intersection / polygon area)
- Downloading Google Street View images for proof-of-concept areas
- Implementing computer vision models to quantify changes over time

## Roadmap

### Primary

- [ ] Connect panorama points to the OpenStreetMaps street grid (https://github.com/rpasquini/street-level-change/issues/3) and calculate area of coverage in meters (https://github.com/rpasquini/street-level-change/issues/1)

### Secondary

- [ ] Generate a street grid from Google Street View images (https://github.com/rpasquini/street-level-change/issues/4)
