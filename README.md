# street-level-change

`street-level-change` is a Python-based pipeline to map and measure urban infrastructure change over time using Google Street View imagery.

## Overview

**Step 1: Define a Region**

Provide a polygon representing the area of interest.

[Region of interest](./assets/region.png)

**Step 2: Create a Grid of Points**
Generate evenly spaced points (e.g., every 10 meters) within the polygon using geospatial transformation to UTM coordinates for accurate spacing.

[Grid of points](./assets/region_points.png)

**Step 3: Query Street View API**
Each point is queried to check for nearby Google Street View panoramas, and metadata such as pano ID, location, and date are collected.

[Panos](./assets/region_points_panos.png)

## Dependencies

This project uses Poetry for dependency and environment management. To get started:

```bash
poetry install
```

To run the project:

```bash
poetry run python run.py
```
