# street-level-change

`street-level-change` is a Python-based pipeline to map and measure urban infrastructure change over time using Google Street View imagery.

## Dependencies

This project uses Poetry for dependency and environment management. To get started:

```bash
poetry install
```

To run the project:

```bash
poetry run python run.py
```

## Workflow Overview

**Step 1: Define a Region**:

Provide a polygon representing the area of interest.

![Region of interest](./assets/region.png "Region of interest")

**Step 2: Create a Grid of Points**:

Generate evenly spaced points (e.g., every 10 meters) within the polygon using geospatial transformation to UTM coordinates for accurate spacing.

```python
points = src.utils.create_point_grid(region, 10)
```

![Grid of points](./assets/region_points.png "Grid of points")

**Step 3: Query Street View API**:

Each point is queried to check for nearby Google Street View panoramas, and metadata such as pano ID, location, and date are collected.

```python
panos = src.sv.get_panos(points)
```

![Panos](./assets/region_points_panos.png "Resulting panoramas")

```python
panos.head()
```

```
|    | id                     |      lat |      lon | date       |
|---:|:-----------------------|---------:|---------:|:-----------|
|  0 | m87GTCsrVJFOTPCR8gVSKw | -34.586  | -58.4259 | 2013-12-01 |
|  1 | OlCYn_A6FTJ9K87oORf9ag | -34.5859 | -58.4259 | 2014-07-01 |
|  2 | GRS6JbAMjoHOt_e-DgYMAw | -34.586  | -58.426  | 2015-06-01 |
|  3 | 3so_z7DA7VMLeseqeBIcPQ | -34.5859 | -58.4259 | 2016-12-01 |
|  4 | bIZOorYAaq2zuoxBLiYTmA | -34.5859 | -58.4259 | 2017-06-01 |
```

## Roadmap

### Primary

- [ ] Connect panorama points to the OpenStreetMaps street grid (https://github.com/rpasquini/street-level-change/issues/3) and calculate area of coverage in meters (https://github.com/rpasquini/street-level-change/issues/1).

### Secondary

- [ ] Generate a street grid from Google Street View images (https://github.com/rpasquini/street-level-change/issues/4).
