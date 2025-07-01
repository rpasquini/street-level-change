# Pull Request: Add Dedicated Plotting Module with Temporal Analysis

## Overview
This PR introduces a dedicated plotting module to improve code organization and modularity in the street-level-change project. The plotting functionality has been extracted from the point unification script and moved to a dedicated module to enhance reusability and maintainability. Additionally, a new temporal analysis visualization has been added to track panorama image acquisition over time.

## Changes

### New Features
- Created a new `plotting` module in the `src` directory
- Implemented dedicated plotting functions for H3 and DBSCAN results
- Added new temporal analysis visualization to track panorama image counts by date
- Updated the `test_point_unification.py` script to use the new plotting functions

### Files Changed
- **New Files:**
  - `src/plotting/__init__.py`: Module initialization with exports
  - `src/plotting/static_plotting.py`: Contains dedicated plotting functions
- **Modified Files:**
  - `scripts/test_point_unification.py`: Removed embedded plotting function and updated imports
  - `README.md`: Updated with comprehensive project information

### Technical Details
- Split the original `plot_results` function into specialized functions:
  - `plot_h3_results`: For visualizing H3 unification results
  - `plot_dbscan_results`: For visualizing DBSCAN unification results
  - `plot_panorama_dates`: For visualizing temporal distribution of panorama images
- The new date plotting function provides both monthly and yearly aggregations
- Added proper closing of matplotlib figures to prevent memory leaks
- Improved documentation with detailed docstrings
- Implemented robust date parsing to handle different date formats

## Benefits
- **Improved Modularity**: Separates visualization logic from data processing
- **Enhanced Reusability**: Plotting functions can be used across different scripts
- **Better Organization**: Creates a dedicated location for all visualization code
- **Future Extensibility**: Makes it easier to add new visualization methods
- **Temporal Analysis**: Provides insights into the temporal distribution of panorama data
- **Data Quality Assessment**: Helps identify potential gaps or biases in temporal coverage

## Testing
The changes have been tested by running:
```bash
make unify
```
The script executes successfully and produces the same visualization outputs as before.

## Screenshots

### New Visualization: Panorama Image Count by Date
The new plotting function generates two visualizations:

1. **Monthly Distribution**: Shows the number of panorama images available for each month
2. **Yearly Distribution**: Shows the yearly aggregation of panorama images

These visualizations help identify temporal patterns and potential gaps in the data collection.

## Related Issues
N/A

## Next Steps
- Consider adding more advanced visualization functions for geospatial data
- Implement interactive plotting capabilities using libraries like Plotly
- Expand temporal analysis to include seasonal patterns and trends
- Add visualization functions for comparing changes between time periods
