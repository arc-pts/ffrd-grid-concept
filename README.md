# ffrd-grid-concept
FFRD multidimensional gridded data concept.

## Setup
Create a conda environment with the required dependencies:

  ```
  conda env create -f environment.yml
  conda activate ffrd-grid-concept
  ```

Get the `2D_model_output_sample_data` from PTS Innovations SharePoint, extract it, and place the contents in the `data` directory.

## Notebooks
* [01 - Preprocessing](01-preprocess.ipynb): Compile example water surface elevation grids
  and terrain from the sample PFRA data into a single Zarr store for analysis.
* [02 - Analysis](02-analyze.ipynb): Estimate the depth of flooding at different recurrence
  intervals for each grid cell.
* [03 - Visualization](03-visualization.ipynb): Visualize the results of the analysis.
