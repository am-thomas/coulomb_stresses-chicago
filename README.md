This is a copy of the chicago branch of https://github.com/lucas-schirbel/coulomb_stresses/ (as of 2026 Jan 13). It was created to generate a Zenodo link. Please see the chicago branch of https://github.com/lucas-schirbel/coulomb_stresses/ for any updates since Jan 13, 2026. 

# Static Stress Perturbations from Mass Unloading

This project is building of the main branch of https://github.com/lucas-schirbel/coulomb_stresses/. It computes static stress perturbations from long-term mass removal (e.g., rock extraction at a quarry), using Boussinesq's solution for a surface load distribution on an isotropic elastic half-space. This branch is modified from the main branch to incorporate elevation raster data and parameters specific to an industrial corridor of the Chicago area (McCook, IL), containing two quarrying operations and a flood-control reservoir. The code assumes a pre-excavated elevation of 185 m and uses processed data from existing Digital Elevation Models (DEMs) to estimate the total volume of rock removed between 1992-2022. Processed DEM data can be downloaded here: https://zenodo.org/records/18236078. By modifying the simulation parameters and using your own processed DEMs, this code can be adapted to any industrial environment. Asssuming a constant pre-excavated elevation may not be appropriate for your locality. In such cases, we recommend adding a topographic differencing step to the DEM processsing procedure. 

---
**Acknowledgment**:
The original code for this simulation was written in MATLAB by Elias Heimisson and translated to Python by Lucas Schirbel.
---


## Prerequisites

This project requires the following Python libraries:

- numpy
- matplotlib
- rasterio

## Usage

1. **Prepare Input Data: Processed Elevation Data**:
   - Download processed elevation data from https://zenodo.org/records/18236078 and place files in the `data/` directory.
   - For testing, there is a sample file (with 100 meter resolution) already existing in the `data/100m` directory.

2. **Run the Simulation**:
   Execute the `main_DEM_1timestep.py` script for the processed 1992 data at a depth of 100m:
   ```bash
   python main_DEM_1timestep.py --elevation_raster 1992_GMRT_30m_reproj_resamp_crop_masked.tif --raster_year 1992 --cbar_vmin -0.0 --cbar_vmax 1.5 --depth 100 --og_elev_units meters
   ```
   You can run 'python main_DEM_1timestep.py --help' to view descriptions of each command-line parameters. 'cbar_vmin' and 'cbar_vmax' are the minimum and maximum color values to plot in the visualizations. 

   Repeat the script for each 'elevation_raster file' in the `data/30m` directory. For 2017 and 2022 DEM files, the original units are in feet please change the end of the script from '--og_elev_units meters' to '--og_elev_units feet'.  See the Note section below if you have any trouble running the script.

3. **Outputs**:
   - Figures and output files are saved in the `output/` directory.
   - Numerical outputs include stress tensors, maximum Coulomb stress, and more, saved as `.npy` files.

## Example Simulation Parameters

Outside of command-line inputs, constant simulation parameters are defined in the `sim_data` dictionary in `main_DEM_1timestep.py`. Example:
```python
sim_data = {     
    'G': 11e9,                           # Shear modulus (Pa)
    'nu': 0.25,                          # Poisson's ratio
    'c': 1.0e-2,                         # Diffusivity m^2/s
    'B': 0.7,                            # Skempton's coefficient
    'mu': 0.7,                           # Friction coefficient for Coulomb stress calculations
    'density': -2700.}                   # Material density (kg/m^3)
```

## Note
- The 30-m resolution files are very large so you might not be able to execute the script on a standard PC. To test if your issue is related to file size, run the following script for a 100-m resolution file. 
```bash
   python main_DEM_1timestep.py --elevation_raster 100m/2022_ISGS_100m_reproj_resamp_masked.tif --raster_year 2022 --cbar_vmin -0.0 --cbar_vmax 1.5 --depth 100
   ```

