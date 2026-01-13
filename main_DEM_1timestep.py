from simulation_DEM_1timestep import time_varied_regional_loading
import argparse

# DEM parameters
parser = argparse.ArgumentParser()
parser.add_argument("--elevation_raster", default='output_USGS100m_resampled_crop.tif', 
                    type=str, help='filename of elevation raster stored in data folder')
parser.add_argument("--raster_year", required=True, type=str, 
                    help='acquisition year of elevation data')
parser.add_argument("--og_elev_units", default='meters',
                    help='units of elevation values. If feet, program will convert feet to meters. Program is only compatible with meters or feet')
parser.add_argument("--depth", default=100, type=int,
                    help='depth in meters, where the stresses should be computed')
parser.add_argument("--strike", default=-5, type=int,
                    help='strike in degrees')
parser.add_argument("--dip", default=5, type=int,
                    help='dip in degrees')
parser.add_argument("--rake", default=80, type=int,
                    help='rake in degrees')

# visualization parameters
parser.add_argument("--cbar_vmin", default='Auto', type=str, 
                    help='minimum value (MPa) of the stress colorbar. Default is matplolib automatically choosing vmin and vmax')
parser.add_argument("--cbar_vmax", default='Auto', type=str, 
                    help='max value (MPa) of the stress colorbar. Not applicable if cba_vmin = Auto')

args = parser.parse_args()

# store a dictionary of all relevant input parameters
sim_data = {     
    'G': 11e9,                           # Shear modulus (Pa)
    'nu': 0.25,                          # Poisson's ratio
    'c': 1.0e-2,                         # Diffusivity m^2/s
    'B': 0.7,                            # Skempton's coefficient
    'mu': 0.7,                           # Friction coefficient for Coulomb stress calculations
    'density': -2700.}                   # Material density (kg/m^3)

sim_data.update({'depth': args.depth,
                 'strike': args.strike,
                 'dip':args.dip,
                 'rake':args.rake,
                 'elevation_raster': args.elevation_raster, 
                 'raster_year': args.raster_year,
                 'og_elevation_units': args.og_elev_units,
                 'cbar_vmin': args.cbar_vmin,
                 'cbar_vmax': args.cbar_vmax})

# run the simulation
time_varied_regional_loading(sim_data)
