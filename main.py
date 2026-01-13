from simulation import time_varied_regional_loading

# Simulation parameters
sim_data = {
    'N': 2 * 20 * 2**1,  # Approximate number of nodes in x or y dimension
    'ext': 1.0,          # Grid is extended in ext*1000 meters in x, y, and z dimension
    'G': 31.4e9,         # Shear modulus (Pa)
    'nu': 0.25,          # Poisson's ratio
    'depth': 100,        # Depth at which stresses are evaluated (m)
    'NT': 2**10,         # Number of time-steps in total
    't': 0.20 * 365 * 24 * 60 * 60,  # Total simulation time in seconds
    'c': 1.0e-2,         # Diffusivity m^2/s
    'B': 0.7,            # Skempton's coefficient
    'downsample': 10,    # Downsample time-steps
    'mu': 0.4,           # Friction coefficient for Coulomb stress calculations
    'strike': 155,       # Fault strike angle (degrees)
    'dip': 45,           # Fault dip angle (degrees)
    'rake': 90,          # Fault rake angle (degrees)
    'density': -2800.,   # Material density (kg/m^3)
}

# Run the simulation
time_varied_regional_loading(sim_data)
