import numpy as np
# from scipy.interpolate import interp1d
# from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
import pickle

# Utility functions

def save_file(filename, file):
    """
    Save file in pickle format
    Params:
        file (any object): any Python object
        filename (str): name of pickle file
    """
    with open(filename, 'wb') as f:
        pickle.dump(file, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_file(filename):
    """
    Load a pickle file
    Args:
        filename (str): Name of the file
    Returns (Python obj): Returns the loaded pickle file
    """
    with open(filename, 'rb') as f:
        file = pickle.load(f)

    return file

def lonlat2local(lonlat, origin):
    """
    Convert from longitude and latitude to local coordinates.
    
    Parameters:
    ll (numpy.ndarray): Nx2 matrix of [longitude, latitude] in degrees.
    origin (numpy.ndarray): 1x2 array [longitude, latitude] for the origin in degrees.
    
    Returns:
    numpy.ndarray: Nx2 matrix of distances relative to origin in kilometers.
    
    Written by Andy Hooper (2008) (MATLAB)
    Adapted from original code of Peter Cervelli	
    """
    # Convert to radians
    ll = np.radians(lonlat).T 
    origin = np.radians(origin)
    
    # Set ellipsoid constants (WGS84)
    a = 6378137.0  # Semi-major axis (meters)
    e = 0.08209443794970  # Eccentricity
    
    # Initialize xy
    xy = np.zeros_like(ll)
    
    # Projection for non-zero latitude
    z = ll[1, :] != 0
    
    dlambda = ll[0, z] - origin[0]
    
    # Meridian arc length M calculation
    M = a * ((1 - e**2 / 4 - 3 * e**4 / 64 - 5 * e**6 / 256) * ll[1, z] - 
             (3 * e**2 / 8 + 3 * e**4 / 32 + 45 * e**6 / 1024) * np.sin(2 * ll[1, z]) +
             (15 * e**4 / 256 + 45 * e**6 / 1024) * np.sin(4 * ll[1, z]) - 
             (35 * e**6 / 3072) * np.sin(6 * ll[1, z]))
    
    M0 = a * ((1 - e**2 / 4 - 3 * e**4 / 64 - 5 * e**6 / 256) * origin[1] - 
              (3 * e**2 / 8 + 3 * e**4 / 32 + 45 * e**6 / 1024) * np.sin(2 * origin[1]) +
              (15 * e**4 / 256 + 45 * e**6 / 1024) * np.sin(4 * origin[1]) - 
              (35 * e**6 / 3072) * np.sin(6 * origin[1]))
    
    # Calculate N (radius of curvature in the prime vertical)
    N = a / np.sqrt(1 - e**2 * np.sin(ll[1, z])**2)
    
    # Calculate Easting (E)
    E = dlambda * np.sin(ll[1, z])
    
    # Local coordinates for latitudes not equal to zero
    xy[0, z] = N * (1/np.tan(ll[1, z])) * np.sin(E)
    xy[1, z] = M - M0 + N * (1/np.tan(ll[1, z])) * (1 - np.cos(E))
    
    # Special case: latitude = 0
    xy[0, ~z] = a * dlambda[~z]
    xy[1, ~z] = -M0
    
    # Convert from meters to kilometers
    xy = xy.T / 1000  # Transpose back and divide by 1000 to convert to km
    
    return xy
    
def surface_halfspace(ls, lo, nu, G):
    """
    Compute stress tensor components due to surface loads in a half-space.
    """
    # ls: location of sources (Nx2) (z coordinate assumed 0)
    # lo: location of observation points (Mx3) (z > 0 down)
    # G: shear modulus [Pa]
    # nu: Poisson ratio
    
    M = len(lo)  # number of observation points
    N = len(ls)  # number of sources
    
    # Initialize arrays to store results
    sixx = np.zeros((M, N))
    siyy = np.zeros((M, N))
    sizz = np.zeros((M, N))
    ux = np.zeros((M, N))
    uy = np.zeros((M, N))
    uz = np.zeros((M, N))
    sixy = np.zeros((M, N))
    sixz = np.zeros((M, N))
    siyz = np.zeros((M, N))

    # Loop over each observation point
    for i in range(M):
        # Compute the displacement vector from the source to the observation point
        xyz = lo[i, :] - np.column_stack((ls, np.zeros(N)))  # Subtract ls from lo, add z=0 for sources
        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]
        r = np.sqrt(np.sum(xyz**2, axis=1))  # Compute the distance r

        # Compute the stress components
        sixx[i, :] = 1 / (2 * np.pi) * ( 3 * x**2 * z / r**5 + (1 - 2 * nu) * (y**2 + z**2) / (r**3 * (z + r)) - (1 - 2 * nu) * z / r**3 - (1 - 2 * nu) * x**2 / (r**2 * (z + r)**2) )
        siyy[i, :] = 1 / (2 * np.pi) * ( 3 * y**2 * z / r**5 + (1 - 2 * nu) * (x**2 + z**2) / (r**3 * (z + r)) - (1 - 2 * nu) * z / r**3 - (1 - 2 * nu) * y**2 / (r**2 * (z + r)**2) )
        sizz[i, :] = 1 / (2 * np.pi) * ( 3 * z**3 / r**5 )
        sixy[i, :] = 1 / (2 * np.pi) * ( 3 * z * y * x / r**5 - (1 - 2 * nu) * (x * y * (z + 2 * r)) / (r**3 * (z + r)**2) )
        siyz[i, :] = 1 / (2 * np.pi) * ( 3 * y * z**2 / r**5 )
        sixz[i, :] = 1 / (2 * np.pi) * ( 3 * x * z**2 / r**5 )
        ux[i, :] = 1 / (4 * np.pi * G) * ( (1 - 2 * nu) * x / (r * (z + r)) - x * z / r**3 )
        uy[i, :] = 1 / (4 * np.pi * G) * ( (1 - 2 * nu) * y / (r * (z + r)) - y * z / r**3 )
        uz[i, :] = -1 / (4 * np.pi * G) * ( 2 * (1 - nu) / r + z**2 / r**3 )

    return sixx, siyy, sizz, sixy, sixz, siyz, ux, uy, uz
    
def strikedip2norm(strike, dip, *args):
    """
    Converts strike and dip of a fault plane to a normal vector [North, East, Up].

    Args:
    - strike (float or np.ndarray): Strike of the fault in degrees (clockwise from North).
    - dip (float or np.ndarray): Dip of the fault in degrees (positive downward from horizontal).
    - *args: Optional additional arguments (to handle cases with multiple fault planes).

    Returns:
    - n, e, u: The components of the normal vector in North, East, Up.
    - Or if nargout <= 1, returns the full normal vector as an Nx3 array.
    
    Written by Garrett Euler (MATLAB)
    """
    
    # Handle the case where the input is a 2D array [strike, dip] (Nx2)
    if isinstance(strike, np.ndarray) and strike.ndim == 2 and strike.shape[1] == 2:
        strike, dip = strike[:, 0], strike[:, 1]
    elif isinstance(strike, (float, int)) and isinstance(dip, (float, int)):
        # Convert scalar inputs to arrays for consistency
        strike = np.array([strike])
        dip = np.array([dip])
    elif isinstance(strike, np.ndarray) and isinstance(dip, np.ndarray):
        # Ensure they have the same length
        if strike.shape[0] != dip.shape[0]:
            raise ValueError('strike and dip must have the same length!')
    
    # Calculate North, East, Up components of the normal vector
    n = -np.sin(np.radians(dip)) * np.sin(np.radians(strike))
    e = np.sin(np.radians(dip)) * np.cos(np.radians(strike))
    u = np.cos(np.radians(dip))
    
    # Otherwise, return North, East, and Up separately
    return n, e, u

def sdr2slip(strike, dip, rake):
    """
    Calculates the slip vector for a given strike, dip, and rake.

    Args:
    - strike (float or np.ndarray): Strike of the fault in degrees (clockwise from North).
    - dip (float or np.ndarray): Dip of the fault in degrees (positive downward from horizontal).
    - rake (float or np.ndarray): Rake of the fault in degrees (positive counter-clockwise from the strike direction).

    Returns:
    - n, e, u: The components of the slip vector in North, East, Up (or a single Nx3 array if requested).
    
    Written by Garrett Euler (MATLAB)
    """
    # Handle different input formats
    if isinstance(strike, np.ndarray) and strike.ndim == 2 and strike.shape[1] == 3:
        # If input is an Nx3 array
        strike, dip, rake = strike[:, 0], strike[:, 1], strike[:, 2]
    elif isinstance(strike, (float, int)) and isinstance(dip, (float, int)) and isinstance(rake, (float, int)):
        # If inputs are scalars, convert to arrays for consistency
        strike = np.array([strike])
        dip = np.array([dip])
        rake = np.array([rake])
    elif isinstance(strike, np.ndarray) and isinstance(dip, np.ndarray) and isinstance(rake, np.ndarray):
        # If all inputs are arrays, make sure they have the same length
        if strike.shape[0] != dip.shape[0] or dip.shape[0] != rake.shape[0]:
            raise ValueError("strike, dip, and rake must have the same length!")

    # Compute the components of the slip vector
    n = np.cos(np.radians(rake)) * np.cos(np.radians(strike)) + np.sin(np.radians(rake)) * np.cos(np.radians(dip)) * np.sin(np.radians(strike))
    e = np.cos(np.radians(rake)) * np.sin(np.radians(strike)) - np.sin(np.radians(rake)) * np.cos(np.radians(dip)) * np.cos(np.radians(strike))
    u = np.sin(np.radians(rake)) * np.sin(np.radians(dip))
    

    # If we have multiple faults, return n, e, u as separate arrays
    return n, e, u


def plot_fault_and_distances(ls, azimuth=155, L=600, cmap='viridis', show_plot=True, save_path=None):
    """
    Plot a fault line at the given azimuth and color-code the ls points by their distance from the line.
    
    Parameters:
    -----------
    ls : array-like, shape (N, 2)
        Coordinates of the grid points inside the pit.
    azimuth : float, optional
        Fault line azimuth in degrees clockwise from north. Default is 155°.
    L : float, optional
        Half-length of the plotted fault line in the same units as ls. Default is 200.
    cmap : str, optional
        Matplotlib colormap for distance. Default is 'viridis'.
        
    Returns:
    --------
    distances : ndarray, shape (N,)
        Array of distances of each point in ls from the fault line.
    """

    # Compute the direction vector of the fault line from the azimuth
    azimuth_radians = np.radians(azimuth)
    dx = np.sin(azimuth_radians)  # x-direction
    dy = np.cos(azimuth_radians)  # y-direction

    # Normalize the direction vector
    length = np.sqrt(dx**2 + dy**2)
    dxu = dx / length
    dyu = dy / length

    # Parametric line coordinates (centered on 0,0)
    t = np.linspace(-L, L, 100)
    x_line = dx * t
    y_line = dy * t

    # Extract coordinates from ls
    Xgr = ls[:, 0]
    Ygr = ls[:, 1]

    # Distance calculation: distance from (Xgr,Ygr) to line through (0,0)
    # with direction (dxu, dyu) is | -dyu*Xgr + dxu*Ygr |
    distances = np.abs(-dyu * Xgr + dxu * Ygr)

    # Plot the grid points colored by distance
    plt.figure(2,figsize=(8, 8))
    scatter = plt.scatter(Xgr, Ygr, c=distances, cmap=cmap, s=20, edgecolor='none')
    cbar = plt.colorbar(scatter)
    cbar.set_label('Distance from Pit Bottom (m)', rotation=90)

    # Plot the fault line
    plt.plot(x_line, y_line, 'r-', linewidth=2, label='Pit Bottom')

    # Set equal aspect ratio and labels
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('X Coordinate (m)')
    plt.ylabel('Y Coordinate (m)')
    plt.title(f'Grid Points Colored by Distance from Pit Bottom (Azimuth={azimuth}°)')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    if show_plot:
        plt.show()

    return distances

def compute_weights_from_distances(distances):
    dist_min = distances.min()
    dist_max = distances.max()
    # Handle case if all distances are the same to avoid division by zero
    if dist_min == dist_max:
        return np.ones_like(distances)
    # Compute weights linearly
    weights = 1 - (distances - dist_min) / (dist_max - dist_min)
    return weights
    
def plot_weights(ls, weights, show_plot=True, save_path=None):
    Xgr = ls[:, 0]
    Ygr = ls[:, 1]
    
    plt.figure(figsize=(8, 8))
    scatter_weights = plt.scatter(Xgr, Ygr, c=weights, cmap='viridis', s=20, edgecolor='none')
    cbar = plt.colorbar(scatter_weights)
    cbar.set_label('Weight', rotation=90)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('X Coordinate (m)')
    plt.ylabel('Y Coordinate (m)')
    plt.title('Weights Assigned to Points Based on Distance to Pit Bottom')
    
    # Optionally, plot the fault line as before:
    # (Assuming x_line, y_line already computed)
    #plt.plot(x_line, y_line, 'r-', linewidth=2, label='Fault Line')
    #plt.legend()
    if save_path:
        plt.savefig(save_path)
    if show_plot:
        plt.show()
    
