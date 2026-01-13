import numpy as np
import matplotlib.pyplot as plt
from utils import lonlat2local, surface_halfspace, strikedip2norm, sdr2slip, plot_fault_and_distances, compute_weights_from_distances, plot_weights, save_file
from shapely.geometry import Polygon, Point
from scipy.interpolate import interp1d
import os

def time_varied_regional_loading(sim_data, preview=True):
    """
    Simulate time-varied regional loading, compute stresses, and plot results.
    """
    # Ensure output directory exists
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Load the pit contour data
    T = np.loadtxt('data/pit_border.csv', skiprows=1, delimiter=',')

    # Assign the dictionary values to variables
    N = sim_data['N']
    ext = sim_data['ext'] 
    G = sim_data['G']
    nu = sim_data['nu']
    depth = sim_data['depth'] 
    NT = sim_data['NT']
    t = sim_data['t']
    downsample = sim_data['downsample'] 
    mu = sim_data['mu']
    density = sim_data['density']

    # Here we downsample the number of time steps
    NTfull = NT
    NTsub = np.arange(1, NT, downsample)
    NT = len(NTsub)

    lonlat = np.column_stack((T[:,0], T[:,1]))

    # Compute the origin of the xy coordinate system
    origin = np.mean(lonlat, axis=0)  # Compute the mean of each column (LON, LAT)

    # Convert longitude and latitude to local coordinates
    xy = 1000 * lonlat2local(lonlat, origin)

    # Plot the initial points
    fig, ax = plt.subplots()
    #ax.plot(xy[:, 0], xy[:, 1], 'b.')
    #plt.gca().set_aspect('equal', adjustable='box')

    # Find the bounding box of the points
    xbeg = np.min(xy[:, 0])
    xend = np.max(xy[:, 0])
    ybeg = np.min(xy[:, 1])
    yend = np.max(xy[:, 1])

    # Domain size
    DX = xend - xbeg
    DY = yend - ybeg

    # Distribute nodes more equally
    NX = int(np.ceil(N * 2 * DX / (DX + DY)))
    NY = int(np.ceil(N * 2 * DY / (DX + DY)))

    # Extend the computational grid by ext * 1000
    xg = np.linspace(-ext * 1000 + xbeg, ext * 1000 + xend, NX)
    yg = np.linspace(-ext * 1000 + ybeg, ext * 1000 + yend, NY)

    # Area element in grid
    dA = (xg[1] - xg[0]) * (yg[1] - yg[0])

    # Generate the [NxM] grid
    Xg, Yg = np.meshgrid(xg, yg)
    

    # Xgr, Ygr are the grid points (observation points)
    Xgr = Xg.copy()
    Ygr = Yg.copy()

    # These are all observation points
    points = list(zip(Xgr.flatten(), Ygr.flatten()))
    contour = [tuple(row) for row in xy]
    polygon = Polygon(contour)
    IN = [polygon.contains(Point(p)) for p in points]
    IN = np.array(IN).reshape(Xgr.shape)

    Xgr = Xgr[IN]
    Ygr = Ygr[IN]
    
    # Plot the polygon
    x, y = polygon.exterior.xy
    ax.plot(x, y, alpha=0.5, color='blue', label="Polygon")

    # locations of the sources
    ls = np.column_stack((Xgr, Ygr))

    # locations of the observation points (where stresses are evaluated)
    lo = np.column_stack((Xg.flatten(), Yg.flatten(), depth * np.ones_like(Yg.flatten())))
    
    # Draw figure to make sure that evertything is working as it should
    ax.scatter(ls[:,0], ls[:,1], color='blue', marker='.', label='Inside',zorder=5) 
    #ax.scatter(lo[:,0], lo[:,1], color='red', marker='.', label='Outside')  # Points outside
    plt.gca().set_aspect('equal', adjustable='box')
    
    distances = plot_fault_and_distances(ls, show_plot=True, save_path=os.path.join(output_dir, "distances_plot.png"))
    weights = compute_weights_from_distances(distances)
    plot_weights(ls, weights, show_plot=True, save_path=os.path.join(output_dir, "weights_plot.png"))

    sixx, siyy, sizz, sixy, sixz, siyz, ux, uy, uz = surface_halfspace(ls, lo, nu, G)
   
    # Multiply by the density and gravity to get F per unit height
    gforce_per_meter = 9.8 * density * -dA
    sixx_Fperh = gforce_per_meter * sixx
    siyy_Fperh = gforce_per_meter * siyy
    sizz_Fperh = gforce_per_meter * sizz
    sixy_Fperh = gforce_per_meter * sixy
    siyz_Fperh = gforce_per_meter * siyz
    sixz_Fperh = gforce_per_meter * sixz 

    # Load the water level or rock column depth information
    WL = np.loadtxt('data/pit_cota.txt')

    # Define the height function using 1D interpolation
    height = interp1d(WL[:, 0], WL[:, 1], kind='linear', fill_value='extrapolate')

    strike = sim_data['strike']
    dip = sim_data['dip']
    rake = sim_data['rake']
    
    # Compute vector normal to fault plane
    nn, en, un = strikedip2norm(strike, dip)

    # Combine normal components into a single array (3xN)
    normal = np.vstack([nn, en, un])
    
    # Compute the slip vector
    ns, es, us = sdr2slip(strike, dip, rake)
    
    # Combine slip components into a single array (3xN)
    slip = np.vstack([ns, es, us])

    # Load filtered catalog data
    #cata = np.load('filtered_catalog.npy', allow_pickle=True)  # assuming the file is saved as a .npy format
    
    # Convert longitude and latitude to local coordinates (assuming `lonlat2local` is a custom function)
    #xyeq = 1000 * lonlat2local(cata[:, 1:3], origin)
    
    # Define the height function
    def load(t):
        return height(t)
    
    # Time vector where stresses are evaluated
    time = np.min(WL[:, 0]) + np.linspace(0, t / (365 * 24 * 60 * 60), NTfull)
    time = time[NTsub]  # Select the times that stresses are stored

    # if preview:
    #     preview_plot(WL,Xgr,Ygr,x,y)
    
    # Pre-allocate shear stress (tau) and normal stress (sig) arrays
    tau = np.zeros((len(Xg.flatten()), len(time)))
    sig = np.zeros_like(tau)  # Normal stress array
    sigma1 = np.zeros_like(tau)  # Maximum compressive stress
    sigma3 = np.zeros_like(tau)  # Minimum compressive stress
    maxcoul = np.zeros_like(tau)  # Assuming you need max Coulomb stress
    
    # Iterate over the grid
    for i in range(NT):
        # multiply stress components (F/m) by load height to get units of F
        sixx = weights * load(time[i]) * sixx_Fperh
        siyy = weights * load(time[i]) * siyy_Fperh
        sizz = weights * load(time[i]) * sizz_Fperh
        sixy = weights * load(time[i]) * sixy_Fperh
        siyz = weights * load(time[i]) * siyz_Fperh
        sixz = weights * load(time[i]) * sixz_Fperh

        # Sum over all sources to get F for each observation point
        sixx = np.sum(sixx, axis=1)  
        siyy = np.sum(siyy, axis=1)  
        sizz = np.sum(sizz, axis=1)  
        sixy = np.sum(sixy, axis=1) 
        siyz = np.sum(siyz, axis=1) 
        sixz = np.sum(sixz, axis=1) 

        for j in range(len(Xg.flatten())):
            # store the stress tensor for one point
            sigmaT = np.array([[sixx[j], sixy[j], sixz[j]],
                                        [sixy[j], siyy[j], siyz[j]],
                                        [sixz[j], siyz[j], sizz[j]]])
            
            # Flip to the compression-positive convention of King et al.
            sigmaTK = np.array([[-sixx[j], sixy[j], sixz[j]],
                                        [sixy[j], -siyy[j], siyz[j]],
                                        [sixz[j], siyz[j], -sizz[j]]])
            
            # Calculate eigenvalues of the stress tensor
            eig_vals = np.real(np.linalg.eigvals(sigmaTK))
            
            # Maximum and minimum compressive stress
            maxe1 = np.max(eig_vals)
            mine1 = np.min(eig_vals)
            
            # Assign values to the stress arrays
            sigma1[j, i] = maxe1  # Maximum compressive stress
            sigma3[j, i] = mine1  # Minimum compressive stress
            tract = np.dot(sigmaT, normal) # Traction vector on the plane
            sig[j, i] = np.sum(np.multiply(tract,normal))  # Normal stress given strike and dip
            tau[j, i] = np.sum(np.multiply(tract,slip))  # Shear stress given strike, dip, and rake

    # Reshape sig and tau for plotting the last stored value
    sigp = np.reshape(sig[:, -1], (NY, NX))  # last column for sig
    taup = np.reshape(tau[:, -1], (NY, NX))  # last column for tau
    
    # Create a figure for Coulomb stress
    plt.figure(4)
    plt.pcolor(Xg, Yg, (taup + mu * sigp) / 1.0e6, shading='nearest')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar()
    plt.plot(xy[:, 0], xy[:, 1], 'w.')  # Contour outline 
    #plt.plot(xyeq[:, 0], xyeq[:, 1], 'k.')  # Events
    plt.title('Coulomb Stress fixed fault orientation [MPa]')
    plt.savefig(os.path.join(output_dir, "coulomb_stress.png"))
    plt.show()
    

    # Define the angle twobeta
    twobeta = np.arctan(-1 / mu)  # from King et al. 1994, but FIXED!!!
    
    # Initialize the maxcoul array
    maxcoul = np.zeros((len(sigma1), NT))
    
    # Loop over NT and compute the maximum Coulomb stress
    for i in range(NT):
        taumax = 0.5 * (sigma1[:, i] - sigma3[:, i])
        sigmax = 0.5 * (sigma1[:, i] + sigma3[:, i])
        sigmab = sigmax - taumax * np.cos(twobeta)
        taub = -taumax * np.sin(twobeta)
        ctest1 = taub - mu * sigmab
        maxcoul[:, i] = ctest1
    
    # Plotting the Max computed Coulomb stress for the last iteration
    plt.figure(5)
    plt.pcolor(Xg, Yg, np.real(maxcoul[:, -1].reshape(NY, NX)) / 1.0e6, shading='nearest')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar()
    plt.plot(xy[:, 0], xy[:, 1], 'w.')
    plt.title('Max Coulomb Stress [MPa]')
    #plt.plot(xyeq[:, 0], xyeq[:, 1], 'k.')
    plt.savefig(os.path.join(output_dir, "max_coulomb_stress.png"))
    plt.show()
    
    # Save numerical outputs
    np.save(os.path.join(output_dir, "tau.npy"), tau)
    np.save(os.path.join(output_dir, "sig.npy"), sig)
    np.save(os.path.join(output_dir, "maxcoul.npy"), maxcoul)
    np.save(os.path.join(output_dir, "Xg.npy"), Xg)
    np.save(os.path.join(output_dir, "Yg.npy"), Yg)
    np.save(os.path.join(output_dir, "xy.npy"), xy)
    np.save(os.path.join(output_dir, "time.npy"), time)

    # Save input parameters 
    sim_data.update({'NX': NX, 'NY': NY, 'NT_final': NT})
    save_file('input_parameters.pkl', sim_data)

