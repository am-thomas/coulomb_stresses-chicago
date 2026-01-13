import numpy as np
import matplotlib.pyplot as plt
from utils import surface_halfspace, strikedip2norm, sdr2slip, save_file
import os
import rasterio

def time_varied_regional_loading(sim_data, show_plot=False):
    """
    Simulate time-varied regional loading, compute stresses, and plot results.
    """
    # Ensure output directory exists
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # load input parameters
    raster_filename = sim_data['elevation_raster']
    raster_year = sim_data['raster_year']
    og_elevation_units = sim_data['og_elevation_units']
    G = sim_data['G']
    nu = sim_data['nu']
    depth = sim_data['depth']
    mu = sim_data['mu']
    density = sim_data['density']
    cbar_vmin = sim_data['cbar_vmin']
    cbar_vmax = sim_data['cbar_vmax']

    # read in elevation data and store decentered coordinates (in m)
    elev_raster = rasterio.open(f"data/{raster_filename}")
    cols, rows = np.meshgrid(np.arange(elev_raster.width), np.arange(elev_raster.height))
    Xg, Yg = rasterio.transform.xy(elev_raster.transform, rows, cols)
    xcenter, ycenter = elev_raster.xy(elev_raster.height // 2, elev_raster.width // 2)
    Xg = Xg - xcenter
    Yg = Yg - ycenter
    Xg_reshaped = np.reshape(Xg, (elev_raster.height, elev_raster.width))
    Yg_reshaped = np.reshape(Yg, (elev_raster.height, elev_raster.width))

    # created an adjusted matrix of elevation values (in meters)
    elevs = elev_raster.read(1)
    nodata = elev_raster.nodata
    if og_elevation_units == 'feet':
        elevs = elevs*0.3048006096012192
    loads_preflatten = np.zeros((elev_raster.height, elev_raster.width))
    for i in range(elev_raster.height):
        for j in range(elev_raster.width):
            val = elevs[i,j]

            # set nan/nodata values as zero
            if nodata is not None and val == nodata:
                loads_preflatten[i, j] = 0
                continue
            
            # another condition to catch nodata/invalid values
            if val <= 0 or val > 185:
                loads_preflatten[i,j] = 0
            # compute depth from surface (185 m)
            else:
                loads_preflatten[i,j] = 185 - val

    #print('shape of loads', np.shape(loads))
    plt.pcolor(Xg_reshaped, Yg_reshaped, loads_preflatten, shading='nearest')
    plt.colorbar()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(os.path.join(output_dir, f"{raster_year}_DEM_demean185meters.png"))
    if show_plot == True:
        plt.show()
    loads = loads_preflatten.flatten()

    # area element in grid
    dA = elev_raster.res[0] * elev_raster.res[1]

    # locations of the sources
    ls = np.column_stack((Xg, Yg))

    # locations of the observation points (where stresses are evaluated)
    lo = np.column_stack((Xg.flatten(), Yg.flatten(), depth * np.ones_like(Yg.flatten())))
   
    # get stress tensor components in F per unit height
    sixx, siyy, sizz, sixy, sixz, siyz, ux, uy, uz = surface_halfspace(ls, lo, nu, G)
    agr_constant = -dA * 9.8 * density
    sixx_Fperh = agr_constant * sixx
    siyy_Fperh = agr_constant * siyy
    sizz_Fperh = agr_constant * sizz
    sixy_Fperh = agr_constant * sixy
    siyz_Fperh = agr_constant * siyz
    sixz_Fperh = agr_constant * sixz 

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

    # Time vector where stresses are evaluated (just one-element to keep a similar structure to simulation.py)
    times = [raster_year]
    len_times = len(times)

    # Pre-allocate shear stress (tau) and normal stress (sig) arrays
    tau = np.zeros((len(Xg.flatten()), len_times))
    sig = np.zeros_like(tau)  # Normal stress array
    sigma1 = np.zeros_like(tau)  # Maximum compressive stress
    sigma3 = np.zeros_like(tau)  # Minimum compressive stress
    maxcoul = np.zeros_like(tau)  # Assuming you need max Coulomb stress
    
    # Iterate over the grid (outer loop is just one iteration for one timestep)
    for i in range(len(times)):
        # multiply stress components (F/m) by load height to get units of F
        sixx = loads * sixx_Fperh
        siyy = loads * siyy_Fperh
        sizz = loads * sizz_Fperh
        sixy = loads * sixy_Fperh
        siyz = loads * siyz_Fperh
        sixz = loads * sixz_Fperh

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
    sigp = np.reshape(sig[:, -1], (elev_raster.height, elev_raster.width))  # last column for sig
    taup = np.reshape(tau[:, -1], (elev_raster.height, elev_raster.width))  # last column for tau
    #sigp = sig[:,-1]
    #taup = tau[:,-1]

    # Create a figure for Coulomb stress
    # NOTE: the equations for stresses at a fixed orientation and the optimal orientation use different 
    # sign conventions (fixed = tau + mu * sigma while max = tau - mu * sigma)
    fixed_coul_MPa = (taup + mu * sigp) / 1.0e6
    plt.figure(4)
    if cbar_vmin == 'Auto':
        plt.pcolor(Xg_reshaped, Yg_reshaped, fixed_coul_MPa, shading='nearest')
    else:
        cbar_vmin = float(cbar_vmin)
        cbar_vmax = float(cbar_vmax)
        plt.pcolor(Xg_reshaped, Yg_reshaped, fixed_coul_MPa, shading='nearest',
                    vmin=cbar_vmin, vmax=cbar_vmax)
        # plt.pcolor(Xg_reshaped, Yg_reshaped, (taup) / 1.0e6, shading='nearest',
        #     vmin=cbar_vmin, vmax=cbar_vmax)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar()
    # plt.plot(xy[:, 0], xy[:, 1], 'w.')  # Contour outline 
    #plt.plot(xyeq[:, 0], xyeq[:, 1], 'k.')  # Events
    plt.title('Coulomb Stress fixed fault orientation [MPa]')
    plt.savefig(os.path.join(output_dir, f"{raster_year}_{depth}m_fixedfault_stress.png"))
    if show_plot:
        plt.show()

    # define bounds around preferred earthquake epicenter and print stresses in this region
    x_eq = -1329.39
    y_eq = 1163.82
    buffer = 200
    x_min = x_eq - buffer
    x_max = x_eq + buffer
    y_min = y_eq - buffer
    y_max = y_eq + buffer
    mask = (Xg_reshaped >= x_min) & (Xg_reshaped <= x_max) & (Yg_reshaped >= y_min) & (Yg_reshaped <= y_max)
    masked_fixedcoul_MPa = fixed_coul_MPa[mask]
    ix = np.abs(Xg_reshaped[0, :] - x_eq).argmin()    #nearest x index to x_eq
    iy = np.abs(Yg_reshaped[:, 0] - y_eq).argmin()    #nearest y index to y_eq   
    fixedstress_near_eq = [fixed_coul_MPa[iy, ix], fixed_coul_MPa[iy-1, ix], fixed_coul_MPa[iy+1, ix],
                      fixed_coul_MPa[iy, ix-1], fixed_coul_MPa[iy, ix+1], fixed_coul_MPa[iy-1, ix-1], fixed_coul_MPa[iy+1, ix+1]]
    max_idx = np.unravel_index(np.argmax(fixed_coul_MPa), fixed_coul_MPa.shape)
    closest_grid_x, closest_grid_y = Xg_reshaped[iy, ix], Yg_reshaped[iy, ix]
    max_grid_x, max_grid_y = Xg_reshaped[max_idx], Yg_reshaped[max_idx]
    distance_closest_max = np.sqrt( (closest_grid_x-max_grid_x)**2 + (closest_grid_y-max_grid_y)**2)
    print('Buffer [m]', buffer)
    print('Fixed fault orientation')
    print('Max stress [all, masked]', np.max(fixed_coul_MPa), np.max(masked_fixedcoul_MPa))
    print('Avg stress [all, masked]', np.mean(fixed_coul_MPa), np.mean(masked_fixedcoul_MPa))
    print(f"Closest grid point: x = {closest_grid_x}, y = {closest_grid_y}")
    print(f"Grid point at max stress: x = {max_grid_x}, y = {max_grid_y}")
    print("Distance between the prior two points:", distance_closest_max)
    print("Coulomb stress at closest grid points",fixedstress_near_eq)
    print("Average coulomb stress at closest grid points",np.mean(fixedstress_near_eq))

    # plot only subset
    # plt.clf()
    # plt.scatter(Xg_reshaped[mask], Yg_reshaped[mask], c=masked_coul_MPa)
    # plt.colorbar(label='Load (m)')
    # plt.xlabel('X (m)')
    # plt.ylabel('Y (m)')
    # plt.title('Subset region of loads_preflatten')
    # plt.show()
    

    # Define the angle twobeta
    twobeta = np.arctan(-1 / mu)  # from King et al. 1994, but FIXED!!!
    
    # Initialize the maxcoul array
    maxcoul = np.zeros((len(sigma1), len_times))
    
    # Loop over NT and compute the maximum Coulomb stress
    for i in range(len_times):
        taumax = 0.5 * (sigma1[:, i] - sigma3[:, i])
        sigmax = 0.5 * (sigma1[:, i] + sigma3[:, i])
        sigmab = sigmax - taumax * np.cos(twobeta)
        taub = -taumax * np.sin(twobeta)
        ctest1 = taub - mu * sigmab
        maxcoul[:, i] = ctest1
    
    # Plotting the Max computed Coulomb stress for the last iteration
    plt.figure(5)
    max_coul_MPa = np.real(maxcoul[:, -1].reshape(elev_raster.height, elev_raster.width)) / 1.0e6
    if cbar_vmin == 'Auto':
        plt.pcolor(Xg_reshaped, Yg_reshaped, max_coul_MPa, 
                shading='nearest')
    else:
        plt.pcolor(Xg_reshaped, Yg_reshaped, max_coul_MPa, 
                shading='nearest', vmin=cbar_vmin, vmax=cbar_vmax)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar()
    #plt.plot(xy[:, 0], xy[:, 1], 'w.')
    plt.title('Max Coulomb Stress [MPa]')
    #plt.plot(xyeq[:, 0], xyeq[:, 1], 'k.')
    plt.savefig(os.path.join(output_dir, f"{raster_year}_{depth}m_max_stress.png"))
    if show_plot:
        plt.show()
    
    masked_maxcoul_MPa = max_coul_MPa[mask]
    maxstress_near_eq = [max_coul_MPa[iy, ix], max_coul_MPa[iy-1, ix], max_coul_MPa[iy+1, ix],
                      max_coul_MPa[iy, ix-1], max_coul_MPa[iy, ix+1], max_coul_MPa[iy-1, ix-1], max_coul_MPa[iy+1, ix+1]]
    print('')
    print('Optimal orientation')
    print('Max stress [all, masked]', np.max(max_coul_MPa), np.max(masked_maxcoul_MPa))
    print('Avg stress [all, masked]', np.mean(max_coul_MPa), np.mean(masked_maxcoul_MPa))
    print("Coulomb stress at closest grid points",maxstress_near_eq)
    print("Average coulomb stress at closest grid points",np.mean(maxstress_near_eq))
    print('')
    
    # Save numerical outputs
    np.save(os.path.join(output_dir, f"{raster_year}_{depth}m_tau.npy"), tau)
    np.save(os.path.join(output_dir, f"{raster_year}_{depth}m_sig.npy"), sig)
    np.save(os.path.join(output_dir, f"{raster_year}_{depth}m_maxcoul.npy"), maxcoul)
    np.save(os.path.join(output_dir, f"{raster_year}_{depth}m_Xg.npy"), Xg_reshaped)
    np.save(os.path.join(output_dir, f"{raster_year}_{depth}m_Yg.npy"), Yg_reshaped)

    # Save input parameters 
    #sim_data.update({'NX': NX, 'NY': NY, 'NT_final': NT})
    save_file('input_parameters.pkl', sim_data)
    np.save(os.path.join(output_dir, f"{raster_year}_{depth}m_loads.npy"), loads)

