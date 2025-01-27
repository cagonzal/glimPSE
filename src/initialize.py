import sys
import os

from curvOperators import *
import numMethod as nm
import flow as bf
import inputOutput as io
from inputOutput import print_rz, read_surface_data, read_velocity_field
from mpi4py import MPI

def initialize_flatplate(config):

    '''
    Initializes a flat plate geometry for PSE computation.

        Parameters:
            Params (dictionary): Dictionary of parameters specified by the input file

        Returns:
            helper_mats (dictionary): Dictionary of matrices that extract u,v,w,p from q-vector
            Grid (grid class): Mesh object that is used by the PSE solver
            EQS (equation class): PSE class object
    '''

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    flow = config['flow']
    disturbance = config['disturbance']
    geometry = config['geometry']
    grid = config['grid']

    # need a temporary spacing of hx = 1 
    hx_temp = 1.0
    Grid = nm.Grid(grid['Nx'], grid['Ny'], hx_temp, flow['x0'], geometry['ymax'], method=grid['method'])
    Grid.kappa = np.ones((grid['Nx'], 1)) * geometry['curvature']['kappa']
    Grid.dtheta_dxi = -1.0 * Grid.kappa

    helper_mats = {}
    helper_mats['zero'] = np.zeros((grid['Ny'], grid['Ny']))
    helper_mats['I'] = np.eye(grid['Ny'])
    I = helper_mats['I']
    zero = helper_mats['zero']

    # define more SPE helper functions 
    helper_mats['uvwP_from_LST'] = np.block([ [np.eye(4*grid["Ny"]),np.zeros((4*grid["Ny"],4*grid["Ny"]))] ])
    helper_mats['u_from_SPE'] = np.block([[I,zero,zero,zero]])
    helper_mats['v_from_SPE'] = np.block([[zero,I,zero,zero]])
    helper_mats['w_from_SPE'] = np.block([[zero,zero,I,zero]])
    helper_mats['P_from_SPE'] = np.block([[zero,zero,zero,I]])

    Baseflow = bf.Baseflow(Grid)
    Baseflow.Blasius(Grid.ygrid, x=flow['x0'], Uinf = flow['Uinf'], nu = flow['nu'])
    print_rz(f"Blasius baseflow initialized\n")

    # Initialize equation instance
    if config["simulation"]['type'] == "LST":
        EQS = LST(Grid, config, Baseflow, helper_mats)
    elif config["simulation"]['type'] == "GORTLER":
        EQS = Gortler(Grid, config, Baseflow, helper_mats)
    elif config['simulation']['type'] == "LPSE": 
        EQS = LPSE(Grid, config, Baseflow)
        config['modes']['temporal'] = 1
        config['modes']['spanwise'] = 1
    elif config['simulation']['type'] == "NLPSE":
        # increment by one for MFD
        config['modes']['temporal'] += 1
        config['modes']['spanwise'] += 1
        EQS = NLPSE(Grid, config, Baseflow)
        EQS.helper_mats = helper_mats

    dirExists = os.path.exists('./nlpse_data')
    if rank == 0:
        if not dirExists:
            os.makedirs('nlpse_data')

    # solve initial station 
    station = 0

    EQS.Solve(station)
    # Solved station zero with either LST or Gortler equations 
    print_rz("Solved station 0")

    if "hx" in grid:
        EQS.hx = grid['hx']
    else: 
        simulation_type = config['simulation']['type'] 
        if config['simulation']['type'] == "NLPSE":
            alpha_r = np.real(EQS.alpha[station, 1:, :])
            idxs = np.nonzero(alpha_r)
            if idxs[0].size > 0:
                min_alpha = np.min(alpha_r[idxs])
                hx_min = 1.0 / min_alpha
            else:
                hx_min = 1.0 / disturbance["beta"]
        else:
            hx_min = 1.0 / np.real(EQS.alpha[station])

        hx = grid["hx_factor"] * hx_min
        EQS.hx = hx
    print_rz(f"hx = {EQS.hx}\n")

    Grid = nm.Grid(grid['Nx'], grid['Ny'], EQS.hx, flow['x0'], geometry['ymax'], method=grid['method'])
    Grid.kappa = np.ones((grid['Nx'],1)) * geometry['curvature']['kappa']
    Grid.dtheta_dxi = -1.0 * Grid.kappa # this can be shown 

    EQS.sampleGrid(Grid)
    # EQS.Grid = Grid

    return helper_mats, Grid, EQS

def initialize_curved_surface(config):
    '''
    Initializes a pre-defined curved surface geometry for PSE computation.

    Parameters:
        config (dictionary): Dictionary of parameters specified by the input file

    Returns:
        helper_mats (dictionary): Dictionary of matrices that extract u,v,w,p from q-vector
        Grid (CurvedSurfaceGrid): Mesh object that is used by the PSE solver
        EQS (equation class): PSE class object
    '''

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    flow = config['flow']
    disturbance = config['disturbance']
    geometry = config['geometry']
    grid = config['grid']

    # Read surface and velocity field data
    x_field, y_field, u_field, v_field, p_field, kappa_field = read_velocity_field(flow['velocity_file'])

    # Create CurvedSurfaceGrid object
    Grid = nm.surfaceImport(x_field, y_field, u_field, v_field, p_field, grid['Nx'], geometry['ymax'], grid['Ny'], method = grid['method'], need_map=True, Uinf=flow['Uinf'], nu=flow['nu'])

    helper_mats = {}
    helper_mats['zero'] = np.zeros((grid['Ny'], grid['Ny']))
    helper_mats['I'] = np.eye(grid['Ny'])
    I = helper_mats['I']
    zero = helper_mats['zero']

    # Define more SPE helper functions 
    helper_mats['uvwP_from_LST'] = np.block([ [np.eye(4*grid["Ny"]),np.zeros((4*grid["Ny"],4*grid["Ny"]))] ])
    helper_mats['u_from_SPE'] = np.block([[I,zero,zero,zero]])
    helper_mats['v_from_SPE'] = np.block([[zero,I,zero,zero]])
    helper_mats['w_from_SPE'] = np.block([[zero,zero,I,zero]])
    helper_mats['P_from_SPE'] = np.block([[zero,zero,zero,I]])

    # Initialize baseflow using interpolated data
    Baseflow = bf.Baseflow(Grid)
    Baseflow.set_velocity_field(Grid.xi_grid, Grid.eta_grid, Grid.u_grid, Grid.v_grid, Grid.p_grid)
    print_rz(f"Curved surface baseflow initialized\n")

    # Initialize equation instance
    if config["simulation"]['type'] == "LST":
        EQS = LST(Grid, config, Baseflow, helper_mats)
    elif config["simulation"]['type'] == "GORTLER":
        EQS = Gortler(Grid, config, Baseflow, helper_mats)
    elif config['simulation']['type'] == "LPSE": 
        EQS = LPSE(Grid, config, Baseflow)
        config['modes']['temporal'] = 1
        config['modes']['spanwise'] = 1
    elif config['simulation']['type'] == "NLPSE":
        config['modes']['temporal'] += 1
        config['modes']['spanwise'] += 1
        EQS = NLPSE(Grid, config, Baseflow)
        EQS.helper_mats = helper_mats

    dirExists = os.path.exists('./nlpse_data')
    if rank == 0:
        if not dirExists:
            os.makedirs('nlpse_data')

    # Solve initial station 
    station = 0
    EQS.Solve(station)
    R = np.sqrt(config['flow']['Uinf'] * Grid.xgrid / config['flow']['nu'])
    print_rz("Solved station 0")
    # if rank == 0:
    #     io.save_data(EQS, Grid, config, helper_mats, R, station, pickling=False)


    return helper_mats, Grid, EQS

def initialize_gortler(config):
    '''
    Initializes a pre-defined curved surface geometry for PSE computation.

    Parameters:
        config (dictionary): Dictionary of parameters specified by the input file

    Returns:
        helper_mats (dictionary): Dictionary of matrices that extract u,v,w,p from q-vector
        Grid (CurvedSurfaceGrid): Mesh object that is used by the PSE solver
        EQS (equation class): PSE class object
    '''

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    flow = config['flow']
    disturbance = config['disturbance']
    geometry = config['geometry']
    grid = config['grid']

    # Here I generate the surface described in the paper by Li and Malik.
    l0 = 0.0005504818825631803 # sqrt(nu X_0 / U_e)
    x0_star = 0.1
    x0 = x0_star / l0 # this is the same value as in the config file
    x_max_star = 1.25
    x_max = x_max_star / l0
    R = np.abs(1.0 / geometry['curvature']['kappa'])
    n_surface = 1000
    x_surface = np.linspace(x0, x_max, n_surface)
    y_surface = -1.0 * np.sqrt(R**2 - x_surface**2) + R

    Grid = nm.gortlerGrid(x_surface, y_surface, grid['hx'], flow['x0'], geometry['ymax'], grid['Ny'], geometry['curvature']['kappa'], method = grid['method'], need_map=True, Uinf=flow['Uinf'], nu=flow['nu'])

    helper_mats = {}
    helper_mats['zero'] = np.zeros((grid['Ny'], grid['Ny']))
    helper_mats['I'] = np.eye(grid['Ny'])
    I = helper_mats['I']
    zero = helper_mats['zero']

    # Define more SPE helper functions 
    helper_mats['uvwP_from_LST'] = np.block([ [np.eye(4*grid["Ny"]),np.zeros((4*grid["Ny"],4*grid["Ny"]))] ])
    helper_mats['u_from_SPE'] = np.block([[I,zero,zero,zero]])
    helper_mats['v_from_SPE'] = np.block([[zero,I,zero,zero]])
    helper_mats['w_from_SPE'] = np.block([[zero,zero,I,zero]])
    helper_mats['P_from_SPE'] = np.block([[zero,zero,zero,I]])

    # Initialize baseflow using interpolated data
    # Grid.transform_velocities()
    Baseflow = bf.Baseflow(Grid)
    Baseflow.Blasius(Grid.ygrid, x=flow['x0'], Uinf = flow['Uinf'], nu = flow['nu'])
    print_rz(f"Curved surface baseflow initialized\n")

    # Store grid metrics in the baseflow object
    # Baseflow.metrics = Grid.get_metrics()

    # Initialize equation instance
    if config["simulation"]['type'] == "LST":
        EQS = LST(Grid, config, Baseflow, helper_mats)
    elif config["simulation"]['type'] == "GORTLER":
        EQS = Gortler(Grid, config, Baseflow, helper_mats)
    elif config['simulation']['type'] == "LPSE": 
        EQS = LPSE(Grid, config, Baseflow)
        config['modes']['temporal'] = 1
        config['modes']['spanwise'] = 1
    elif config['simulation']['type'] == "NLPSE":
        config['modes']['temporal'] += 1
        config['modes']['spanwise'] += 1
        EQS = NLPSE(Grid, config, Baseflow)
        EQS.helper_mats = helper_mats

    dirExists = os.path.exists('./nlpse_data')
    if rank == 0:
        if not dirExists:
            os.makedirs('nlpse_data')

    # Solve initial station 
    station = 0
    EQS.Solve(station)
    R = np.sqrt(config['flow']['Uinf'] * Grid.xgrid / config['flow']['nu'])
    print_rz("Solved station 0")
    if rank == 0:
        io.save_data(EQS, Grid, config, helper_mats, R, station, pickling=False)


    return helper_mats, Grid, EQS
