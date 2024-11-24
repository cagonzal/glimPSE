import yaml
import h5py
import os
import numpy as np
from mpi4py import MPI

def print_rz(message, rank_in=0):
    """
    Prints a message if the current MPI rank matches the specified rank.

    Parameters:
        message (str): The message to print.
        rank_in (int): The rank to match. Default is 0.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank_in == rank:
        print(message)

def validate_config(config):
    """
    Validates the user's configuration file.

    Args:
        config (dict): Configuration dictionary containing flow parameters.

    Raises:
        AssertionError: If any of the required parameters are not positive.
    """
    assert config['flow']['Re'] > 0, "Reynolds number must be positive"
    assert config['flow']['Uinf'] > 0, "Freestream velocity must be positive"
    assert config['flow']['x0'] >= 0, "Reference x0 must be nonnegative"

    if "hx" in config["grid"]:
        assert config["grid"]["hx"] > 0, "Grid spacing must be greater than zero"
        if "hx_factor" in config["grid"]:
            raise Exception("Cannot specify both hx_factor and hx in the configuration")

    if "hx_factor" in config["grid"]:
        assert config["grid"]["hx_factor"] > 0, "Grid spacing factor must be greater than zero"
        if "hx" in config['grid']:
            raise Exception("Cannot specify both hx_factor and hx in the configuration")

    return

def read_input_parameters(file_path):
    """
    Reads and processes input parameters from a YAML file.

    Args:
        file_path (str): Path to the YAML file containing configuration parameters.

    Returns:
        dict: Processed configuration dictionary.
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    print_rz(f'Reading parameters from {file_path}')

    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)

    # Calculate derived parameters
    config['disturbance']['omega'] = config['disturbance']['F'] * config['flow']['Re'] / (1e6)
    config['disturbance']['beta'] = config['disturbance']['b'] * config['flow']['Re'] / (1e3)

    # Convert init_modes to numpy array
    config['disturbance']['init_modes'] = np.array(config['disturbance']['init_modes'])
    
    # set some default values 
    if "compute_eigs" not in config["numerical"]:
        config["numerical"]["compute_eigs"] = False

    if "method" not in config["numerical"]:
        config["numerical"]["method"] = "IE"

    validate_config(config)

    return config

def save_data(EQS, Grid, config, helper_mats, R, station, pickling=True):
    """
    Saves simulation data to an HDF5 file.

    Args:
        EQS: Object containing simulation results.
        Grid: Object containing grid information.
        config (dict): Configuration dictionary.
        helper_mats: Helper matrices used in the simulation.
        R: Array of R values.
        station (int): Station index up to which data is saved.
        pickling (bool, optional): Flag to enable/disable pickling. Defaults to True.

    Raises:
        IOError: If there is an issue writing to the file.
    """

    filename = 'nlpse_data/nlpse_data.h5'
    temp_filename = 'nlpse_data/nlpse_data_temp.h5'
    
    try:
        # Write to a temporary file first
        with h5py.File(temp_filename, 'w') as hf:
            # Save alpha
            alpha_group = hf.create_group('alpha')
            for m in range(config['modes']['temporal']):
                for n in range(config['modes']['spanwise']):
                    alpha_group.create_dataset(f'alpha_{m}{n}', data=EQS.alpha[0:station, m, n])
            
            # Save other datasets
            hf.create_dataset('x_alpha', data=Grid.xgrid[0:station])
            hf.create_dataset('R', data=R[0:station])
            hf.create_dataset('x', data=Grid.xgrid[0:station])
            hf.create_dataset('U', data=EQS.U[:,0:station])
            hf.create_dataset('V', data=EQS.V[:,0:station])
            hf.create_dataset('Uy', data=EQS.Uy[:,0:station])
            hf.create_dataset('y', data=EQS.ygrid)
            hf.create_dataset('xi', data=Grid.xi_grid[0:station])
            try:
                hf.create_dataset('xx', data=Grid.physicalX[0:station,0])
                hf.create_dataset('yy', data=Grid.physicalY[0,:])
            except:
                pass
            # hf.create_dataset('theta', data=Grid.theta)
            
            # Save disturbances
            hf.create_dataset('u', data=EQS.getDisturbance(helper_mats, station, 'u')[:,:,0:station,:])
            hf.create_dataset('v', data=EQS.getDisturbance(helper_mats, station, 'v')[:,:,0:station,:])
            hf.create_dataset('w', data=EQS.getDisturbance(helper_mats, station, 'w')[:,:,0:station,:])
            hf.create_dataset('p', data=EQS.getDisturbance(helper_mats, station, 'p')[:,:,0:station,:])

            # save eigenvalues
            hf.create_dataset('eigs', data=EQS.opEigs[0:station, :, :, :])
        
        # If writing to temp file was successful, rename it to the final filename
        if os.path.exists(filename):
            os.remove(filename)
        os.rename(temp_filename, filename)
        
        print(f"Data saved to {filename}")
        
        # Verify the file was written correctly
        with h5py.File(filename, 'r') as hf:
            print("Keys in the saved file:", list(hf.keys()))
    
    except Exception as e:
        print(f"Error saving data to HDF5 file: {str(e)}")
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
    
    # Save EQS object (if pickling is enabled)
    if pickling:
        try:
            import pickle
            with open('nlpse_data/EQS.pkl', 'wb') as f:
                pickle.dump(EQS, f, protocol=4)
            print("EQS object saved successfully.")
        except Exception as e:
            print(f"Error pickling EQS object: {str(e)}")

def read_surface_data(filename):
    """
    Reads surface coordinate data from an HDF5 file.

    Args:
        filename (str): Path to the HDF5 file containing surface coordinates.

    Returns:
        tuple: x and y coordinates of the surface.

    Raises:
        KeyError: If required datasets are not found in the file.
        IOError: If there's an error reading the file.
    """
    try:
        with h5py.File(filename, 'r') as f:
            if 'surface/x' not in f or 'surface/y' not in f:
                raise KeyError("Surface coordinate datasets not found in the file.")
            
            x_surface = f['surface/x'][:]
            y_surface = f['surface/y'][:]

        return x_surface, y_surface
    
    except IOError as e:
        print(f"Error reading file {filename}: {e}")
        raise
    except KeyError as e:
        print(f"Error: {e}")
        raise
    
def read_velocity_field(filename):
    """
    Reads velocity field data from an HDF5 file.

    Args:
        filename (str): Path to the HDF5 file containing velocity field data.

    Returns:
        tuple: x coordinates, y coordinates, u velocity component, and v velocity component.

    Raises:
        KeyError: If required datasets are not found in the file.
        IOError: If there's an error reading the file.
    """
    try:
        with h5py.File(filename, 'r') as f:
            required_datasets = ['grid/x', 'grid/y', 'velocity/u', 'velocity/v']
            for dataset in required_datasets:
                if dataset not in f:
                    raise KeyError(f"Dataset {dataset} not found in the file.")
            
            # x_field = f['grid/x'][:]
            # y_field = f['grid/y'][:]
            # u_field = f['velocity/u'][:]
            # v_field = f['velocity/v'][:]
            # p_field = f['pressure/p'][:]

            x_field = f['grid/X'][:]
            y_field = f['grid/Y'][:]
            u_field = f['velocity/U'][:]
            v_field = f['velocity/V'][:]
            p_field = f['pressure/p'][:]

            if f['grid/kappa']:

                kappa_field = f['grid/kappa'][:]
                return x_field, y_field, u_field, v_field, p_field, kappa_field

            else:

                return x_field, y_field, u_field, v_field, p_field
    
    except IOError as e:
        print(f"Error reading file {filename}: {e}")
        raise
    except KeyError as e:
        print(f"Error: {e}")
        raise
