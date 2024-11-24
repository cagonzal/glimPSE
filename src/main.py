import sys
import time 
import inputOutput as io
from inputOutput import print_rz
import initialize as ini
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpi4py import MPI

def main():
    """
    Main function to run the simulation.

    This function initializes the MPI environment, reads the configuration file,
    sets up the simulation based on the specified geometry, and handles the pressure gradient.
    It also prints relevant information based on the MPI rank.

    The configuration file can be specified as a command-line argument. If not provided,
    'pse_config.yaml' is used by default.

    Raises:
        SystemExit: If an invalid geometry type is specified in the configuration file.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print_rz(f"Running with {size} ranks")

    start = time.time()

    config_file = sys.argv[1] if len(sys.argv) > 1 else 'pse_config.yaml'
    config = io.read_input_parameters(config_file)

    # Handle pressure gradient
    if config['numerical']['dpdx_stabilizer']:
        print_rz("Running with pressure gradient turned off")
    else:
        print_rz("Running with pressure gradient turned on")

    geometry = config['geometry']['type']
    simulation_type = config['simulation']['type']

    if geometry == 'flat_plate': 
        helper_mats, Grid, EQS = ini.initialize_flatplate(config)
        comm.Barrier()
    elif geometry == 'import_geom':
        helper_mats, Grid, EQS = ini.initialize_curved_surface(config)
        comm.Barrier()
    elif geometry == 'gortler':
        helper_mats, Grid, EQS = ini.initialize_gortler(config)
    else:
        print_rz(f"Invalid geometry: {geometry}")
        sys.exit(1)

    R = np.sqrt(config['flow']['Uinf'] * Grid.xgrid / config['flow']['nu'])

    if config['simulation']['restart']:
        with open('nlpse_data/EQS.pkl', 'rb') as f:
            EQS = pickle.load(f)

        print_rz("Starting from restart condition")
        Nfinal = EQS.U.shape[1]
        print_rz(f"Number of stations saved in previous run = {Nfinal}")
        for station in range(Nfinal, config['grid']['Nx']):
            EQS.Solve(station)

            if simulation_type == "NLPSE" and station % config['output']['sync_frequency'] == 0:
                print_rz("=============")
                print_rz("Saving data...")
                print_rz("=============")

                if rank == 0:
                    io.save_data(EQS, Grid, config, helper_mats, R, station, pickling=False)
                print_rz(f"Elapsed time so far = {time.time() - start}")

    else:
        for station in range(1, config['grid']['Nx']):
            EQS.Solve(station)

            if simulation_type == "NLPSE" and station % config['output']['sync_frequency'] == 0:
                print_rz("=============")
                print_rz("Saving data...")
                print_rz("=============")

                if rank == 0:
                    io.save_data(EQS, Grid, config, helper_mats, R, station, pickling=False)
                print_rz(f"Elapsed time so far = {time.time() - start}")

                if "xi_max" in config['grid']:
                    if EQS.xgrid[station] >= config['grid']['xi_max']:
                        print_rz("\nFinal xi location reached. Shutting down the simulation now...")
                        break

    end = time.time()
    print_rz(f"Total elapsed time = {end - start}")

if __name__ == "__main__":
    main()
