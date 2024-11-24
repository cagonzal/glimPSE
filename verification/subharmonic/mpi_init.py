import numpy as np 
import yaml

def viableHarmonics(init_modes, numM, numN, num_repeats=3):
        """
        Determines viable harmonics for the initial modes.

        Parameters:
            init_modes (array-like): Initial modes.
            num_repeats (int, optional): Number of repeats. Defaults to 3.

        Returns:
        list: A list of unique tuples that satisfy the conditions.

        """

        for i in range(init_modes.shape[0]):
            if i == 0:
                result = [tuple(init_modes[0])]
            else:
                result.append(tuple(init_modes[i]))
        for repeat in range(num_repeats):
            loop_arr = list(result)
            for i in range(len(loop_arr)):
                for j in range(i+1, len(loop_arr)):
                    pair_sum = tuple(map(sum, zip(loop_arr[i], loop_arr[j])))  # Calculate the sum of tuple pairs
                    pair_diff = tuple(map(lambda x, y: x - y, loop_arr[i], loop_arr[j]))  # Calculate the difference of tuple pairs
                    result.append(pair_sum)
                    result.append(pair_diff)  # Add subtracted pair to the result
        result = list(set(result))
        
        # Remove tuples whose first entry is greater than M or second entry is greater than N
        # have to subtract by one to account for MFD
        result = [t for t in result if t[0] <= numM-1 and t[1] <= numN-1]
        # Remove tuples with negative numbers
        result = [t for t in result if all(x >= 0 for x in t)]

        result = sorted(result)

        # print(f"Harmonics: {result}")
        
        return np.array(result)

def replace_line(file_name, line_num, text):
    lines = open(file_name, 'r').readlines()
    lines[line_num] = text
    out = open(file_name, 'w')
    out.writelines(lines)
    out.close()

with open('./pse_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

config['disturbance']['init_modes'] = np.array(config['disturbance']['init_modes'])
harmonics = viableHarmonics(config['disturbance']['init_modes'], config['modes']['temporal']+1, config['modes']['spanwise']+1, num_repeats=4)
# print("modes")
# print(harmonics)
print(f"number of modes = {harmonics.shape[0]}")
numModes = harmonics.shape[0]
numModes = harmonics.shape[0]

# 16 cores per node
numNodes = int(np.floor((numModes-1) / 16) + 1)


replace_line("submit.slurm", 5, f"#SBATCH -N {numNodes}\n")
replace_line("submit.slurm", 6, f"#SBATCH -n {numModes}\n")
replace_line("submit.slurm", 14, f"mpirun -n {numModes} python -u ~/glimPSE/glimPSE/src/main.py \n")

# with open('submit.slurm', 'w') as file:
#     lines = file.readlines()
#     lines[5] = f"#SBATCH -N {numNodes}\n"
#     lines[6] = f"#SBATCH -n {numModes}\n"  # Modify line 17
    # modified_code = ''.join(lines)

# print(modified_code)

