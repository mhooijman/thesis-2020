import os
import numpy as np
from utils import create_dir_if_not_exists, write_numpy_to_file


result_dir = './results/activations_combined_sets'
create_dir_if_not_exists(result_dir)

combined_set_activations = []
for set in os.listdir('./results/activations/'):
    set_path = './results/activations/{}'.format(set)
    l2_set_activations = []

    activation_files = [f for f in os.listdir(set_path) if f.endswith('.npy')]
    for f in activation_files:
        activations = np.load('{}/{}'.format(set_path, f))
        sum_activations = np.sum(activations, axis=1)  # cummulated over timesteps
        l2_activations = sum_activations / np.sqrt(np.sum(sum_activations**2))  # l2 normalize
        l2_set_activations.append(l2_activations)

    averaged_set_activations = np.mean(np.array(l2_set_activations), axis=0)  # average over activations per sample

    write_numpy_to_file('{}/activations_{}'.format(result_dir, set), averaged_set_activations)  # write to file

    combined_set_activations.append(averaged_set_activations)

# combine set activations into one array using mean
combined_activations = np.mean(np.array(combined_set_activations), axis=0)
write_numpy_to_file('./results/activations_combined', combined_activations)  # write to file
