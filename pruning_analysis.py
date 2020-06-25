# What is being pruned
# Measure the relation between top X neuron indexes in recording and time-step scores
# Measure the relation between top X neuron indexes in inputSet scores and recording scores.
# Measure the relation between top X neuron indexes in total scores and inputSet scores

import os
import json
import numpy as np
from utils import create_dir_if_not_exists, write_numpy_to_file


set_dir = './results/activations_combined_sets'

combined_set_activations = []

rec_vs_ts_overlap_percentages = {
    '0.05': [],
    '0.1': [],
    '0.2': []
}

set_vs_rec_overlap_percentages = {
    '0.05': [],
    '0.1': [],
    '0.2': []
}

total_vs_set_overlap_percentages = {
    '0.05': [],
    '0.1': [],
    '0.2': []
}

for set in os.listdir('./results/activations/'):
    set_activations = np.load('{}/activations_{}.npy'.format(set_dir, set))
    set_path = './results/activations/{}'.format(set)

    activations_per_timestep = []
    activations_per_recording = []
    activations_per_recording_l2 = []
    activation_files = [f for f in os.listdir(set_path) if f.endswith('.npy')]
    for f in activation_files:
        activations = np.load('{}/{}'.format(set_path, f))
        activations_per_timestep.append(activations)

        sum_activations = np.sum(activations, axis=1)  # cummulated over timesteps
        activations_per_recording.append(sum_activations)

        l2_activations = sum_activations / np.sqrt(np.sum(sum_activations**2))  # l2 normalize
        activations_per_recording_l2.append(l2_activations)

    averaged_set_activations = np.mean(np.array(activations_per_recording_l2), axis=0)  # average over activations per sample
    combined_set_activations.append(averaged_set_activations)

    # recording vs time-step
    for i, act in enumerate(activations_per_timestep):
        recording_act_1d = activations_per_recording[i].flatten()
        n_neurons_total = recording_act_1d.shape[0]
        recording_indexes = recording_act_1d.argsort()

        for n in range(act.shape[1]):  # for each timestep
            ts_act_1d_indexes = act[:,n,:].flatten().argsort()
            
            for p in [0.05, 0.1, 0.2]:
                selection_recording = recording_indexes[-int(n_neurons_total*(1-p)):]
                selection_timestep = ts_act_1d_indexes[-int(n_neurons_total*(1-p)):]

                overlapping_acts = np.intersect1d(selection_recording, selection_timestep)
                n_overlap = overlapping_acts.shape[0]
                percent_overlap = n_overlap / n_neurons_total*(1-p)

                rec_vs_ts_overlap_percentages[str(p)].append(percent_overlap)


    # set vs recording
    set_act_1d = averaged_set_activations.flatten()
    n_neurons_total = set_act_1d.shape[0]
    set_indexes = set_act_1d.argsort()
    for act in activations_per_recording_l2:
        rec_act_1d_indexes = act.flatten().argsort()

        for p in [0.05, 0.1, 0.2]:
            selection_set = set_indexes[-int(n_neurons_total*(1-p)):]
            selection_recordings = rec_act_1d_indexes[-int(n_neurons_total*(1-p)):]

            overlapping_acts = np.intersect1d(selection_set, selection_recordings)
            n_overlap = overlapping_acts.shape[0]
            percent_overlap = n_overlap / n_neurons_total*(1-p)

            set_vs_rec_overlap_percentages[str(p)].append(percent_overlap)


# total vs set
total_act_1d = np.mean(np.array(combined_set_activations), axis=0).flatten()
n_neurons_total = total_act_1d.shape[0]
total_indexes = total_act_1d.argsort()

for act in combined_set_activations:
    set_act_1d_indexes = act.flatten().argsort()

    for p in [0.05, 0.1, 0.2]:
        selection_total = total_indexes[-int(n_neurons_total*(1-p)):]
        selection_set = set_act_1d_indexes[-int(n_neurons_total*(1-p)):]

        overlapping_acts = np.intersect1d(selection_total, selection_set)
        n_overlap = overlapping_acts.shape[0]
        percent_overlap = n_overlap / n_neurons_total*(1-p)

        total_vs_set_overlap_percentages[str(p)].append(percent_overlap)


# Average resuts of overlap calculations
rec_vs_ts_avg_overlap = {}
for p, overlap in rec_vs_ts_overlap_percentages.items():
    rec_vs_ts_avg_overlap[p] = sum(overlap) / len(overlap)

set_vs_rec_avg_overlap = {}
for p, overlap in set_vs_rec_overlap_percentages.items():
    set_vs_rec_avg_overlap[p] = sum(overlap) / len(overlap)

total_vs_set_avg_overlap = {}
for p, overlap in total_vs_set_overlap_percentages.items():
    total_vs_set_avg_overlap[p] = sum(overlap) / len(overlap)


print(rec_vs_ts_avg_overlap, set_vs_rec_avg_overlap, total_vs_set_avg_overlap)
