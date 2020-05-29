import json
from matplotlib import pyplot as plt
from utils import create_dir_if_not_exists

create_dir_if_not_exists('./figures')
color = "#a3a3a3"
rwidth = 0.85

# Distribution of speech recordings per input set

# sets = json.load(open('pertubed_input_sets_balanced.json'))
# n_samples_per_set = [set['set_length'] for set in sets]
n_samples_per_set = [34, 12, 14, 52, 18, 36, 24, 10, 28, 10, 12, 12, 10, 28, 34, 10, 14, 12, 20, 10, 30, 10, 10, 10, 16, 24, 14, 28, 16, 46, 50, 10, 10, 28, 14, 36, 22, 28, 10, 10, 32, 26, 34, 10, 12, 26, 24, 10, 10, 28, 12, 10, 10, 10, 10, 10, 10, 28, 36, 30, 34, 10, 12, 12, 26, 28, 30, 12, 10, 32, 10, 14, 12, 10, 30, 12, 14, 10, 28, 10, 12, 10, 26, 12, 30, 36, 30, 10, 16, 10, 34, 34, 10, 36, 10, 16, 34, 38, 10, 16, 16, 10, 36, 32, 12, 12, 10, 24, 10, 30, 10, 12, 10, 10, 30, 28, 10, 20, 14, 10, 12, 14, 30, 10, 26, 10, 32, 10, 16, 16, 10, 12, 10, 10, 16, 12, 14, 10, 10, 10, 38, 12, 10, 34, 22, 32, 10, 42, 10, 34, 12, 22, 16, 12, 10, 16, 12, 26, 14, 18, 26, 14, 12, 10, 14, 10, 10, 30, 10, 10, 12, 50, 32, 40, 14, 10, 10, 32, 10, 10, 12, 10, 30, 16, 10, 12, 26, 10, 12, 36, 20, 34, 10, 24, 12, 12, 10, 14, 10, 10, 34, 12, 28, 10, 12, 30, 42, 10, 16, 36, 10, 12, 24, 10, 12, 26, 12, 12, 36, 32, 28, 34, 10, 10, 12, 12, 32, 10, 12, 10, 22, 10, 14, 10, 34, 12, 32, 10, 48, 32, 12, 32, 28, 26, 14, 26, 10, 18, 10, 12, 36, 28, 12, 10, 40, 42, 26, 12, 18, 14, 10, 26, 34, 12, 12, 12, 28, 10, 26, 12, 24, 12, 10, 12, 12, 14, 12, 12, 10, 40, 40, 26, 36, 12, 48, 16, 10, 10, 32, 18, 24, 30, 28, 10, 12, 10, 12, 28, 10, 10, 10, 10, 16, 10, 28, 20, 14, 44, 32, 12, 10, 10, 10, 12, 22, 22, 28, 12, 10, 30, 10, 10, 32, 10, 10, 36, 24, 10, 28, 40, 40, 10, 10, 10, 20, 10, 10, 28, 12, 30, 10, 34, 36, 10, 12, 12, 30, 12, 10, 22, 10, 10, 40, 10, 10, 40, 32, 10, 12, 36, 10, 28, 34, 10, 10, 26, 12, 10, 10, 42, 16, 10, 10, 10, 10, 34, 32, 10, 26, 20, 22, 32, 10, 10, 28, 10, 34, 32, 36, 10, 30, 28, 10, 26, 42, 14, 32, 10, 10, 18, 32, 24, 10, 10, 14, 22, 10, 12, 10, 10, 32, 18, 28, 10, 26, 26, 28, 16, 32, 10, 26, 36, 10, 10, 12, 30, 10, 10, 14, 28, 28, 10, 10, 36, 12, 10, 12, 12, 34, 10, 10, 36, 30, 12, 10, 36, 32, 10, 22, 10, 28, 10, 10, 40, 10, 44, 10, 24, 10, 10, 14, 20, 10, 10, 26, 12]

plt.hist(n_samples_per_set, bins=len(set(n_samples_per_set)), color=color, rwidth=rwidth)
plt.xlabel('Number of recordings in input set')
plt.ylabel('Frequency')
plt.savefig('./figures/input-sets-distribution.pdf')  

import numpy as np
a = np.array(n_samples_per_set)
print('Number of sets: {}.'.format(len(n_samples_per_set)))
print('Range of number of recordings per set: {} to {}.'.format(np.min(a), np.max(a)))
print('Average recordings per set: {}.'.format(np.mean(a)))
print('Standard deviation of number of inputs per sets: {}.'.format(np.std(a)))