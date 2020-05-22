import numpy as np
from evaluate_pruned_model import evaluate_with_pruning
import csv
import absl.app
import pandas as pd
import os
from utils import create_dir_if_not_exists, write_numpy_to_file

import sys
sys.path.append('./DeepSpeech')
from util.flags import create_flags
from util.config import initialize_globals


def do_experiment(scores_dir, csv_path, file_info, results_file):
    for filename in os.listdir(scores_dir):

        info = file_info[filename[:-4]]
        csv_file_path = '{}/{}.csv'.format(csv_path, filename[:-4])
        write_csv(csv_file_path, info)

        importance = np.load('{}/{}'.format(scores_dir, filename))
        importance_normalized = []
        print(importance.shape)
        for layer in importance:
            mean = np.mean(layer, axis=0)
            l2 = mean / np.sqrt(np.sum(mean**2))
            importance_normalized.append(l2)
        
        normalized_file = '{}/normalized/{}'.format(scores_dir, filename)
        write_numpy_to_file(normalized_file, np.array(importance_normalized))

        for percentage in [0, .05, .1, .15, .2, .3, .5]:
            evaluate_with_pruning(test_csvs=csv_file_path, prune_percentage=percentage,
                    random=False, scores_file=normalized_file, result_file=results_file)


def get_file_info(path):
    return {i['wav_filename'].split('/')[-1][:-4]: i for i in pd.read_csv(path).T.to_dict().values()}

def write_csv(path, data):
    create_dir_if_not_exists('/'.join(path.split('/')[:-1]))
    pd.DataFrame(data, index=[0]).to_csv(path)



def main(_):
    initialize_globals()
    file_into_path = './data/librivox-test-clean.csv'
    file_info = get_file_info(file_into_path)

    scores_dir = './results/imp_scores_per_timestep'
    csv_path = './results/imp_scores_per_timestep/csv'
    results_file = './results/evaluation_output.txt'
    
    do_experiment(scores_dir, csv_path, file_info, results_file)

if __name__ == "__main__":
    create_flags()
    absl.app.run(main)
