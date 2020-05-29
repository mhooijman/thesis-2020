import numpy as np
from evaluate_pruned_model import evaluate_with_pruning
import csv
import absl.app
import pandas as pd
import os
from utils import create_dir_if_not_exists, write_numpy_to_file, write_to_file

import sys
import json 

sys.path.append('./DeepSpeech')
from util.flags import create_flags
from util.config import initialize_globals


def evaluate(scores_path, prune_percent, evaluate_files):
    
    # Write evaluate files as temporary csv file
    csv_file_path = './data/tmp_evaluation_info.csv'
    write_csv(csv_file_path, list(evaluate_files.values()))

    evaluation_result = {}

    # Evaluate score based
    wer, cer, mean_loss = evaluate_with_pruning(test_csvs=csv_file_path, 
                        prune_percentage=prune_percent, random=False, scores_file=scores_path, 
                                                result_file=None, verbose=False)

    evaluation_result['score-based'] = {
        'wer': float(wer), 'cer': float(cer), 'mean_loss': float(mean_loss)
    }

    # Evaluate random
    wer, cer, mean_loss = evaluate_with_pruning(test_csvs=csv_file_path, 
                        prune_percentage=prune_percent, random=True, scores_file=scores_path, 
                                                result_file=None, verbose=False)

    evaluation_result['random'] = {
        'wer': float(wer), 'cer': float(cer), 'mean_loss': float(mean_loss)
    }  
    
    return evaluation_result 

def get_file_info(path):
    return {i['wav_filename'].split('/')[-1][:-4]: i for i in pd.read_csv(path).T.to_dict().values()}

def write_csv(path, data):
    create_dir_if_not_exists('/'.join(path.split('/')[:-1]))
    pd.DataFrame(data).to_csv(path, index=False)


def main(_):
    initialize_globals()
    
    # Evaluate on:
    # - files used for pruning (per set prune or total prune)
    # - original test set for model

    pertubed_sets = json.load(open('data/pertubed_input_sets_balanced.json'))
    common_voice_info = get_file_info('./data/common-voice-pertubed_sets.csv')
    percents = [0, .05, .1, .2]


    # Prune and evaluate on all pertubed sets combined
    evaluation = {}
    file_info = [common_voice_info[name] for name in 
        [item['path'][:-4] for item in [set for set in pertubed_sets]]]
    for percent in percents:
        pruned_results, random_results = evaluate(
            scores_path='./results/activations_combined.npy', prune_percent=percent,
                                                                    evaluate_files=file_info)
        evaluation['{}'.format(percent)] = results

    json.dump(evaluation, open('./results/evaluations_all_pertubated_sets.json', 'w+'))


    # Prune and evaluate on original test set
    file_info = get_file_info('./data/librivox-test-clean.csv')
    evaluation = {}
    for percent in percents:
        results = evaluate(
            scores_path='./results/activations_combined.npy', prune_percent=percent,
                                                                evaluate_files=file_info)
        
        evaluation['{}'.format(percent)] = results
    
    json.dump(evaluation, open('./results/evaluations_original_test_set.json', 'w+'))


    # Prune and evaluate per pertubed set
    evaluation = {}
    for set in pertubed_sets:
        file_info = [common_voice_info[name] for name in [item['path'][:-4] for item in set]]
        for percent in percents:
            pruned_results, random_results = evaluate(
                scores_path='./results/activations_combined_sets/activations_set_{}.npy'.format(set['set_id']),
                                            prune_percent=percent, evaluate_files=file_info)

            evaluation['{}-{}'.format(set, percent)] = results

    json.dump(evaluation, open('./results/evaluation_per_pertubed_set.json', 'w+'))


if __name__ == "__main__":
    create_flags()
    absl.app.run(main)