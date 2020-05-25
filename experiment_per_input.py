import numpy as np
from evaluate_pruned_model import evaluate_with_pruning
import csv
import absl.app
import pandas as pd
import os
from utils import create_dir_if_not_exists, write_numpy_to_file, write_to_file

import sys
sys.path.append('./DeepSpeech')
from util.flags import create_flags
from util.config import initialize_globals


def do_experiment(scores_dir, csv_path, file_info, results_file, tests):
    '''Evaluate using single scores per input sample'''
    results_score = {'{}_percent'.format(i*100): [] for i in tests}
    results_random = {'{}_percent'.format(i*100): [] for i in tests}
    for n, filename in enumerate([f for f in os.listdir(scores_dir) if f.endswith('.npy')]):

        info = file_info[filename[:-4]]
        csv_file_path = '{}/{}.csv'.format(csv_path, filename[:-4])
        write_csv(csv_file_path, info)

        importance = np.load('{}/{}'.format(scores_dir, filename))
        importance_combined = []
        for i in range(importance.shape[1]):
            layer = importance[:,i]
            steps = []
            for frame in layer:
                for step in frame:
                    steps.append(step)
            steps = np.array(steps)
            sum = np.sum(np.abs(steps), axis=0)
            importance_combined.append(sum)

        normalized_file = '{}/normalized/{}'.format(scores_dir, filename)
        write_numpy_to_file(normalized_file, np.array(importance_combined))

        for p in tests:
            results_score['{}_percent'.format(p*100)].append(evaluate_with_pruning(test_csvs=csv_file_path, prune_percentage=p,
                    random=False, scores_file=normalized_file, result_file=results_file, verbose=False))

        for p in tests:
            results_random['{}_percent'.format(p*100)].append(evaluate_with_pruning(test_csvs=csv_file_path, prune_percentage=p,
                    random=True, scores_file=normalized_file, result_file=results_file, verbose=False))

        if n/50 in range(30):

            print_evaluation_report(results_score, 'Results at sample {} for score based evaluation:'.format(n))
            print_evaluation_report(results_random, 'Results at sample {} for random evaluation:'.format(n))

    print_evaluation_report(results_score, 'Final results for score based evaluation:')
    print_evaluation_report(results_random, 'Final results for random evaluation:')

def do_experiment_2(scores_dir, csv_path, file_info, results_file):
    '''Evaluate using time-step-based scores'''
    tests = [.05, .1, .15, .2]
    results_score = {'{}_percent'.format(i*100): [] for i in tests}
    results_random = {'{}_percent'.format(i*100): [] for i in tests}

    for n, filename in enumerate([f for f in os.listdir(scores_dir) if f.endswith('.npy')]):

        info = file_info[filename[:-4]]
        csv_file_path = '{}/{}.csv'.format(csv_path, filename[:-4])
        write_csv(csv_file_path, info)

        importance = np.load('{}/{}'.format(scores_dir, filename))
        importance_combined = []

        # shape of importange: n_steps, n_layers, n_neurons (?, 5, 2048)
        for layer in importance:
            sum = np.sum(np.abs(layer), axis=0)
            importance_combined.append(sum)

        normalized_file = '{}/normalized/{}'.format(scores_dir, filename)
        write_numpy_to_file(normalized_file, np.array(importance_combined))

        for p in tests:
            results_score['{}_percent'.format(p*100)].append(evaluate_with_pruning(test_csvs=csv_file_path, prune_percentage=p,
                    random=False, scores_file=normalized_file, result_file=results_file, verbose=False))

        for p in tests:
            results_random['{}_percent'.format(p*100)].append(evaluate_with_pruning(test_csvs=csv_file_path, prune_percentage=p,
                    random=True, scores_file=normalized_file, result_file=results_file, verbose=False))

        if n/50 in range(30):

            print_evaluation_report(results_score, 'Results at sample {} for score based evaluation:'.format(n))
            print_evaluation_report(results_random, 'Results at sample {} for random evaluation:'.format(n))

    print_evaluation_report(results_score, 'Final results for score based evaluation:')
    print_evaluation_report(results_random, 'Final results for random evaluation:')

def print_evaluation_report(data, title):
    print(title+"\n")
    write_to_file('./results/evaluation_per_sample.txt', title+"\n", 'a+')
    for key, items in data.items():
        cer = []
        wer = []
        loss = []
        for item in items:
            for i in item:
                cer.append(i.cer)
                wer.append(i.wer)
                loss.append(i.loss)
        res_str = 'Results for {}: cer: {}, wer: {}, loss: {}.'.format(key, sum(cer)/len(cer), sum(wer)/len(wer), sum(loss)/len(loss))
        write_to_file('./results/evaluation_per_sample.txt', res_str+"\n", 'a+')
        print(res_str)

    write_to_file('./results/evaluation_per_sample.txt', "\n", 'a+')

def get_file_info(path):
    return {i['wav_filename'].split('/')[-1][:-4]: i for i in pd.read_csv(path).T.to_dict().values()}

def write_csv(path, data):
    create_dir_if_not_exists('/'.join(path.split('/')[:-1]))
    pd.DataFrame(data, index=[0]).to_csv(path)


def main(_):
    initialize_globals()
    file_into_path = './data/librivox-test-clean.csv'
    file_info = get_file_info(file_into_path)
    tests = [.05, .1, .15, .2]

    scores_dir = './results/imp_scores_per_timestep'
    csv_path = './results/imp_scores_per_timestep/csv'
    results_file = './results/evaluation_output.txt'
    do_experiment(scores_dir, csv_path, file_info, results_file, tests)

    # scores_dir = './results/imp_scores_avg_inputs'
    # csv_path = './results/imp_scores_avg_inputs/csv'
    # results_file = './results/evaluation_output.txt'
    # do_experiment_2(scores_dir, csv_path, file_info, results_file)

if __name__ == "__main__":
    create_flags()
    absl.app.run(main)
