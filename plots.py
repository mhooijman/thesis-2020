import json
from matplotlib import pyplot as plt
from utils import create_dir_if_not_exists
from evaluate_pruned_model import prune_matrices
import numpy as np

# create_dir_if_not_exists('./figures')
mapping = {'0': 'dense_1', '1': 'dense_2', '2': 'dense_3', '3': 'rnn_lstm', '4': 'dense_4'}
importance_scores = np.load('./results/activations_combined.npy')

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


Pruning evaluation on CommonVoice test data
evaluation_stats = json.load(open('evaluations_all_pertubated_sets.json'))
evaluation_stats = {"0.1": {"random": {"cer": 0.463499037072701, "mean_loss": 70.49937438964844, "wer": 0.7953134698944755}, "score-based": {"cer": 0.3095510351468464, "mean_loss": 50.061241149902344, "wer": 0.6128181253879578}}, "0": {"score-based": {"cer": 0.1538878189696678, "mean_loss": 25.395339965820312, "wer": 0.36638733705772814}}, "0.05": {"random": {"cer": 0.291345690900337, "mean_loss": 45.26877212524414, "wer": 0.5966790813159528}, "score-based": {"cer": 0.19463168030813674, "mean_loss": 30.498804092407227, "wer": 0.4346679081315953}}, "0.2": {"random": {"cer": 0.9862481945113144, "mean_loss": 234.3425750732422, "wer": 1.0}, "score-based": {"cer": 0.9834195955705344, "mean_loss": 314.9481201171875, "wer": 0.9964307883302297}}}
for percentage, item in evaluation_stats.items():
    for type, stats in item.items():
        print('Results for {} pruning at {}%: CER: {}.'.format(type, float(percentage)*100, round(stats['cer'], 2)))
    print('\n')


# Pruning evaluation on LibriSpeech test data
evaluation_stats = {"0":{'score-based": {"wer": 0.14748174071819842, "cer": 0.046400028416154586, "mean_loss": 17.554323196411133}}}


# Plot pruning per layer

for p in [.05, .1, .2]:

    score_based_mask_percent = prune_matrices(importance_scores, p)
    pruned_per_layer_score_based = {mapping[str(k)]:np.count_nonzero(layer==0) 
                        for k, layer in enumerate(score_based_mask_percent)}

    random_mask_percent = prune_matrices(importance_scores, p, random=True)
    pruned_per_layer_random = {mapping[str(k)]:np.count_nonzero(layer==0) 
                            for k, layer in enumerate(random_mask_percent)}


    plt.plot(*zip(*pruned_per_layer_score_based.items()), label='imp-score-{}'.format(p*100))
    plt.plot(*zip(*pruned_per_layer_random.items()), label='random-{}'.format(p*100))

    plt.ylabel('Number of neurons pruned')
    plt.legend()
    plt.savefig('./figures/neurons-pruned-{}.pdf'.format(p*100))
    plt.clf()


# Gender encoding result common voice
encoding_common_stats = {
    'imp-score-10': {
        'layer_0': 0.9222222222222223,
        'layer_1': 0.9333333333333333,
        'layer_2': 0.9222222222222223,
        'layer_3': 0.9611111111111111,
        'layer_4': 0.8555555555555555
    },
    'random-10': {
        'layer_0': 0.9222222222222223,
        'layer_1': 0.9611111111111111,
        'layer_2': 0.9277777777777778,
        'layer_3': 0.9333333333333333,
        'layer_4': 0.8555555555555555
    },
    'full': {
        'layer_0': 0.9444444444444444,
        'layer_1': 0.9333333333333333,
        'layer_2': 0.8833333333333333,
        'layer_3': 0.8722222222222222,
        'layer_4': 0.8555555555555555
    },
    'randomly-initialized': {
        'layer_0': 0.8388888888888889,
        'layer_1': 0.85, 
        'layer_2': 0.8444444444444444,
        'layer_3': 0.8555555555555555,
        'layer_4': 0.85
    }
}

for model, results in encoding_common_stats.items():
    plt.plot(*zip(*results.items()), label=model)

plt.ylabel('Accuracy')
plt.legend()
plt.savefig('./figures/gender-encoding-common-voice.pdf')
plt.clf()


# Gender encoding result libri speech
encoding_libri_stats = {
    'imp-score-10': {
        'layer_0': 0.9908396946564886,
        'layer_1': 0.9587786259541985,
        'layer_2': 0.9908396946564886,
        'layer_3': 0.9694656488549618,
        'layer_4': 0.9969465648854962
    },
    'random-10': {
        'layer_0': 0.9908396946564886,
        'layer_1': 0.9984732824427481,
        'layer_2': 0.9984732824427481,
        'layer_3': 0.9801526717557252,
        'layer_4': 0.9969465648854962
    },
    'full': {
        'layer_0': 0.9954198473282443,
        'layer_1': 0.9786259541984733,
        'layer_2': 0.9740458015267176,
        'layer_3': 0.9908396946564886,
        'layer_4': 0.9923664122137404
    }
}

for model, results in encoding_libri_stats.items():
    plt.plot(*zip(*results.items()), label=model)

plt.ylabel('Accuracy')
plt.legend()
plt.savefig('./figures/gender-encoding-libri-speech.pdf')
plt.clf()

# Sentence length encoding experiment Common Voice

score_based_mask_percent = prune_matrices(importance_scores, .1)
pruned_per_layer_score_based = {mapping[str(k)]:np.count_nonzero(layer==0) 
                    for k, layer in enumerate(score_based_mask_percent)}

random_mask_percent = prune_matrices(importance_scores, .1, random=True)
pruned_per_layer_random = {mapping[str(k)]:np.count_nonzero(layer==0) 
                        for k, layer in enumerate(random_mask_percent)}

sentence_encoding_stats = {"libri_speech": {"full": {"layer_1": 0.10989010989010989, "layer_2": 0.17216117216117216, "layer_0": 0.1391941391941392, "layer_3": 0.15018315018315018, "layer_4": 0.16117216117216118}, "imp-score-10": {"layer_1": 0.12454212454212454, "layer_2": 0.10989010989010989, "layer_0": 0.0989010989010989, "layer_3": 0.18315018315018314, "layer_4": 0.15018315018315018}, "random-10": {"layer_1": 0.14285714285714285, "layer_2": 0.16483516483516483, "layer_0": 0.1282051282051282, "layer_3": 0.20146520146520147, "layer_4": 0.15018315018315018}, "largest_class_baseline": {"layer_1": 0.1173235564, "layer_2": 0.1173235564, "layer_0": 0.1173235564, "layer_3": 0.1173235564, "layer_4": 0.1173235564}},"common_voice": {"random-10": {"layer_0": 0.9722222222222222, "layer_1": 0.8, "layer_3": 0.6055555555555555, "layer_2": 0.45555555555555555, "layer_4": 0.9666666666666667}, "full": {"layer_0": 0.6611111111111111, "layer_1": 0.8055555555555556, "layer_3": 0.5666666666666667, "layer_2": 0.9666666666666667, "layer_4": 0.8}, "imp-score-10": {"layer_0": 0.9555555555555556, "layer_1": 0.6055555555555555, "layer_3": 0.8, "layer_2": 0.42777777777777776, "layer_4": 0.9666666666666667}}}

for dataset, models in sentence_encoding_stats.items():
    for model, layers in models.items():
        print('Average encoding accuracy for {} on {} data: {}'.format(
            model, dataset, (sum(list(layers.values()))/len(layers))
        ))

for dataset, models in sentence_encoding_stats.items():

    for model in ['imp-score-10', 'random-10', 'full', 'largest_class_baseline']:  # Fix plot order for color correspondance
        if model == 'largest_class_baseline' and 'largest_class_baseline' not in models: continue
        results = {name: models[model]['layer_{}'.format(key)] for key, name in mapping.items()}
        plt.plot(*zip(*results.items()), label=model)

    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('./figures/sentence-length-encoding-{}.pdf'.format(dataset))
    plt.clf()


# Relative gender  encoding Common Voice

for dataset, results in sentence_encoding_stats.items():

    results_clean = {name: 
        abs(sentence_encoding_stats[dataset]['full']['layer_{}'.format(key)] 
        - results['imp-score-10']['layer_{}'.format(key)])/((2048-pruned_per_layer_score_based[name])/2048) 
                                                                    for key, name in mapping.items()}
    plt.plot(*zip(*results_clean.items()), label='imp-score-10')

    results_clean = {name: 
        abs(sentence_encoding_stats[dataset]['full']['layer_{}'.format(key)] 
        - results['random-10']['layer_{}'.format(key)])/((2048-pruned_per_layer_random[name])/2048) 
                                                                    for key, name in mapping.items()}
    plt.plot(*zip(*results_clean.items()), label='random-10')


    plt.ylabel('Relative encoding effect (REE)')
    plt.legend()
    plt.savefig('./figures/relative-gender-encoding-{}.pdf'.format(dataset))
    plt.clf()


for dataset, results in {'libri_speech': encoding_libri_stats, 'common_voice': encoding_common_stats}.items():

    results_clean = {name: 
        abs(sentence_encoding_stats[dataset]['full']['layer_{}'.format(key)] 
        - results['imp-score-10']['layer_{}'.format(key)])/((2048-pruned_per_layer_score_based[name])/2048) 
                                                                    for key, name in mapping.items()}
    plt.plot(*zip(*results_clean.items()), label='imp-score-10')

    results_clean = {name: 
        abs(sentence_encoding_stats[dataset]['full']['layer_{}'.format(key)] 
        - results['random-10']['layer_{}'.format(key)])/((2048-pruned_per_layer_random[name])/2048) 
                                                                    for key, name in mapping.items()}
    plt.plot(*zip(*results_clean.items()), label='random-10')


    plt.ylabel('Relative encoding effect (REE)')
    plt.legend()
    plt.savefig('./figures/relative-sentence-length-encoding-{}.pdf'.format(dataset))
    plt.clf()



# Number of speakers in common voice sets
common_voice_sets = json.load(open('./data/pertubed_input_sets_balanced.json'))
sets_train = json.load(open('./results/set_ids_used.json'))

speakers_train = []
speakers_test = []
n_train = 0
n_test = 0
for set in common_voice_sets:
    if str(set['set_id']) in sets_train: 
        for item in set['set_items']:
            if item['client_id'] not in speakers_train: speakers_train.append(item['client_id'])
            n_train +=1
    else: 
        for item in set['set_items']:
            if item['client_id'] not in speakers_test: speakers_test.append(item['client_id'])
            n_test += 1

overlap = [s for s in speakers_test if s in speakers_train]

print('Number of unique speakers in Common Voice balanced dataset: {} of {} samples.'.format(len(speakers_train), n_train))
print('Number of unique speakers in Common Voice balanced dataset: {} of {} samples.'.format(len(speakers_test), n_test))
print('Number of overlapping speakers in Common Voice balanced dataset train and test: {}.'.format(len(speakers_test)))
