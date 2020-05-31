import os
import json
import csv
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
import numpy as np
from nltk.tokenize import word_tokenize


random_state = 1203

def prepare_speaker_data(file_path):
    speakers = open(file_path, 'r')
    
    speakers_data = {
        l.split('|')[0].strip(): l.split('|')[1].strip() 
        for l in speakers.readlines() if not l.startswith(';')
                                            and l.split('|')[2].strip() == 'test-clean'}
    
    return speakers_data

def do_gender_encoding_experiment(sets, activations_dir, speakers_data):

    data = []
    for set in sets:
        set_path = '{}/{}'.format(activations_dir, set['set_id'])
        activation_files = [f for f in os.listdir(set_path) if f.endswith('.npy')]
        print('{} files to load.'.format(len(activation_files)))
        for f in activation_files:
            print(f)
            data.append(np.load('{}/{}/{}'.format(activations_dir, set['set_id'], f)))

        # data += [np.load('{}/{}/{}'.format(activations_dir, set['set_id'], f)) for f in activation_files if f.endswith('.npy')]
        
    print('{} files found'.format(len(data)))

    labels = [speakers_data[i.split('-')[0]] for i in os.listdir(activations_dir)]

    activations_per_layer = {}
    results = {}
    for item in data:
        for i, layer_act in enumerate(item):
            if i/20 in range(40): print('Working on {}th input...'.format(i))
            # Average activations over timesteps and L2 normalize
            mean_activations = np.mean(layer_act, axis=0)
            l2_activations = mean_activations / np.sqrt(np.sum(mean_activations**2))

            layer_name = 'layer_{}'.format(i)
            if layer_name not in activations_per_layer: activations_per_layer[layer_name] = []
            activations_per_layer[layer_name].append(l2_activations)

    for name, activations in activations_per_layer.items():
        print('Training Logistic Regression classifier for {} activations'.format(name))
        X_train, X_test, y_train, y_test = train_test_split(activations, labels, test_size=0.25, random_state=random_state)
        classifier = LogisticRegressionCV(Cs=5, random_state=random_state).fit(X_train, y_train)
        test_accuracy = classifier.score(X_test, y_test)
        print('Accuracy for layer {}: {}'.format(name, test_accuracy))

        results[name] = test_accuracy

    return results


def main():
    speaker_data = prepare_speaker_data('./data/LibriSpeech/SPEAKERS.TXT')
    pertubed_sets = json.load(open('data/pertubed_input_sets_balanced.json'))
    train_sets = json.load(open('./results/set_ids_used.json'))
    sets_to_use = [set for set in pertubed_sets if set['set_id'] not in train_sets]

    # Encoding experiment of gender on full model activations
    activations_dir = './results/activations'
    results_full_model = do_gender_encoding_experiment(sets=sets_to_use, 
                    activations_dir=activations_dir, speakers_data=speaker_data)

    # Encoding experiment of gender on 0.1 pruned model activations
    activations_dir = './results/activations/pruned-10'
    results_pruned_model = do_gender_encoding_experiment(sets=sets_to_use, 
                    activations_dir=activations_dir, speakers_data=speaker_data)

    # Encoding experiment of gender on 0.1 pruned model activations
    activations_dir = './results/activations/pruned-10-random'
    results_random_pruned_model = do_gender_encoding_experiment(sets=sets_to_use, 
                    activations_dir=activations_dir, speakers_data=speaker_data)

    total_results = {
        'full': results_full_model, 
        'imp-score-10': results_pruned_model,
        'random-10': results_random_pruned_model    
    }

    json.dump(open('./results/encoding_experiment_results.json', 'w+'), total_results)

if __name__ == "__main__":
    main()

