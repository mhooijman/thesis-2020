import os
import json
import csv
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
    labels = []
    print('{} sets to process...'.format(len(sets)))
    for set in sets:
        for item in set['set_items']:
            path = item['path'][:-4]
            print(path)
            data.append(np.load('{}/{}/{}.npy'.format(activations_dir, set['set_id'], path)))
            labels.append(item['gender'])
        
    print('{} files found'.format(len(data)))

    activations_per_layer = {}
    results = {}
    for item in data:
        for i, layer_act in enumerate(item):
            # Average activations over timesteps and L2 normalize
            mean_activations = np.mean(layer_act, axis=0)
            l2_activations = mean_activations / np.sqrt(np.sum(mean_activations**2))

            layer_name = 'layer_{}'.format(i)
            if layer_name not in activations_per_layer: activations_per_layer[layer_name] = []
            activations_per_layer[layer_name].append(l2_activations)

    for name, activations in activations_per_layer.items():
        print('Training Logistic Regression classifier for {} activations'.format(name))
        X_train, X_test, y_train, y_test = train_test_split(activations, labels, test_size=0.25, random_state=random_state)

        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        classifier = LogisticRegressionCV(Cs=5, max_iter=500, random_state=random_state).fit(X_train, y_train)
        test_accuracy = classifier.score(X_test, y_test)
        print('Accuracy for layer {}: {}'.format(name, test_accuracy))

        results[name] = test_accuracy

    return results


def main():
    # speaker_data_librispeech = prepare_speaker_data('./data/LibriSpeech/SPEAKERS.TXT')
    pertubed_sets = json.load(open('data/pertubed_input_sets_balanced.json'))
    train_sets = json.load(open('./results/set_ids_used.json'))
    sets_to_use = [set for set in pertubed_sets if str(set['set_id']) not in train_sets]

    # Encoding experiment of gender on full model activations
    activations_dir = './results/activations'
    results_full_model = do_gender_encoding_experiment(sets=sets_to_use, 
                    activations_dir=activations_dir, speakers_data=None)

    # Encoding experiment of gender on 0.1 pruned model activations
    activations_dir = './results/activations/pruned-10.0'
    results_pruned_model = do_gender_encoding_experiment(sets=sets_to_use, 
                    activations_dir=activations_dir, speakers_data=None)

    total_results = {
        'full': results_full_model, 
        'imp-score-10': results_pruned_model
    }
    
    json.dump()
    json.dump(total_results, open('./results/encoding_experiment_results.json', 'w+'))

if __name__ == "__main__":
    main()

