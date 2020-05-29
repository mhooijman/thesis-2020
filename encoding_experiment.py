import os
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

def prepare_sentence_data(file_path):
    with open(file_path, 'r') as f:

        reader = csv.reader(f)
        headers = reader[0]
        file_data = [dict(zip(headers,row)) for row in reader[1:]]

    print(file_data[0])
    # print(len(word_tokenize(file_data)))
    import sys
    sys.exit()

    sentence_data = {
        item.split('/')[-1][:-4]:  # filename withouth path and extension
        len(word_tokenize(item[2]))  # transcribings have no punctuation
    for item in file_data}

    return sentence_data


def do_gender_encoding_experiment(activations_dir, speakers_data):

    activation_files = os.listdir(activations_dir)
    data = [np.load('{}/{}'.format(activations_dir, f), 'r') 
        for f in activation_files if f.endswith('.npy')]

    labels = [speakers_data[i.split('-')[0]] for i in os.listdir(activations_dir)]

    activations_per_layer = {}
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
    classifier = LogisticRegressionCV(Cs=5, random_state=random_state).fit(X_train, y_train)
    print('Accuracy for layer {}: {}'.format(name, classifier.score(X_test, y_test)))


def do_sentence_length_encoding_experiment(activations_dir, sentence_data):

    activation_files = os.listdir(activations_dir)
    data = [np.load('{}/{}'.format(activations_dir, f), 'r') 
        for f in activation_files if f.endswith('.npy')]

    labels = [sentence_data[i[:-4]] for i in os.listdir(activations_dir)]

    activations_per_layer = {}
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

        classifier = LogisticRegressionCV(Cs=5, random_state=random_state).fit(X_train, y_train)
        print('Accuracy for layer {}: {}'.format(name, classifier.score(X_test, y_test)))



def main():
    speakers_data = prepare_speaker_data('./data/LibriSpeech/SPEAKERS.TXT')
    sencente_data = prepare_sentence_data('./data/librivox-test-clean-real.csv')

    for model_name in ['full_model', '10_percent_pruned_model']:
        activations_dir = './results/activations/{}'.format(model_name)

        print('Results of gender encoding experiments on {}'.format(model_name))
        do_gender_encoding_experiment(activations_dir, speakers_data)

        print('Results of sentence length encoding experiments on {}'.format(model_name))
        do_sentence_length_encoding_experiment(activations_dir, speakers_data)
        
        
        break

if __name__ == "__main__":
    main()

