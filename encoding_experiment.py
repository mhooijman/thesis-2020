import os
import json
import csv
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


random_state = 1203

def prepare_speaker_data(file_path):
    speakers = open(file_path, 'r')
    speakers_data = {
        l.split('|')[0].strip(): l.split('|')[1].strip() 
        for l in speakers.readlines() if not l.startswith(';')
                                            and l.split('|')[2].strip() == 'test-clean'}

    return speakers_data

def prepare_sentence_data(file_path):
    df = pd.read_csv(file_path)
    sentence_data = {
        r['wav_filename'].split('/')[-1][:-4]:r['transcript'] for i, r in df.iterrows()
    }
    
    return sentence_data

def do_gender_encoding_experiment_common_voice(sets, activations_dir):

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

def do_gender_encoding_experiment_libri_speech(speaker_data, activations_dir):
    activations_per_layer = {}
    labels = []

    files = [f for f in os.listdir(activations_dir) if f.endswith('.npy')]
    for file in files:
        path = file[:-4]
        print(path)
        if path == '2961-961-0022': continue
        # item = np.load('{}/{}.npy'.format(activations_dir, path))
        # for i, layer_act in enumerate(item):
        #     # Average activations over timesteps and L2 normalize
        #     mean_activations = np.mean(layer_act, axis=0)
        #     l2_activations = mean_activations / np.sqrt(np.sum(mean_activations**2))

        #     layer_name = 'layer_{}'.format(i)
        #     if layer_name not in activations_per_layer: activations_per_layer[layer_name] = []
        #     activations_per_layer[layer_name].append(l2_activations)

        labels.append(speaker_data[path.split('-')[0]])
        
    print('{} files found'.format(len(activations_per_layer)))

    results = {}

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


def do_sentence_length_encoding_experiment_common_voice(sets, activations_dir):
    data = []
    labels = []
    print('{} sets to process...'.format(len(sets)))
    for set in sets:
        for item in set['set_items']:
            path = item['path'][:-4]
            print(path)
            data.append(np.load('{}/{}/{}.npy'.format(activations_dir, set['set_id'], path)))

            # Clean up sentences from punctuation
            not_allowed = [',', '.', '!', '?', '"', '-', ':', ';']
            sentence_clean = item['sentence']
            for c in not_allowed:
                sentence_clean = sentence_clean.replace(c, '')

            # Use length of blank splitted as label (as string, classification not regression)
            labels.append(str(len(sentence_clean.split(' '))))
        
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

        classifier = LogisticRegressionCV(Cs=5, max_iter=2000, random_state=random_state).fit(X_train, y_train)
        test_accuracy = classifier.score(X_test, y_test)
        print('Accuracy for layer {}: {}'.format(name, test_accuracy))

        results[name] = test_accuracy

    return results


def do_sentence_encoding_experiment_libri_speech(activations_dir, sentence_data):
    activations_per_layer = {}
    labels = []

    files = [f for f in os.listdir(activations_dir) if f.endswith('.npy')]
    for file in files:
        path = file[:-4]
        print(path)
        if path == '2961-961-0022': continue
        # item = np.load('{}/{}.npy'.format(activations_dir, path))
        # for i, layer_act in enumerate(item):
        #     # Average activations over timesteps and L2 normalize
        #     mean_activations = np.mean(layer_act, axis=0)
        #     l2_activations = mean_activations / np.sqrt(np.sum(mean_activations**2))

        #     layer_name = 'layer_{}'.format(i)
        #     if layer_name not in activations_per_layer: activations_per_layer[layer_name] = []
        #     activations_per_layer[layer_name].append(l2_activations)

        # Use length of blank splitted as label (as string, classification not regression)
        labels.append(str(len(sentence_data[path].split(' '))))

    counter = {}
    for label in set(labels):
        counter[label] = labels.count(label)

    print(sorted(counter.items(), key=lambda kv: kv[1]))

    import sys
    sys.exit(1)
    print('{} files found'.format(len(activations_per_layer)))

    results = {}

    for name, activations in activations_per_layer.items():
        print('Training Logistic Regression classifier for {} activations'.format(name))
        X_train, X_test, y_train, y_test = train_test_split(activations, labels, test_size=0.25, random_state=random_state)

        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        classifier = LogisticRegressionCV(Cs=5, max_iter=2000, random_state=random_state).fit(X_train, y_train)
        test_accuracy = classifier.score(X_test, y_test)
        print('Accuracy for layer {}: {}'.format(name, test_accuracy))

        results[name] = test_accuracy

    return results

def main():
    speaker_data_librispeech = prepare_speaker_data('./data/LibriSpeech/SPEAKERS.TXT')
    sentence_data_librispeech = prepare_sentence_data('./data/librivox-test-clean.csv')
    pertubed_sets = json.load(open('data/pertubed_input_sets_balanced.json'))
    train_sets = json.load(open('./results/set_ids_used.json'))
    sets_to_use = [set for set in pertubed_sets if str(set['set_id']) not in train_sets]

    # ### Gender encoding experiment ###

    # # Encoding experiment of gender on full model activations of common voice
    # activations_dir = './results/activations'
    # results_full_model_common = do_gender_encoding_experiment_common_voice(sets=sets_to_use, 
    #                 activations_dir=activations_dir)

    # # Encoding experiment of gender on 0.1 pruned model activations of common voice
    # activations_dir = './results/activations/pruned-10.0'
    # results_pruned_model_common = do_gender_encoding_experiment_common_voice(sets=sets_to_use, 
    #                 activations_dir=activations_dir)

    # # Encoding experiment of gender on 0.1 pruned model activations of common voice
    # activations_dir = './results/activations/pruned-10.0-random'
    # results_random_pruned_model_common = do_gender_encoding_experiment_common_voice(sets=sets_to_use, 
    #                 activations_dir=activations_dir)

    # # Encoding experiment of gender on full model activations of librispeech
    # activations_dir = './results/activations/libri'
    # results_full_model_libri = do_gender_encoding_experiment_libri_speech( 
    #                 activations_dir=activations_dir, speaker_data=speaker_data_librispeech)

    # # Encoding experiment of gender on 0.1 pruned model activations of librispeech
    # activations_dir = './results/activations/libri/pruned-10.0'
    # results_pruned_model_libri = do_gender_encoding_experiment_libri_speech(
    #                 activations_dir=activations_dir, speaker_data=speaker_data_librispeech)

    # # Encoding experiment of gender on 0.1 pruned model activations of librispeech
    # activations_dir = './results/activations/libri/pruned-10.0-random'
    # results_random_pruned_model_libri = do_gender_encoding_experiment_libri_speech(
    #                 activations_dir=activations_dir, speaker_data=speaker_data_librispeech)

    # total_results = {
    #     'common_voice': {
    #         'full': results_full_model_common, 
    #         'imp-score-10': results_pruned_model_common,
    #         'random-10': results_random_pruned_model_common
    #     },
    #     'libri_speech': {
    #         'full': results_full_model_libri,
    #         'imp-score-10': results_pruned_model_libri,
    #         'random-10': results_random_pruned_model_libri
    #     }
    # }
    
    # json.dump(total_results, open('./results/gender_encoding_experiment_results.json', 'w+'))



    ### Sentence encoding experiment ###

    # # Encoding experiment of gender on full model activations of common voice
    # activations_dir = './results/activations'
    # results_full_model_common = do_sentence_length_encoding_experiment_common_voice(sets=sets_to_use, 
    #                 activations_dir=activations_dir)

    # # Encoding experiment of gender on 0.1 pruned model activations of common voice
    # activations_dir = './results/activations/pruned-10.0'
    # results_pruned_model_common = do_sentence_length_encoding_experiment_common_voice(sets=sets_to_use, 
    #                 activations_dir=activations_dir)

    # # Encoding experiment of gender on 0.1 pruned model activations of common voice
    # activations_dir = './results/activations/pruned-10.0-random'
    # results_random_pruned_model_common = do_sentence_length_encoding_experiment_common_voice(sets=sets_to_use, 
    #                 activations_dir=activations_dir)

    # Encoding experiment of gender on full model activations of librispeech
    activations_dir = './results/activations/libri'
    results_full_model_libri = do_sentence_encoding_experiment_libri_speech( 
                    activations_dir=activations_dir, sentence_data=sentence_data_librispeech)

    # Encoding experiment of gender on 0.1 pruned model activations of librispeech
    activations_dir = './results/activations/libri/pruned-10.0'
    results_pruned_model_libri = do_sentence_encoding_experiment_libri_speech( 
                    activations_dir=activations_dir, sentence_data=sentence_data_librispeech)

    # Encoding experiment of gender on 0.1 pruned model activations of librispeech
    activations_dir = './results/activations/libri/pruned-10.0-random'
    results_random_pruned_model_libri = do_sentence_encoding_experiment_libri_speech( 
                    activations_dir=activations_dir, sentence_data=sentence_data_librispeech)

    total_results = {
        # 'common_voice': {
        #     'full': results_full_model_common, 
        #     'imp-score-10': results_pruned_model_common,
        #     'random-10': results_random_pruned_model_common
        # }
        # ,
        'libri_speech': {
            'full': results_full_model_libri,
            'imp-score-10': results_pruned_model_libri,
            'random-10': results_random_pruned_model_libri
        }
    }
    
    json.dump(total_results, open('./results/sentence_length_encoding_experiment_results.json', 'w+'))


if __name__ == "__main__":
    main()

