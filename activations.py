from __future__ import absolute_import, division, print_function

import sys
import os
import json
from datetime import datetime 

LOG_LEVEL_INDEX = sys.argv.index('--log_level') + 1 if '--log_level' in sys.argv else 0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = sys.argv[LOG_LEVEL_INDEX] if 0 < LOG_LEVEL_INDEX < len(sys.argv) else '3'

import absl.app
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tfv1
import pickle as pkl
from utils import write_numpy_to_file, create_dir_if_not_exists
from evaluate_pruned_model import prune_matrices

sys.path.append('./DeepSpeech')
from util.config import Config, initialize_globals
from util.flags import create_flags, FLAGS
from DeepSpeech import create_inference_graph, try_loading, create_overlapping_windows
from util.feeding import audiofile_to_features
import numpy as np


def activations_common_voice_pertubed_sets(input_dir, output_dir, test_only=False, prune_percentage=0, scores_file=None, random=False, verbose=True):
    '''Obtains activations for wavs in input_dir and saves them to output_dir'''
    inputs, outputs, layers = create_inference_graph(batch_size=1, n_steps=-1)
    intermediate_layer_names = ['layer_1', 'layer_2', 'layer_3', 'rnn_output', 'layer_4', 'layer_5']
    intermediate_layers = [l for n,l in layers.items() if n in intermediate_layer_names]
    
    pertubed_sets = json.load(open('data/pertubed_input_sets_balanced.json'))
    skip_sets = []
    if test_only: skip_sets = json.load(open('./results/set_ids_used.json'))
    
    if not prune_percentage: base_path = '{}/activations'.format(output_dir)
    else: base_path = '{}/activations/pruned-{}'.format(output_dir, prune_percentage*100)
    if random: base_path += '-random'
    
    with tfv1.Session(config=Config.session_config) as session:
        # Create a saver using variables from the above newly created graph
        # saver = tfv1.train.Saver()

        # # Restore variables from training checkpoint
        # loaded = False
        # if not loaded and FLAGS.load in ['auto', 'last']:
        #     loaded = try_loading(session, saver, 'checkpoint', 'most recent', load_step=False)
        # if not loaded and FLAGS.load in ['auto', 'best']:
        #     loaded = try_loading(session, saver, 'best_dev_checkpoint', 'best validation', load_step=False)
        # if not loaded:
        #     print('Could not load checkpoint from {}'.format(FLAGS.checkpoint_dir))
        #     sys.exit(1)


        ###### PRUNING PART ######

        if verbose: 
            if not prune_percentage: print('No pruning done.')
        else:
            if verbose: print('-'*80)
            if verbose: print('pruning with {}%...'.format(prune_percentage))
            scores_per_layer = np.load(scores_file)
            layer_masks = prune_matrices(scores_per_layer, prune_percentage=prune_percentage, random=random, verbose=verbose, skip_lstm=False)

            n_layers_to_prune = len(layer_masks)
            i=0
            for index, v in enumerate(tf.trainable_variables()):
                lstm_layer_name = 'cudnn_lstm/rnn/multi_rnn_cell/cell_0/cudnn_compatible_lstm_cell/kernel:0'
                if 'weights' not in v.name and v.name != lstm_layer_name: continue
                if(i >= n_layers_to_prune): break  # if i < total_ops, it is not yet the last layer
                # make mask into the shape of the weights                
                if v.name == lstm_layer_name:
                    if skip_lstm: continue
                    # Shape of LSTM weights: [(2*neurons), (4*neurons)]
                    cell_template = np.ones((2, 4))
                    mask = np.repeat(layer_masks[i], v.shape[0]//2, axis=0)
                    mask = mask.reshape([layer_masks[i].shape[0], v.shape[0]//2])
                    mask = np.swapaxes(mask, 0, 1)
                    mask = np.kron(mask, cell_template)
                else:
                    idx = layer_masks[i] == 1
                    mask = np.repeat(layer_masks[i], v.shape[0], axis=0)
                    mask = mask.reshape([layer_masks[i].shape[0], v.shape[0]])
                    mask = np.swapaxes(mask, 0, 1)

                # apply mask to weights
                session.run(v.assign(tf.multiply(v, mask)))
                i+=1

        ###### END PRUNING PART ######

        # Default states for LSTM cell
        previous_state_c = np.zeros([1, Config.n_cell_dim])
        previous_state_h = np.zeros([1, Config.n_cell_dim])

        sets_to_process = [set for set in pertubed_sets if str(set['set_id']) not in skip_sets]
        print('{} sets found'.format(len(sets_to_process)))
        for set in sets_to_process:
            print('Processing set {}, {} items...'.format(set['set_id'], set['set_length']))

            # Only process files that are not yet available in results directory
            create_dir_if_not_exists('{}/{}'.format(base_path, set['set_id']))  # Check if directory exists
            files_done = [f[:-4] for f in os.listdir('{}/{}'.format(base_path, set['set_id'])) if f.endswith('.npy')]

            for item in set['set_items']:
                file_name = item['path'][:-4]
                print(file_name)
                if file_name in files_done: 
                    print('Skipped.')
                    continue
                print('current file: {}'.format(file_name))

                input_file_path = '{}/{}.wav'.format(input_dir, file_name)

                # Prepare features
                features, features_len = audiofile_to_features(input_file_path)
                features = tf.expand_dims(features, 0)
                features_len = tf.expand_dims(features_len, 0)
                features = create_overlapping_windows(features).eval(session=session)
                features_len = features_len.eval(session=session)

                feed_dict = {
                    inputs['input']: features,
                    inputs['input_lengths']: features_len,
                    inputs['previous_state_c']: previous_state_c,
                    inputs['previous_state_h']: previous_state_h,
                }
                intermediate_activations = session.run(intermediate_layers, feed_dict=feed_dict)

                # Save activations of actual input
                save_to_path_activations = '{}/{}/{}.npy'.format(base_path, set['set_id'], file_name)
                write_numpy_to_file(save_to_path_activations, np.array(intermediate_activations))
                print('Activations for {} are saved to: {}'.format(file_name, save_to_path_activations))

    return True


def activations_libri_speech_test_set(input_dir, output_dir, test_only=False, prune_percentage=0, scores_file=None, random=False, verbose=True):
    '''Obtains activations for wavs in input_dir and saves them to output_dir'''
    
    inputs, outputs, layers = create_inference_graph(batch_size=1, n_steps=-1)
    intermediate_layer_names = ['layer_1', 'layer_2', 'layer_3', 'rnn_output', 'layer_4', 'layer_5']
    intermediate_layers = [l for n,l in layers.items() if n in intermediate_layer_names]
    
    if not prune_percentage: base_path = '{}/activations/libri'.format(output_dir)
    else: base_path = '{}/activations/libri/pruned-{}'.format(output_dir, prune_percentage*100)
    if random: base_path += '-random'
    
    with tfv1.Session(config=Config.session_config) as session:
        # Create a saver using variables from the above newly created graph
        saver = tfv1.train.Saver()

        # Restore variables from training checkpoint
        loaded = False
        if not loaded and FLAGS.load in ['auto', 'last']:
            loaded = try_loading(session, saver, 'checkpoint', 'most recent', load_step=False)
        if not loaded and FLAGS.load in ['auto', 'best']:
            loaded = try_loading(session, saver, 'best_dev_checkpoint', 'best validation', load_step=False)
        if not loaded:
            print('Could not load checkpoint from {}'.format(FLAGS.checkpoint_dir))
            sys.exit(1)


        ###### PRUNING PART ######

        if verbose: 
            if not prune_percentage: print('No pruning done.')
        else:
            if verbose: print('-'*80)
            if verbose: print('pruning with {}%...'.format(prune_percentage))
            scores_per_layer = np.load(scores_file)
            layer_masks = prune_matrices(scores_per_layer, prune_percentage=prune_percentage, random=random, verbose=verbose, skip_lstm=False)

            n_layers_to_prune = len(layer_masks)
            i=0
            for index, v in enumerate(tf.trainable_variables()):
                lstm_layer_name = 'cudnn_lstm/rnn/multi_rnn_cell/cell_0/cudnn_compatible_lstm_cell/kernel:0'
                if 'weights' not in v.name and v.name != lstm_layer_name: continue
                if(i >= n_layers_to_prune): break  # if i < total_ops, it is not yet the last layer
                # make mask into the shape of the weights                
                if v.name == lstm_layer_name:
                    if skip_lstm: continue
                    # Shape of LSTM weights: [(2*neurons), (4*neurons)]
                    cell_template = np.ones((2, 4))
                    mask = np.repeat(layer_masks[i], v.shape[0]//2, axis=0)
                    mask = mask.reshape([layer_masks[i].shape[0], v.shape[0]//2])
                    mask = np.swapaxes(mask, 0, 1)
                    mask = np.kron(mask, cell_template)
                else:
                    idx = layer_masks[i] == 1
                    mask = np.repeat(layer_masks[i], v.shape[0], axis=0)
                    mask = mask.reshape([layer_masks[i].shape[0], v.shape[0]])
                    mask = np.swapaxes(mask, 0, 1)

                # apply mask to weights
                session.run(v.assign(tf.multiply(v, mask)))
                i+=1

        ###### END PRUNING PART ######

        # Default states for LSTM cell
        previous_state_c = np.zeros([1, Config.n_cell_dim])
        previous_state_h = np.zeros([1, Config.n_cell_dim])

        for file_name in [f for f in os.listdir(input_dir) if f.endswith('.wav')]:

            # Only process files that are not yet available in results directory
            create_dir_if_not_exists(base_path)  # Check if directory exists
            files_done = [f[:-4] for f in os.listdir(base_path) if f.endswith('.npy')]

            if file_name[:-4] in files_done: 
                print('Skipped.')
                continue
            print('current file: {}'.format(file_name))

            input_file_path = '{}/{}'.format(input_dir, file_name)

            # Prepare features
            features, features_len = audiofile_to_features(input_file_path)
            features = tf.expand_dims(features, 0)
            features_len = tf.expand_dims(features_len, 0)
            features = create_overlapping_windows(features).eval(session=session)
            features_len = features_len.eval(session=session)

            feed_dict = {
                inputs['input']: features,
                inputs['input_lengths']: features_len,
                inputs['previous_state_c']: previous_state_c,
                inputs['previous_state_h']: previous_state_h,
            }
            intermediate_activations = session.run(intermediate_layers, feed_dict=feed_dict)

            # Save activations of actual input
            save_to_path_activations = '{}/{}.npy'.format(base_path, file_name[:-4])
            write_numpy_to_file(save_to_path_activations, np.array(intermediate_activations))
            print('Activations for {} are saved to: {}'.format(file_name, save_to_path_activations))

    return True

def main(_):
    initialize_globals()
    input_dir = './data/CommonVoice/pertubed_sets'
    
    
    output_dir = './results/randomly-initialized'

    tfv1.reset_default_graph()
    activations_common_voice_pertubed_sets(input_dir=input_dir, output_dir=output_dir)


    # output_dir = './results'

    # # Obtain activations for all sets without pruning of common voice test set
    # tfv1.reset_default_graph()
    # activations_common_voice_pertubed_sets(input_dir=input_dir, output_dir=output_dir)

    # # Obtain activations for non-training sets with pruning of common voice test set
    # tfv1.reset_default_graph()
    # activations_peractivations_common_voice_pertubed_setstubed_sets(
    #     input_dir=input_dir, output_dir=output_dir, test_only=True, 
    #     prune_percentage=.1, scores_file='./results/activations_combined.npy')

    # # Obtain activations for non-training sets with pruning of common voice test set
    # tfv1.reset_default_graph()
    # activations_peractivations_common_voice_pertubed_setstubed_sets(
    #     input_dir=input_dir, output_dir=output_dir, test_only=True, random=True, 
    #     prune_percentage=.1, scores_file='./results/activations_combined.npy')


    # input_dir = './data/LibriSpeech/test-clean-wav'

    # # Obtain activations for all sets without pruning of librispeech validation set
    # tfv1.reset_default_graph()
    # activations_libri_speech_test_set(input_dir=input_dir, output_dir=output_dir)

    # # Obtain activations for non-training sets with pruning of librispeech validation set
    # tfv1.reset_default_graph()
    # activations_libri_speech_test_set(
    #     input_dir=input_dir, output_dir=output_dir, test_only=True, 
    #     prune_percentage=.1, scores_file='./results/activations_combined.npy')

    # # Obtain activations for non-training sets with random pruning of librispeech validation set
    # tfv1.reset_default_graph()
    # activations_libri_speech_test_set(
    #     input_dir=input_dir, output_dir=output_dir, test_only=True, random=True, 
    #     prune_percentage=.1, scores_file='./results/activations_combined.npy')

    


if __name__ == "__main__":
    create_flags()
    absl.app.run(main)