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

sys.path.append('./DeepSpeech')
from util.config import Config, initialize_globals
from util.flags import create_flags, FLAGS
from DeepSpeech import create_inference_graph, try_loading, create_overlapping_windows
from util.feeding import audiofile_to_features
import numpy as np


def activations_pertubed_sets(input_dir, output_dir):
    '''Obtains activations for wavs in input_dir and saves them to output_dir'''
    inputs, outputs, layers = create_inference_graph(batch_size=1, n_steps=-1)
    intermediate_layer_names = ['layer_1', 'layer_2', 'layer_3', 'rnn_output', 'layer_4', 'layer_5']
    intermediate_layers = [l for n,l in layers.items() if n in intermediate_layer_names]
    
    pertubed_sets = json.load(open('data/pertubed_input_sets_balanced.json'))

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

        # Default states for LSTM cell
        previous_state_c = np.zeros([1, Config.n_cell_dim])
        previous_state_h = np.zeros([1, Config.n_cell_dim])

        for set in pertubed_sets:
            print('Processing set {}, {} items...'.format(set['set_id'], set['set_length']))

            # Only process files that are not yet available in results directory
            create_dir_if_not_exists('{}/activations/{}'.format(output_dir, set['set_id']))  # Check if directory exists
            files_done = [f[:-4] for f in os.listdir('{}/activations/{}'.format(output_dir, set['set_id'])) if f.endswith('.npy')]

            for item in set['set_items']:
                file_name = item['path'][:-4]
                if file_name in files_done: continue
                print('current file: '.format(file_name))

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
                save_to_path_activations = '{}/activations/{}/{}.npy'.format(output_dir, set['set_id'], file_name[:-4])
                write_numpy_to_file(save_to_path_activations, np.array(intermediate_activations))
                print('Activations for {} are saved to: {}'.format(file_name, save_to_path_activations))

    return True

def main(_):
    input_dir = './data/CommonVoice/pertubed_sets'
    output_dir = './results'

    initialize_globals()
    tfv1.reset_default_graph()
    if activations_pertubed_sets(input_dir=input_dir, output_dir=output_dir): print('Done.')

if __name__ == "__main__":
    create_flags()
    absl.app.run(main)