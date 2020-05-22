from __future__ import absolute_import, division, print_function

import sys
import os

LOG_LEVEL_INDEX = sys.argv.index('--log_level') + 1 if '--log_level' in sys.argv else 0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = sys.argv[LOG_LEVEL_INDEX] if 0 < LOG_LEVEL_INDEX < len(sys.argv) else '3'

import absl.app
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tfv1
from utils import write_numpy_to_file, create_dir_if_not_exists

sys.path.append('./DeepSpeech')
from util.config import Config, initialize_globals
from util.flags import create_flags, FLAGS
from DeepSpeech import create_inference_graph, try_loading, create_overlapping_windows
from util.feeding import audiofile_to_features



def record_multi_file_activations(inputs_directory, output_directory):
    with tfv1.Session(config=Config.session_config) as session:
        inputs, outputs, layers = create_inference_graph(batch_size=1, n_steps=-1)

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

        if inputs_directory[-1] != '/': inputs_directory += '/'
        for file_name in [filename for filename in os.listdir(inputs_directory) if filename.endswith('.wav')]:
            input_file_path = f"{inputs_directory}{file_name}"
            print('Fetching activations for {}...'.format(input_file_path))
            features, features_len = audiofile_to_features(input_file_path)
            previous_state_c = np.zeros([1, Config.n_cell_dim])
            previous_state_h = np.zeros([1, Config.n_cell_dim])

            # Add batch dimension
            features = tf.expand_dims(features, 0)
            features_len = tf.expand_dims(features_len, 0)

            # Evaluate
            features = create_overlapping_windows(features).eval(session=session)
            features_len = features_len.eval(session=session)    

            
            # Code for obtaining activations for each of the hidden layers of the model
            hidden_layers = [l for name, l in layers.items() if name not in ['input_reshaped', 'rnn_output_state', 'layer_6', 'raw_logits']]

            feed_dict = {
                inputs['input']: features,
                inputs['input_lengths']: features_len,
                inputs['previous_state_c']: previous_state_c,
                inputs['previous_state_h']: previous_state_h,
            }
            
            activations = np.array(session.run(hidden_layers, feed_dict=feed_dict))

            write_numpy_to_file('{}/{}'.format(output_directory, file_name.replace('.wav', '.npy')), activations)



def main(_):
    initialize_globals()
    tfv1.reset_default_graph()

    inputs_directory = './data/LibriSpeech/test-clean-wav'
    
    for model_name in ['full_model', '10_percent_pruned_model']:
        output_directory = './results/activations/{}'.format(model_name)
        record_multi_file_activations(inputs_directory, output_directory)

if __name__ == "__main__":
    create_flags()
    absl.app.run(main)