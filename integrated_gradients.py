
from __future__ import absolute_import, division, print_function

import sys
import os
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

def integrated_gradients(inp, target_label_index, predictions_and_gradients, baseline, steps=50):

    if baseline is None:
        baseline = 0*inp
    assert(baseline.shape == inp.shape)

    # Scale input and compute gradients.
    scaled_inputs = [baseline + (float(i)/steps)*(inp-baseline) for i in range(0, steps+1)]

    # result = [predictions_and_gradients(np.expand_dims(input, axis=0), target_label_index) for input in scaled_inputs]
    # predictions = np.array([r[0][0] for r in result])
    # grads = np.squeeze(np.array([r[1][0] for r in result]), axis=1)


    predictions, grads = predictions_and_gradients(scaled_inputs, target_label_index)  # shapes: <steps+1>, <steps+1, inp.shape>
    predictions = np.array(predictions)
    grads = np.squeeze(np.array(grads), axis=0)

    # Use trapezoidal rule to approximate the integral.
    # See Section 4 of the following paper for an accuracy comparison between
    # left, right, and trapezoidal IG approximations:
    # "Computing Linear Restrictions of Neural Networks", Matthew Sotoudeh, Aditya V. Thakur
    # https://arxiv.org/abs/1908.06214

    grads = (grads[:-1] + grads[1:]) / 2.0
    avg_grads = np.average(grads, axis=0)
    integrated_gradients = (inp-baseline)*avg_grads  # shape: <inp.shape>
    return integrated_gradients, predictions, _get_ig_error(integrated_gradients, predictions)

def _get_ig_error(integrated_gradients, predictions):
    sum_attributions = 0
    sum_attributions += np.sum(integrated_gradients)

    delta_prediction = predictions[-1] - predictions[0]

    error_percentage = \
        100 * (delta_prediction - sum_attributions) / delta_prediction

    return error_percentage

def neuron_importance(input_dir, output_dir, riemann_steps):
    '''Computes neuron importance scores for the given input_dir'''
    inputs, outputs, layers = create_inference_graph(batch_size=1, n_steps=-1)
    
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

        importance_scores = {}
        n = riemann_steps  # Number of steps in Riemann sum approximation of integration

        # Default states for LSTM cell
        previous_state_c = np.zeros([1, Config.n_cell_dim])
        previous_state_h = np.zeros([1, Config.n_cell_dim])

        # Only process files that are not yet available in results directory
        create_dir_if_not_exists('{}/imp_scores_per_timestep'.format(output_dir))  # Check if directory exists
        files_done = [f[:-4] for f in os.listdir('{}/imp_scores_per_timestep'.format(output_dir)) if f.endswith('.npy')]
        input_files = [f for f in os.listdir(input_dir) if f.endswith('.wav') and f[:-4] not in files_done]

        print('{} audio files found. Start computing neuron importance...'.format(len(input_files)))
        
        # Calculate neuron scores for each input file
        for input_count, file_name in enumerate(input_files):
            print(file_name)
            # layers = {k:l for k, l in layers.items() if k not in ['rnn_output_state', 'raw_logits']}
            if input_count in range(len(input_files), 10): print('... {} audio files processed...'.format(input_count))
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

            logits = outputs['outputs'].eval(feed_dict=feed_dict, session=session)
            logits = np.squeeze(logits)
            target_per_sequence_unit = np.argmax(logits, axis=1)  # holds the predicted output index per sequence unit
            # ouput_value_per_seq_unit = [logits[i,t] for i, t in enumerate(target_per_sequence_unit)]

            def make_predictions_and_gradients(sess, feeds, layers):
                """Returns a function that can be used to obtain the predictions and gradients
                from the Inception network for a set of inputs. 
                
                The function is meant to be provided as an argument to the integrated_gradients
                method.
                """
                label_index = tf.placeholder(tf.int32, [])
                label_prediction = layers['layer_6'][:, label_index]
                predictions = layers['layer_6']
                grads = tf.gradients(label_prediction, feeds[0])

                run_graph = sess.make_callable([predictions, grads], feed_list=feeds+[label_index])
                def f(inputs, target_label_index):
                    inputs = np.array(inputs)
                    shape = inputs.shape
                    inputs = np.squeeze(inputs, axis=1)
                    len = shape[0]
                    return run_graph(inputs, np.full(len, 1), np.zeros([len, 2048]), np.zeros([len, 2048]), target_label_index)
                return f

            ig_steps = 30
            tfv1.get_variable_scope().reuse_variables()
            inputs, outputs, layers = create_inference_graph(batch_size=ig_steps+1, n_steps=-1)
            baseline_correction = np.mean(features, axis=1)
            error_rate = []
            for i in range(features_len[0]):
                if i/20 in range(30):
                    print('working on {}th example'.format(i))
                input = features[:,i,:,:]
                input = input.reshape(1, *input.shape)  # fit shape to models expectated input shape
                # baseline = input - baseline_correction

                feeds = [inputs['input'], inputs['input_lengths'], inputs['previous_state_c'],inputs['previous_state_h']]
                pred_and_grad_f = make_predictions_and_gradients(session, feeds, layers)

                igs, predictions, error = integrated_gradients(inp=input, target_label_index=target_per_sequence_unit[i],
                                                predictions_and_gradients=pred_and_grad_f, baseline=None, steps=ig_steps)
                print(error)
                error_rate.append(error)

            print('average error: {}', sum(error_rate)/len(error_rate))

            sys.exit(1)


            # To compute conductance we need:
            # - Gradients of the predicted output value with resprect to the intermediate layers
            scores = []
            # prev_target_idx = target_per_sequence_unit[0]
            # n_same_targets = 0

            for i in range(features.shape[1]):

                target_idx = target_per_sequence_unit[i]  # target for current timestep
                # if prev_target_idx == target_idx:
                #     n_same_targets += 1
                #     continue
                
                print('Computing {} (of {})'.format(i, features.shape[1]))
                gradients_per_layer = []
                
                output_tensor_timesteps = output_tensor[i,target_idx]
                for inter_tensor in intermediate_layers:  # get gradients for intermediate layers for current timestep
                    gradients_per_layer.append(tf.gradients(output_tensor_timesteps, inter_tensor)[0])

                input = features[:,i,:,:]  # input for current timestep 
                input = input.reshape(1, *input.shape)  # fit shape to models expectated input shape
                feed_dict = {
                    inputs['input']: input,
                    inputs['input_lengths']: [1],
                    inputs['previous_state_c']: previous_state_c,
                    inputs['previous_state_h']: previous_state_h,
                }

                scores_for_timestep, _ = inter_intgrads(input, riemann_steps, feed_dict, inputs['input'],
                                                    intermediate_layers, gradients_per_layer, session)

                scores.append(scores_for_timestep)

                # n_same_targets = 0
                
            
            scores = np.array(scores).squeeze()
            # Save neuron importance scores to file
            save_to_path_scores = '{}/imp_scores_per_timestep/{}.npy'.format(output_dir, file_name[:-4])
            write_numpy_to_file(save_to_path_scores, scores)
            print('Layer scores for {} are saved to: {}'.format(file_name, save_to_path_scores))
            
            # Save activations of actual input
            # save_to_path_activations = '{}/activations/full_model/{}.npy'.format(output_dir, file_name[:-4])
            # write_numpy_to_file(save_to_path_activations, np.array(input_activations))
            # print('Activations for {} are saved to: {}'.format(file_name, save_to_path_activations))

    return True

def main(_):
    input_dir = './data/test'
    output_dir = './results'
    riemann_steps = 20

    initialize_globals()
    tfv1.reset_default_graph()
    neuron_importance(input_dir=input_dir, output_dir=output_dir, riemann_steps=riemann_steps)

if __name__ == "__main__":
    create_flags()
    absl.app.run(main)