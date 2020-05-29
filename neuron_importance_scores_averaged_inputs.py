
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


def inter_intgrads(input_value, reference_value, grads_and_activation_func, 
                n, feed_dict, input_tensor, inter_tensor, gradients, session):  
    '''Code originaly comes from the following paper: https://arxiv.org/pdf/1807.09946.pdf.
    The code is adapted for computing scores for multiple layers in the same tf session.'''


    # print(type(input_value), input_value.shape)
    # print(type(reference_value), reference_value.shape)
    #(1) Interpolate between reference_value and input_value at n+1 points
    step_size = (input_value - reference_value)/float(n)
    intermediate_values = [reference_value + i*step_size
                           for i in range(n+1)] 

    #(2) Compute the gradient and activation on the
    # neurons at each of the n+1 points
    # gradients_at_intermediate_values, activations_at_intermediate_values = grads_and_activation_func([intermediate_values]) #input should be in list

    gradients_at_intermediate_values = []
    activations_at_intermediate_values = []
    i = 0
    while intermediate_values:
        print('Compute gradients for intermediate values {}'.format(i))
        i += 1
        feed_dict[input_tensor] = intermediate_values.pop()

        # inter_tensor and gradients are lists with respectively input tensors and gradient ops per layer
        # input for session.run() = [inter_tensor_layer_1, i_t_layer_2, ..., i_t_layer_n] + [gradients_layer_1, ... grad_layer_2, grad_layer_n]
        output = session.run(inter_tensor+gradients, feed_dict=feed_dict)  
        activations_at_intermediate_values.append(output[0:len(inter_tensor)])
        gradients_at_intermediate_values.append(output[len(inter_tensor):])

        if i == n:  # above computations are on the actual input
            # capture activations for further experiments
            input_activations = np.array(output[0:len(inter_tensor)])

    # Group activations and gradients per layer
    grouped_per_layer = {}
    for input_i in range(n+1):
        for layer_i in range(len(inter_tensor)):
            layer_name = 'layer_{}'.format(layer_i)
            if layer_name not in grouped_per_layer: grouped_per_layer[layer_name] = {'activations': [], 'gradients': []}
            grouped_per_layer[layer_name]['activations'].append(activations_at_intermediate_values[input_i][layer_i])
            grouped_per_layer[layer_name]['gradients'].append(gradients_at_intermediate_values[input_i][layer_i])

    # Compute neuron importance scores per layer
    layer_scores = []  # Holds a score for each layer
    for layer_results in grouped_per_layer.values():
        
        activations_at_intermediate_values = np.array(layer_results['activations'])
        gradients_at_intermediate_values = np.array(layer_results['gradients'])

        #Formula was:
        # int_[alpha=0 to alpha=1] d[output]/d[neurons]
        #                          * d[neurons]/d[alpha] * d[alpha]

        #(3) Compute the delta in activations
        # (empirical estimate of d[neurons]/d[alpha])
        # delta_activations has length n
        delta_activations = activations_at_intermediate_values[1:] - activations_at_intermediate_values[:-1]

        #(4) Get the best estimate of the gradient corresponding
        # to each delta. This is d[output]/d[neurons]
        # corresp_grad has length n
        # The definition below means integrated grads agrees with
        # grad*input at n=1
        corresp_grad = gradients_at_intermediate_values[1:]

        #(5) Multiply the delta in activation with the gradient, take the mean
        contrib = np.sum(delta_activations*corresp_grad, axis=0)
        
        layer_scores.append(contrib)

    return layer_scores, input_activations

def neuron_importance(input_dir, output_dir, riemann_steps):
    '''Computes neuron importance scores for the given input_dir'''

    with tfv1.Session(config=Config.session_config) as session:
        inputs, outputs, layers = create_inference_graph(batch_size=1, n_steps=-1)
        layers = {k:l for k, l in layers.items() if k not in ['rnn_output_state', 'raw_logits']}

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

        # Only process files that are not yet available in results directory
        create_dir_if_not_exists('{}/imp_scores_avg_inputs'.format(output_dir))  # Check if directory exists
        files_done = [f[:-4] for f in os.listdir('{}/imp_scores_avg_inputs'.format(output_dir)) if f.endswith('.npy')]
        input_files = [f for f in os.listdir(input_dir) if f.endswith('.wav') and f[:-4] not in files_done]

        print('{} audio files found. Start computing neuron importance...'.format(len(input_files)))
        
        # Calculate neuron scores for each input file
        for input_count, file_name in enumerate(input_files):
            print(file_name)
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

            
            # avg_logits = np.average(logits, axis=0)
            avg_features = np.average(features, axis=1).reshape(-1, 1, 19, 26)

            # Layer stuff
            intermediate_layers = [item for key,item in layers.items() if key not in ['input_reshaped', 'layer_6']]        
            input_tensor = layers['input_reshaped']
            output_tensor = layers['layer_6']

            # To compute conductance we need:
            # - Gradients of the predicted output value with resprect to the intermediate layers

            gradients_for_all_layers = []
            # Get gradient graphs for all interm. layers: output with respect interm layer
            for inter_tensor in intermediate_layers:
                gradients_for_all_layers.append(tf.gradients(tf.reduce_sum(output_tensor, axis=0), inter_tensor)[0])
            
            feed_dict = {
                inputs['input']: avg_features,
                inputs['input_lengths']: [1],
                inputs['previous_state_c']: previous_state_c,
                inputs['previous_state_h']: previous_state_h,
            }

            # Computing importance scores for current X
            layer_scores, input_activations = inter_intgrads(
                        input_value=np.array(avg_features),
                        reference_value=np.array(np.zeros_like(avg_features)),
                        grads_and_activation_func=None,
                        n=riemann_steps,
                        feed_dict=feed_dict,
                        input_tensor=inputs['input'],
                        inter_tensor=intermediate_layers,
                        gradients=gradients_for_all_layers,
                        session=session)

            # Save neuron importance scores to file
            save_to_path_scores = '{}/imp_scores_avg_inputs/{}.npy'.format(output_dir, file_name[:-4])
            write_numpy_to_file(save_to_path_scores, layer_scores)
            print('Layer scores for {} are saved to: {}'.format(file_name, save_to_path_scores))

    return True

def main(_):
    input_dir = './data/LibriSpeech/test-clean-wav'
    output_dir = './results'
    riemann_steps = 20

    initialize_globals()
    tfv1.reset_default_graph()
    neuron_importance(input_dir=input_dir, output_dir=output_dir, riemann_steps=riemann_steps)

if __name__ == "__main__":
    create_flags()
    absl.app.run(main)