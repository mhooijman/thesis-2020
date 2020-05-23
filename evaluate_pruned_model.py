from __future__ import absolute_import, division, print_function

import sys
import os

LOG_LEVEL_INDEX = sys.argv.index('--log_level') + 1 if '--log_level' in sys.argv else 0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = sys.argv[LOG_LEVEL_INDEX] if 0 < LOG_LEVEL_INDEX < len(sys.argv) else '3'

import absl.app
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tfv1
import progressbar
from six.moves import zip
import itertools
from multiprocessing import cpu_count

from utils import write_to_file

sys.path.append('./DeepSpeech')
from util.config import Config, initialize_globals
from util.flags import create_flags, FLAGS
from DeepSpeech import try_loading, create_model
from ds_ctcdecoder import ctc_beam_search_decoder_batch, Scorer
from util.evaluate_tools import calculate_report
from util.feeding import create_dataset
from util.logging import log_error, log_progress, create_progressbar
from evaluate import sparse_tensor_value_to_texts, sparse_tuple_to_texts


def prune_matrices(input_array, prune_percentage=0, random=False, verbose=True):
    '''Returns a matrix with ones and zeros for each layers
    Supports input to be a numpy array. The layers of the DeepSpeech
    model have an equal number of neurons.'''

    # Calculste total number of neurons
    n_neurons_total = 1
    for n in input_array.shape:
      n_neurons_total *= n

    n_neurons_prune = int(n_neurons_total * prune_percentage)
    if verbose: print('Neurons to prune: {}'.format(n_neurons_prune))
    
    # rm lstm layer
    input_array = np.delete(input_array, 4, 0)
    n_neurons_total = 1
    for n in input_array.shape:
      n_neurons_total *= n

    scores_1d = input_array.flatten()

    if random:
        if verbose:print('Creating random pruning masks...')
        mask = np.ones_like(scores_1d)
        mask[:n_neurons_prune] = 0
        np.random.shuffle(mask)
    else:
        if verbose:print('Creating score-based pruning masks...')
        scores_1d = np.abs(scores_1d)
        keep_indexes = scores_1d.argsort()[-(n_neurons_total-n_neurons_prune):]
        mask = np.zeros_like(scores_1d)
        mask[keep_indexes] = 1

    try:
        assert n_neurons_total-n_neurons_prune == np.sum(mask)
    except:
        print('WARNING: neurons to keep and true values in scores are not equal.',
                      n_neurons_total-n_neurons_prune, len(mask[mask>0]))

    # Reshape 1d array to its original shape
    scores_muted = np.reshape(mask, input_array.shape)

    for i, layer in enumerate(scores_muted):
        if verbose: print('Neurons to be pruned in layer_{}: {}.'.format(i+1, len(layer)-len(layer[layer>0])))

    return [layer for layer in scores_muted]


def evaluate_with_pruning(test_csvs, prune_percentage, random, scores_file, result_file, verbose=True):
    '''Code originaly comes from the DeepSpeech repository (./DeepSpeech/evaluate.py).
    The code is adapted for evaluation on pruned versions of the DeepSpeech model.
    '''
    tfv1.reset_default_graph()
    if FLAGS.lm_binary_path:
        scorer = Scorer(FLAGS.lm_alpha, FLAGS.lm_beta,
                        FLAGS.lm_binary_path, FLAGS.lm_trie_path,
                        Config.alphabet)
    else:
        scorer = None

    test_csvs = test_csvs.split(',')
    test_sets = [create_dataset([csv], batch_size=FLAGS.test_batch_size, train_phase=False) for csv in test_csvs]
    iterator = tfv1.data.Iterator.from_structure(tfv1.data.get_output_types(test_sets[0]),
                                                 tfv1.data.get_output_shapes(test_sets[0]),
                                                 output_classes=tfv1.data.get_output_classes(test_sets[0]))
    test_init_ops = [iterator.make_initializer(test_set) for test_set in test_sets]

    batch_wav_filename, (batch_x, batch_x_len), batch_y = iterator.get_next()

    # One rate per layer
    no_dropout = [None] * 6
    logits, _ = create_model(batch_x=batch_x,
                             batch_size=FLAGS.test_batch_size,
                             seq_length=batch_x_len,
                             dropout=no_dropout)

    # Transpose to batch major and apply softmax for decoder
    transposed = tf.nn.softmax(tf.transpose(a=logits, perm=[1, 0, 2]))

    loss = tfv1.nn.ctc_loss(labels=batch_y,
                          inputs=logits,
                          sequence_length=batch_x_len)

    tfv1.train.get_or_create_global_step()

    # Get number of accessible CPU cores for this process
    try:
        num_processes = cpu_count()
    except NotImplementedError:
        num_processes = 1

    # Create a saver using variables from the above newly created graph
    saver = tfv1.train.Saver()

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
            layer_masks = prune_matrices(scores_per_layer, prune_percentage=prune_percentage, random=random, verbose=verbose)

            n_layers_to_prune = len(layer_masks)
            i=0
            for index, v in enumerate(tf.trainable_variables()):
                if 'weights' not in v.name and v.name != 'cudnn_lstm/rnn/multi_rnn_cell/cell_0/cudnn_compatible_lstm_cell/kernel:0': continue
                if(i >= n_layers_to_prune): break  # if i < total_ops, it is not yet the last layer
                if v.name == 'cudnn_lstm/rnn/multi_rnn_cell/cell_0/cudnn_compatible_lstm_cell/kernel:0':
                    # i+=1
                    continue
                # make mask into the shape of the weights                
                # if v.name == 'cudnn_lstm/rnn/multi_rnn_cell/cell_0/cudnn_compatible_lstm_cell/kernel:0':
                #     # Shape of LSTM weights: [(2*neurons), (4*neurons)]
                #     cell_template = np.ones((2, 4))
                #     mask = np.repeat(layer_masks[i], v.shape[0]//2, axis=0)
                #     mask = mask.reshape([layer_masks[i].shape[0], v.shape[0]//2])
                #     mask = np.swapaxes(mask, 0, 1)
                #     mask = np.kron(mask, cell_template)
                else:
                    idx = layer_masks[i] == 1
                    mask = np.repeat(layer_masks[i], v.shape[0], axis=0)
                    mask = mask.reshape([layer_masks[i].shape[0], v.shape[0]])
                    mask = np.swapaxes(mask, 0, 1)

                # apply mask to weights
                session.run(v.assign(tf.multiply(v, mask)))
                i+=1

        ###### END PRUNING PART ######

        def run_test(init_op, dataset):
            wav_filenames = []
            losses = []
            predictions = []
            ground_truths = []

            bar = create_progressbar(prefix='Test epoch | ',
                                     widgets=['Steps: ', progressbar.Counter(), ' | ', progressbar.Timer()]).start()
            log_progress('Test epoch...')

            step_count = 0

            # Initialize iterator to the appropriate dataset
            session.run(init_op)

            # First pass, compute losses and transposed logits for decoding
            while True:
                try:
                    batch_wav_filenames, batch_logits, batch_loss, batch_lengths, batch_transcripts = \
                        session.run([batch_wav_filename, transposed, loss, batch_x_len, batch_y])
                except tf.errors.OutOfRangeError:
                    break

                decoded = ctc_beam_search_decoder_batch(batch_logits, batch_lengths, Config.alphabet, FLAGS.beam_width,
                                                        num_processes=num_processes, scorer=scorer,
                                                        cutoff_prob=FLAGS.cutoff_prob, cutoff_top_n=FLAGS.cutoff_top_n)
                predictions.extend(d[0][1] for d in decoded)
                ground_truths.extend(sparse_tensor_value_to_texts(batch_transcripts, Config.alphabet))
                wav_filenames.extend(wav_filename.decode('UTF-8') for wav_filename in batch_wav_filenames)
                losses.extend(batch_loss)

                step_count += 1
                bar.update(step_count)

            bar.finish()

            wer, cer, samples = calculate_report(wav_filenames, ground_truths, predictions, losses)
            mean_loss = np.mean(losses)

            # Take only the first report_count items
            report_samples = itertools.islice(samples, FLAGS.report_count)
            
            if verbose: print('Test on %s - WER: %f, CER: %f, loss: %f' %
                  (dataset, wer, cer, mean_loss))
            if verbose: print('-' * 80)

            pruning_type = 'score-based' if not random else 'random'
            result_string = '''Results for evaluating model with pruning percentage of {}% and {} pruning:
            Test on {} - WER: {}, CER: {}, loss: {}

            '''.format(prune_percentage*100, pruning_type, dataset, wer, cer, mean_loss)
            write_to_file(result_file, result_string, 'a+')
            
            for sample in report_samples:
                if verbose: 
                    print('WER: %f, CER: %f, loss: %f' %
                        (sample.wer, sample.cer, sample.loss))
                    print(' - wav: file://%s' % sample.wav_filename)
                    print(' - src: "%s"' % sample.src)
                    print(' - res: "%s"' % sample.res)
                    print('-' * 80)

            return samples

        samples = []
        for csv, init_op in zip(test_csvs, test_init_ops):
            if verbose: print('Testing model on {}'.format(csv))
            samples.extend(run_test(init_op, dataset=csv))
        return samples


def main(_):
    initialize_globals()

    evaluation_csv = './data/librivox-test-clean.csv'
    results_file = './results/evaluation_output.txt'
    scores_file = './results/final_imp_scores.npy'

    for prune_settings in [(0, False), (.05, False), (.05, True), (.1, False), (.1, True), (.15, False), (.15, True)]:
        evaluate_with_pruning(evaluation_csv, create_model, try_loading,
            prune_settings[0], random=prune_settings[1], scores_file=scores_file, result_file=results_file)

if __name__ == "__main__":
    create_flags()
    absl.app.run(main)
