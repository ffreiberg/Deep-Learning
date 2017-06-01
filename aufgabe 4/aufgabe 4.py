import logging
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne
from numpy.distutils.system_info import lapack_src_info

chars = 'BTSXPVE'

graph = [[(1, 5), ('T', 'P')], [(1, 2), ('S', 'X')],
         [(3, 5), ('S', 'X')], [(6,), ('E')],
         [(3, 2), ('V', 'P')], [(4, 5), ('V', 'T')]]


def in_grammar(word):
    if word[0] != 'B':
        return False
    node = 0
    for c in word[1:]:
        transitions = graph[node]
        try:
            node = transitions[0][transitions[1].index(c)]
        except ValueError:  # using exceptions for flow control in python is common
            return False
    return True


def sequenceToWord(sequence):
    """
    converts a sequence (one-hot) in a reber string
    """
    reberString = ''
    for s in sequence:
        index = np.where(s == 1.)[0][0]
        reberString += chars[index]
    return reberString


def generateSequences(len_seq=10):
    while True:
        inchars = ['B']
        node = 0
        outchars = []
        while node != 6:
            transitions = graph[node]
            i = np.random.randint(0, len(transitions[0]))
            inchars.append(transitions[1][i])
            outchars.append(transitions[1])
            node = transitions[0][i]
        if len(inchars) == len_seq+1:
            return inchars, outchars


def get_one_example(len_seq):
    inchars, outchars = generateSequences(len_seq)
    inseq = []
    outseq = []
    for i, o in zip(inchars, outchars):
        inpt = np.zeros(7)
        inpt[chars.find(i)] = 1.
        outpt = np.zeros(7)
        for oo in o:
            outpt[chars.find(oo)] = 1.
        inseq.append(inpt)
        outseq.append(outpt)
    return inseq, outseq


def get_char_one_hot(char):
    char_oh = np.zeros(7)
    for c in char:
        char_oh[chars.find(c)] = 1.
    return [char_oh]


def get_n_examples(n, len_seq=10):
    examples = []
    for i in range(n):
        examples.append(get_one_example(len_seq))
    return examples


def rnn(vocab_len, len_seq, inputs=None):

    net = {}

    net['input'] = lasagne.layers.InputLayer((None, len_seq, vocab_len), input_var=inputs)

    net['lstm1'] = lasagne.layers.LSTMLayer(net['input'], num_units=vocab_len, nonlinearity=lasagne.nonlinearities.tanh)

    net['out'] = lasagne.layers.NINLayer(net['lstm1'], num_units=len_seq, W=lasagne.init.GlorotUniform())

    return net


def minibatches(inputs, targets, mbs, shuffle):

    if(shuffle == True):
        idx = np.arange(len(inputs))
        np.random.shuffle(idx)
    for i in range(0, len(inputs) - mbs + 1, mbs):
        if(shuffle == True):
            batchIdx = idx[i:i+mbs]
        else:
            batchIdx = slice(i, i + mbs)
        yield inputs[batchIdx], targets[batchIdx]


def calc_acc(pred, targets, mbs, len_seq, vocab_len):
    rnd = np.round(pred)
    sum_eq = np.sum(targets == rnd, axis=2)
    sum_cols = np.sum(sum_eq, axis=1)
    valid = np.count_nonzero(sum_cols == len_seq * vocab_len)
    acc = valid / mbs
    return acc

def main():

    num_samples = 100000
    len_seq = 10
    vocab_len = 7
    epochs = 100
    _mbs = [16, 32, 64, 128, 256, 512]
    _gd = [lasagne.updates.nesterov_momentum, lasagne.updates.rmsprop, lasagne.updates.adadelta]
    _eta = [.5, .1, .01]
    eps = .95
    rho = 1e-6
    mom = .75

    for gd in _gd:
        for mbs in _mbs:
            for eta in _eta:

                if (gd == lasagne.updates.adadelta or gd == lasagne.updates.rmsprop):
                    log = logging.FileHandler('rnn_mbs={}_gd={}_epochs={}_eta={}_eps={}_rho={}.log'.format(
                        mbs, gd.__name__, epochs, eta, eps, rho))
                elif (gd == lasagne.updates.nesterov_momentum):
                    log = logging.FileHandler('rnn_mbs={}_gd={}_epochs={}_eta={}_mom={}.log'.format(
                        mbs, gd.__name__, epochs, eta, mom))
                logger.addHandler(log)

                logger.info('Generating {} sequences of length {}...'.format(num_samples, len_seq))
                data = np.array(get_n_examples(num_samples, len_seq))
                # data[sample, 0: data 1: label, seq, letter]
                X_train, y_train, X_val, y_val, X_test, y_test = data[:int(num_samples * 0.8), 0, :, :], data[:int(num_samples * 0.8), 1, :, :], data[int(num_samples * 0.8):int(num_samples * 0.9), 0, :, :], \
                                                                 data[int(num_samples * 0.8):int(num_samples * 0.9), 1, :, :], data[int(num_samples * 0.9):, 0, :, :], data[int(num_samples * 0.9):, 1, :, :]
                # print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)
                inputs = T.tensor3('input')
                targets = T.tensor3('output')

                logger.info('Creating RNN...')
                net = rnn(vocab_len, len_seq, inputs)

                logger.info('Compiling Theano...')
                prediction = lasagne.layers.get_output(net['out'])
                loss = lasagne.objectives.squared_error(targets, prediction)
                loss = loss.mean()

                trainAcc = T.mean(T.eq(T.argmax(prediction, axis=1), T.argmax(T.round(targets), axis=1)), dtype=theano.config.floatX)

                params = lasagne.layers.get_all_params(net['out'], trainable=True)
                #updates = lasagne.updates.adam(loss, params, learning_rate=.001)

                #updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=.01)

                if (gd == lasagne.updates.adadelta or gd == lasagne.updates.rmsprop):
                    #        print("Using {} for updates with learning rate: {}, epsilon: {}, rho: {}".format(gd.__name__, eta, eps, rho))
                    logger.info(
                        "Using {} for updates with learning rate: {}, epsilon: {}, rho: {}".format(gd.__name__, eta,
                                                                                                   eps, rho))
                    updates = gd(loss, params, learning_rate=eta, rho=rho, epsilon=eps)
                elif (gd == lasagne.updates.nesterov_momentum):
                    #        print("Using {} for updates with learning rate: {}, momentum: {}".format(gd.__name__, eta, mom))
                    logger.info(
                        "Using {} for updates with learning rate: {}, momentum: {}".format(gd.__name__, eta, mom))
                    updates = gd(loss, params, learning_rate=eta, momentum=mom)

                testPrediction = lasagne.layers.get_output(net['out'], deterministic=True)
                testLoss = lasagne.objectives.squared_error(targets, testPrediction)
                testLoss = testLoss.mean()
                testAcc = T.mean(T.eq(testPrediction, T.round(targets)), dtype=theano.config.floatX)

                fit = theano.function([inputs, targets], [loss, prediction], updates=updates, allow_input_downcast=True)
                #fit = theano.function([inputs, targets], loss, updates=updates, allow_input_downcast=True)
                test = theano.function([inputs, targets], [testLoss, testPrediction], allow_input_downcast=True)
                #test = theano.function([inputs, targets], testLoss, allow_input_downcast=True)

                logger.info('Starting training...')
                for e in range(epochs):
                    trainErr, trainBatches, trainAcc = 0, 0, 0
                    startTime = time.time()

                    for b in minibatches(X_train, y_train, mbs, True):
                        batchInputs, batchTargets = b
                        #print(batchInputs.shape, batchTargets.shape)
                        err, pred = fit(batchInputs, batchTargets)
                        acc = calc_acc(pred, batchTargets, mbs, len_seq, vocab_len)
                        trainErr += err
                        trainAcc += acc
                        trainBatches += 1

                    val_err, val_batches, val_acc = 0, 0, 0

                    for b in minibatches(X_val, y_val, mbs, shuffle=False):
                        batchInputs, batchTargets = b
                        err, pred = test(batchInputs, batchTargets)
                        # val accuracy
                        acc = calc_acc(pred, batchTargets, mbs, len_seq, vocab_len)
                        val_err += err
                        val_acc += acc
                        val_batches += 1

                    logger.info("Epoch {} of {} took {:.3f}s".format(e + 1, epochs, time.time() - startTime))
                    logger.info("  training loss:\t\t{:.6f}".format(trainErr / trainBatches))
                    logger.info("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
                    logger.info("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))

                testErr, testAcc, testBatches = 0, 0, 0

                for b in minibatches(X_test, y_test, mbs, shuffle=False):
                    batchInputs, batchTargets = b
                    err, pred = test(batchInputs, batchTargets)
                    testErr += err
                    acc = calc_acc(pred, batchTargets, mbs, len_seq, vocab_len)
                    testAcc += acc
                    testBatches += 1

                logger.info("Final results:")
                logger.info("  test loss:\t\t\t{:.6f}".format(testErr / testBatches))
                logger.info("  test accuracy:\t\t{:.2f} %".format(testAcc / testBatches * 100))

if __name__ == '__main__':

    logger = logging.getLogger('rnn')
    logger.setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    main()