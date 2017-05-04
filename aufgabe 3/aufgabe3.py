import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne

def loadMnist():

    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)

        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)

        data = data.reshape(-1, 1, 28, 28)

        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)

        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)

        labels = np.zeros((len(data), 10))
        labels[np.arange(len(data)), data] = 1
        return labels

    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')


    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_test, y_test

def cnn(inputs=None, activation=lasagne.nonlinearities.rectify, w_init=lasagne.init.GlorotUniform()):

    net = {}
    net['input'] = lasagne.layers.InputLayer(shape=(None, 1, 28, 28), input_var=inputs)

    #first conv and pool layer
    net['conv1']    = lasagne.layers.Conv2DLayer(   net['input'], num_filters=32, filter_size=(5, 5), nonlinearity=activation, W=w_init)
    net['pool1']    = lasagne.layers.Pool2DLayer(   net['conv1'], pool_size=(2,2))
#    net['pool1'] = lasagne.layers.MaxPool2DLayer(net['conv1'], pool_size=(2,2))

    #second conv and pool layer
    net['conv2']    = lasagne.layers.Conv2DLayer(   net['pool1'], num_filters=32, filter_size=(5, 5), nonlinearity=activation)
    net['pool2']    = lasagne.layers.Pool2DLayer(   net['conv2'], pool_size=(2, 2))
#    net['pool2'] = lasagne.layers.MaxPool2DLayer(net['conv2'], pool_size=(2,2))

    net['out']      = lasagne.layers.DenseLayer(    net['pool2'], num_units=10, nonlinearity=lasagne.nonlinearities.sigmoid)
#    net['out']      = lasagne.layers.DenseLayer(    net['pool2'], num_units=10, nonlinearity=lasagne.nonlinearities.softmax)

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


def main(mbs=128, gd=lasagne.updates.rmsprop, epochs=60, eta=.01, eps=.95, rho=1e-6, mom=.9):

    #Loading MNIST, taken from Theano example mnist.py
    print('Loading MNIST dataset...')
    X_train, y_train, X_test, y_test = loadMnist()
    #print('Loading MNIST finished!')

    inputs = T.tensor4('inputs')
    targets = T.matrix('targets')

    #Building up cnn
    print('Creating network...')
    net = cnn(inputs)
    #print('Creating network finished!')

    #prediction of network
    prediction = lasagne.layers.get_output(net['out'])
    loss = lasagne.objectives.squared_error(targets, prediction)
    loss = loss.mean()

    trainAcc = T.mean(T.eq(T.argmax(prediction, axis=1), T.argmax(targets, axis=1)), dtype=theano.config.floatX)

    #updates
    params = lasagne.layers.get_all_params(net['out'], trainable=True)

    if(gd == lasagne.updates.adadelta or gd == lasagne.updates.rmsprop):
        print("Using {} for updates with learning rate: {}, epsilon: {}, rho: {}".format(gd.__name__, eta, eps, rho))
        updates = gd(loss, params, learning_rate=eta, rho=rho, epsilon=eps)
    elif(gd == lasagne.updates.momentum):
        print("Using {} for updates with learning rate: {}, momentum: {}".format(gd.__name__, eta, mom))
        updates = gd(loss, params, learning_rate=eta, momentum=mom)

    #monitoring progress during training
    testPrediction = lasagne.layers.get_output(net['out'], deterministic=True)
    testLoss = lasagne.objectives.squared_error(targets, testPrediction)
    testLoss = testLoss.mean()

    testAcc = T.mean(T.eq(T.argmax(testPrediction, axis=1), T.argmax(targets, axis=1)), dtype=theano.config.floatX)

    fit = theano.function([inputs, targets], [loss, trainAcc], updates=updates, allow_input_downcast=True)
    test = theano.function([inputs, targets], [testLoss, testAcc], allow_input_downcast=True)

    print('Starting training...')
    for e in range(epochs):

        trainErr, trainBatches, trainAcc = 0, 0, 0
        startTime = time.time()

        for b in minibatches(X_train, y_train, mbs, True):
            batchInputs, batchTargets = b
            err, acc = fit(batchInputs, batchTargets)
            trainErr += err
            trainAcc += acc
            trainBatches += 1

        print("Epoch {} of {} took {:.3f}s".format(e + 1, epochs, time.time() - startTime))
        print("  training loss:\t\t{:.6f}".format(trainErr / trainBatches))

    print("Training accuracy:\t\t{:.2f} %".format(trainAcc / trainBatches * 100))
    #run on test set

    testErr, testAcc, testBatches = 0, 0, 0

    for b in minibatches(X_test, y_test, mbs, shuffle=False):
        batchInputs, batchTargets = b
        err, acc = test(batchInputs, batchTargets)
        testErr += err
        testAcc += acc
        testBatches += 1

    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(testErr / testBatches))
    print("  test accuracy:\t\t{:.2f} %".format(testAcc / testBatches * 100))


if __name__ == '__main__':
    main()
