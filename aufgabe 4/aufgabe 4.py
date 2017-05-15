import numpy as np
import theano
import theano.tensor as T

import lasagne

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


def generateSequences(minLength):
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
        if len(inchars) > minLength:
            return inchars, outchars


def get_one_example(minLength):
    inchars, outchars = generateSequences(minLength)
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


def get_n_examples(n, minLength=10):
    examples = []
    for i in range(n):
        examples.append(get_one_example(minLength))
    return examples


def rnn(num_inputs):
    net = {}

    net['input'] = lasagne.layers.InputLayer((None, None, num_inputs))

    net['lstm1'] = lasagne.layers.LSTMLayer(net['input'],)

    pass


def main():

    pass

if __name__ == '__main__':
    x = get_n_examples(1000, 5)

    for e in x:
        i,o = e
        print(sequenceToWord(i), sequenceToWord(o))
        print(in_grammar(sequenceToWord(i)), in_grammar(sequenceToWord(o)))