'''

http://andrew.gibiansky.com/blog/machine-learning/speech-recognition-neural-networks/

https://gab41.lab41.org/speech-recognition-you-down-with-ctc-8d3b558943f0

https://www.cs.toronto.edu/~graves/preprint.pdf

'''

import numpy as np
import matplotlib.pyplot as plt

def logAdd(a,b):
    return np.log(a) + np.log(1 + np.exp(np.log(b) - np.log(a)))
    pass

def logMul(a,b):
    return np.log(a) + np.log(b)
    pass

def ctc(probs, seq, name):

    # Thus, instead of considering a label ℓ, we consider a modified label L,
    # which is just ℓ with blanks inserted between all letters, as well as at the beginning and end.
    L = 2 * len(seq) + 1     # length of sequence [9]
    T = probs.shape[1]       # timesteps [1, ..., 12]
    blank = 0

    a = np.zeros((L, T))
    b = np.zeros((L, T))

    seqProb = ord(seq[0]) - 96

    a[0, 0] = probs[blank, 0]
    a[1, 0] = probs[seqProb, 0]

    c = np.sum(a[:, 0])
    a[:, 0] /= c

    forward = np.log(c)

    for t in range(1, T):
        start = max(0, L - 2 * (T - t) - 1)
        end = min(2 * t + 2, L)
        for s in range(start, L):
            l = (s - 1) / 2

            if(s % 2 == 0):
                if(s == 0):
                    a[s, t] = a[s, t - 1] * probs[blank, t]
                    #a[s, t] = logMul(a[s, t - 1], probs[blank, t])
                else:
                    a[s, t] = (a[s, t - 1] + a[s - 1, t - 1]) * probs[blank, t]
                    #a[s, t] = logMul((logAdd(a[s, t - 1], a[s - 1, t - 1])), probs[blank, t])
            elif(s == 1 or seq[int(l)] == seq[int(l - 1)]):
                a[s, t] = (a[s, t - 1] + a[s - 1, t - 1]) * probs[(ord(seq[int(l)]) - 96), t]
                #a[s, t] = logMul(logAdd(a[s, t - 1], a[s - 1, t - 1]), probs[(ord(seq[int(l)]) - 96), t])
            else:
                a[s, t] = (a[s, t - 1] + a[s - 1, t - 1] + a[s - 2, t - 1]) * probs[ord(seq[int(l)]) - 96, t]
                #a[s, t] = logMul(logAdd(logAdd(a[s, t - 1], a[s - 1, t - 1]), a[s - 2, t - 1]), probs[ord(seq[int(l)]) - 96, t])

        c = np.sum(a[start:end, t])
        a[start:end, t] /= c
        forward += np.log(c)

    seqProb = ord(seq[-1]) - 96

    b[-1, -1] = probs[0, -1]
    b[-2, -1] = probs[seqProb, -1]

    c = np.sum(b[:, -1])
    b[:, -1] /= c

    backward = np.log(c)

    for t in range(T - 2, -1, -1):
        start = max(0, L - 2 * (T - t) - 1)
        end = min(2 * t + 2, L)
        for s in range(end - 1, -1, -1):
            l = (s - 1) / 2

            if(s % 2 == 0):
                if(s == L - 1):
                    b[s, t] = b[s, t + 1] * probs[0, t]
                    #b[s, t] = logMul(b[s, t + 1], probs[0, t])
                else:
                    b[s, t] = (b[s, t + 1] + b[s + 1, t + 1] * probs[0, t])
                    #b[s, t] = logMul(logAdd(b[s, t + 1], b[s + 1, t + 1]), probs[0, t])
            elif(s == L - 2 or seq[int(l)] == seq[int(l + 1)]):
                b[s, t] = (b[s, t + 1] + b[s + 1, t + 1]) * probs[ord(seq[int(l)]) - 96, t]
                #b[s, t] = logMul(logAdd(b[s, t + 1], b[s + 1, t + 1]), probs[ord(seq[int(l)]) - 96, t])
            else:
                b[s, t] = (b[s, t + 1] + b[s + 1, t + 1] + b[s + 2, t + 1]) * probs[ord(seq[int(l)]) - 96, t]
                #b[s, t] = logMul(logAdd(logAdd(b[s, t + 1], b[s + 1, t + 1]), b[s + 2, t + 1]), probs[ord(seq[int(l)]) - 96, t])

        c = np.sum(b[start:end, t])
        b[start:end, t] /= c
        backward += np.log(c)

    grad = np.zeros(probs.shape)
    p = a * b
    #p = logMul(a,b)
    for s in range(L):
        if(s % 2 == 0):
            grad[0, :] += p[s, :]
            p[s, :] /= probs[0, :]
        else:
            grad[ord(seq[int((s - 1) / 2)]) - 96, :] += p[s, :]
            p[s, :] /= probs[ord(seq[int((s - 1) / 2)]) - 96, :]

    #grad = probs - grad / (probs * np.sum(p, axis=0))

    print(forward, backward, '\n', grad)
    axes = plt.gca()
    axes.set_xlim([0, 11])
    #axes.set_ylim([0,  1])
    plt.ylabel('probability')
    plt.xlabel('timesteps')
    plt.title('CTC output of {} distribution'.format(name))
    plt.plot(grad[0].T, 'k--', label='$\\varepsilon$')
    plt.plot(grad[1].T, label='a')
    plt.plot(grad[2].T, label='b')
    plt.plot(grad[3].T, label='c')
    plt.legend()
    plt.savefig('ctc_{}_distribution'.format(name))
    plt.show()

def main():

    '''
    uniform distribution:

    ykt     t=1  t=2  t=3  t=4  t=5  t=6  t=7  t=8  t=9  t=10 t=11 t=12
    k=ε     0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25
    k=“a“   0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25
    k=“b“   0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25
    k=“c“   0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25


    weighted distribution:

    ykt     t=1  t=2  t=3 t=4 t=5 t=6 t=7 t=8  t=9 t=10 t=11 t=12
    k=ε     0.25 0.25 0.0 0.0 0.0 0.5 0.5 0.25 0.0 0.0  0.25 0.25
    k=“a“   0.75 0.75 0.5 0.5 0.0 0.0 0.0 0.0  0.0 0.0  0.0  0.0
    k=“b“   0.0  0.0  0.5 0.5 1.0 0.5 0.5 0.5  0.5 0.5  0.25 0.0
    k=“c“   0.0  0.0  0.0 0.0 0.0 0.0 0.0 0.25 0.5 0.5  0.5  0.75
    '''

    seq = 'abbc'
    '''
    probUniform = np.array([
        [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
    ])
    '''
    probUniform = np.full((4, 12), 0.25)

    probWeighted = np.array([
        [.25, .25, .0, .0, .0, .5, .5, .25, .0, .0, .25, .25],
        [.75, .75, .5, .5, .0, .0, .0,  .0, .0, .0,  .0,  .0],
        [ .0,  .0, .5, .5, 1., .5, .5,  .5, .5, .5, .25,  .0],
        [ .0,  .0, .0, .0, .0, .0, .0, .25, .5, .5,  .5, .75],
    ])

    probs = [probUniform, probWeighted]
    names = ['uniform',     'weighted']

    tpl = zip(probs, names)

    for i in tpl:
        p, n = i[0], i[1]
        ctc(p, seq, n)


if __name__ == '__main__':
    main()