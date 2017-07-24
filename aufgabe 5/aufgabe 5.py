'''

http://andrew.gibiansky.com/blog/machine-learning/speech-recognition-neural-networks/

https://gab41.lab41.org/speech-recognition-you-down-with-ctc-8d3b558943f0

https://www.cs.toronto.edu/~graves/preprint.pdf

'''

import numpy as np
import matplotlib.pyplot as plt

def int_ctc(probs, seqq, name):

    seq = np.array([ord(s) - 96 for s in seqq])
    # L = 2 * len(seq) + 1    # length of sequence [9] (U')
    L = len(seq)
    T = probs.shape[1]  # timesteps [1, ..., 12]
    blank = 0

    #    alpha = np.zeros((L,T))
    alpha = np.zeros((T, L))
    #    beta = np.zeros((L,T))
    beta = np.zeros((T, L))

    alpha[0, 0] = 1.0
    alpha[1, 1] = probs[blank, 0]
    alpha[1, 2] = probs[0, 0]

    for t in range(1, T):

        start = max(1, L - ((2 * (T - t)) - 1))

        # for u in range(start, L):
        for u in range(0, L):

            if u == 0:
                alpha[t, u] = alpha[t - 1, u] * probs.T[t, seq[u]]
            else:
                if seq[u] == 0 or seq[u - 2] == seq[u]:
                    alpha[t, u] += probs.T[t, seq[u]] * (alpha[t - 1, u - 1] + alpha[t - 1, u])
                else:
                    alpha[t, u] += probs.T[t, seq[u]] * (
                        alpha[t - 1, u - 2] + alpha[t - 1, u - 1] + alpha[t - 1, u])

        n = np.sum(alpha[t, :], axis=0)
        if n != 0:
            alpha[t, :] /= n

    print('alpha: \n', alpha, '\n')

    beta[-1, -1] = 1
    beta[-1, -2] = 1

    for t in range(T - 2, 0, -1):

        end = min(2 * t + 1, L)

        for u in range(L - 2, 0, -1):

            if u == L-2:
                beta[t, u] = beta[t + 1, u] * probs.T[t, seq[u]]
            else:
                if seq[u] == 0 or seq[u + 2] == seq[u]:
                    beta[t, u] = probs.T[t + 1, seq[u]] * (beta[t + 1, u + 1] + beta[t + 1, u])
                else:
                    beta[t, u] = probs.T[t + 1, seq[u]] * (beta[t + 1, u + 2] + beta[t + 1, u + 1] + beta[t + 1, u])

        n = np.sum(beta[t, :], axis=0)
        if n != 0:
            beta[t, :] /= n

    print('beta: \n', beta, '\n')

    sumuB = np.zeros(probs.shape)
    # absum = np.zeros(T)
    grad = np.zeros(probs.shape)
    # p = np.empty(T)
    ab = alpha * beta

    axes = plt.gca()
    # axes.set_xlim([0, 11])
    # axes.set_ylim([0,  1])
    plt.ylabel('probability')
    plt.xlabel('timesteps')
    plt.title('$\\alpha * \\beta$ output of {} distribution'.format(name))
    for i in range(ab.shape[1]):
        plt.plot(ab.T[i], label='{}'.format(i))

    plt.legend()
    plt.savefig('ab_{}_distribution'.format(name))
    plt.show()

    # for t in range(T):
    #     absum[t] = np.sum(ab[t, :])

    for u in range(L):
        if u % 2 == 0:
            sumuB[0, :] += ab[:, u]
            for t in range(T):
                if probs[0, t] != 0:
                    ab[t, u] = ab[t, u] / probs[0, t]
                else:
                    ab[t, u] = ab[t, u]
        else:
            sumuB[seq[u], :] += ab[:, u]

            for t in range(T):
                if probs[seq[u], t] != 0:
                    ab[t, u] = ab[t, u] / probs[seq[u], t]
                else:
                    ab[t, u] = ab[t, u]

    absum = np.sum(ab, axis=1)
    p = absum * probs

    for t in range(T):
        for u in range(L):
            if p[seq[u], t] != 0 or probs[seq[u], t] != 0:
                grad[seq[u], t] = (sumuB[seq[u], t] / p[seq[u], t] * probs[seq[u], t])
            else:
                grad[seq[u], t] = sumuB[seq[u], t]

    axes = plt.gca()
    # axes.set_xlim([0, 11])
    # axes.set_ylim([-1,  1])
    plt.ylabel('probability')
    plt.xlabel('timesteps')
    plt.title('CTC output of {} distribution'.format(name))

    plt.plot(grad[0], 'k--', label='$\\varepsilon$')
    plt.plot(grad[1], label='a')
    plt.plot(grad[2], label='b')
    plt.plot(grad[3], label='c')

    plt.legend()
    plt.savefig('ctc_{}_distribution'.format(name))
    plt.show()

    print('ab:\n', ab, '\n\nabsum:\n', absum, '\n\nsumuB:\n', sumuB, '\n\ngrad:\n', grad)

    # for t in range(T):
    #     for u in range(L):
    #         grad[seq[u],t] = -(1 / (p.T[t, seq[u]] * probs.T[t, seq[u]]) * absum[t])

    # print(ab.shape, ab)


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

    seq = '`a`b`b`c`'
    # seq = 'abbc'
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
        int_ctc(p, seq, n)


if __name__ == '__main__':
    main()