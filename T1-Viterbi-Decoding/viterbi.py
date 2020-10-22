#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Xingchen Song @ 2020-10-12

import numpy as np


class HMM:
    """Hidden Markov Model

    HMM with 3 states and 2 observation categories.

    Attributes:
        ob_category (list, with length 2): observation categories
        total_states (int): number of states, default=3
        pi (array, with shape (3,)): initial state probability
        A (array, with shape (3, 3)): transition probability. A.sum(axis=1) must be all ones.
                                      A[i, j] means transition prob from state i to state j.
                                      A.T[i, j] means transition prob from state j to state i.
        B (array, with shape (3, 2)): emitting probability, B.sum(axis=1) must be all ones.
                                      B[i, k] means emitting prob from state i to observation k.

    """

    def __init__(self):
        self.ob_category = ['THU', 'PKU']  # 0: THU, 1: PKU
        self.total_states = 3
        self.pi = np.array([0.2, 0.4, 0.4])
        self.A = np.array([[0.1, 0.6, 0.3],
                           [0.3, 0.5, 0.2],
                           [0.7, 0.2, 0.1]])
        self.B = np.array([[0.5, 0.5],
                           [0.4, 0.6],
                           [0.7, 0.3]])

    def forward(self, ob):
        """HMM Forward Algorithm.

        Args:
            ob (array, with shape(T,)): (o1, o2, ..., oT), observations

        Returns:
            fwd (array, with shape(T, 3)): fwd[t, s] means full-path forward probability torwards state s at
                                           timestep t given the observation ob[0:t+1].
                                           给定观察ob[0:t+1]情况下t时刻到达状态s的所有可能路径的概率和
            prob: the probability of HMM model generating observations.

        """
        T = ob.shape[0]
        fwd = np.zeros((T, self.total_states))

        # Begin Assignment

        # PUT YOUR CODE HERE.

        # End Assignment

        prob = fwd[-1, :].sum()

        return fwd, prob

    def backward(self, ob):
        """HMM Backward Algorithm.

        Args:
            ob (array, with shape(T,)): (o1, o2, ..., oT), observations

        Returns:
            bwd (array, with shape(T, 3)): bwd[t, s] means full-path backward probability torwards state s at
                                           timestep t given the observation ob[t+1::]
                                           给定观察ob[t+1::]情况下t时刻到达状态s的所有可能路径的概率和
            prob: the probability of HMM model generating observations.

        """
        T = ob.shape[0]
        bwd = np.zeros((T, self.total_states))

        # Begin Assignment

        # PUT YOUR CODE HERE.

        # End Assignment

        prob = (bwd[0, :] * self.B[:, ob[0]] * self.pi).sum()

        return bwd, prob

    def viterbi(self, ob):
        """Viterbi Decoding Algorithm.

        Args:
            ob (array, with shape(T,)): (o1, o2, ..., oT), observations

        Variables:
            delta (array, with shape(T, 3)): delta[t, s] means max probability torwards state s at
                                             timestep t given the observation ob[0:t+1]
                                             给定观察ob[0:t+1]情况下t时刻到达状态s的概率最大的路径的概率
            phi (array, with shape(T, 3)): phi[t, s] means prior state s' for delta[t, s]
                                           给定观察ob[0:t+1]情况下t时刻到达状态s的概率最大的路径的t-1时刻的状态s'

        Returns:
            best_prob: the probability of the best state sequence
            best_path: the best state sequence

        """
        T = ob.shape[0]
        delta = np.zeros((T, self.total_states))
        phi = np.zeros((T, self.total_states), np.int)
        best_prob, best_path = 0.0, np.zeros(T, dtype=np.int)

        # Begin Assignment

        # PUT YOUR CODE HERE.

        # End Assignment

        best_path[T-1] = delta[T-1, :].argmax(0)
        best_prob = delta[T-1, best_path[T-1]]
        for t in reversed(range(T-1)):
            best_path[t] = phi[t+1, best_path[t+1]]

        return best_prob, best_path


if __name__ == "__main__":
    model = HMM()
    observations = np.array([0, 1, 0, 1, 1])  # [THU, PKU, THU, PKU, PKU]
    fwd, p = model.forward(observations)
    print(p, '\n', fwd)
    bwd, p = model.backward(observations)
    print(p, '\n', bwd)
    prob, path = model.viterbi(observations)
    print(prob, '\n', path)
