#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 20:49:26 2017

@author: gazula
"""

import numpy as np


def Hbeta(D=np.array([]), beta=1.0):
    """Compute the perplexity and the P-row for a specific value of the
    precision of a Gaussian distribution.

    Args:
        D (float): matrix of euclidean distances between every pair of points
        beta (float): Precision of Gaussian distribution
                        Given as -1/(2*(sigma**2))

    Returns:
        H (float): Entropy
        P (float): Similarity matrix (matrix of conditional probabilities)

    """
    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """Performs a binary search to get P-values in such a way that each
    conditional Gaussian has the same perplexity.

    Args:

    Returns:

    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point ", i, " of ", n, "...")

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2
                else:
                    beta[i] = (beta[i] + betamax) / 2
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2
                else:
                    beta[i] = (beta[i] + betamin) / 2

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries = tries + 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: ", np.mean(np.sqrt(1 / beta)))
    return P


def pca(X=np.array([]), no_dims=50):
    """Runs PCA on the NxD array X in order to reduce its dimensionality to
    no_dims dimensions.

    Args:
        X (float): training data of size [examples, features]
        no_dims (int): integer representing the number of dimensions to reduce
                        the data to

    Returns:
        Y (float): reduced training data of size [examples, no_dims]

    """

    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y


def tsne(X=np.array([]),
         Y=np.array([]),
         Shared_length=0,
         no_dims=2,
         initial_dims=50,
         perplexity=30.0,
         computation_phase='remote'):
    """Runs t-SNE on the dataset in the NxD array X to reduce its
    dimensionality to no_dims dimensions. The syntaxis of the function is
    Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array."""

    def updateS(Y, G):
        return Y

    def updateL(Y, G):
        return Y + G

    def demeanS(Y):
        return Y

    def demeanL(Y):
        return Y - np.tile(np.mean(Y, 0), (Y.shape[0], 1))

    # Check inputs
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = pca(X, initial_dims).real
    (n, d) = X.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 7  # early exaggeration
    P = np.maximum(P, 1e-12)

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = 1 / (1 + np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y))
        num[list(range(n)), list(range(n))] = 0
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = np.sum(
                np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y),
                0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * (
            (dY > 0) == (iY > 0))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)

        if computation_phase is 'remote':
            Y = Y + iY
            Y = Y - np.tile(np.mean(Y, 0), (n, 1))
        else:
            Y[:Shared_length, :] = updateS(Y[:Shared_length, :],
                                           iY[:Shared_length, :])
            Y[Shared_length:, :] = updateL(Y[Shared_length:, :],
                                           iY[Shared_length:, :])
            Y[:Shared_length, :] = demeanS(Y[:Shared_length, :])
            Y[Shared_length:, :] = demeanL(Y[Shared_length:, :])

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration ", (iter + 1), ": error is ", C)

        # Stop lying about P-values
        if iter == 100:
            P = P / 4

    # Return solution
    return Y


def normalize_columns(X=Math.array([])):
	minimum = Math.min(X);
	X = X - minimum;
	maximum = Math.max(X)
	X = X / maximum
	rows, cols = X.shape
	for cols in xrange(cols):
		p = Math.mean((X[:, cols]))
		X[:, cols] = X[:, cols] - p;
	return X;
