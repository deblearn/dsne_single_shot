#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 20:48:24 2017

@author: gazula
"""

import numpy as np
from tsneFunctions import normalize_columns, tsne


def remote_site(args, computation_phase):
    shared_X = np.loadtxt(args["shared_X"])
    #    sharedLabel = np.loadtxt(args.run["shared_Label"])
    no_dims = args["no_dims"]
    initial_dims = args["initial_dims"]
    perplexity = args["perplexity"]

    shared_X = normalize_columns(shared_X)
    (sharedRows, sharedColumns) = shared_X.shape

    init_Y = np.random.randn(sharedRows, no_dims)

    # shared data computation in tsne
    shared_Y = tsne(
        shared_X,
        init_Y,
        sharedRows,
        no_dims,
        initial_dims,
        perplexity,
        computation_phase=computation_phase)

    with open("Y_values.txt", "w") as f:
        for i in range(0, len(shared_Y)):
            f.write(str(shared_Y[i][0]) + '\t')
            f.write(str(shared_Y[i][1]) + '\n')

    args["shared_Y"] = "Y_values.txt"

    return args
