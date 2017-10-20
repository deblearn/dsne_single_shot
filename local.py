#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 20:49:04 2017

@author: gazula
"""
import numpy as np
import argparse
import json
from tsneFunctions import normalize_columns, tsne


def local_site(args, computation_phase):

    shared_X = np.loadtxt(args["shared_X"])
    shared_Y = np.loadtxt(args["shared_Y"])
    no_dims = args["no_dims"]
    initial_dims = args["initial_dims"]
    perplexity = args["perplexity"]
    sharedRows, sharedColumns = shared_X.shape

    # load high dimensional site 1 data
    parser = argparse.ArgumentParser(
        description='''read in coinstac args for local computation''')
    parser.add_argument('--run', type=json.loads, help='grab coinstac args')
    localSite1_Data = ''' {
        "site1_Data": "Site_1_Mnist_X.txt",
        "site1_Label": "Site_1_Label.txt"
    } '''
    site1args = parser.parse_args(['--run', localSite1_Data])
    Site1Data = np.loadtxt(site1args.run["site1_Data"])
    (site1Rows, site1Columns) = Site1Data.shape

    # create combinded list by local and remote data
    combined_X = np.concatenate((shared_X, Site1Data), axis=0)
    combined_X = normalize_columns(combined_X)

    # create low dimensional position
    combined_Y = np.random.randn(combined_X.shape[0], no_dims)
    combined_Y[:shared_Y.shape[0], :] = shared_Y

    Y_plot = tsne(
        combined_X,
        combined_Y,
        sharedRows,
        no_dims=no_dims,
        initial_dims=initial_dims,
        perplexity=perplexity,
        computation_phase=computation_phase)

    # save local site data into file
    with open("local_site1.txt", "w") as f1:
        for i in range(sharedRows, len(Y_plot)):
            f1.write(str(Y_plot[i][0]) + '\t')
            f1.write(str(Y_plot[i][1]) + '\n')

    # pass data to remote in json format
    localJson = ''' {"local": "local_site1.txt"} '''
    localY = parser.parse_args(['--run', localJson])

    return (localY.run)
