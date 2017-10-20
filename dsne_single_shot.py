#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 08:59:22 2017

@author: dsaha
"""
import numpy as np
import json
import argparse

from remote import remote_site
from local import local_site

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='''read in coinstac args for remote computation''')
    parser.add_argument('--run', type=json.loads, help='grab coinstac args')

    sharedData = ''' {
        "shared_X": "Shared_Mnist_X.txt",
        "shared_Label": "Shared_Label.txt",
        "no_dims": 2,
        "initial_dims": 50,
        "perplexity" : 20.0
    } '''

    args = parser.parse_args(['--run', sharedData])

    remote_output = remote_site(args.run, computation_phase='remote')
    local_output = local_site(remote_output, computation_phase='local')

    #Receive local site data
    LY = np.loadtxt(local_output["local"])
