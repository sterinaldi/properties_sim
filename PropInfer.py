#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
import lal

from ranking import *


'''
Folder structure:
path-to-data/lalinference_mcmc/id/posterior_samples.hdf5

id is a progressive integer from 1 to n_detections, labelling the folders
'''

catalog_file = 'm-catalog.txt'
path_to_data = '.'

catalog = pd.read_csv(catalog_file, sep = '\t')
catalog = catalog[catalog['z']<0.1]

n_detections = len([f for f in os.listdir(path_to_data+'/lalinference_mcmc') if not f.startswith('.')])

z_bounds = [0.0001,0.1]
omega = lal.CreateCosmologicalParameters(0.7,0.3,0.7,-1.,0.,0.)

for i in range(n_detections):
    id = str(i+1)
    posterior_file = path_to_data+'/lalinference_mcmc/'+id+'/posterior_samples.hdf5'
    M = ranking(omega, z_bounds, catalog, id)
    M.run(file = positions, run_sampling = True, show_output = True)
    
