#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 10:01:27 2025

@author: u0890475
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py

from all_functions import *
"""
FIle paths:
 some files pathsare need to be changed
 also listed all variable names
"""
gc_path = "/home/u1474458/Documents/BLF_Model_Matts/h5_results/T_300_P_100v1.h5" # path to g_c vs phi data (precalculated with model Sl^2/ag = 1)

outdir = "/media/u1474458/Data/phi375_amr3_plt/Analysis/" # where to put results (stored as .h5 dicts)
DAT_folder = "/media/u1474458/Data/phi375_amr3_plt/Analysis/DAT_files/"	# where the .dat files are stored
CSV_folder = "/media/u1474458/Data/phi375_amr3_plt/plt/" # where the .csv files are stored
CURV_folder = "/media/u1474458/Data/phi375_amr3_plt/Analysis/Curvature/DAT_files/" # where the curvature .dat isocontours are stored


SD = "phi375" # what to call Saved Data in h5 data structure
dat_prefix = "CH2_0.8"
U_prefix = "U_wall_phi375"
K_prefix = "CH2_0.8"
phi = 0.375

"""
get files for each type of file
file_list = os.listdir(DAT_folder)

speed_wall_files = [x for x in os.listdir(CSV_folder) if U_prefix in x]
curv_files = [x for x in os.listdir(CURV_folder) if K_prefix in x]
dat_files =  [x for x in os.listdir(DAT_folder) if dat_prefix  in x]
"""
speed_files = file_findall(U_prefix,CSV_folder)
dat_files = file_findall(dat_prefix,DAT_folder)
curv_files = file_findall(K_prefix,CURV_folder)

speed_dict = dict(speed_files)
dat_dict = dict(dat_files)
curv_dict  = dict(curv_files)
match_all = [ [x,speed_dict[x],dat_dict [x],curv_dict[x]] for x in speed_dict.keys() ]
match_both.sort(key=lambda x:x[0])