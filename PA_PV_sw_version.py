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
import pandas as pd
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
match_all.sort(key=lambda x:x[0])
ttl_files = len(match_all)

"""
define the parameters that needs
"""
t = []
Xf = []
Yf = []
Zf = []
Uf = []
g = []
L_sz = []
kf = []
Gkf = []

for i in range(ttl_files):
    # get min values for dat_files
    dat_data = pd.read_csv(match_all[i][2])
    dat_x_velocity_min_loc = dat_data["x_velocity"].argmin()
    dat_x_min,dat_y_min,dat_z_min = dat_data.loc[dat_x_velocity_min_loc,"X"], dat_data.loc[dat_x_velocity_min_loc,"Y"], dat_data.loc[dat_x_velocity_min_loc,"Z"]
    # append these min values to the array

    # get min values for speed profile
    speed_data = pd.read_csv(match_all[i][1])
    speed_x_velocity_min_loc = speed_data["x_velocity"].argmin()
    speed_x_min,speed_y_min,speed_z_min =  speed_data.loc[speed_x_velocity_min_loc,"Points:0"],speed_data.loc[speed_x_velocity_min_loc,"Points:1"],speed_data.loc[speed_x_velocity_min_loc,"Points:2"]
    # need these min values to do further operation

    #modify the speed_data profile, set all xyz coord of positive velocity profile to zero
    speed_data["Points:0"]=np.where(speed_data["x_velocity"]<=0,speed_data["Points:0"],0)
    speed_data["Points:1"]=np.where(speed_data["x_velocity"]<=0,speed_data["Points:1"],0)
    speed_data["Points:2"]=np.where(speed_data["x_velocity"]<=0,speed_data["Points:2"],0)
    xback = speed_data["Points:0"].min()

    # i dont understand what he is trying to do.
    l_sepzone = Xf[-1] -xback
    delta = l_sepzone*5
    X_mes = Xf[-1] - delta
    speed_data["x_diff"] = abs(speed_data["Points:0"]- X_mes)
    min_x_diff_loc = speed_data["x_diff"].argmin()
    v_y_ratio = speed_data.loc[min_x_diff_loc,"x_velocity"] / speed_data.loc[min_x_diff_loc,"Points:1"]


    # get curvature info
    curv_data = pd.read_csv(match_all[i][3])
    curv_x_col_min_loc = curv_data["X"].argmin()
    mean_curv_y_h2 = curv_data.loc[curv_x_col_min_loc,"MeanCurvature_Y(H2)"]
    gaus_curv_y_h2 = curv_data.loc[curv_x_col_min_loc,"GaussianCurvature_Y(H2)"]



"""
plot starts here 

"""









