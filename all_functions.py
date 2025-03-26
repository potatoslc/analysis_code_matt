#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 11:28:59 2025

@author: u0890475
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import time


import re

def file_findall_complex(header,directory):
    file_seq = []
    file_match_patten = header+r"_(\d+)\.csv"
    file_list = os.listdir(directory)
    uin_files =[x for x in file_list if header in x]
    total_files = len(uin_files)

    
    for i in range(total_files):
        temp_filename = uin_files[i]
        match1= re.search(file_match_patten,temp_filename)
        if(match1):
            file_seq.append([int(match1.group(1)),temp_filename])
            
        else:
            continue
    
    file_seq.sort(key=lambda x:x[0])
    return file_seq

def file_findall_lite(header,directory):
    #file_seq = []
    file_list = os.listdir(directory)
    uin_files =[x for x in file_list if header in x]
    #total_files = len(uin_files)
    sorted(uin_files)
    return uin_files


def get_info(paradata_filename,isosurface_filename):
    paradata_df = pd.read_csv(paradata_filename)
    isosurface_df = pd.read_csv(isosurface_filename)
    """
    get u info from isosurface file
    x: Points:0
    y: Points:1
    x velocity: x_velocity
    """
    minx_loc = isosurface_df["Points:0"].argmin()
    x_minx ,y_minx, xvel_minx= isosurface_df.loc[minx_loc,["Points:0"]],isosurface_df.loc[minx_loc,["Points:1"]],isosurface_df.loc[minx_loc,["x_velocity"]]




    """
    get speed profile from paraview file
    """
    flame_minx_loc = paradata_df["Points:0"].argmin()
    x_flame_minx,y_flame_minx,z_flame_minx= paradata_df.loc[flame_minx_loc,["Points:0"]],paradata_df.loc[flame_minx_loc,["Points:1"]],paradata_df.loc[flame_minx_loc,["Points:2"]]
    #x_vel_flame_minx=paradata_df.loc[flame_minx_loc,["x_velocity"]]
    x_vel_filter = np.where(paradata_df["x_velocity"]<=0,1,0)
    paradata_df["x_filter"] = x_vel_filter*paradata_df["Points:0"]
    #paradata_df["y_filter"] = x_vel_filter*paradata_df["Points:1"]
    #paradata_df["z_filter"] = x_vel_filter*paradata_df["Points:2"]
    x_fb = min(paradata_df[paradata_df["x_filter"]>0]["Points:0"])
    
    len_sep = max(paradata_df["x_filter"] - x_fb)
    #x_diff = abs(paradata_df["Points:0"] - (1.25*len_sep - x_minx))
    x_comp = (1.25*len_sep - x_minx)
    #t1= time.perf_counter()
    x_diff = [ (x-x_comp ) for x in paradata_df["Points:0"]]
    #t2 = time.perf_counter()
    #print("takes:" +str(t2-t1))
    min_diff_loc = np.array(x_diff).argmin()
    dv = float(paradata_df["x_velocity"][min_diff_loc])
    dz = float(paradata_df["Points:1"][min_diff_loc])
    g = dv/dz


    return [g, x_flame_minx,y_flame_minx,x_minx,y_minx]

#paradata_df[paradata_df["x_filter"]>0]
#







