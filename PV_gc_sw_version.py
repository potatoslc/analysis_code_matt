"""
Author: u0890475
potatoslc version of PV_gc.py
Need to rewrtie most of the code


"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
import re
import os

"""
# define the file path over here
"""
#path = "/media/u1474458/Data/phi375_amr3_plt/plt/" previous file path
path ="/media/u0890475/6f7f5b18-6951-4d23-9e1b-146d3d4c2671/Matt_Old/phi375_amr3_plt/plt/"
U_title = "U_wall_phi375" # name of all CSV files before timestep
Flame_title = "CH2_0.8_phi375" # name of all CSV files before timestep


"""
change if needed
For the current case, the names of x,y coord are:
x-coordinate: Points:0
y-coordinate: Points:1
"""


"""
get all files with needed prefix
also filter out file with errors

combine file names with same series number
as:[0, 'U_wall_phi375_0.csv', 'CH2_0.8_phi375_0.csv']
"""
def file_findall(header,directory):
    file_seq = []
    file_match_patten = header+r"_(\d+)\.csv"
    file_list = os.listdir(directory)
    all_files =[x for x in file_list if header in x]
    total_files = len(all_files)

    
    for i in range(total_files):
        temp_filename = all_files[i]
        match1= re.search(file_match_patten,temp_filename)
        if(match1):
            file_seq.append([int(match1.group(1)),temp_filename]) 
        else:
            continue
    
    #file_seq.sort(key=lambda x:x[0])
    return file_seq

speed_files = file_findall(U_title, path)
flame_files = file_findall(Flame_title, path)

speed_dict = dict(speed_files)
flame_dict = dict(flame_files)
match_both = [ [x,speed_dict[x],flame_dict[x]] for x in speed_dict.keys() ]
match_both.sort(key=lambda x:x[0])



