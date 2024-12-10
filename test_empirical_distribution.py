# -*- coding: utf-8 -*-
"""
INPUT:
- local path to the dataset folder

Example:
python test_empirical_distribution.py NADA_Disentanglement

Utility for visualizing empirical distributions of the provided data.
@author: Giulio Del Corso
"""

#%% Libraries
import argparse
import sys
import matplotlib.pyplot as plt
import os
import shutil
import random
import cv2 
import pandas as pd
import numpy as np
import seaborn
from itertools import product
from scipy.stats import spearmanr


#%% Define an aux function to annotate
def aux_add_corr(x, y, hue=None, ax=None, **kws):
    '''
    Plot the correlation coefficient in the top left hand corner of a plot.
    '''
    
    r, _ = spearmanr(x, y)
    ax = ax or plt.gca()
    ax.annotate(f'ρ = {r:.2f}', xy=(.1, .9), xycoords=ax.transAxes)
    
    
#%% Call the parser to receive the input path
try:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to the dataset folder",
                type=str)
    args = parser.parse_args()
except:
    e = sys.exc_info()[0]
    
#%% Generate a folder to evaluate images    
path_save = os.path.join(os.getcwd(),'test_empirical')

if os.path.isdir(path_save):
    shutil.rmtree(path_save)
    os.mkdir(path_save)
else:    
    os.mkdir(path_save)

path_to_folder = os.path.join(os.getcwd(),sys.argv[1])

path_to_dataframe = os.path.join(path_to_folder, 'combined_dataframe.csv')
df = pd.read_csv(path_to_dataframe)

#%% Set up styles
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.bottom'] = False
plt.rcParams['axes.spines.left'] = False


#%% Extract the possible shapes and colors:
full_list_colors = [str(i) for i in list(set(df['color']))]
full_list_shapes = [str(i) for i in list(set(df['shape']))]


#%% Import field by field and represent the empirical/theoretical graph:
#%% Shapes and colors
fig, ax = plt.subplots(1,2, figsize=(12, 12))

list_shapes = [str(i) for i in list(set(df['shape']))]
shapes_values = np.array([len(df[df['shape']==int(i)]) for i in list_shapes])
total_value = sum(shapes_values)
shapes_values = 100*shapes_values/total_value

ax[0].bar(list_shapes, shapes_values, color = 'gray')
ax[0].set_ylim([0,100])
ax[0].set_title("Shapes")

list_colors = [str(i) for i in list(set(df['color']))]
colors_values = np.array([len(df[df['color']==str(i)]) for i in list_colors])
total_value = sum(colors_values)
colors_values = 100*colors_values/total_value

ax[1].bar(list_colors, colors_values, color = 'gray')
ax[1].set_ylim([0,100])
ax[1].set_title("Colors")


fig_name = os.path.join(path_save, 'test_shapes_colors.png')
plt.savefig(fig_name)
plt.close('all')


#%% Cycle on each subset of shape and color:
for s, c in product(full_list_shapes, full_list_colors):
    tmp_df = df
    
    # Select the shape:
    tmp_df = tmp_df[tmp_df['shape']==int(s)]
    
    # Select the color:
    tmp_df = tmp_df[tmp_df['color']==str(c)]    
    
    
    #%% Positions and properties
    fig, ax = plt.subplots(2,2, figsize=(12, 12))
    # Center x
    ax[0,0].hist(tmp_df['centre_x'], density = True, color = 'gray', edgecolor="black")
    ax[0,0].set_xlim([0,100])
    ax[0,0].set_title("Center X")
    
    # Center y
    ax[0,1].hist(tmp_df['centre_y'], density = True, color = 'gray', edgecolor="black")
    ax[0,1].set_xlim([0,100])
    ax[0,1].set_title("Center Y")
    
    # Radius
    ax[1,0].hist(tmp_df['radius'], density = True, color = 'gray', edgecolor="black")
    ax[1,0].set_xlim([0,50])
    ax[1,0].set_title("Radius")
    
    # Rotation
    ax[1,1].hist(tmp_df['rotation_(degrees)'], density = True, color = 'gray', edgecolor="black")
    ax[1,1].set_xlim([0,360])
    ax[1,1].set_title("Rotation")
    
    fig_name = os.path.join(path_save, 'test_position_radius_rotation'+str(s)+'_'+str(c)+'.png')
    plt.savefig(fig_name)
    plt.close("all")
    
    #%% Uncertainties
    fig, ax = plt.subplots(1,3, figsize=(12, 6))
    # Deformation
    #df['deformation'] = df['deformation']*100
    ax[0].hist(tmp_df['deformation'], density = True, color = 'gray', edgecolor="black")
    ax[0].set_xlim([0,1])
    ax[0].set_title("Deformation")
    
    # Blur
    ax[1].hist(tmp_df['blur'], density = True, color = 'gray', edgecolor="black")
    ax[1].set_xlim([0,1])
    ax[1].set_title("Blur")
    
    # White noise
    ax[2].hist(tmp_df['white_noise'], density = True, color = 'gray', edgecolor="black")
    ax[2].set_xlim([0,1]) 
    ax[2].set_title("White Noise")
    
    
    fig_name = os.path.join(path_save, 'test_uncertainties_'+str(s)+'_'+str(c)+'.png')
    plt.savefig(fig_name)
    plt.close("all")
    
    #%% Correlation graph (For each combination)
    # Center x, center y, radius, rotation
    df_correlation = tmp_df[['centre_x','centre_y', 'radius', 'rotation_(degrees)']]
    df_correlation.columns = ['Center x','Center y', 'Radius', 'Rotation']
    
    
    s_corplot = seaborn.pairplot(df_correlation, plot_kws={'color':'gray', 'alpha':0.05}, corner = True, diag_kind='hist',  diag_kws={'color':'gray'})
    s_corplot.map_lower(aux_add_corr)
    
    #s_corplot = seaborn.pairplot(df_correlation, kind = 'hist', corner = True)
    s_fig = s_corplot.fig
    s_fig.savefig(os.path.join(path_save, 'test_correlation_'+str(s)+'_'+str(c)+'.png')) 