# -*- coding: utf-8 -*-
"""
data_load test

@author: Giulio Del Corso
"""

#%% Libraries
import os
from aux_data_load import load_img_name, load_img_index
import matplotlib.pyplot as plt
import pandas as pd


#%% Test data loader by image name
path_folder = os.path.join(os.getcwd(), 'output', 'test-2', 'dataset_images')
img = 'ID_3_(5, 5, 255)_21'

output = load_img_name(img, path_folder)
plt.imshow(output)


#%% Test data loader by image index
path_df = os.path.join(os.getcwd(), 'output', 'test-2', 'combined_dataframe.csv')
df = pd.read_csv(path_df)
selected_index = 16


#%% Use load_img_name
selected_image = df.iloc[selected_index]['ID_image']
output = load_img_name(selected_image, path_folder)
plt.imshow(output)


#%% Use load_img_idex
output = load_img_index(selected_index, df , path_folder)
plt.imshow(output)