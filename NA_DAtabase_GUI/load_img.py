# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 07:24:28 2025

@author: Giulio Del Corso
"""

#%% Libraries
import os
import pandas as pd
import math
from PIL import Image
import matplotlib.pyplot as plt

path_folder = os.path.join(os.getcwd(), 'dataset_images')
img = 'ID_4_(5, 5, 255)_499'

#%% Load img aux function
def load_img(img, path_folder, n_row = 20, n_columns = 20):
    
    # Identify the index, color and the shape
    shape = img.split('_')[1]
    color = img.split('_')[2].strip("()").replace(" ","").replace(",","-")
    index = int(img.split('_')[3])
    
    # Sub Folder path
    path_sub_folder = os.path.join(path_folder,
                                   "shape-"+str(shape)+"_color-"+str(color))
    list_combined_img = os.listdir(path_sub_folder)
    
    # Having the index we can define the combined image containing the img
    selected_list = [i for i in list_combined_img if (index>=int(i.split("_")[1].replace('.png',"")))and(index<=int(i.split("_")[2].replace('.png',"")))]
    selected_combined_img = Image.open(os.path.join(path_sub_folder, selected_list[0]))
    
    # Find image size:
    img_height = int(selected_combined_img.size[1]/n_row)
    img_width = int(selected_combined_img.size[0]/n_columns)
    
    # Rescale index
    index = index-int(selected_list[0].split("_")[1].replace('.png',""))
    
    # Selected subsquare
    row = index // n_columns
    col = index % n_columns
    
    x = col * img_width
    y = row * img_height
    selected_img = selected_combined_img.crop((x, y, x + img_width, y + img_height))
 
    
    return selected_img

#%% Test
output = load_img(img, path_folder)

plt.imshow(output)