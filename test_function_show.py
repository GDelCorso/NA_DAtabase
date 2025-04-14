# -*- coding: utf-8 -*-
"""
INPUT:
- local path to the dataset folder
- number of views to check (each containing 12 random samples)

Example:
python test_function_show.py NADA_Disentanglement_t/NADA_Disentanglement_t 6

Utility for visualizing samples from data sets. Each sample is 
with its description to check the quality of the synthetic data.

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
from aux_data_load import load_img_name

#%% Call the parser to receive the input path
try:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to the dataset folder",
                type=str)
    parser.add_argument("views", help="number of views to check",
                type=int)
    args = parser.parse_args()
except:
    e = sys.exc_info()[0]
    
#%% Generate a folder to evaluate images    
path_save = os.path.join(os.getcwd(),'test_views')

if os.path.isdir(path_save):
    shutil.rmtree(path_save)
    os.mkdir(path_save)
else:    
    os.mkdir(path_save)

path_to_folder = os.path.join(os.getcwd(),sys.argv[1])
path_to_images = os.path.join(path_to_folder, 'dataset_images')


path_to_dataframe = os.path.join(path_to_folder, 'combined_dataframe.csv')
df = pd.read_csv(path_to_dataframe)

list_images = df['ID_image']

#%% Repeat for the number of views:
for i in range(int(sys.argv[2])):
    # Sample 12 images:
    path_to_images_selected = random.choices(list_images, k=12)      
    
    # Import the 12 images
    image_list = []
    name_list = []
    row_list = []
    for j in path_to_images_selected:
        tmp_image = load_img_name(j, path_to_images)
        
        image_list.append(tmp_image)

        name_list.append(j)
        row_list.append(df[df['ID_image']==j.split('.')[0]])
        

    # Combine the 12 images in one view
    fig, ax = plt.subplots(4,3, figsize=(12, 8))
    
    combined_index = 0
    for row in range(4):
        for col in range(3):
            row_tmp = row_list[combined_index]

            ax[row,col].imshow(image_list[combined_index])
            ax[row,col].axis('off')
            ax[row,col].title.set_text(name_list[combined_index])
            ax[row,col].text(0.31+col*0.274, 0.86-0.2*row, "Color:"+str(row_tmp['color'].iloc[0]), fontsize=8, color = 'black', va='top', ha='left', transform=plt.gcf().transFigure)
            ax[row,col].text(0.31+col*0.274, 0.86-0.2*row-0.015*1, "Shape:"+str(round(row_tmp['shape'].iloc[0]))+' vertices', fontsize=8, color = 'black', va='top', ha='left', transform=plt.gcf().transFigure)
            ax[row,col].text(0.31+col*0.274, 0.86-0.2*row-0.015*2, "Position x:"+str(round(row_tmp['centre_x'].iloc[0],1))+"%", fontsize=8, color = 'black', va='top', ha='left', transform=plt.gcf().transFigure)
            ax[row,col].text(0.31+col*0.274, 0.86-0.2*row-0.015*3, "Position y:"+str(round(row_tmp['centre_y'].iloc[0],1))+"%", fontsize=8, color = 'black', va='top', ha='left', transform=plt.gcf().transFigure)
            ax[row,col].text(0.31+col*0.274, 0.86-0.2*row-0.015*4, "Radius:"+str(round(row_tmp['radius'].iloc[0],1))+"%", fontsize=8, color = 'black', va='top', ha='left', transform=plt.gcf().transFigure)
            ax[row,col].text(0.31+col*0.274, 0.86-0.2*row-0.015*5, "Rotation:"+str(round(row_tmp['rotation_(degrees)'].iloc[0]))+"°", fontsize=8, color = 'black', va='top', ha='left', transform=plt.gcf().transFigure)
            ax[row,col].text(0.31+col*0.274, 0.86-0.2*row-0.015*6, "Deformation:"+str(round(row_tmp['deformation'].iloc[0],1))+"%", fontsize=8, color = 'black', va='top', ha='left', transform=plt.gcf().transFigure)
            ax[row,col].text(0.31+col*0.274, 0.86-0.2*row-0.015*7, "White Noise:"+str(round(row_tmp['white_noise'].iloc[0],1))+"%", fontsize=8, color = 'black', va='top', ha='left', transform=plt.gcf().transFigure)
            ax[row,col].text(0.31+col*0.274, 0.86-0.2*row-0.015*8, "Blur:"+str(round(row_tmp['blur'].iloc[0],1))+"%", fontsize=8, color = 'black', va='top', ha='left', transform=plt.gcf().transFigure)
            combined_index+=1 
            
    fig_name = os.path.join(path_save, 'test_'+str(i)+'.png')
    plt.savefig(fig_name)