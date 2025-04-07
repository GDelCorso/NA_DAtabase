# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 14:16:47 2025

@author: Giulio Del Corso
"""

import os
import pandas as pd
import math
from PIL import Image

path_imgs = os.path.join(os.getcwd(), 'dataset_images')

#%% Aux function to compress
def compress_images(path_imgs, n_row = 20, n_columns = 20):
    '''
    Aux script to convert the original databaset to an efficient representation
    All the images contained in path_imgs subfolder are compressed in larger
    images (shape = n_row*n_columns). Original images are removed.
    
    path_imgs (str) - path to the images folder
    n_row (int) = 20 - number of rows
    n_columns (int) = 20 - number of columns
    '''
    
    # List of all (sub) folders
    list_folders = os.listdir(path_imgs)
    
    # Iterate on each subfolder
    for folder in list_folders:
        # Define the path to the subfolder
        path_folder = os.path.join(path_imgs, folder)
        
        # List images and define a dataframe containing required information
        list_imgs = os.listdir(path_folder)
        list_img =  [i for i in list_imgs if ('.png' in i)]
        list_path = [os.path.join(path_folder, i) for i in list_imgs]
        list_id = [int(i.split('_')[-1].split('.')[0]) for i in list_imgs]
        
        df_ref = pd.DataFrame()
        df_ref['img'] = list_img
        df_ref['id'] = list_id
        df_ref['path'] = list_path
        df_ref = df_ref.sort_values(by='id')
        
        # Load the images
        number_imgs = n_row*n_columns
        total_imgs = len(df_ref)
        
        # Initialize an empty img
        for i in range(math.ceil(total_imgs/number_imgs)):
            # Combined image name
            img_name = 'combined_'+str(df_ref.iloc[i*number_imgs]['id'])+'_'+\
                str(df_ref.iloc[min((i+1)*number_imgs-1,total_imgs-1)]['id'])+'.png'
            
            # Select the sub-df of the n_row*n_columns images to be combined
            sub_df = df_ref.iloc[i*number_imgs:min((i+1)*number_imgs,total_imgs)]
            
            # Select all the corresponding images
            sub_img_list = [Image.open(f) for f in sub_df['path']]
            
            # Define a canvas (grid n_rows, n_columns)
            img_width, img_height = sub_img_list[0].size
            combined_image = Image.new('RGBA', (n_columns*img_width, 
                                                n_row*img_height), color=(0, 0, 0, 255))
            
            # Paste each image into the grid
            for idx, img in enumerate(sub_img_list):
                row = idx // n_columns
                col = idx % n_columns
                x = col * img_width
                y = row * img_height
                combined_image.paste(img, (x, y))
            
            # Save the final image
            combined_image.save(os.path.join(path_folder, img_name))
            
            # Remove all the remaining images
            [os.remove(f) for f in sub_df['path']]
            
        
        
#%% Call the aux function
compress_images(path_imgs)
