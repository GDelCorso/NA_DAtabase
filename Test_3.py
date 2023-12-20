#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 12:24:59 2023

@author: claudiacaudai
"""


import NA_Database as nad


# TEST 2

# At the beginning the the dataset distributions have default values:
    
# - two figures: circles and triangles with equal probability of being generated
# - one color: red
# - Constant distribution with mean 50 for the x and y coordinates of the centers (figures always in the middle)
# - constant distribution with mean 10 for the ray length (10% of the image)
# - zero constant distributions for rotations, deformations, blur, white noise, holes, additive noise regression and multiplicative noise regression

# The command to generate the dictionary with the default distributions is:

d=nad.dataset_sampler()

# ----------------------------------------- #

# In order to perform a distributional non-zero transformation over the edge curvature, we can use the following command:
    
print(d.dictionary['deformation'])

d.alter('deformation',[0,80,40,10])

print(d.dictionary['deformation'])

# ----------------------------------------- #

# In order to perform a distributional non-zero transformation over the edge curvature, we can use the following command:
 
# lower bound, upper bound, average and sigma of distribution of blur (in strength, from 0=no noise to 100=max noise)
d.alter('blur',[0,60,30,5])

# lower bound, upper bound, average and sigma of distribution of white noise (in strength, from 0=no noise to 100=max noise)
d.alter('white_noise',[30,50,None,None])

# lower bound, upper bound, average and sigma of distribution of holes (in strength, from 0=no noise to 100=max noise)
d.alter('holes',[0,30,40,10])

# lower bound, upper bound, average and sigma of distribution of amount of additive random error (in strength, from 0=no noise to 100=max noise)
d.alter('additive_noise_regression',[None,None,40,10])

# lower bound, upper bound, average and sigma of distribution of amount of multiplicative random error  (in strength, from 0=no noise to 100=max noise).
d.alter('multiplicative_noise_regression',[None,None,30,5])


# ----------------------------------------- #

# In order to save dataset_properties.csv, shapes_colors_probabilities.csv and dataset_correlation.csv', we can use:
    
d.save_data()




















# ----------------------------------------- #

# In order to save dataset_properties.csv, shapes_colors_probabilities.csv and dataset_correlation.csv', we can use:
    
d.save_data()
