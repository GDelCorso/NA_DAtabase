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

# in order to change the moments of the distributions (lower bound, upper boud, average, standard deviation) of the features we can use this command:

print(d.dictionary['blur'])

d.alter('blur',[None,None,10,None])

print(d.dictionary['blur'])

# ----------------------------------------- #

# The following function:
    
#d.control('blur')

# control just verify if the feature distribution is an admitted distribution. 
# Admitted distributions are:
#    - Uniform: lower bound and upper bound != 0 | lower bound < upper bound | average and std = None
#    - Gaussian: lower bound and upper bound = None | average != None | std > 0
#    - Constant: lower bound, upper bound and std = None | average != None
#    - Truncated Gaussian: lower bound or upper bound or both != None | average != None | std > 0

# ----------------------------------------- #

# In order to change for example colours, from red to red and green, we can use the following command:
    
d.alter('colours',[[1,0,0],[0,1,0]])

# ----------------------------------------- #

# It's necessary to change also the probabilities of the colours (their number have to match the number of colours):
    
d.alter('prob_colours',[0.3,0.7])

# ----------------------------------------- #

# In order to add a new shape, for example square, we can use the following command:
    
d.alter('shapes',[0,3,4])

# ----------------------------------------- #

# It's necessary to change also the probabilities of the shapes (their number have to match the number of colours):
    
d.alter('prob_shapes',[0.2,0.2,0.6])

# ----------------------------------------- #

# in order to normalize unbalanced probabilities we can use this command:
    
d.set_prob_standard('colours')

# ----------------------------------------- #

# to fix a certain probability (the others will be authomatically normalized) we can use this command:

d.set_prob_fix('shapes', [3], [0.1])

# ----------------------------------------- #

# in order to set a specific correlation between two features, we can use this command:
    
d.set_correl('centre_x','centre_y',0.7)

# ----------------------------------------- #

# In order to save dataset_properties.csv, shapes_colors_probabilities.csv and dataset_correlation.csv', we can use:
    
d.save_data()







