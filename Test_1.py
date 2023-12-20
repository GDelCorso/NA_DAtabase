#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 12:24:59 2023

@author: claudiacaudai
"""

import NA_Database as nad

# TEST 1

# Test 1 has the default values ​​of the dataset distributions:
    
# - two figures: circles and triangles with equal probability of being generated
# - one color: red
# - Constant distribution with mean 50 for the x and y coordinates of the centers (figures always in the middle)
# - constant distribution with mean 10 for the ray length (10% of the image)
# - zero constant distributions for rotations, deformations, blur, white noise, holes, additive noise regression and multiplicative noise regression

# The command to generate the dictionary with the default distributions is:

    
d=nad.dataset_sampler()

# ----------------------------------------- #

# In order to save dataset_properties.csv, shapes_colors_probabilities.csv and dataset_correlation.csv', we can use:
    
d.save_data()

