#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Package to analyze echo.

@author: Giulio Del Corso
"""

#%% Import the package:
    
import NA_DAtabase as NA_D




#%% Initialize the class

#my_sampler = NA_D.random_sampler(dataset_name='prova')
#my_sampler.import_dataset_properties()    
#my_sampler.call_sampler()
#my_sampler.generate_images()
###############################################################################

NA_D.random_sampler(dataset_name='prova').auto_process()

