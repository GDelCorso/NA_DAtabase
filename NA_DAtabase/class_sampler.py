# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 21:00:10 2023

2) SISTEMARE TODO

4) IMPORTARE MATRICE CORRELAZIONE

5) FARE CONTROLLO STOCASTICO DAL CENTRO - QUI CI SONO I TRUCCHI DI COME IMPLEMENTARE IL CONTROLLO "RAGGIO VS CENTRO"
https://openturns.github.io/openturns/latest/auto_probabilistic_modeling/distributions/plot_truncated_distribution.html#sphx-glr-auto-probabilistic-modeling-distributions-plot-truncated-distribution-py

@author: Claudia Caudai, Giulio Del Corso, and Federico Volpini
"""

###############################################################################






###############################################################################
class random_sampler:
    '''
    random_sampler class. It defines an object (random_sampler) that produces
    the database based on the csv files saved in the input folder.
    '''

    def __init__(self, 
                 dataset_properties = None, 
                 dataset_correlation = None, 
                 shapes_colors = None,
                 sampler_properties = None, 
                 dataset_name = None,
                 main_path = None):
        '''
        dataset_properties = None : csv file of dataset properties
        dataset_correlation = None : csv file of correlation matrices
        shapes_colors = None : csv file of shapes/colors probabilities
        sampler_properties = None : csv file of sampler properties 
        dataset_name = None : name of the dataset
        main_path = None : main path (if None -> main_path = current directory)
        '''
        
        # Libraries:
        import os
        import shutil
        
        # Define the actual path
        if main_path == None:
            self.path_cwd = os.getcwd()
        else:
            self.path_cwd = main_path
            
        # Initialize an empty save folder 
        if dataset_name == None:
            self.dataset_name = 'dataset'
        else:
            self.dataset_name = 'dataset_'+dataset_name
            
        self.path_save_folder = os.path.join(self.path_cwd, self.dataset_name)
        
        if not os.path.isdir(self.path_save_folder):
            os.mkdir(self.path_save_folder)

        # Define the file paths
        # Properties of the dataset:
        self.path_dataset_properties = os.path.join(self.path_save_folder, 
                                                      'dataset_properties.csv')
        # Probabilites of colors and shapes:
        self.path_shapes_colors = os.path.join(self.path_save_folder, 
                                             'shapes_colors_probabilities.csv')
        # Correlation matrix:
        self.path_dataset_correlation = os.path.join(self.path_save_folder, 
                                                     'dataset_correlation.csv')
        # Sampler properties::
        self.path_sampler_properties = os.path.join(self.path_save_folder, 
                                                      'sampler_properties.csv')
        
        # Initialize with non-standard csv files:
        # Marginals    
        if dataset_properties != None:
            self.path_save_dataset_properties = dataset_properties
            
        # Probabilities of colors and shapes:
        if dataset_correlation != None:
            self.path_shapes_colors = shapes_colors
        
        # Correlation matrix of matrices
        if dataset_correlation != None:
            self.path_dataset_correlation = dataset_correlation
            
        if sampler_properties != None:
            self.path_sampler_properties = sampler_properties
        
        # Save folder:    
        if dataset_name == None:
            self.path_save_dataset = os.path.join(self.path_save_folder,
                                                             'dataset_partial')
        else:
            self.path_save_dataset = os.path.join(self.path_save_folder,
                                            'dataset_'+dataset_name+'_partial')
            
        # If it already exists - remove it:
        if not os.path.isdir(self.path_save_dataset):
            os.mkdir(self.path_save_dataset) 
        else:
            shutil.rmtree(self.path_save_dataset)
            os.mkdir(self.path_save_dataset)    
            
        # Define the maximum amount of resampling:
        self.max_find_another = 100000
            
        
    def import_dataset_properties(self, 
                                  path_dataset_properties = None, 
                                  path_dataset_correlation = None, 
                                  path_shapes_colors = None, 
                                  path_sampler_properties = None):
        '''
        (Method) import the .cvs with the dataset properties
        dataset_properties = None : csv file of dataset properties
        dataset_correlation = None : csv file of correlation matrices
        shapes_colors = None : csv file of shapes/colors probabilities
        sampler_properties = None : csv file of sampler properties 
        '''
        
        # Libraries
        import os
        import pandas as pd
        import numpy as np
        
        # Update datasets with different name
        if path_dataset_properties != None:  
            self.path_dataset_properties = path_dataset_properties
        if path_shapes_colors != None:  
            self.path_shapes_colors = path_shapes_colors
        if path_dataset_correlation != None:
            self.path_dataset_correlation = path_dataset_correlation
        if path_sampler_properties != None:
            self.path_sampler_properties = path_sampler_properties
         
        # Read csv with marginals
        self.dataset_properties = \
                 pd.read_csv(self.path_dataset_properties)
         
        try:
            # Import dataset:
            # Dataset properties:
            self.dataset_properties = \
                  pd.read_csv(self.path_dataset_properties)
            
            # Colors and shape probabilities:
            self.shapes_colors = \
                  pd.read_csv(self.path_shapes_colors)       
        
            # Correlation matrix:
            self.dataset_correlation = \
                pd.read_csv(self.path_dataset_correlation, header=None)
            
            # Sampler properties:
            self.sampler_properties = \
                pd.read_csv(self.path_sampler_properties)
                
            # Define the useful variables:
            self.dataset_size = self.sampler_properties['dataset_size'][0]
            self.sampling_strategy = \
                self.sampler_properties['sampling_strategy'][0]
                
            if (not self.sampling_strategy in ['MC', 'LHC', 'LDS']):
                print("WARNING: uncorrect sampling strategy. Set up as MC")
                self.sampling_strategy = 'MC'
                
            self.random_seed = \
                int(self.sampler_properties['random_seed'][0])
                
            self.pixel_resolution_x = \
                int(self.sampler_properties['pixel_resolution_x'][0])
                
            self.pixel_resolution_y = \
                int(self.sampler_properties['pixel_resolution_y'][0])
                
            self.correct_class = \
                self.sampler_properties['correct_classes']\
                    .replace(np.nan, None).to_list()
            self.correct_class = [i for i in self.correct_class if i is not None]    
                
            self.classification_noise = \
                self.sampler_properties['classification_noise'].to_list()
            
            self.out_of_border = self.sampler_properties['out_of_border']\
                .replace(np.nan, None).to_list()
            self.out_of_border = \
                [i for i in self.out_of_border if i is not None][0]   
            self.out_of_border = True
            
            self.background_color = \
                self.sampler_properties['background_color'][0]
            
        except:
            if (not os.path.isfile(self.path_dataset_properties)):
                print("WARNING: wrong path_dataset_properties.")
            elif (not os.path.isfile(self.path_shapes_colors)):
                print("WARNING: wrong path_shapes_colors.")
            elif (not os.path.isfile(self.path_dataset_correlation)):
                print("WARNING: wrong path_dataset_correlation.")
            elif (not os.path.isfile(self.path_sampler_properties)):
                print("WARNING: wrong path_sampler_properties.")
            else:
                print("WARNING: import_dataset_properties.")
    
    
    def define_marginals(self, shape = None, colour = None):
        '''
        (method) Define the marginals of the continuous distribution of a given
        combination of shape and colour.
        shape = None : shape of the image
        colour = None : colour of the image
        '''     
        
        # Libraries
        import openturns as ot 
        import numpy as np
        
        # Convert nan to None:
        self.dataset_properties = self.dataset_properties.replace(np.nan, None)
        
        # Define marginals
        def marginal_continuous_distribution(lower_bound=None, 
                                             upper_bound=None, 
                                             mu=None, 
                                             sigma=None):
            
            # Dirac constant distribution (non-null mu)- 
            if (mu!=None)and(lower_bound==None)and(upper_bound==None)\
                         and(sigma==None):
                # Dirac
                mu = float(mu)
                distribution = ot.Dirac(mu)       
                
            # Uniform distribution (non-null left and right bounds)
            elif (mu==None)and(lower_bound!=None)and(upper_bound!=None)\
                           and(sigma==None):
                lower_bound = float(lower_bound)
                upper_bound = float(upper_bound)
                distribution = ot.Uniform(lower_bound, upper_bound)
                
            # Gaussian distribution (non-null mu and sigma)
            elif (mu!=None)and(lower_bound==None)and(upper_bound==None)\
                           and(sigma!=None):
                mu = float(mu)
                sigma = float(sigma)
                distribution = ot.Normal(mu, sigma)
                
            # Truncated Gaussian distribution (non-null mu, sigma, and bounds)
            elif (mu!=None)and(sigma!=None)and\
            ((lower_bound!=None)or(upper_bound!=None)):
                if lower_bound==None:
                    lower_bound = -10e6
                if upper_bound==None:
                    upper_bound = 10e6
                lower_bound = float(lower_bound)
                upper_bound = float(upper_bound)
                mu = float(mu)
                sigma = float(sigma)
                distribution = ot.TruncatedNormal(mu, sigma, lower_bound,upper_bound)                
            else:
                print("WARNING: undefined distribution.")
                
            return distribution

        marginal_list = []          # List of marginal distributions
        marginal_list_names = []    # List of marginal names
        
        # Define the variables:
        # centre	 - Continuous X
        temp_variable_name = 'centre_x'
        distribution = marginal_continuous_distribution(
            self.dataset_properties[temp_variable_name][0], 
            self.dataset_properties[temp_variable_name][1], 
            self.dataset_properties[temp_variable_name][2], 
            self.dataset_properties[temp_variable_name][3])
        
        marginal_list_names.append(temp_variable_name)
        marginal_list.append(distribution)
        
        # centre	 - Continuous Y
        temp_variable_name = 'centre_y'
        distribution = marginal_continuous_distribution(
                self.dataset_properties[temp_variable_name][0], 
                self.dataset_properties[temp_variable_name][1], 
                self.dataset_properties[temp_variable_name][2], 
                self.dataset_properties[temp_variable_name][3])
        
        marginal_list_names.append(temp_variable_name)
        marginal_list.append(distribution)        
        
        # radius - Continuous	
        temp_variable_name = 'radius'
        distribution = marginal_continuous_distribution(
            self.dataset_properties[temp_variable_name][0], 
            self.dataset_properties[temp_variable_name][1], 
            self.dataset_properties[temp_variable_name][2], 
            self.dataset_properties[temp_variable_name][3])
        
        marginal_list_names.append(temp_variable_name)
        marginal_list.append(distribution)
        
        
        # rotation - Continuous - IF CIRCLE, set up to 0
        temp_variable_name = 'rotation'
        if shape == '1':
            # Circles are rotation invariant
            distribution = marginal_continuous_distribution(mu=0)
            marginal_list_names.append(temp_variable_name)
            marginal_list.append(distribution)
        else:
            distribution = marginal_continuous_distribution(
                self.dataset_properties[temp_variable_name][0], 
                self.dataset_properties[temp_variable_name][1], 
                self.dataset_properties[temp_variable_name][2], 
                self.dataset_properties[temp_variable_name][3])
            
            marginal_list_names.append(temp_variable_name)
            marginal_list.append(distribution)
        
        # deformation - Continuous
        temp_variable_name = 'deformation'
        if shape == '1':
            # Circles are deformation invariant
            distribution = marginal_continuous_distribution(mu=0)
            marginal_list_names.append(temp_variable_name)
            marginal_list.append(distribution)
        else:
            distribution = marginal_continuous_distribution(
                self.dataset_properties[temp_variable_name][0], 
                self.dataset_properties[temp_variable_name][1], 
                self.dataset_properties[temp_variable_name][2], 
                self.dataset_properties[temp_variable_name][3])
            
            marginal_list_names.append(temp_variable_name)
            marginal_list.append(distribution)
        
        # blur - Continuous
        temp_variable_name = 'blur'
        distribution = marginal_continuous_distribution(
            self.dataset_properties[temp_variable_name][0], 
            self.dataset_properties[temp_variable_name][1], 
            self.dataset_properties[temp_variable_name][2], 
            self.dataset_properties[temp_variable_name][3])
        
        marginal_list_names.append(temp_variable_name)
        marginal_list.append(distribution)
        
        # white_noise - Continuous	
        temp_variable_name = 'white_noise'
        distribution = marginal_continuous_distribution(
            self.dataset_properties[temp_variable_name][0], 
            self.dataset_properties[temp_variable_name][1], 
            self.dataset_properties[temp_variable_name][2], 
            self.dataset_properties[temp_variable_name][3])
        
        marginal_list_names.append(temp_variable_name)
        marginal_list.append(distribution)
        
        # holes	- Continuous
        temp_variable_name = 'holes'
        distribution = marginal_continuous_distribution(
            self.dataset_properties[temp_variable_name][0], 
            self.dataset_properties[temp_variable_name][1], 
            self.dataset_properties[temp_variable_name][2], 
            self.dataset_properties[temp_variable_name][3])
        
        marginal_list_names.append(temp_variable_name)
        marginal_list.append(distribution)
        
        # additive_noise_regression	
        temp_variable_name = 'additive_noise_regression'
        distribution = marginal_continuous_distribution(
            self.dataset_properties[temp_variable_name][0], 
            self.dataset_properties[temp_variable_name][1], 
            self.dataset_properties[temp_variable_name][2], 
            self.dataset_properties[temp_variable_name][3])
        
        marginal_list_names.append(temp_variable_name)
        marginal_list.append(distribution)
        
        # multiplicative_noise_regression	
        temp_variable_name = 'multiplicative_noise_regression'
        distribution = marginal_continuous_distribution(
            self.dataset_properties[temp_variable_name][0], 
            self.dataset_properties[temp_variable_name][1], 
            self.dataset_properties[temp_variable_name][2], 
            self.dataset_properties[temp_variable_name][3])
        
        marginal_list_names.append(temp_variable_name)
        marginal_list.append(distribution)    
        
        self.marginal_list_names = marginal_list_names
        self.marginal_list = marginal_list
        
        

    def define_multivariate_distribution(self, corr_matrix, shape, color):
        '''
        Define a multivariate distribution.
        corr_matrix : (panda dataframe) correlation matrices
        shape : (str) shape under examination
        color : (str) color under examination
        '''

        # Libraries
        import openturns as ot 
        
        # Define the associated multivariate distribution, 
        # they are the same for each class:        
        index_color = self.list_colors.index(color) # Rows
        index_shape = self.list_shapes.index(shape) # Columns
        size_sub_matrix = len(self.marginal_list_names)
        
        
        try:
            # From the full correlation matrix, extract the submatrix:
            tmp_matrix = corr_matrix.to_numpy()[
                (index_shape)*size_sub_matrix:(index_shape+1)*size_sub_matrix,
                (index_color)*size_sub_matrix:(index_color+1)*size_sub_matrix]
            
            # Define the correlation matrix
            corr_matrix = ot.CorrelationMatrix(tmp_matrix)
            
        except:
            # Initialize a standard identical Copula
            corr_matrix = ot.CorrelationMatrix(len(self.marginal_list_names))

        # Define the Copula:
        Copula = ot.NormalCopula(corr_matrix)
        
        # Define the multivariate distribution:
        multi_distribution = ot.ComposedDistribution(self.marginal_list, 
                                                     Copula)    
        multi_distribution.setDescription(self.marginal_list_names)
        
        return multi_distribution
    
    
    
    def define_csv(self, sample_df, shape = 'no_shape', color = 'no_color'):
        '''
        Define the csv files of the dataset.
        sample_df : dataset to save (temporary sample dataset)
        shape = 'no_shape' : save the dataset of the given shape
        color = 'no_color' : save the dataset of the given color
        '''
        
        # Libraries
        import os
        import pandas as pd
        import numpy as np
        import random
        
        random.seed(self.random_seed)
        
        # Define the dictionary to define the correct list name
        sample_df = pd.DataFrame(sample_df.to_numpy(), 
                                 columns = self.marginal_list_names)
               
        # Add index for each element:
        sample_df['ID_image'] = ['ID_'+str(shape)+'_'+str(color)+'_'+str(i) 
                                 for i in range(len(sample_df))]
        
        # Add the shape and color identity
        sample_df['shape'] = shape        
        sample_df['color'] = color
        
        # Add the background color
        sample_df['background_color'] = self.background_color
        
        # Add pixel resolution
        sample_df['pixel_resolution_x'] = self.pixel_resolution_x
        sample_df['pixel_resolution_y'] = self.pixel_resolution_y
        
        # Define the regression ground truth - radius
        sample_df['regression_radius'] = sample_df['radius']
        sample_df['regression_radius_noise'] = sample_df['radius']*\
            (100+sample_df['multiplicative_noise_regression'])/100\
                +sample_df['additive_noise_regression']
        
        # Define the regression ground truth - centering
        sample_df['regression_centering'] = \
           np.sqrt((sample_df['centre_x']-50)**2+(sample_df['centre_y']-50)**2)
        sample_df['regression_centering_noise'] = \
            sample_df['regression_centering']*\
            (100+sample_df['multiplicative_noise_regression'])/100\
                +sample_df['additive_noise_regression']
                
        # Define the correct shapes and colors
        correct_shapes = [] # ONLY shapes
        correct_colors = [] # ONLY color
        correct_shapes_and_colors = [] # Shapes and colors
        
        
        for i_list in range(len(self.correct_class)):
            tmp_correct_class = self.correct_class[i_list].split('/')
            tmp_correct_class =  [i for i in tmp_correct_class if i != '']   

            if len(tmp_correct_class) >1:
                correct_shapes_and_colors.append(\
                                                 [int(tmp_correct_class[0]),\
                                                 tmp_correct_class[1].replace(' ','')])
            else:
                if ('(' in self.correct_class[i_list]): # Color
                    correct_colors.append(self.correct_class[i_list].split('/')[1].replace(' ',''))
                else:
                    correct_shapes.append(int(self.correct_class[i_list].split('/')[0].replace(' ','')))
        
        # Noisy classification class                                 
        sample_df['shape_noisy'] = sample_df['shape']
        sample_df['color_noisy'] = sample_df['color']
                
        # Correct/Noisy classification class - Classes can be shape/color/combination
        # Iterate on the dataframe:
        sample_df['correct_class'] = False
        sample_df['noisy_class'] = False
        
        for i_row in range(len(sample_df)):
            row = sample_df.iloc[i_row]

            # Check if shape is correct
            if int(row['shape']) in list(correct_shapes):
                sample_df.loc[i_row,'correct_class'] = True
                
            # Define the "noisy" classification label
            # The noisy classification is defined from a class (shape, color)
            # to another class. 'all'/'' means that every class is changed:
            for noisy_class in self.classification_noise:
                try:
                    noisy_class_split = noisy_class.split(';')                
                    first_class_shape = noisy_class_split[0].replace('[','').replace(']','').split('/')[0] # If empty len() == 0
                    first_class_color = noisy_class_split[0].replace('[','').replace(']','').split('/')[1]
                    second_class_shape = noisy_class_split[1].replace('[','').replace(']','').split('/')[0]
                    second_class_color = noisy_class_split[1].replace('[','').replace(']','').split('/')[1]
                    change_probability = noisy_class_split[2]
                    
                    # Check the original shape:
                    # The input should be: 
                    #[original_shape/original_color];[new_shape/new_color];prob
                    # If an original element is empty, implies that every choices
                    # are acceptable
                    # Viceversa, if output is empty, it implies that remains the same
                    # If output is *, it sample randomly
                    
                    if (row['shape'] == first_class_shape)\
                                                 or(len(first_class_shape)==0):
                                                         # Check also color (or empty (all) color)
                        if (row['color'] == first_class_color)\
                                                 or(len(first_class_color)==0):
                                                     
                            # Randomly sample a number between [0,1]
                            
                            random_sampled = random.randint(1,100)
                            if random_sampled <= \
                                            int(float(change_probability)*100):
                                
                                # We want to change to second shape and second
                                # color. But if ALL is active, we must resample
                                if len(second_class_shape)>0:
                                    # Or remain the same or randomly resample:
                                    if second_class_shape == '*':
                                        candidate_shape = self.list_shapes\
                                            [random.randint(0, \
                                                      len(self.list_shapes)-1)]
                                    else:
                                        candidate_shape = second_class_shape
                                else:
                                    # Mantain the original value:
                                    candidate_shape = \
                                                   sample_df.loc[i_row,'shape']
                                
                                if len(second_class_color)>0:
                                    if (second_class_color == '*'):
                                        candidate_color = self.list_colors\
                                            [random.randint(0, \
                                                      len(self.list_colors)-1)]
                                    else:
                                        candidate_color = second_class_color
                                else:
                                    candidate_color = \
                                                   sample_df.loc[i_row,'color']
                                
                                sample_df.loc[i_row,'shape_noisy'] = \
                                                                candidate_shape
                                sample_df.loc[i_row,'color_noisy'] = \
                                                                candidate_color
                except:
                    pass
            
            # Redefine the row (updated)
            row = sample_df.iloc[i_row]
            
            # Check if shape is correct
            if int(row['shape']) in list(correct_shapes):
                sample_df.loc[i_row,'correct_class'] = True
            if int(row['shape_noisy']) in list(correct_shapes):
                sample_df.loc[i_row,'noisy_class'] = True
                
            # Check if color is correct
            if str(row['color']) in list(correct_colors):
                sample_df.loc[i_row,'correct_class'] = True
            if str(row['color_noisy']) in list(correct_colors):
                sample_df.loc[i_row,'noisy_class'] = True
                
            # Check if shape AND color is correct
            for element in correct_shapes_and_colors:
                element_shape = element[0]
                element_color = element[1]
                
                if (int(row['shape']) == element_shape)and(str(row['color']) \
                                                             == element_color):
                    sample_df.loc[i_row,'correct_class'] = True    
                if (int(row['shape_noisy']) == element_shape)and\
                                    (str(row['color_noisy']) == element_color):
                    sample_df.loc[i_row,'noisy_class'] = True   
                    
        path_save_csv = os.path.join(self.path_save_dataset, \
                                   'partial_'+str(shape)+'_'+str(color)+'.csv')
        sample_df.to_csv(path_save_csv, index = False)
    
    
    
    def merge_dataset(self):
        '''
        Merge all the saved datasets.
        '''
        
        import os
        import pandas as pd
        
        list_csv = os.listdir(self.path_save_dataset)
        try:
            list_csv.remove('combined_dataframe.csv')
        except:
            pass
        
        path_dataframe = os.path.join(self.path_save_dataset, list_csv[0])
        df_combined = pd.read_csv(path_dataframe)
        
        try:
            list_csv.remove(list_csv[0])
        except:
            pass
        
        for element in list_csv:
            path_dataframe = os.path.join(self.path_save_dataset, element)
            df_combined = pd.concat([df_combined,pd.read_csv(path_dataframe)])
        
        df_combined.to_csv(os.path.join(self.path_save_folder,
                                      'combined_dataframe.csv'), index = False)
        
    
    
    def call_sampler(self):
        '''
        Call the sampler and define the whole dataset.
        '''
        
        import openturns as ot
        
        # Define the list of shapes and colors:
        self.list_shapes = (self.shapes_colors.columns).to_list()[1:]
        self.list_colors = (self.shapes_colors).iloc[:,0].values.tolist()
        
        # Define the combined distribution with the given copula
        seed_modifier = 1        
        
        for shape in self.list_shapes:
            for color in self.list_colors:
                # Re-define marginals for each combination:
                self.define_marginals(shape, color)
                
                # Recover the property from shapes_colors
                # Probability
                tmp_probability = (self.shapes_colors.loc[:,shape][
                    self.shapes_colors.iloc[:,0]==color]).to_list()[0]
                tmp_dataset_size = int(self.dataset_size*tmp_probability)

                # Define the multivariate distribution
                multi_distribution = self.define_multivariate_distribution(
                    corr_matrix = self.dataset_correlation,
                    shape = shape, color = color)
                
                # Call the sampler: Modified to avoid same results among shapes
                ot.RandomGenerator.SetSeed(self.random_seed*seed_modifier)
                seed_modifier += 1
                
                # Standard Monte Carlo
                if self.sampling_strategy == 'MC':  
                    tmp_sample_df = multi_distribution.getSample\
                                            (tmp_dataset_size).asDataFrame()
                                            
                # If the method is Monte Carlo and out_of_border is False, 
                # center outside the border are not allowed and resampled
                    resampler_seed = 1
                    if (self.out_of_border == False):
                        # Find elements which have center outside the border:
                        for i_row in range(len(tmp_sample_df)):
                            resampler_seed += 1
                            row = tmp_sample_df.iloc[i_row]
                            centre_x = float(row['centre_x'])
                            centre_y = float(row['centre_x'])
                            radius = float(row['radius'])
                            
                            # Check if centre is too close to the border:
                            find_another = 0
                            if (centre_x<radius)or(centre_x>100-radius)or\
                                    (centre_y<radius)or(centre_y>100-radius):
                                        
                                while find_another < self.max_find_another:
                                    find_another += 1
                                    ot.RandomGenerator.SetSeed(\
                                                             self.random_seed*\
                                                                seed_modifier*\
                                                                 find_another*\
                                                                resampler_seed)
                                    # Find another candidate:
                                    tmp_resampling_df = \
                                                  multi_distribution.getSample\
                                                  (1).asDataFrame()    
                                    row_resampled = tmp_resampling_df.iloc[0]
                                    centre_x_resampled = \
                                               float(row_resampled['centre_x'])
                                    centre_y_resampled = \
                                               float(row_resampled['centre_x'])
                                    radius_resampled = \
                                                 float(row_resampled['radius'])
                                    
                                    if (centre_x_resampled>radius_resampled)and\
                                       (centre_x_resampled<100-radius_resampled)and\
                                       (centre_y_resampled>radius_resampled)and\
                                       (centre_y_resampled<100-radius_resampled):
                                         # Stop the while cycle:
                                         find_another = self.max_find_another+1       
                                         
                                         # Change the original row:
                                         tmp_sample_df.iloc[i_row] = \
                                                                  row_resampled 
                # Latin Hyper Cube Sampling
                elif self.sampling_strategy == 'LHC':
                    exp = ot.LHSExperiment(multi_distribution, \
                                           tmp_dataset_size, False, False)
                    tmp_sample_df = exp.generate().asDataFrame()
                    
                # Low Discrepancy Sequence (SOBOL)    
                elif self.sampling_strategy == 'LDS':
                    sequence = ot.SobolSequence(len(self.marginal_list_names))
                    exp = ot.LowDiscrepancyExperiment(sequence,\
                                multi_distribution, tmp_dataset_size, False)
                    tmp_sample_df = exp.generate().asDataFrame()
                else:
                    print("WARNING: sampling strategy definition.")
                    tmp_sample_df = multi_distribution.getSample\
                                            (tmp_dataset_size).asDataFrame()
                                            
                # Convert the float variable (holes) to an integer one:
                for i_row in range(len(tmp_sample_df)):
                    row = tmp_sample_df.iloc[i_row]
                    tmp_sample_df.loc[i_row,'holes'] = int(round(row['holes']))
                
                # Call the method to produce the csv for the given sample
                self.define_csv(tmp_sample_df, shape = shape, color = color)
        
        self.merge_dataset()
        
        
        
        
                
        
        
        
        
        







