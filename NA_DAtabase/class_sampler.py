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
     
        
    def generate_images(self):
        '''
        Generate the images from the defined dataset.
        '''

        MDB = MorphShapes_DB_Builder(self.path_save_folder)
        MDB.generate()
        
    def auto_process(self):
        '''
        Execute all the standard instructions to produce the dataset,.
        '''
        
        self.import_dataset_properties()    
        self.call_sampler()
        self.generate_images()




            
                            
        
#%% Class to generate pictures
class MorphShapes_DB_Builder:
	def __init__(self, csv_path):
		''' 
		Load the data from the given csv_path
		''' 
        
		import pandas as pd
		import os
		import shutil
        
		path_csv_to_read = os.path.join(csv_path,
                                      'combined_dataframe.csv')
		self.df = pd.read_csv(path_csv_to_read)
        
		image_save_path = os.path.join(csv_path,'dataset_images')
        
        # If exists, destroy
		if os.path.isdir(image_save_path):
		    shutil.rmtree(image_save_path)
		    os.mkdir(image_save_path)  
		else:
		    os.mkdir(image_save_path)  
        
		self.output_path = image_save_path
		self.output_csv_filename = path_csv_to_read
		
	def generate(self):
		'''
		Generates the shape in the canvas based on csv data
		'''

		area = []
		area_noise = []
		
		for index, row in self.df.iterrows():
			
			w, h = row['pixel_resolution_x'], row['pixel_resolution_y']
			M = Main_Surface(w, h) 

			background = self._get_tuple_or_value(row['background_color'], 
                                                             [(0, 1),(0, 255)])

			#
			# the backbround can be 
			# - a color (tuple of R,G,B values)
			# - a path to an image (that will be resized to the canvas size) 
			# 
			if type(background) is tuple:
				M.setBackgroundColor(background)
			else:
				M.setBackgroundImage(background)

			color = self._get_tuple_or_value(
				row['color'], 
				[(0, 1),(0, 255)]
			)

			sides = row['shape']

			# 
			# two sides is a segment so skip it
			# 
			if sides == 2:
				area.append(0)
				area_noise.append(0)
				continue

			center = (
				self._remap(
					row['centre_x'],
					[(0, 100), (0, w)]
				),
				self._remap(
					row['centre_y'],
					[(0, 100), (0, h)]
				)
			) 

			radius = self._remap(
				row['radius'],
				[(0, 100), (0, w)]
			)

			rotation = row['rotation']
			morph_percentage = row['deformation']

			M.drawShape(
			    Shape(
			        center = center,
			        radius = radius,
			        color = color, 
			        sides = sides, 
			        rotation = rotation,
			        morph_percentage = morph_percentage
			    )

			)

			# 
			# add holes
			# 
			holes = int(row['holes'])
			if(holes > 1):
				M.emmental(holes, index)

			area.append(round(M.filled_area, 3))

			mnr, anr = row['multiplicative_noise_regression'], row['additive_noise_regression']
			area_noise.append(round(M.filled_area * (100 + mnr) + anr, 3))

			# 
			# add blur
			# 
			if(row['blur'] > 0):
				blur = self._remap(
					row['blur'],
					[(0, 100), (0, 10)],
					rounded = False
				)
				M.blur(blur)

			# 
			# add noise
			# 
			if(row['white_noise'] > 0):
				noise_power = self._remap(
					row['white_noise'],
					[(0, 100), (0, 1)],
					rounded = False
				) 
				M.addNoise(noise_power)
			
			# 
			# save output image
			# 
			filename = self.output_path+"/"+row['ID_image']+".png"
			M.save(filename)
			print (filename+" generated.")

		# 
		# add the shape area to the data frame
		# and save the output csv
		# 
		self.df['regression_area'] = area
		self.df['regression_area_noise'] = area_noise
		self.df.to_csv(self.output_csv_filename)

	
	def _get_tuple_or_value(self, cell, map_range):
		# 
		# check a cell string value if is a tuple
		# and eventually convert it, remapping the values
		# 
		if cell.startswith("("):
			cell = cell.strip("()")
			return self._remap(list(map(float, cell.split(','))), map_range)
		return cell

	def _remap(self, value, map_range, rounded = True):
		# 
		# remap a single value or tuple using a range defined as
		# [ (min_value, max_value), (new_min, new_max) ]
		# 
		# es: to remap range 0,1 to 0,255 map_range is:
		# [ (0, 1), (0, 255) ]
		# 
		_mra, _mrb = map_range
		_min, _max, = _mra
		_a, _b = _mrb
		
		if type(value) is list:
			value = tuple(map(lambda v: round((_b -_a) * (v - _min) / (_max - _min) +_a), value))
			return value 

		value = (_b - _a) * (value - _min) / (_max - _min) +_a 
		return round(value) if rounded else value        
        
        
                
#%% Auxiliary functions
        
# 
# this class is used to reproduce OpenCV shapes formulas
# because OpenCV circlular shapes are weird 
# 
class PIL_Drawing:
    def __init__(self, width, height):
        # 
        # The drawing surface
        # 
        from PIL import Image, ImageDraw
        
        self._image = Image.new('RGB', (width, height))
        self._draw = ImageDraw.Draw(self._image)
        cx = width // 2
        cy = height // 2
        self._center = (cx, cy)
    
    #
    # image getter
    #
    def image(self):
        return self._image

    # 
    # image setter
    # 
    def set_image(self, image):
        self._image = image

    # 
    # center getter
    # 
    def center(self):
        return self._center

    def fillPoly(self, ptx, color):
        # 
        # cv2.fillPoly override
        # 
        self._draw.polygon(
            ptx, 
            fill = color, 
            outline = None
        )

    def circle(self, center, radius, color, width = 1):
        # 
        # cv2.circle override
        # 
        x, y = center
        fill_color = None if width > 0 else color
        
        self._draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill_color,
            color
        )

    def arc(self, center, axes, start_angle, end_angle, color, filled = True):
        # 
        # cv2.ellipse-like override
        # 
        x, y = center
        a1, a2 = axes
        
        arc_center = [(x - a1, y - a2), (x + a1, y + a1)]
                
        if filled:
            self._draw.chord(
                arc_center,
                start_angle - 90, 
                end_angle - 90, 
                color
            )
        else: 
            self._draw.arc(
                arc_center.
                start_angle - 90, 
                end_angle - 90, 
                color
            )

    def to_openCV_Image(self):
        '''
        Convert PIL image to OpenCV format 
        ''' 
        
        import cv2
        import numpy as np
        
        return cv2.cvtColor(np.array(self._image), cv2.COLOR_RGB2BGR)

    def paste(self, image, coords = (0,0)):
        # 
        # paste image into canvas at top-left coords
        # 
        self._image.paste(image, coords)

class Shape:
    def __init__(self, center, radius, color, sides = 1, rotation = 0, morph_percentage = 0):
        # 
        # the shapes are all build inscribed inside a circle
        # so each shape has a center and a radius.
        # 
        # if sides is not defined or equals zero,
        # the result shape is a circle.
        # 
        # the shape can be rotated and have a morph percentage 
        # that makes it stick to a circle
        # 
        # each shape is pre-drawn on a square surface 
        # with dimensions equal to the diameter (+1px) of the 
        # circle in which it is inscribed, multiplied by 
        # a variable scale factor depending on the radius itself
        # 
        self._center = center
        self._radius = radius
        self._color = color
        self._sides = sides
        self._rotation = rotation
        self._morph_percentage = morph_percentage 
        self._up_factor = 10 if radius < 50 else 2
        
    def center(self):
        return self._center

    def radius(self):
        return self._radius

    def shape_canvas(self):
        return self._shape_canvas

    def draw(self, anti_aliasing = True):
        from PIL import Image
        
        canvas_side = (self._radius * 2 + 1) * self._up_factor
        self._shape_canvas = PIL_Drawing(canvas_side, canvas_side)

        # 
        # if the number of sides equals zero
        # the shape is a circle, else a polygon
        # 
        if self._sides == 1:
            self._shape_canvas.circle(
                self._shape_canvas.center(), 
                self._radius * self._up_factor, 
                self._color, 
                -1
            )
        else:
            ptx = self._generate_vertices()
            self._shape_canvas.fillPoly(ptx, self._color)
            # 
            # the poly is possibly morphed and rotated
            # 
            self._morph()
            self._rotate(anti_aliasing)
                    
        # 
        # the final image is resized to its original size,
        # using AA filter if required
        # 
        resampling = Image.Resampling.BICUBIC if anti_aliasing else Image.Resampling.NEAREST

        original_size = self._shape_canvas.image().size[0] // self._up_factor

        self._shape_canvas.set_image(
            self._shape_canvas.image().resize(
                (original_size, original_size), 
                resampling
            )
        )

    def _rotate(self, anti_aliasing):    
        from PIL import Image
        
        if self._rotation == 0 or self._sides == 0:
            return

        resampling = Image.Resampling.BICUBIC if anti_aliasing else Image.Resampling.NEAREST

        rot = self._shape_canvas.image().rotate(
            -self._rotation, 
            resampling, 
            expand =False
        )

        self._shape_canvas.set_image(
            rot
        )

    def _morph(self):
        # 
        # the morph is always calculated from the original shape towards a circle.
        # the idea is that the canvas area of the figure with morph = 100%
        # is equal to the canvas of the circle that inscribes it.
        #  
        
        from PIL import Image, ImageDraw
        from math import pi, sin, radians, cos, dist, sqrt, acos
        
        if self._morph_percentage == 0 or self._sides == 0:
            return 

        # 
        # set the radius
        # 
        R = self._radius * self._up_factor

        # 
        # set the maximum distance from the center of the circle
        # to calculate the maximum bending arc
        # 
        max_cdy = R * 5

        # 
        # the rotation angle of the bending arc is 
        # calculated using the number of sides,
        # then the position of a vertex is set so as 
        # to make sure that the dx and dy components are respectively
        # parallel to the Y axis (dy) and the X axis (dx)
        #  
        alpha = 180 // self._sides
        
        dx = R * sin(radians(alpha))
        dy = R * cos(radians(alpha))

        cx, cy = self._shape_canvas.center()
        vertex = cx + dx, cy + dy

        mask = Image.new("L", self._shape_canvas.image().size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((cx - R, cy - R, cx + R, cy + R), fill=255)
        
        # 
        # the deviation from the center is calculated
        # in order to have a constant curvature variation 
        # as the morph percentage varies#
        # 
        cdy = self._calcCDY(max_cdy)
        
        # 
        # move the center of the circle and 
        # recalculated the radius and the dy component
        # 
        center = (cx, cy - cdy)

        R = round(dist(center, vertex))
        dy = sqrt(R**2 - dx**2)
        
        #
        # the subtended arc angle increased by an X value
        # to make sure all the arcs close the edges
        #   
        alpha_arc = round(acos((dy**2 + R**2 - dx**2) / (2 * dy * R)) * 180 / pi) + 0.5
        # 
        # the arc will have to be drawn and rotated N times around the center
        # based on the number of vertices of the shape
        # 
        rotation_step = round(360 / self._sides)

        for side in range(0, 360 - rotation_step + 1 , rotation_step):
            self._shape_canvas.arc(
                center, 
                (R, R), 
                180 + side - alpha_arc, 
                180 + side + alpha_arc, 
                self._color, 
                True
            ) 
            center = self._rotate_point(
                center, 
                self._shape_canvas.center(), 
                -rotation_step
            )
        
        # 
        # the final result is masked
        # to remove edges noise
        # 
        blank = self._shape_canvas.image().point(lambda _: 0)
        c = Image.composite(
            self._shape_canvas.image(), 
            blank, 
            mask
        )

        self._shape_canvas.set_image(c)

    def refillblack(self, image, background_color):
        d = image.getdata()
        new_image = []

        for item in d:
            if item[0] == 0:
                new_image.append(background_color)
            else:
                new_image.append(item)
         
        image.putdata(new_image)
        return image

    def _calcCDY(self, max_cdy):
        # 
        # the main formula to calcolate the arc of morphing
        # 
        
        from math import cos, radians, sqrt
        
        R = self._radius * self._up_factor
        n = self._sides
        l = R * sqrt(2 - 2 * cos(radians(360 / n)))
        R1 = sqrt(R**2 - (l / 2)**2)
        d1 = self._morph_percentage / 100 * (R - R1)
        d2 = ((l/2)**2 - 2 * d1 * R1 - d1**2) / (2 * d1)
        return round(d2) if d2 < max_cdy else max_cdy 

    def _rotate_point(self, point, origin, degrees):
        #
        # Rotate a point counterclockwise by a given angle around a given origin.
        #
        
        from math import radians, cos, sin
        
        rad = radians(degrees)
        x,y = point
        offset_x, offset_y = origin
        adjusted_x = (x - offset_x)
        adjusted_y = (y - offset_y)
        cos_rad = cos(rad)
        sin_rad = sin(rad)
        qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
        qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y
        return round(qx), round(qy)
        
    def _generate_vertices(self):
        vertices = []
        center = self._shape_canvas.center()

        # 
        # the first vertex is always the top one
        # 
        top = (center[0], center[1] - self._radius * self._up_factor)
        
        # 
        # if the shape has an even number of sides,
        # a pre-rotation is applied
        # 
        if self._sides % 2 == 0:
            top = self._rotate_point(top, center, 180 // self._sides)
        
        for i in range(self._sides):
            vertices.append(self._rotate_point(top, center, i * 360 / self._sides))

        return vertices

class Main_Surface:
    def __init__(self, width, height):
        # 
        # The main canvas where the shape(s) is/are drawn
        # 
        self._main_surface = PIL_Drawing(width, height)
        self.shape = None
        self.filled_area = 0
        self.background_color = (0, 0, 0)
        self.backgroundImage = None
        self.image_set = False
        self.noise_power = 0 

    def setBackgroundColor(self, background_color):
        self.background_color = background_color

    def setBackgroundImage(self, path):
        from PIL import Image
        
        self.backgroundImage = Image.open(path)
        self.backgroundImage = self.backgroundImage.resize(
            self._main_surface.image().size, 
            Image.Resampling.BICUBIC
        )
        
    def drawShape(self, shape, sides = 0, anti_aliasing = True):
        if self.shape != None:
            return
        # 
        # add a shape and draw it in the main canvas
        # 
        shape.draw(anti_aliasing)

        cx, cy = shape.center()
        R = shape.radius()
           
        self._main_surface.paste(
            shape.shape_canvas().image(), 
            (cx - R, cy - R)
        )
        
        self.shape = shape
        self.filled_area = self._calcArea()
        
    def _prepareImage(self):
        from PIL import Image
        import numpy as np
        import cv2

        if (self.backgroundImage == None and self.background_color == None) or self.image_set:
            return

        mask = self._getImageMask()
        mask = Image.fromarray(mask)
    
        background = self.backgroundImage if self.backgroundImage != None else Image.new("RGB",self._main_surface.image().size,self.background_color)
      
        self._main_surface.set_image(
            Image.composite(
                self._main_surface.image(), 
                background, 
                mask
            )
        )

        if self.noise_power != None:
            # 
            # add weighted gaussian noise to the original image
            # power is the alpha channel of the noise and
            # its value must be between 0 and 1
            # 
            im = np.array(self._main_surface.image())
            width, height = self._main_surface.image().size

            mean = 0
            stddev = 128

            noise = np.zeros((width, height, 3), dtype = np.uint8)

            for i in range(len(noise) - 1):
                cv2.randn(noise[i], mean, stddev)

            im_new = cv2.addWeighted(
                noise, 
                self.noise_power, 
                im, 
                1, 
                0, 
                im
            );

            self._main_surface.set_image(
                Image.fromarray(im_new)
            )

        self.image_set = True
       
    def blur(self, power = 2):
        # 
        # blur the final image
        # 
        from PIL import ImageFilter
        
        self._main_surface.set_image(
            self._main_surface.image().filter(ImageFilter.GaussianBlur(power))
        )

    def addNoise(self, noise_power = 0.5):
        self.noise_power = noise_power

    def emmental(self, holes = 2, seed_index = None):
        # 
        # create hole(s) on the main canvas
        # 
        
        from random import randint, seed
        from math import sin, cos, radians
        
        if seed_index != None:
            seed(seed_index)

        shape = self.shape
        cx, cy = shape.center()
        R = shape.radius()
            
        for hole in range(holes):
            alpha = randint (0,359)
            dx = round(R * sin(radians(alpha)))
            dy = round(R * cos(radians(alpha)))

            center = (cx + dx , cy + dy)
            hole_radius = randint(R // 2, R * 2 // 3)

            self._main_surface.circle(
                center,
                hole_radius,
                (0, 0, 0),
                -1
            )


    def show(self):
        # 
        # show the result on screen
        # 
        
        import cv2
        
        self._prepareImage()
        cv2.imshow(
            "Final Result", 
            self._main_surface.to_openCV_Image()
        )
        cv2.waitKey(0)

    def save(self, filename):
        # 
        # saves the result on the filesystem
        #
        self._prepareImage()
        self._main_surface.image().save(filename)

    def _getImageMask(self):
        # 
        # return the mask of the shape
        # calculated with a threshold value
        # 
        
        import cv2
        import numpy as np
        
        L= cv2.cvtColor(np.array(self._main_surface.image()), cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(L)
        min_v, max_v = np.min(v), np.max(v)
        mask = np.zeros((L.shape[0], L.shape[1]), dtype=np.uint8)

        for i in range(L.shape[0]):
            for j in range(L.shape[1]):
                mask[i,j] = int((v[i,j] - min_v) * 255 / (max_v - min_v)) if ((max_v - min_v) > 0) else 255
                if mask[i,j] >= 240:
                    mask[i,j] = 255

        return mask

    def _calcArea(self):
        # 
        # calculate the area of the shape in %
        # based on non zero pixel
        # 
        
        import numpy as np
        
        mask = self._getImageMask()
        w,h = self._main_surface.image().size
        pixels = np.count_nonzero(mask)
        return pixels / (w * h) * 100                
        
        
        







