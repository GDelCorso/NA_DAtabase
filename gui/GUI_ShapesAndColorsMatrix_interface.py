# -*- coding: utf-8 -*-
"""
Comandi per GUI Volpini

Created on Wed Feb  7 10:26:32 2024

@author: Giulio Del Corso
"""
import numpy as np
import pandas as pd
import os
#internal import
from GUI_helper import *
              
class GUI_ShapesAndColorsMatrix_interface:

    
    def __init__(self, dataset_name = 'outputs_folder', round_normalization=5):
        '''
        Parameters
        ----------
        dataset_name : TYPE, optional
            Name of the dataset. The default is 'outputs_folder'.
        round_normalization : TYPE, optional
            Precision. The default is 5.
        '''
        self.dataset_name = dataset_name
        self.round_normalization = round_normalization
        self.probability_matrix = np.zeros((0,0))
        self.history_probability_matrix = [self.probability_matrix.copy()]    # List for undo
        
        self.lock_matrix = np.zeros((0,0)).astype(np.uint8)
        self.history_lock_matrix = [self.lock_matrix.copy()]    # List for undo
        
        self.shape_order = []   # Column
        self.history_shape_order = [self.shape_order.copy()]
        
        self.color_order = []   # Row
        self.history_color_order = [self.color_order.copy()]
        
        self.prob_shape = []    # Prob shape
        self.prob_color = []    # Prob color
        
        print("%s initialized" % dataset_name)
        
        
    
    def add_to_memory(self):
        '''
        Add to the memory the old version.
        '''
        self.history_probability_matrix.append(self.probability_matrix.copy())
        self.history_lock_matrix.append(self.lock_matrix.copy())
        self.history_shape_order.append(self.shape_order.copy())
        self.history_color_order.append(self.color_order.copy())
        
        # If the length is excessive, delete them:
        if len(self.history_probability_matrix)>20:
            self.history_probability_matrix = self.history_probability_matrix[len(self.history_probability_matrix)-20:]
            self.history_lock_matrix = self.history_lock_matrix[len(self.history_lock_matrix)-20:]
            self.history_shape_order = self.history_shape_order[len(self.history_shape_order)-20:]
            self.history_color_order = self.history_color_order[len(self.history_color_order)-20:]
    
        
    
    def undo(self):
        '''
        Undo the last operation
        '''
        
        #print("\n\nVEDERE")
        #print(self.probability_matrix)
        #print(self.history_probability_matrix)
        
        self.probability_matrix = self.history_probability_matrix[-1]
        self.lock_matrix = self.history_lock_matrix[-1].copy()
        self.shape_order = self.history_shape_order[-1].copy()
        self.color_order = self.history_color_order[-1].copy()

        # Remove the last element
        self.history_probability_matrix = self.history_probability_matrix[:-1]
        self.history_lock_matrix = self.history_lock_matrix[:-1]
        self.history_shape_order = self.history_shape_order[:-1]
        self.history_color_order = self.history_color_order[:-1]
        
        #print(self.probability_matrix)
        #print("VISTO\n\n")
        
        
        
      
    def upd_probability(self):
        '''
        Update the probabilities of shapes and colors.
        '''
        self.prob_color = np.sum(self.probability_matrix,1)
        self.prob_shape = np.sum(self.probability_matrix,0)
        
        
    
    def lock_cell(self, shape, color):        
        '''
        Lock/Unlock a given cell.
        '''
        self.add_to_memory()
        
        # Find the index of the given shape and color:
        col_shape = self.shape_order.index(shape)
        row_color= self.color_order.index(color)
        
        
        # Lock/Unlock
        if self.lock_matrix[row_color,col_shape] == 0:
            self.lock_matrix[row_color,col_shape] = 1
        else:
            self.lock_matrix[row_color,col_shape] = 0
        #self.lock_matrix[row_color,col_shape] = value
        
        #print('%slocking shape %s color %s' % ("un" if value == 0 else "",shape, color) )
        
        
            
    def unlock_all(self):
        '''
        Unlock all the fields.
        '''
        self.add_to_memory()
        
        self.lock_matrix[self.lock_matrix==1] =0
        
        
    
    def lock_shape(self, shape):        
        '''
        Lock a whole column (shape). If already locked, unclock it.
        '''
        self.add_to_memory()
        
        # Find the index of the given shape:
        col_shape = self.shape_order.index(shape)
        
        locked_this_col_probability = np.sum(self.lock_matrix[:,col_shape])
        
        # If all already locked: unlock
        if locked_this_col_probability == len(self.color_order):
            self.lock_matrix[:,col_shape] = 0
        else:   # Otherwise: lock them
            self.lock_matrix[:,col_shape] = 1
        
        #self.lock_matrix[:,col_shape] = value
        
        #print('%slocking shape %s' % ("un" if value == 0 else "", shape))
        



    def lock_color(self, color):   
        '''
        Lock a whole column (shape). If already locked, unclock it.
        '''
        self.add_to_memory()
        
        # Find the index of the given shape and color:
        row_color= self.color_order.index(color)
    
        locked_this_row_probability = np.sum(self.lock_matrix[row_color,:])
        
        
        # If all already locked: unlock
        if locked_this_row_probability == len(self.shape_order):
            self.lock_matrix[row_color,:] = 0
        else:   # Otherwise: lock them
            self.lock_matrix[row_color,:] = 1
        
        #print('%slocking color %s' % ("un" if value == 0 else "", color))
        
        
        
        
    def normalize(self):
        '''
        Normalize the unlocked cells

        '''
        how_many_locked = int(self.lock_matrix.sum())
        how_many_cells = int(np.size(self.lock_matrix))

        # Probability of locked cells
        locked_probability = np.sum(self.probability_matrix[self.lock_matrix ==1])
        
        if (how_many_cells-how_many_locked)<=0:
            c_normalization = 1
        else:
            c_normalization = \
                    (1-locked_probability)/(how_many_cells-how_many_locked)
        
        self.probability_matrix[self.lock_matrix ==0] = \
                    np.round(c_normalization,self.round_normalization)
    
        self.upd_probability() # Update probability of colors and shape
        
    def new_color(self, color_name):
        '''
        Add a new shape to the matrix.
        '''
        
        # Check if the color is new:
        if not(str(color_name) in self.color_order):
            self.add_to_memory()
            
            self.color_order.append(str(color_name))
            
            self.probability_matrix = np.vstack((self.probability_matrix, \
                                              np.zeros((1,len(self.shape_order)))))
            self.lock_matrix = np.vstack((self.lock_matrix, \
                             np.zeros((1,len(self.shape_order))))).astype(np.uint8)
            self.normalize()
        
    
    
    def new_shape(self, shape_name):
        '''
        Add a new shape to the matrix
        '''
        
        # Check if the shape is new:
        if not(str(shape_name) in self.shape_order):
            self.add_to_memory()
        
            self.shape_order.append(str(shape_name))
            self.probability_matrix = np.hstack((self.probability_matrix, \
                                         np.zeros((len(self.color_order), 1))))
            self.lock_matrix = np.hstack((self.lock_matrix, \
                        np.zeros((len(self.color_order), 1)))).astype(np.uint8)
            self.normalize()
        
        
    
    def modify_cell(self, shape, color, value):
        '''
        Modify a given cell. Value is the probability value to define in that
        cell.
        '''
        self.add_to_memory()
        
        # Find the index of the given shape and color:
        col_shape = self.shape_order.index(shape)
        row_color = self.color_order.index(color)
        
        # Only if not locked
        if self.lock_matrix[row_color,col_shape] == 0:
            # Admissible:
            locked_probability = np.sum(\
                                 self.probability_matrix[self.lock_matrix ==1])
            if value + locked_probability <= 1:
                self.probability_matrix[row_color,col_shape] = \
                                                            max(min(value,1),0)
                self.lock_matrix[row_color,col_shape] = 1
                
        self.normalize()
        
        
    
    def modify_shape(self, shape, value):
        '''
        Modify shape. This method changes the value of the whole shape column.
        '''
        self.add_to_memory()
        
        col_shape = self.shape_order.index(shape)
        
        # Check if value can be applied to the column/row:
        locked_probability = \
                          np.sum(self.probability_matrix[self.lock_matrix ==1])
        locked_this_row_probability = np.sum(self.probability_matrix\
                              [:,col_shape][self.lock_matrix[:,col_shape] ==1])
        
        how_many_locked = int(self.lock_matrix[:,col_shape].sum())
        how_many_cells = int(np.size(self.lock_matrix[:,col_shape]))
        
    
        if ((locked_probability-locked_this_row_probability)+value <= 1)\
                and(how_many_cells-how_many_locked>0) \
                and(locked_this_row_probability<value):
            value_to_update = value-locked_this_row_probability

            c_normalization = value_to_update/(how_many_cells-how_many_locked)            
            self.probability_matrix[:,col_shape]\
                         [self.lock_matrix[:,col_shape] == 0] = c_normalization
            
            self.lock_shape(shape)  # Lock the row/column:
            self.normalize()        # Normalize
            
    
    
    def modify_color(self, color, value):
        '''
        Modify color. This method changes the value of the whole color row.
        '''
        self.add_to_memory()
        
        row_color= self.color_order.index(color)
        
        # Check if value can be applied to the column/row:
        locked_probability = np.sum(self.probability_matrix\
                                                        [self.lock_matrix ==1])
        locked_this_row_probability = np.sum(self.probability_matrix\
                              [row_color,:][self.lock_matrix[row_color,:] ==1])
        
        how_many_locked = int(self.lock_matrix[row_color,:].sum())
        how_many_cells = int(np.size(self.lock_matrix[row_color,:]))
        
        if ((locked_probability-locked_this_row_probability)+value <= 1) \
                and(how_many_cells-how_many_locked>0)\
                and(locked_this_row_probability<value):
            value_to_update = value-locked_this_row_probability

            c_normalization = value_to_update/(how_many_cells-how_many_locked)            
            self.probability_matrix[row_color,:]\
                           [self.lock_matrix[row_color,:]==0] = c_normalization
            
            self.lock_color(color)  # Lock the row/column:
            self.normalize()        # Normalize
            
            
    
    def reset(self):
        '''
        Reset the probability/lock matrices.
        '''
        
        self.lock_matrix[self.lock_matrix==1] = 0
        self.normalize()
        
        
        
    def hard_reset(self):
        '''
        Reset the probability/lock matrices AND the shapes/colors lists.
        '''
        self.add_to_memory()
        
        self.probability_matrix = np.zeros((0,0))
        self.lock_matrix = np.zeros((0,0)).astype(np.uint8)
        self.shape_order = []   # Column
        self.prob_shape = []    # Probability shape
        self.color_order = []   # Row
        self.prob_color = []    # Probability color
        
        
        
    def save_data(self):
        '''
        Save the matrix as a csv file.
        '''
        path_save_folder = self.get_save_folder()
        
        if not os.path.isdir(path_save_folder):
            os.mkdir(path_save_folder) 
        
        M_shape_colours_prob=pd.DataFrame(self.probability_matrix,\
                                            list(map(ColorHelper.hexToRGB,self.color_order)), self.shape_order)
        M_shape_colours_prob.to_csv(self.get_save_date_path())


    def get_save_date_path(self):
        return os.path.join(self.get_save_folder(),\
                                            'shapes_and_colors_matrix.csv')

    def get_save_folder(self):
        path_data = os.getcwd()
        return os.path.join(path_data, self.dataset_name)
