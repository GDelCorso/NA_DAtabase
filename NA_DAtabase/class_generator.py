#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 15:48:09 2023

@author: claudiacaudai
"""


class dataset_sampler:
    'Docstring'
    
    import numpy as np
    import pandas as pd
    import os
    import argparse
    import copy


    def __init__(self):
        self.M_dataset_properties = None, 
        self.M_correlation = None, 
        self.M_shape_colours_prob = None,
        self.dictionary = None
        
        import argparse
        import pandas as pd
        import numpy as np
        
        '''
        dataset_properties = None : csv file with dataset properties
        '''

    
    
        parser = argparse.ArgumentParser()
        # arguments
        parser.add_argument("-s", "--shapes", help="array with desired shapes (defined by number of vertices, es circle=0, triangle=3, square=4, etc...)", type=float,default=[0,3])
        parser.add_argument("-c", "--colours", help="array with desired colours (defined by rgb values, es red=[1,0,0], green=[0,1,0], blue=[0,0,1], etc...)", type=float, default=[(1,0,0)])
        parser.add_argument("-ps", "--prob_shapes", help="array with probabilities of shapes (sum must be 1).", type=float, default=[0.5,0.5])
        parser.add_argument("-pc", "--prob_colours", help="earray with probabilities of colours (sum must be 1).", type=float, default=[1])
        parser.add_argument("-centx", "--centre_x", help="lower bound, upper bound, average and sigma of distribution of x coord of centres (in percentage in respect to x axis).", type=float, default=[None,None,50,None])
        parser.add_argument("-centy", "--centre_y", help="lower bound, upper bound, average and sigma of distribution of y coord of centres (in percentage in respect to y axis).", type=float, default=[None,None,50,None])
        parser.add_argument("-rad", "--radius", help="lower bound, upper bound, average and sigma of distribution of radii (in percentage in respect to x axis).", type=float, default=[None,None,10,None])
        parser.add_argument("-rot", "--rotation", help="lower bound, upper bound, average and sigma of distribution of rotations (in degrees).", type=float, default=[None,None,0,None])
        parser.add_argument("-def", "--deformation", help="lower bound, upper bound, average and sigma of distribution of deformations (percentage of curvature of edges, 100=circle).", type=float, default=[None,None,0,None])
        parser.add_argument("-blur", "--blur", help="lower bound, upper bound, average and sigma of distribution of blur (in strength, from 0=no noise to 100=max noise).", type=float, default=[None,None,0,None])
        parser.add_argument("-wn", "--white_noise", help="lower bound, upper bound, average and sigma of distribution of white noise (in strength, from 0=no noise to 100=max noise).", type=float, default=[None,None,0,None])
        parser.add_argument("-ho", "--holes", help="lower bound, upper bound, average and sigma of distribution of holes noise (in strength, from 0=no noise to 100=max noise).", type=float, default=[None,None,0,None])
        parser.add_argument("-anr", "--additive_noise_regression", help="lower bound, upper bound, average and sigma of distribution of amount of additive random error (in strength, from 0=no noise to 100=max noise).", type=float, default=[None,None,0,None])
        parser.add_argument("-mnr", "--multiplicative_noise_regression", help="lower bound, upper bound, average and sigma of distribution of amount of multiplicative random error  (in strength, from 0=no noise to 100=max noise).", type=float, default=[None,None,0,None])
        parser.add_argument("-name", "--outputs_folder_name", help='name of outputs folder', type=str, default="dataset_csv_files")
        
        
        #%%
        
        
        
        
        args = parser.parse_args()
        
        self.dictionary = {'shapes': args.shapes, 
             'colours' : args.colours,
             'prob_shapes': args.prob_shapes,
             'prob_colours': args.prob_colours,
             'centre_x': args.centre_x,
             'centre_y': args.centre_y,
             'radius': args.radius,
             'rotation': args.rotation,
             'deformation': args.deformation,
             'blur': args.blur,
             'white_noise': args.white_noise,
             'holes': args.holes,
             'additive_noise_regression': args.additive_noise_regression,
             'multiplicative_noise_regression': args.multiplicative_noise_regression,
             'outputs_folder_name': args.outputs_folder_name
            }
    
        
        self.keys_prop = ['centre_x','centre_y','radius','rotation','deformation','blur','white_noise','holes','additive_noise_regression','multiplicative_noise_regression']
        
        for key in self.keys_prop:
            self.control(key)
        
        self.M_dataset_properties = {key: self.dictionary[key] for key in self.keys_prop}
        self.M_dataset_properties = pd.DataFrame(data=self.M_dataset_properties)
        
        self.M_shape_colours_prob=np.zeros((len(self.dictionary['prob_colours']),len(self.dictionary['prob_shapes'])))
        for i in range(len(self.dictionary['prob_shapes'])):
            for j in range(len(self.dictionary['prob_colours'])):
                self.M_shape_colours_prob[j,i]=self.dictionary['prob_shapes'][i]*self.dictionary['prob_colours'][j]
        self.M_shape_colours_prob=pd.DataFrame(data=self.M_shape_colours_prob,columns=self.dictionary['shapes'],index=self.dictionary['colours'])
        
        self.M_correlation=np.zeros((self.M_dataset_properties.shape[1]*len(self.dictionary['shapes']), self.M_dataset_properties.shape[1]*len(self.dictionary['colours'])))
        for i in range(len(self.dictionary['shapes'])):
            for j in range(len(self.dictionary['colours'])):
                for z in range(self.M_dataset_properties.shape[1]):
                    self.M_correlation[z+i*self.M_dataset_properties.shape[1],z+j*self.M_dataset_properties.shape[1]]=1
        
        
    
    def save_data(self):
        
        import numpy as np
        import os
    
        path_data = os.getcwd()
        path_save_folder = os.path.join(path_data, self.dictionary['outputs_folder_name'])
        if not os.path.isdir(path_save_folder):
            os.mkdir(path_save_folder) 
    
        self.M_dataset_properties.to_csv(os.path.join(path_save_folder,'dataset_properties.csv'),index=False)
    
        self.M_shape_colours_prob.to_csv(os.path.join(path_save_folder,'shapes_colors_probabilities.csv'))
        
        np.savetxt(os.path.join(path_save_folder,'dataset_correlation.csv'),self.M_correlation,delimiter=",")

    
    def control(self,name):
        if (self.dictionary[name][0]!=None and self.dictionary[name][1]!=None and self.dictionary[name][0]<self.dictionary[name][1] and (self.dictionary[name][2]==None or self.dictionary[name][3]==None)):
            print("Uniform Distribution")  
        elif (self.dictionary[name][0]==None and self.dictionary[name][1]==None and self.dictionary[name][2]!=None and self.dictionary[name][3]!=None and self.dictionary[name][3]>0):
            print("Gaussian Distribution") 
        elif (self.dictionary[name][0]==None and self.dictionary[name][1]==None and self.dictionary[name][2]!=None or self.dictionary[name][3]==None):
            print("Constant Distribution")
        elif (((self.dictionary[name][0]!=None and self.dictionary[name][1]!=None and self.dictionary[name][0]<self.dictionary[name][1]) or\
              (self.dictionary[name][0]!=None and self.dictionary[name][1]==None) or\
              (self.dictionary[name][0]==None and self.dictionary[name][1]!=None)) and self.dictionary[name][2]!=None and self.dictionary[name][3]>0):
            print("Truncated Gaussian Distribution")
        else:
            print("warning: invalid distribution")
           
    
    def alter(self,name,value):
        
        try: 
            len(value)==len(self.dictionary[name])
        except ValueError:
            print('set correct value format')  
        if name in self.key_prop:
            self.control(value)
        name=str(name)
        self.dictionary[name]=value
    
    
    def set_prob_standard(self,name):
        
        # name can be 'shapes' or 'colours'
        try:
            name =='colours' or name == 'shapes'
        except ValueError:
            print('name must be shapes or colours')
        self.dictionary[str('prob_'+name)]=[]
        for i in range(len(self.dictionary[name])):
            prob_c=1/len(self.dictionary[name])
            self.dictionary[str('prob_'+name)].append(prob_c)
    
    
    def set_prob_fix(self, name, values, probs):
        import numpy as np
        
        if type(values[0])!=list:
            values=[values]
        if type(probs)!=list:
            probs=[probs]
        try:
            len(values)==len(probs)
        except ValueError:
            print('values and probs must have same length')
        try:
            name =='colours' or name == 'shapes'
        except ValueError:
            print('name must be shapes or colours')
        try:
            for i in range(len(values)):
                values[i] in self.dictionary[name]
        except ValueError:
            print('the value is not in the string')
        self.dictionary[str('prob_'+name)]=[]
        for i in range(len(self.dictionary[name])):
            for j in range(len(values)):
                if self.dictionary[name][i] == values[j]:
                    prob_c = probs[j]
                    self.dictionary[str('prob_'+name)].append(prob_c)
                else:
                    if len(self.dictionary[name])!=len(values):
                        prob_c=round(1-np.sum(probs)/(len(self.dictionary[name])-len(values)),2)
                        self.dictionary[str('prob_'+name)].append(prob_c)
                        
    
    def set_correl(self,prop1,prop2,p):
        try:
            prop1 in self.dictionary
        except ValueError:
            print('the key is not in the dictionary')
        try:
            prop2 in self.dictionary
        except ValueError:
            print('the key is not in the dictionary')
            
        p1=list(self.dictionary).index(prop1)
        p2=list(self.dictionary).index(prop2)
        for i in range(len(self.dictionary['shapes'])):
            for j in range(len(self.dictionary['colours'])):
                self.M_correlation[p1+i*self.properties.shape[1],p2+j*self.properties.shape[1]]=p
                self.M_correlation[p2+i*self.properties.shape[1],p1+j*self.properties.shape[1]]=p
                
            

            




