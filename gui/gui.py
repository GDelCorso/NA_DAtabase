
# python import
import re
import numpy as np
from os import sys

# custom tkinter import
from customtkinter import *
from CTkMessagebox import CTkMessagebox
import tkinter.messagebox as tkmb

#internal import 
from GUI_helper import *
from GUI_preset import *
from GUI_ShapesAndColorsMatrix import *
from GUI_SamplerPropertiesMatrix import *
from GUI_UncertantiesMatrix import *
from GUI_ContinuousDistributionMatrix import *
from GUI_MultivariateDistributionMatrix import *

class App(CTk):
	'''
	Main App Class

	This class build the entire gui
	'''

	# Short description for each tab
	tabview_info = {
		'Shapes and Colors': "Provides the probability of each color/shape combination.\n\nP value (row on the left): Cumulative probability of the color among different shapes. If changed, it will update all unlocked cells corresponding to this color.\nP value (column on the top): Cumulative probability of the shape among different colors. If changed, it will update all unlocked cells corresponding to this shape.\nLock/Unlock symbol: Lock/unlock the cell/row/column value to prevent the probability from being updated automatically.\nReset p-matrix: Unlock all cells and sample each shape/color combination with equal probability",
		'Sampler Properties': "Contains the properties of the dataset and optionally the addresses for classification ground truth.\n\nCorrect classes:  [Facultative] Choose a correct class for classification (colors and/or shapes).\nAllow out of border: Allow for production of shapes with center into the image and radii eventually out of borders. If enabled the only sampling strategy allowed is MC.",
		'Uncertanties': "Provides the marginal distribution of each continuous random variable and classification noise for every couple of shapes and colors.\nClassification noise (manage aleatoric classification uncertainty over a specific class):  [Facultative] Choose a noise percentage (range from 0 to 1) over classes for classification (colors and/or shapes).\nPossible distributions include: Constant, Uniform, Gaussian and Truncated Gaussian. To set up a single bound for a Truncated Gaussian Distribution, it is sufficient to keep the default infinity bound.\n\nDeformation: Continuous deformation [0-100%] of the original shape into a circle. A value of 0 means that no deformation is applied to the shape, while a deformation of 100% converts the original image into a circle.",
		'Continuous distribution': "Provides the marginal distribution of each continuous random variable. Possible distributions include: Constant, Uniform, Gaussian and Truncated Gaussian. To set up a single bound for a Truncated Gaussian Distribution, it is sufficient to keep the default infinity bound.\n\nCenter X: Relative position [0 – 100%] on the horizontal axis of the center of the shape.\nCenter Y: Relative position [0 – 100%] on the vertical axis of the center of the shape.\nRadius:Radius [0-100%] of the given shape. If the random radius exceeds the boundaries of the image and Allow Out Of Border(s) matrix is not checked, the image will be resampled.\nRotation: Rotation (0-360°) of the given shape.",
		'Multivariate distribution': "Multivariate distribution is the correlation matrix between continuous variables for every couple of shapes and colors.\n\nEdit selected: allows to set correlations between continuous variables for each of the selected color/shape couples. Values showing * imply that some color/shape couples have different values; modifying these values in the matrix updates each couple to the provided value.\nM: clicking on m (matrix) allows us to update the correlation values for that specific matrix (corresponding to a precise color/shape couple)."
	}

	def __init__(self):
		super().__init__()
		
		# initialize the window		
		
		self.titleString = "NA Database Configurator"
		self.title(self.titleString)
		self.geometry("1200x640")
		self.grid_columnconfigure(0, weight=1)
		self.grid_rowconfigure(1, weight=1)
		
		bf = CTkFrame(self)
		bf.grid(row=0, column=0, padx=10,sticky="we")
		
		bf.grid_columnconfigure(2, weight=1)
		
		# add button to show tab info
		CTkButton(bf, text='Tab info', command=self.info, width=1).grid(row=0, column=0, pady=10, padx=5)
		
		# add button to save all csv files
		self.save = CTkButton(bf, text='Save Database', command=self.save)
		self.save.grid(row=0, column=1, pady=10)
		
		self.message_box = CTkLabel(bf, bg_color="#222", anchor="w", text="")
		self.message_box.grid(row=0, column=2, padx=10, sticky="ew")
		
		# create the tab view
		self.tabview = CTkTabview(master=self)
		self.tabview.add("Shapes and Colors")
		self.tabview.add("Sampler Properties")
		self.tabview.add("Uncertanties")
		self.tabview.add("Continuous distribution")
		self.tabview.add("Multivariate distribution")
		self.tabview.grid(row=1, column=0, padx=10, pady=5, sticky="nswe")

		# add contents for every tab
		
		self.SamplerPropertiesMatrix = SamplerPropertiesMatrix(self)
		self.UncertantiesMatrix = UncertantiesMatrix(self)
		self.ContinuousDistributionMatrix = ContinuousDistributionMatrix(self)
		self.MultivariateDistributionMatrix = MultivariateDistributionMatrix(self)
		self.ShapesAndColorsMatrix = ShapesAndColorsMatrix(self)
		
		self.focus_force()

		# ask for db name 
		self.ask_db_name()
		# TODO: shows choose-preset window
		# self.PresetWindow = PresetWindow(self)
		self.mainloop()

	def info(self):
		'''
		pops up an info Message box containing info about the active tab
		'''
		self.info_msg(msg = "%s - %s" % (self.tabview.get(), self.tabview_info[self.tabview.get()]))
		return

	def ask_db_name(self):
		'''
		ask for db name and initialize the GUI_interface
		'''
		dialog = CTkInputDialog(title="Database name", text="Enter Database name:")
		self.db_name = dialog.get_input()
		self.focus_force()

		if(self.db_name is None):
			self.destroy()
			sys.exit(0)

		self.db_name = self.db_name.strip()

		if(len(self.db_name) == 0):
			self.destroy()
			sys.exit(0)

		self.title(self.titleString + " - " + self.db_name)
		self.ShapesAndColorsMatrix.init_G(self.db_name)

	def save(self):
		'''
		save all csv data
		'''
		if self.ShapesAndColorsMatrix.save(self.db_name) and \
			self.SamplerPropertiesMatrix.save(self.db_name) and \
			self.ContinuousDistributionMatrix.save(self.db_name) and \
			self.MultivariateDistributionMatrix.save(self.db_name) and \
			self.UncertantiesMatrix.save(self.db_name):
			self.success_msg("Data successfully saved in folder %s" % self.db_name, True)

	def success_msg(self, msg, alert=False):
		'''
		show a success message on message box
		''' 
		self.message_box.configure(text=msg, text_color="#6f6")
		if(alert):
			CTkMessagebox(title="Success", message=msg, icon="check")
			self.focus_force()
			#tkmb.showinfo(title='Success', message=msg, icon='info')

	def error_msg(self, msg, alert = True):
		'''
		show an error message on message box
		'''
		self.message_box.configure(text=msg, text_color="#f66")
		if(alert):
			CTkMessagebox(title="Error", message=msg, icon="cancel")
			self.focus_force()
			
			#tkmb.showinfo(title='Error', message=msg, icon='error')
	
	def info_msg(self, msg):
		'''
		show an info message on message box
		'''
		CTkMessagebox(title="Info", message=msg, icon="info", width=600)
		self.focus_force()
		#tkmb.showinfo(title='Info', message=msg, icon='info')
	
# set the dark mode
set_appearance_mode("dark")  
app = App()