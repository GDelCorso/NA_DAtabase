# python import
import re
import numpy as np
import pandas as pd
from os import sys

# custom tkinter import
from customtkinter import *

#internal import
from GUI_helper import *
from GUI_UncertantiesMatrixTop import *

class UncertantiesMatrix():

	continuous_variables = [
		'deformation',
		'blur',
		'white_noise',
		'holes',
		'additive_noise_regression',
		'multiplicative_noise_regression'
	]

	index = [
		'lower_bound', 
		'mean', 
		'sigma', 
		'upper_bound'
	]

	oob = -2

	def __init__(self, tab, error_msg, success_msg):
		# callcack di
		self.error_msg = error_msg
		self.success_msg = success_msg
		self.G = None
		self.u_matrix = np.empty((0,0), dtype=object)
		self.cells = np.array([[]])
		
		self.UncertantiesMatrixTop = UncertantiesMatrixTop(self)
		self.UncertantiesMatrixTop.withdraw()

		bf = CTkFrame(tab)
		bf.grid(row=0, column=0, padx=10,sticky="we")
		
		self.edit = CTkButton(bf, text='Edit selected', state="disabled", command=self.edit_selected)
		self.edit.grid(row=0, column=0, pady=10)
		
		self.f = CTkFrame(tab)
		self.f.grid(row=1, column=0, pady=10, rowspan=2, sticky="ew")

	def edit_selected(self):
		if len(self._groupToModify) == 1:
			return self._mod_matrix(self._groupToModify[0]['row'],self._groupToModify[0]['col'])

		row = list(map(lambda e: e['row'], self._groupToModify))
		col = list(map(lambda e: e['col'], self._groupToModify))

		self.UncertantiesMatrixTop.modify(self._empty_cell_matrix(), row, col, True)
		return
	
	def new_shape(self, vertices, G):
		self.G = G
		self.u_matrix = np.hstack((self.u_matrix, np.empty((len(G.color_order), 1), dtype=object)))

		sf = CTkFrame(self.f)
		sf.grid_columnconfigure((0,1), weight=1)
		# add shape label 
		e=CTkLabel(sf, text="Circle" if vertices == '0' else "%s sides poly" % vertices, width=80)
		e.grid(row=0, column=0, padx=2, pady=2)

		# place the frame in the gui
		sf.grid(row=0, column=len(G.shape_order))
		self._add_cells('shape')
		return    

	def new_color(self, color_picked, G):
		self.G = G
		self.u_matrix = np.vstack((self.u_matrix, np.empty((1,len(G.shape_order)), dtype=object)))
		
		cf = CTkFrame(self.f)
		cf.grid_columnconfigure((0,1), weight=1)
		# add shape label 
		e = CTkLabel(cf, fg_color=color_picked, text_color=ColorHelper.getTextColor(color_picked), text=color_picked,width=80)
		e.grid(row=0, column=2, padx=2, pady=2)
		
		# place the frame in the gui
		cf.grid(row=len(G.color_order), column=0)
		self._add_cells('color')
		return    

	def update(self,G):
		print("Updating");
		
		self.UncertantiesMatrixTop.update(G);

		for r in range(self.u_matrix.shape[0]):
			for c in range(self.u_matrix.shape[1]):
				if self.u_matrix[r][c] is None:
					self.u_matrix[r][c] = self._empty_cell_matrix()
		return
		
	def _empty_cell_matrix(self):
		entries = np.empty((len(self.continuous_variables), len(self.index)), dtype=object)
		entries.fill('')
		
		EV = {
			'list': [],
			'cbox': ['constant' for i in range(len(self.continuous_variables))],
			'entries' : entries
		}

		EV['entries'][:, 1] = 0

		return EV


	def update_multiple_matrix(self, matrix, row, col):
		for ci in range(len(row)):
			self.u_matrix[row[ci]][col[ci]] = matrix
		self._checked()
		#print(self.u_matrix)
		'''
		if np.all(np.linalg.eigvals(self.u_matrix) > 0):
			self.parent.success_msg("Matrix updated successfully.", True)
		else:
			self.parent.error_msg("Warning, the matrix is not positive definite. A random perturbation is going to be applied to ensure a proper Cholesky decomposition.")
		'''
	def _map_color(self, hex):
		rgb = str(ColorHelper.hexToRGB(hex))
		rgb = rgb.replace("(","")
		rgb = rgb.replace(")","")
		rgb = rgb.replace(",","-")
		rgb = rgb.replace(" ","")
		return rgb

	def save(self, db_name):
		colors = list(map(self._map_color, self.G.color_order))
		
		for ci in range(self.u_matrix.shape[0]):
			for si in range(self.u_matrix.shape[1]):
				filename = "uncertanties_distribution_matrix_%s_%s.csv" % ( self.G.shape_order[si], colors[ci])
				print(filename)

		'''
		try:
			csvdata = [[e.get().replace(self.infinity, "inf") for e in sub] for sub in self.entries]
			csvdata = np.transpose(csvdata)
			path_data = os.getcwd()
			path_data = os.path.join(path_data, db_name)

			filename = os.path.join(path_data, 'uncertanties_distribution_matrix.csv')
			#print(filename)
			df = pd.DataFrame(csvdata, index=self.index, columns=self.continuous_variables)
			df.to_csv(filename)
		except:
			msg = "Unable to save continuous_distribution_matrix.csv"
			self.error_msg(msg)
			return False
		'''
		return True
		
	def _add_cells(self,what):
		'''
		Add a row/column of cells
		'''
		if self.u_matrix.shape[0] == 0 or self.u_matrix.shape[1] == 0:
			return

		row, col = len(self.G.color_order), len(self.G.shape_order)
		data = []
		if what == 'shape':
			# shape added
			for r in range(1,row+1):
				data.append(self._add_cell(r,col))
			# adding rows depending on first shape added or not
			if col == 1:
				# first shape added
				self.cells = np.append(self.cells, np.array([[data[0]]]), axis=1)
				for i in range(1,len(data)):
					self.cells = np.append(self.cells, np.array([[data[i]]]), axis=0)
			else:
				self.cells = np.hstack((self.cells, np.atleast_2d(np.array(data)).T))
			
		if what == 'color':
			# color added
			for c in range(1,col+1):
				data.append(self._add_cell(row,c))
			# adding cols depending on first color added or not
			if row == 1:
				# first color added
				for i in range(0,len(data)):
					self.cells = np.append(self.cells, np.array([[data[i]]]), axis=1)
			else:
				self.cells = np.r_[self.cells,[data]]	

	def _add_cell(self, row, col):
		'''
		add a single cell in gui
		'''
		inner_f= CTkFrame(self.f)
		inner_f.grid(row=row,column=col)
		CTkButton(inner_f, text="m", width=1, command=lambda: self._mod_matrix(row-1, col-1)).grid(row=0, column=0, padx=1)
		cb =CTkCheckBox(inner_f, text="",width=1, command=self._checked)
		cb.grid(row=0, column=1, padx=1)
		return cb

	def _checked(self):
		self._groupToModify = []

		for i in range(self.u_matrix.shape[0]):
			for j in range(self.u_matrix.shape[1]):
				if self.cells[i][j].get() == 1:
					self._groupToModify.append({'matrix': self.u_matrix[i][j].copy(), 'row': i, 'col': j})

		self.edit.configure(state="normal" if len(self._groupToModify) else "disabled")
		return

	def _mod_matrix(self, row, col):
		self.UncertantiesMatrixTop.modify(self.u_matrix[row][col], row, col)
		

