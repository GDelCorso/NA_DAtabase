# python import
import re
import numpy as np
import pandas as pd
from os import sys

# custom tkinter import
from customtkinter import *

#internal import
from GUI_helper import *
from GUI_v_matrix import *
from GUI_d_matrix_top import *

class dMatrix():

	oob = -2

	def __init__(self, tab, error_msg, success_msg):
		# callcack di
		self.error_msg = error_msg
		self.success_msg = success_msg
		self.G = None
		self.d_matrix = np.empty((0,0), dtype=object)
		self.cells = np.array([[]])
		
		self.d_matrixTop = dMatrixTop(self, self._empty_cell_matrix())
		self.d_matrixTop.withdraw()

		bf = CTkFrame(tab)
		bf.grid(row=0, column=0, padx=10,sticky="we")
		
		self.edit = CTkButton(bf, text='Edit selected', state="disabled", command=self.edit_selected)
		self.edit.grid(row=0, column=0, pady=10)
		
		self.normalize = CTkButton(bf, text='Normalize selected', state="disabled", command=self.normalize_selected)
		self.normalize.grid(row=0, column=1, pady=10, padx=5)
		
		self.f = CTkFrame(tab)
		self.f.grid(row=1, column=0, pady=10, rowspan=2, sticky="ew")

	def edit_selected(self):
		if len(self._groupToModify) == 1:
			return self._mod_matrix(self._groupToModify[0]['row'],self._groupToModify[0]['col'])
		matrix_to_send = self._merge(list(map(lambda e: e['matrix'], self._groupToModify)))
		row = list(map(lambda e: e['row'], self._groupToModify))
		col = list(map(lambda e: e['col'], self._groupToModify))
		self.d_matrixTop.modify(matrix_to_send, row, col, multiple=True)
		return

	def normalize_selected(self):
		
		return

	def new_shape(self, vertices, G):
		self.G = G
		self.d_matrix = np.hstack((self.d_matrix, np.empty((len(G.color_order), 1), dtype=object)))
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
		self.d_matrix = np.vstack((self.d_matrix, np.empty((1,len(G.shape_order)), dtype=object)))
		
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
		for r in range(self.d_matrix.shape[0]):
			for c in range(self.d_matrix.shape[1]):
				if self.d_matrix[r][c] is None:
					self.d_matrix[r][c] = self._empty_cell_matrix()
		return

	def update_matrix(self, matrix, row, col):
		if type(row) is list:
			for ci in range(len(row)):
				for i in range(matrix.shape[0]):
					for j in range(matrix.shape[1]):
						self.d_matrix[row[ci]][col[ci]][i][j] = matrix[i][j] if matrix[i][j] != self.oob else self.d_matrix[row[ci]][col[ci]][i][j]
		else:
			self.d_matrix[row][col] = matrix.copy()
		self._checked()
		print(self.d_matrix)
		'''
		if np.all(np.linalg.eigvals(self.d_matrix) > 0):
			self.parent.success_msg("Matrix updated successfully.", True)
		else:
			self.parent.error_msg("Warning, the matrix is not positive definite. A random perturbation is going to be applied to ensure a proper Cholesky decomposition.")
		'''
	def save(self, db_name):
		row, col = self.d_matrix.shape
		columns = vMatrix.continuous_variables*col
		index = vMatrix.continuous_variables*row

		csvdata = np.zeros((len(index),len(columns)))

		for i in range(row):
			for j in range(col):
				i_row, i_col = self.d_matrix[i][j].shape
				for i_i in range(i_row):
					x = i_row * i + i_i
					for i_j in range(i_col):
						y = i_col * j + i_j
						csvdata[x][y] = self.d_matrix[i][j][i_i][i_j]

		df = pd.DataFrame(csvdata, index=index, columns=columns)
		path_data = os.getcwd()
		path_data = os.path.join(path_data, db_name)
		filename = os.path.join(path_data, 'shapes_colors_d_matrix.csv')
		#print(filename)
		df.to_csv(filename)
		return True

	def _empty_cell_matrix(self):
		l = len(vMatrix.continuous_variables)
		cm = np.zeros((l,l))
		for i in range(0,l):
			cm[i][i] = 1
		return cm

	def _merge(self, matrix_array):
		first = matrix_array[0]

		for mi in range(1,len(matrix_array)):
			if (first == matrix_array[mi]).all() == False:
				for i in range(first.shape[0]):
					for j in range(first.shape[1]):
						first[i][j] = first[i][j] if first[i][j] == matrix_array[mi][i][j] else self.oob

		return first

		
	def _add_cells(self,what):
		'''
		Add a row/column of cells
		'''
		if self.d_matrix.shape[0] == 0 or self.d_matrix.shape[1] == 0:
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

		for i in range(self.d_matrix.shape[0]):
			for j in range(self.d_matrix.shape[1]):
				if self.cells[i][j].get() == 1:
					self._groupToModify.append({'matrix': self.d_matrix[i][j].copy(), 'row': i, 'col': j})

		self.edit.configure(state="normal" if len(self._groupToModify) else "disabled")
		self.normalize.configure(state="normal" if len(self._groupToModify) else "disabled")
		return

	def _mod_matrix(self, row, col):
		self.d_matrixTop.modify(self.d_matrix[row][col].copy(), row, col)
		
