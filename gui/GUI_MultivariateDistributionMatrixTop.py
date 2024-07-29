# python import
import numpy as np
import pandas as pd
from os import sys

# custom tkinter import
from customtkinter import *

#internal import
from GUI_helper import *
from GUI_ContinuousDistributionMatrix import *

class MultivariateDistributionMatrixTop(CTkToplevel):
	'''
	Genertate the "Choose a Preset" window
	'''
	def __init__(self, parent, d_matrix):
		super().__init__()
		self.parent = parent
		self.title("Multivariate distribution")
		self.protocol("WM_DELETE_WINDOW", self._close)
		self.d_matrix = d_matrix
		self._last_value = None

		l = len(ContinuousDistributionMatrix.continuous_variables)
		
		self.entries = np.empty((l,l), dtype=object)

		pos=1
		for v in ContinuousDistributionMatrix.continuous_variables:
			CTkLabel(self, text=self._title(v)).grid(row=0, column=pos, padx=5, pady=5, sticky='we')
			CTkLabel(self, text=self._title(v)).grid(row=pos, column=0, padx=5, pady=5, sticky='we')
			pos = pos + 1

		for i in range(d_matrix.shape[0]):
			for j in range (d_matrix.shape[1]):
				self.entries[i][j] = self._entry(i,j)

		CTkButton(self, text="Update", font=(None,16), command=self.save).grid(row=11, column=0, columnspan=11,pady=10,padx=10,sticky="e")
		CTkLabel(self, text="Remember to hit enter key,tab key or change focus by clicking on other cells to update the values.").grid(row=12, column=0, columnspan=11,pady=10,padx=10,sticky="ew")
		
		self.warning = CTkLabel(self, text="", bg_color="#222", text_color="#f66")
		self.warning.grid(row=13, column=0, columnspan=11,pady=10,padx=10,sticky="ew")

	def save(self):
		self.parent.update_matrix(self.d_matrix, self.row, self.col)
		self._close()
		
	def _close(self):
		self.grab_release()
		self.withdraw()

	def modify(self, d_matrix, row, col, multiple=False):
		# print(d_matrix)
		self.d_matrix = d_matrix
		self.row = row
		self.col = col
		self.multiple = multiple
		for i in range(d_matrix.shape[0]):
			for j in range (d_matrix.shape[1]):
				EntryHelper.update_value(self.entries[i][j], '*' if int(d_matrix[i][j]) == self.parent.oob else d_matrix[i][j])

		self.deiconify()
		self._positive()
		self.focus_force()
		self.grab_release()
		self.grab_set()

	def _title(self, v):
		v = v.title().replace("_", " ")
		if v.__contains__(' ') and v.__contains__('(') == False:
			v = ''.join(map(lambda e: e[0],v.split()))
		elif len(v)>3:
			v=v[:3]
		return v

	def _entry(self, i, j):
		e= CTkEntry(self, justify="center", fg_color="#000", width=50)
		e.bind('<FocusIn>', lambda evt: self._set_last_value(e.get()))
		EntryHelper.update_value(e, self.d_matrix[i][j])
		if i==j:
			v = IntVar()
			v.set(1)
			CellHelper.lock_cell(e,v)
		e.grid(row=i+1, column=j+1)
		CellHelper.bind_cell(e, lambda evt: self._check(i, j))
		return e	

	def _set_last_value(self, value):
		self._last_value = value
	
	def _check(self, i, j):

		e = self.entries[i][j]
		value = e.get().strip()
		
		if len(value) == 0:
			return
		
		if(self._v_float(value)):
			value = value.lstrip('0')
			if value[0] == '.':
				value = "0%s" % value
			
			if value == '1':
				value = "1.0"
			
			if value == '-1':
				value = "-1.0"

			self.d_matrix[i][j] = value
			self.d_matrix[j][i] = value
			EntryHelper.update_value(self.entries[i][j], value)
			EntryHelper.update_value(self.entries[j][i], value)
		else:
			EntryHelper.update_value(e,self._last_value)
		
		if self.multiple == False:
			self._positive()

	def _positive(self):
		if np.all(np.linalg.eigvals(self.d_matrix) > 0):
			self.warning.configure(text="")
		else:
			self.warning.configure(text="Warning, the matrix is not positive definite. We reccommend to normalize it.")

	def _v_float(self, value):
		try:
			f = float(value)
			if f >= -1 and f <= 1:
				return True
			self.parent.parent.error_msg("Error: matrix value must be a float number between -1 and 1.")
			return False
		except ValueError:
			self.parent.parent.error_msg("Error: matrix value must be a float number between -1 and 1.")
			return False
	


