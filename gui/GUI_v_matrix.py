# python import
import re
import numpy as np
import pandas as pd
from os import sys

# custom tkinter import
from customtkinter import *

#internal import
from GUI_helper import *

class vMatrix():
	continuous_variables = [
		'centre_x',
		'centre_y',
		'radius',
		'rotation_(degrees)',
		'deformation',
		'blur',
		'white_noise',
		'holes',
		'additive_noise_regression',
		'multiplicative_noise_regression'
	]

	def __init__(self, tab, error_msg, success_msg):
		# callcack di
		self.error_msg = error_msg
		self.success_msg = success_msg

		self.infinity = "∞"

		# last cell value focused on
		self._last_value = None
			
		self.index = ['lower_bound', 'mean', 'sigma', 'upper_bound']

		self.entries = np.empty((len(self.continuous_variables), len(self.index)), dtype=object)
		
		self.cbox_lock_values = {
			'constant' : ['lower_bound', 'sigma', 'upper_bound'],
			'uniform': ['mean', 'sigma'],
			'gaussian': ['lower_bound', 'upper_bound'],
			'truncated_gaussian': []
		}


		self.default_values = {
			'centre_x': {
				'cbox': 'uniform',
				'lower_bound': 25,
				'upper_bound': 75
			},
			'centre_y': {
				'cbox': 'uniform',
				'lower_bound': 25,
				'upper_bound': 75
			},
			'radius': {
				'cbox': 'truncated_gaussian',
				'lower_bound': 5,
				'mean': 15,
				'sigma': 3,
				'upper_bound': 25
			},
			'rotation_(degrees)': {
				'cbox': 'constant',
				'mean': 0
			},
			'deformation': {
				'cbox': 'constant',
				'mean': 0
			},
			'blur': {
				'cbox': 'constant',
				'mean': 0
			},
			'white_noise': {
				'cbox': 'constant',
				'mean': 0
			},
			'holes': {
				'cbox': 'constant',
				'mean': 0
			},
			'deformation': {
				'cbox': 'constant',
				'mean': 0
			},
			'additive_noise_regression': {
				'cbox': 'constant',
				'mean': 0
			},
			'multiplicative_noise_regression': {
				'cbox': 'constant',
				'mean': 0
			},
		}

		self.ranges = {
			'rotation_(degrees)': {
				'upper_bound': 360
			}
		}
		
		tab.grid_columnconfigure((0,1), weight=1)
		tab.grid_rowconfigure((0,1), weight=1)
		
		f = CTkFrame(tab)
		f.grid_columnconfigure((0,1,2,3,4,5), weight=1)
		f.grid(row=0, column=0, pady=10, rowspan=2, sticky="nswe")

		c=2
		for i in self.index:
			CTkLabel(f, text=i.title().replace("_", " ")).grid(row=0, column=c, padx=5, pady=5, sticky='we')
			c = c + 1
		
		r=1
		for i in self.continuous_variables:
			self.add_row(f, i, r)
			r = r + 1
		
	def add_row(self, master, what, row):
		CTkLabel(master, text='%s:' % what.title().replace("_", " "), anchor="e").grid(row=row, column=0, padx=5, pady=5, sticky='we')
		values = list(self.cbox_lock_values.keys())
		cb = CTkComboBox(master, values=values, state="readonly", command=lambda v:self.lock(v, row-1))
		cb.grid(row=row, column=1, padx=5, sticky='we')
		
		for i in range(0,len(self.index)):
			self.entries[row-1][i] = self._entry(master, row, i, what)

		if what in self.default_values:
			cb.set(self.default_values[what]['cbox'])
			self.lock(self.default_values[what]['cbox'], row-1)

		
		
	def check(self, row, i, min_value, max_value):

		e = self.entries[row][i]
		value = e.get().strip()
		
		if len(value) == 0:
			return
		
		if i==0 and value == "-%s" % self.infinity:
			return

		if i==3 and value == "+%s" % self.infinity:
			return

		if i==0 and value == "+%s" % self.infinity or i==3 and value == "-%s" % self.infinity:
			self.error_msg("Error: %s can't be %s." % (self.index[i], value))
			EntryHelper.update_value(e,self._last_value)
			return

		try:
			value = float(value)
			
			if value < min_value:
				self.error_msg("Error: %s must be greater than %s." % (self.index[i], str(min_value)))
				EntryHelper.update_value(e,self._last_value)
				return
		
			if value > max_value:
				self.error_msg("Error: %s must be lower than %s." % (self.index[i], str(max_value)))
				EntryHelper.update_value(e,self._last_value)
				return

			l = self.entries[row][0]
			l_value = l.get().strip()
				
			u = self.entries[row][3]			
			u_value = u.get().strip()

			if i==2 and value <= 0:
				self.error_msg("Error: %s must be greater than zero." % self.index[i])
				EntryHelper.update_value(e,self._last_value)
				return
			
			if len(l_value) > 0 and len(u_value) > 0:
				if float(l_value) >= float(u_value):
					self.error_msg("Error: %s must be greater than %s." % (self.index[3],self.index[0]))
					EntryHelper.update_value(e,self._last_value)
					return False

			self.success_msg("%s inserted successfully." % self.index[i])
			return True
		except ValueError:
			self.error_msg("Error: %s must be a numeric value." % self.index[i])
			EntryHelper.update_value(e,self._last_value)
			return False



	def lock(self,v, r):
		to_locked = self.cbox_lock_values[v]
		for i in range(0,len(self.index)):
			s = IntVar()
			cell = self.entries[r][i]
			if self.index[i] in to_locked:
				s.set(1)
			else:
				s.set(0)
			
			CellHelper.lock_cell(cell, s, delete=True)
		
	def save(self, db_name):
		try:
			csvdata = [[e.get().replace(self.infinity, "inf") for e in sub] for sub in self.entries]
			csvdata = np.transpose(csvdata)
			path_data = os.getcwd()
			path_data = os.path.join(path_data, db_name)
			filename = os.path.join(path_data, 'shapes_colors_v_matrix.csv')
			#print(filename)
			df = pd.DataFrame(csvdata, index=self.index, columns=self.continuous_variables)
			df.to_csv(filename)
		except:
			msg = "Unable to save v-matrix CSV"
			self.error_msg(msg)
			return False

		return True

	
	def _entry(self, master, row, i, what):
		e = CTkEntry(master, fg_color="#000", justify="center")
		e.bind('<FocusIn>', lambda evt: self._set_last_value(e.get()))
		e.grid(row=row, column=i+2, padx=5, sticky='we')
		if i == 0:
			EntryHelper.update_value(e,"-%s" % self.infinity)
		if i == 3:
			EntryHelper.update_value(e,"+%s" % self.infinity)

		if what in self.default_values:
			if self.index[i] in self.default_values[what]:
				#f_value = 0 if self.default_values[what][self.index[i]] == 0 else "%.2f" % self.default_values[what][self.index[i]]
				f_value = self.default_values[what][self.index[i]]
				EntryHelper.update_value(e, f_value)

		min_value = 0
		max_value = 100

		if what in self.ranges.keys():
			if 'lower_bound' in self.ranges[what].keys():
				min_value = self.ranges[what]['lower_bound']
		
			if 'upper_bound' in self.ranges[what].keys():
				max_value = self.ranges[what]['upper_bound']

		CellHelper.bind_cell(e, lambda evt: self.check(row-1,i, min_value, max_value))
		return e	

	def _set_last_value(self, value):
		self._last_value = value