# python import
import re
import numpy as np
import pandas as pd
import time
from os import sys

# custom tkinter import
from customtkinter import *

#internal import
from GUI_helper import *

class ContinuousDistributionMatrix():
	'''
	This matrix 
	'''
	continuous_variables = [
		'centre_x',
		'centre_y',
		'radius',
		'rotation_(degrees)'
	]

	def __init__(self, parent):

		self.csv = 'continuous_distribution_matrix.csv'

		self.parent = parent
		
		# variables storing last value and last position
		self._last_value = None
		self._coords = (0, 0)
			
		self.entries = np.empty((len(self.continuous_variables), len(ContinuousVariableHelper.index)), dtype=object)
		self.cbox = []

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
			}
		}

		self.ranges = {
			'radius': {
				'upper_bound': 50
			},
			'rotation_(degrees)': {
				'upper_bound': 360
			}
		}
		
		tab = parent.tabview.tab('Continuous distribution')
		tab.grid_columnconfigure((0,1), weight=1)
		tab.grid_rowconfigure((0,1), weight=1)
		
		f = CTkFrame(tab)
		f.grid_columnconfigure((0,1,2,3,4,5), weight=1)
		f.grid(row=0, column=0, pady=10, rowspan=2, sticky="nswe")

		c=2
		for i in ContinuousVariableHelper.index:
			CTkLabel(f, text=i.title().replace("_", " ")).grid(row=0, column=c, padx=5, pady=5, sticky='we')
			c = c + 1
		
		r=1
		for i in self.continuous_variables:
			self.add_row(f, i, r)
			r = r + 1

		#CTkButton(f, text='-%s' % ContinuousVariableHelper.infinity, command=self._add_negative_inf).grid(row=r, column=2, padx=5, pady=5,sticky='we')
		
		#CTkButton(f, text='+%s' % ContinuousVariableHelper.infinity, command=self._add_positive_inf).grid(row=r, column=5, padx=5, pady=5,sticky='we')
		
	def _add_negative_inf(self):
		x,y = self._coords
		EntryHelper.update_value(self.entries[x][y], '-%s' % ContinuousVariableHelper.infinity)
		return;
	
	def _add_positive_inf(self):
		x,y = self._coords
		EntryHelper.update_value(self.entries[x][y], '+%s' % ContinuousVariableHelper.infinity)
		return;

	def add_row(self, master, what, row):
		CTkLabel(master, text='%s:' % what.title().replace("_", " "), anchor="e").grid(row=row, column=0, padx=5, pady=5, sticky='we')
		values = list(ContinuousVariableHelper.cbox_lock_values.keys())
		cb = CTkComboBox(master, values=values, state="readonly", command=lambda v:self.lock(v, row-1))
		cb.grid(row=row, column=1, padx=5, sticky='we')
		
		for i in range(0,len(ContinuousVariableHelper.index)):
			self.entries[row-1][i] = self._entry(master, row, i, what)

		if what in self.default_values:
			cb.set(self.default_values[what]['cbox'])
			self.lock(self.default_values[what]['cbox'], row-1)

		self.cbox.append(cb)

	def error(self, e, msg):
		EntryHelper.update_value(e,self._last_value)
		self.parent.error_msg(msg)
		return False
		
	def check(self, row, i, min_value, max_value):

		e = self.entries[row][i]
		value = e.get().strip()
		
		if len(value) == 0:
			return
		
		if (value == ContinuousVariableHelper.N_INF() or value == ContinuousVariableHelper.P_INF()):
			return self.error(e, "Error: %s can't be %s." % (ContinuousVariableHelper.title(ContinuousVariableHelper.index[i]), value))

		'''
		if i==3 and value == ContinuousVariableHelper.P_INF():
			return

		if i==0 and value == ContinuousVariableHelper.P_INF() or i==3 and value == ContinuousVariableHelper.N_INF():
			return self.error(e, "Error: %s can't be %s." % (ContinuousVariableHelper.title(ContinuousVariableHelper.index[i]), value))
		'''
		try:
			value = float(value)
			
			if value < min_value:
				return self.error(e, "Error: %s must be greater than or equal %s." % (ContinuousVariableHelper.title(ContinuousVariableHelper.index[i]), str(min_value)))
		
			if value > max_value:
				return self.error(e, "Error: %s must be lower than or equal %s." % (ContinuousVariableHelper.title(ContinuousVariableHelper.index[i]), str(max_value)))

			l = self.entries[row][0]
			l_value = l.get().strip()
				
			u = self.entries[row][3]			
			u_value = u.get().strip()
			
			if i==2 and value <= 0:
				return self.error(e,"Error: %s must be greater than zero." % ContinuousVariableHelper.title(ContinuousVariableHelper.index[i]))
			
			if len(l_value) > 0 and len(u_value) > 0:
				if (l_value != ContinuousVariableHelper.N_INF() and u_value != ContinuousVariableHelper.P_INF() and float(l_value) >= float(u_value)):
					return self.error(e, "Error: %s must be greater than or equal %s." % (ContinuousVariableHelper.title(ContinuousVariableHelper.index[3]),ContinuousVariableHelper.title(ContinuousVariableHelper.index[0])))

			#self.success_msg("%s inserted successfully." % ContinuousVariableHelper.title(ContinuousVariableHelper.index[i]))
			return True
		except ValueError:
			return self.error(e, "Error: %s must be a numeric value." % ContinuousVariableHelper.title(ContinuousVariableHelper.index[i]))

	def lock(self,v, r):
		to_locked = ContinuousVariableHelper.cbox_lock_values[v]
		for i in range(0,len(ContinuousVariableHelper.index)):
			s = IntVar()
			cell = self.entries[r][i]
			if ContinuousVariableHelper.index[i] in to_locked:
				s.set(1)
			else:
				s.set(0)
			
			CellHelper.lock_cell(cell, s, delete=True)
		
	def save(self, db_name):
		errors = False
		for row in range(self.entries.shape[0]):
			for col in range(self.entries.shape[1]):
				e = self.entries[row][col]
				if e.cget('state') == 'normal' and e.get().strip() == '':
					errors = True

		if errors:
			return self._throw_error()
		
		i,j = self._coords
		min_value, max_value = self._min_max(self.continuous_variables[i])
		e = self.entries[i][j]
		if e.cget('state') == 'normal' and self.check(i,j,min_value,max_value) == False:
			return

		try:
			csvdata = [[e.get().replace(ContinuousVariableHelper.infinity, "inf") for e in sub] for sub in self.entries]
			csvdata = np.transpose(csvdata)
			path_data = PathHelper.get_db_path()
			path_data = os.path.join(path_data, db_name)
			filename = os.path.join(path_data, self.csv)
			#print(filename)
			df = pd.DataFrame(csvdata, index=ContinuousVariableHelper.index, columns=self.continuous_variables)
			df.to_csv(filename)
		except:
			
			return self._throw_error()

		return True

	def load(self, path):
		cd = pd.read_csv(os.path.join(path, self.csv), dtype=object).fillna('')
		# print(cd)
		for i in range(len(self.continuous_variables)):
			l = cd[self.continuous_variables[i]].tolist()
			self._set_cbox_value(i, l)
			self.lock(ContinuousVariableHelper.get_cbox_value(l), i)
			for j in range(len(l)):
				e = self.entries[i][j]
				EntryHelper.update_value(e,str(l[j]).replace("inf", ContinuousVariableHelper.infinity,))
	
	def _set_cbox_value(self, i, l):
		cbox_value = 'truncated_gaussian' 
		if l[0] == '' and l[3] == '':
			cbox_value = 'constant'	if l[2] == '' else 'gaussian'
		if l[1] == '' and l[2] == '':
			cbox_value = 'uniform' 
		self.cbox[i].set(cbox_value)


	def _throw_error(self):
		msg = "Unable to save %s" % self.csv
		self.parent.error_msg(msg)
		return False
			
	def _entry(self, master, row, i, what):
		e = CTkEntry(master, fg_color="#000", justify="center")
		e.bind('<FocusIn>', lambda evt: self._set_last_value(e.get(),row-1, i))
		e.grid(row=row, column=i+2, padx=5, sticky='we')
		if i == 0:
			EntryHelper.update_value(e,ContinuousVariableHelper.N_INF())
		if i == 3:
			EntryHelper.update_value(e,ContinuousVariableHelper.P_INF())

		if what in self.default_values:
			if ContinuousVariableHelper.index[i] in self.default_values[what]:
				#f_value = 0 if self.default_values[what][ContinuousVariableHelper.index[i]] == 0 else "%.2f" % self.default_values[what][ContinuousVariableHelper.index[i]]
				f_value = self.default_values[what][ContinuousVariableHelper.index[i]]
				EntryHelper.update_value(e, f_value)

		min_value, max_value = self._min_max(what)

		CellHelper.bind_cell(e, lambda evt: self.check(row-1,i, min_value, max_value))
		return e	

	def _min_max(self, what):
		min_value = 0
		max_value = 100

		if what in self.ranges.keys():
			if 'lower_bound' in self.ranges[what].keys():
				min_value = self.ranges[what]['lower_bound']
		
			if 'upper_bound' in self.ranges[what].keys():
				max_value = self.ranges[what]['upper_bound']

		return (min_value, max_value)

	def _set_last_value(self, value, row, col):
		self._last_value = value
		self._coords = (row,col)