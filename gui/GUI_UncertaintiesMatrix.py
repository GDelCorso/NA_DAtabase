# python import
import re
import numpy as np
import pandas as pd
from os import sys

# custom tkinter import
from customtkinter import *

#internal import
from GUI_helper import *
from GUI_UncertaintiesMatrixTop import *

class UncertaintiesMatrix():

	continuous_variables = [
		'deformation',
		'blur',
		'white_noise',
		'holes',
		'additive_noise_regression',
		'multiplicative_noise_regression'
	]

	oob = -2

	def __init__(self, parent):

		self.parent = parent

		self.csv_cn = "uncertainties_classification_noise_%s_%s.csv"
		self.csv_dm = "uncertainties_distribution_matrix_%s_%s.csv"
		
		self.cn = {
			'old_shape': StringVar(),
			'new_shape': StringVar(),
			'cb_shape': StringVar(),
			'old_color': StringVar(),
			'new_color': StringVar(),
			'cb_color': StringVar(),
			'probability': StringVar(),
			'textbox': StringVar()
		}

		self.cbox = {
			'shape': [],
			'color': []
		}


		self.G = None
		self.u_matrix = np.empty((0,0), dtype=object)
		self.cells = np.array([[]])

		self.UncertaintiesMatrixTop = UncertaintiesMatrixTop(self)
		self.UncertaintiesMatrixTop.withdraw()

		tab = parent.tabview.tab('Uncertainties')

		tf = CTkFrame(tab)
		tf.grid(row=0, column=0, padx=10,sticky="we")
		
		self.edit = CTkButton(tf, text='Edit selected', width=1, state="disabled", command=self.edit_selected, anchor="nw")
		self.edit.grid(row=0, column=0, pady=10)
		
		self.cs_f = CTkFrame(tab)
		self.cs_f.grid(row=1, column=0, pady=10, rowspan=2, sticky="ew")
		
		rfb = CTkFrame(tab)
		rfb.grid_columnconfigure((1,2,3,4,5,6), weight=1)
		rfb.grid(row=3, column=0, pady=5, padx=5,sticky="nswe")
		
		CTkLabel(rfb, text='Classification noise:').grid(row=0, column=0, columnspan="7", padx=5, pady=0, sticky='we')
		
		self.add_cbox(rfb, 'shape1',1,0)
		
		CTkLabel(rfb, text='->').grid(row=1, column=2, padx=5, pady=5, sticky='we')

		cb_shape = self.add_cbox(rfb, 'shape2',1,3)
		
		self.c1=CTkCheckBox(rfb, text="R", width=1, variable=self.cn['cb_shape'], onvalue="disabled", offvalue="normal",command=lambda: cb_shape.configure(state=self.cn['cb_shape'].get()))
		self.c1.deselect()
		self.c1.grid(row=1, column=5, padx=5)

		self.add_cbox(rfb, 'color1',2,0)

		CTkLabel(rfb, text='->').grid(row=2, column=2, padx=5, pady=5, sticky='we')

		cb_color = self.add_cbox(rfb, 'color2',2,3)

		self.c2=CTkCheckBox(rfb, text="R", width=1, variable=self.cn['cb_color'], onvalue="disabled", offvalue="normal",command=lambda: cb_color.configure(state=self.cn['cb_color'].get()))
		self.c2.deselect()
		self.c2.grid(row=2, column=5, padx=5)

		
		self.p = self.add_entry(rfb, 'Probability', 3, self.cn['probability'], self.v_float)
		
		CTkButton(rfb, text='+', command=self.add_cn).grid(row=3, column=4, padx=5, pady=5,sticky='we')
		
		CTkButton(rfb, text='Reset', command=self.reset_cn).grid(row=4, column=4, padx=5, pady=5,sticky='we')
		
		self.cn_textbox = CTkTextbox(rfb, wrap="word", width=300, bg_color="#333", state="disabled")
		self.cn_textbox.grid(row=1, column=6, pady=5, rowspan="4",padx=5,sticky='nswe')
		
	def add_cbox(self, master, what, row, column):
		CTkLabel(master, text="%s:" % what.title(), anchor="e").grid(row=row, column=column, padx=5, pady=5, sticky='we')
		what = re.sub(r'\d+', '', what)
		s = CTkComboBox(master, values=[''], state="readonly")
		if what=='color':
			s.configure(command=lambda e:self._update_bg_color(s))
		s.grid(row=row, column=(column+1), padx=5, sticky='we')
		self.cbox[what].append(s)
		return s
	
	def add_entry(self, master, what, row, textvariable, callback = None, default = None):
		if callback == None:
			callback= self.v_int
		CTkLabel(master, text='%s:' % what, anchor="e").grid(row=row, column=0, padx=5, pady=5, sticky='we')
		e = CTkEntry(master,state = "normal",fg_color="#000", justify="right", textvariable=textvariable)
		
		CellHelper.bind_cell(e, callback)
		e.grid(row=row, column=1, padx=5, sticky='we')

		if(default is not None):
			EntryHelper.update_value(e, default)
		
		return e;


	def edit_selected(self):
		if len(self._groupToModify) == 1:
			return self._mod_matrix(self._groupToModify[0]['row'],self._groupToModify[0]['col'])

		row = list(map(lambda e: e['row'], self._groupToModify))
		col = list(map(lambda e: e['col'], self._groupToModify))

		self.UncertaintiesMatrixTop.modify(self._empty_cell_matrix(), row, col, True)
		return
	
	def new_shape(self, vertices, G):
		self.G = G
		self.u_matrix = np.hstack((self.u_matrix, np.empty((len(G.color_order), 1), dtype=object)))

		sf = CTkFrame(self.cs_f)
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
		
		cf = CTkFrame(self.cs_f)
		cf.grid_columnconfigure((0,1), weight=1)
		# add shape label 
		e = CTkLabel(cf, fg_color=color_picked, text_color=ColorHelper.getTextColor(color_picked), text=color_picked,width=80)
		e.grid(row=0, column=2, padx=2, pady=2)
		
		# place the frame in the gui
		cf.grid(row=len(G.color_order), column=0)
		self._add_cells('color')
		return    

	def update(self,G):
		# print("Updating");
		
		# self.UncertaintiesMatrixTop.update(G);

		for s in self.cbox['shape']:
			s.configure(values=[''] + G.shape_order)
		for c in self.cbox['color']:
			c.configure(values=[''] + G.color_order)    
		
		for r in range(self.u_matrix.shape[0]):
			for c in range(self.u_matrix.shape[1]):
				if self.u_matrix[r][c] is None:
					self.u_matrix[r][c] = self._empty_cell_matrix()
		return

	def _update_bg_color(self, c):
		color = c.get()
		if color == '':
			color='#343638'
		c.configure(fg_color=color, text_color=ColorHelper.getTextColor(color))
		
	def _empty_cell_matrix(self):
		entries = np.empty((len(self.continuous_variables), len(ContinuousVariableHelper.index)), dtype=object)
		entries.fill('')
		
		EV = {
			'cbox': ['constant' for i in range(len(self.continuous_variables))],
			'entries' : entries
		}

		EV['entries'][:, 1] = 0

		return EV


	def update_multiple_matrix(self, matrix, row, col):
		for ci in range(len(row)):
			self.u_matrix[row[ci]][col[ci]] = matrix.copy()
		self._checked()
		#print(self.u_matrix)
		'''
		if np.all(np.linalg.eigvals(self.u_matrix) > 0):
			self.parent.success_msg("Matrix updated successfully.", True)
		else:
			self.parent.error_msg("Warning, the matrix is not positive definite. A random perturbation is going to be applied to ensure a proper Cholesky decomposition.")
		'''
	
	

	def _map_color(self, hex):
		'''
		rgb = str(ColorHelper.hexToRGB(hex))
		rgb = rgb.strip("()")
		rgb = rgb.replace(",","-")
		rgb = rgb.replace(" ","")
		'''
		rgb = '-'.join(str(x) for x in ColorHelper.hexToRGB(hex))
		return rgb

	def save(self, db_name):
		try:
			colors = list(map(self._map_color, self.G.color_order))
		except:
			return False
		
		for ci in range(self.u_matrix.shape[0]):
			for si in range(self.u_matrix.shape[1]):
				
				entries = self.u_matrix[ci][si]['entries'].copy()
				
				csvdata = [[str(e).replace(ContinuousVariableHelper.infinity, "inf") for e in sub] for sub in entries]
				csvdata = np.transpose(csvdata)
				
				path_data = os.getcwd()
				path_data = os.path.join(path_data, db_name)
				
				filename = self.csv_dm % ( self.G.shape_order[si], colors[ci])
				filename = os.path.join(path_data, filename)
				
				try:
					pd.DataFrame(csvdata, index=ContinuousVariableHelper.index, columns=self.continuous_variables).to_csv(filename)
				except:
					msg = "Unable to write %s" % os.path.basename(filename)
					self.parent.error_msg(msg)
					return False	

				list_cn = self.list
				
				rows = len(list_cn)
				if rows > 0:
					csvdata = np.empty((rows,1), dtype=object)
					csvdata[0][0] = list_cn[0] if len(list_cn) else None

					for row in range(1,len(list_cn)):
						csvdata[row][0] = list_cn[row] if row < len(list_cn) else None
				else:
					csvdata = []

				path_data = os.getcwd()
				path_data = os.path.join(path_data, db_name)
				
				filename = self.csv_cn % ( self.G.shape_order[si], colors[ci])
				filename = os.path.join(path_data, filename)
				
				try:
					pd.DataFrame(csvdata, columns=['classification_noise']).to_csv(filename, index=False)	
				except:
					msg = "Unable to write %s" % os.path.basename(filename)
					self.parent.error_msg(msg)
					return False	

		return True
	
	def load(self, path):
		list_loaded = False
		for ci, c in enumerate(self.G.color_order):
			for si, s in enumerate(self.G.shape_order):
				color = self._map_color(c)
				ucn = pd.read_csv(os.path.join(path, self.csv_cn % (s, color)))['classification_noise'].tolist()
				udm = pd.read_csv(os.path.join(path, self.csv_dm % (s, color)), dtype=object).fillna('')
				entries = np.empty((len(self.continuous_variables), len(ContinuousVariableHelper.index)), dtype=object)
				entries.fill('')
				
				cbox = []
				for cvi, cv in enumerate(self.continuous_variables):
					
					l = [str(x).replace("inf", ContinuousVariableHelper.infinity) for x in udm[cv].tolist()]
					cbox.append(ContinuousVariableHelper.get_cbox_value(udm[cv].tolist()))
					for li, udl in enumerate(l):
						entries[cvi][li] = udl

				if list_loaded == False:
					self.list = ucn
					list_loaded = True

				self.u_matrix[ci][si]['cbox'] = cbox
				self.u_matrix[ci][si]['entries'] = entries
		
		TextboxHelper.update_value(self.cn_textbox, '\n'.join(self.list))
		return

	def add_cn(self):
		cc_shape1, cc_shape2, cc_color1, cc_color2, p = self.cbox['shape'][0].get(),self.cbox['shape'][1].get(),self.cbox['color'][0].get(),self.cbox['color'][1].get(), self.p
		
		if len(p.get()) == 0:
			self.parent.error_msg("Error: Probability in Classification Noise cannot be empty.")
			return

		if (self.v_float(p) == False):
			return;
		
		if self.cn['cb_shape'].get() == 'disabled':
			cc_shape2 = "*"

		if self.cn['cb_color'].get() == 'disabled':
			cc_color2 = "*"
		

		value_to_check = '[%s/%s]%s[%s/%s]' % (cc_shape1, str(ColorHelper.hexToRGB(cc_color1)), TextboxHelper.SEPARATOR, cc_shape2, str(ColorHelper.hexToRGB(cc_color2)))

		value = '%s%s%s' % (value_to_check, TextboxHelper.SEPARATOR, p.get())

		if value_to_check not in map(self.reduce,self.list):
			self.list.append(value)
		else:
			self.parent.error_msg('%s already present in list.' % value_to_check)

		TextboxHelper.update_value(self.cn_textbox, '\n'.join(self.list))
		
		return 

	def v_float(self, e):
		value = e.get().strip()
		
		if len(value) == 0:
			return
		try:
			f = float(value)
			if f < 0 or f > 1:
				self.parent.error_msg("Error: probability must be a float number between 0 and 1).")
				EntryHelper.update_value(e, '0')
				return False
			else:
				self.parent.success_msg("%s inserted successfully." % value)
				return True
				
		except ValueError:
			self.parent.error_msg("Error: probability must be a float number between 0 and 1).")
			EntryHelper.update_value(e, '0')
			return False

	def reduce(self,array):
		array = array.split(";")
		return "%s;%s" % (array[0],array[1])

	def reset_cn(self):
		if hasattr(self, 'list'):
			self.list = []

		EntryHelper.update_value(self.p, None)
		
		self.c1.deselect()
		self.c2.deselect()

		for key in self.cbox:
			for cb in self.cbox[key]:
				if key != 'cv':
					cb.set('')
					self._update_bg_color(cb)

		TextboxHelper.update_value(self.cn_textbox, '')

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
		inner_f= CTkFrame(self.cs_f)
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
		self.UncertaintiesMatrixTop.modify(self.u_matrix[row][col], row, col)
