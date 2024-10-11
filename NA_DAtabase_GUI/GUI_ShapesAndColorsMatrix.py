# python import
import numpy as np
from os import sys

# custom tkinter import
from customtkinter import *
from CTkMessagebox import CTkMessagebox

#internal import
from GUI_ShapesAndColorsMatrix_interface import *
from GUI_helper import *
from GUI_MyColorPicker import *

class ShapesAndColorsMatrix():
	def __init__(self, parent):

		self.csv = 'shapes_and_colors_matrix.csv'

		self.parent = parent

		# list of shape's p cells
		self.shapes = []
		# list of color's p cells
		self.colors = []
		# list of cell values
		self.cells = np.array([[]])

		# last cell value focused on
		self._last_value = None
		
		'''
		The top frame with buttons and message box
		'''
		tab = parent.tabview.tab('Shapes and Colors')
		bf = CTkFrame(tab)
		bf.grid(row=0, column=0, pady=10, sticky="we")
		#self.undo = CTkButton(bf, text='Undo', width=1, command=self.do_undo, state="disabled")
		#self.undo.grid(row=0, column=0, padx=5)
		add_shape = CTkButton(bf, text='Add Shape', width=1, command=self.add_shape)
		add_shape.grid(row=0, column=1, padx=5)
		add_color = CTkButton(bf, text='Add Color', width=1, command=self.add_color)
		add_color.grid(row=0, column=2, padx=5)
		self.reset = CTkButton(bf, text='Reset p matrix', width=1, command=self.reset, state="disabled")
		self.reset.grid(row=0, column=3, padx=5)
		self.df = CTkFrame(tab)
		self.df.grid(row=1, column=0, padx=0, sticky="ew")

	def init_G(self, db_name):
		self.G = GUI_ShapesAndColorsMatrix_interface(db_name)
		
	def do_undo(self):
		self.G.undo()
		self._refresh_values()
		return

	def add_shape(self, vertices = None):
		'''
		Add a shape in the gui
		'''
		if vertices == None:
			dialog = CTkInputDialog(title="Add Shape",text="Enter Vertices (0=circle):")
			vertices = dialog.get_input();
			self.parent.focus_force()
			
			if vertices == None:
				return
		
		if vertices.isdigit() and (int(vertices) == 0 or int(vertices) >= 3):
			if vertices in self.G.shape_order:
				# if shape already created before then raise error
				msg = "shape with %s vertices" % vertices if int(vertices) > 2 else "circle"
				self.parent.error_msg("Error: "+msg+" already present.")
				return
			
			# create the frame
			sf = CTkFrame(self.df)
			sf.grid_columnconfigure((0,1), weight=1)

			# add probability label and cell
			pl = CTkLabel(sf, text="p",width=1)
			pl.grid(row=0, column=0, columnspan=2)
			p = CTkEntry(sf,state = "normal",fg_color="#000", width=80, justify="center")
			p.bind('<FocusIn>', lambda evt: self._set_last_value(p.get()))
			p.grid(row=1, column=0, columnspan=2)

			EntryHelper.update_value(p, 0.0)
			CellHelper.bind_cell(p, self._update_cell)
		
			# add shape label 
			e = CTkLabel(sf, text="Circle" if vertices == '0' else "%s sides poly" % vertices, width=80)
			e.grid(row=2, column=0)

			# add switch to lock/unlock the shape column
			s = CTkSwitch(sf, text='', progress_color="darkred", width=1, command=lambda: self.lock_shape(vertices))
			s.grid(row=2, column=1)
			
			# append shape data to array
			self.shapes.append({ 
				's': s, 		# switch
				'p': p,  		# shape probability
				'v': vertices	# num of vertices
			})
			# add shape in GUI_interface
			self.G.new_shape(vertices)
			self.parent.MultivariateDistributionMatrix.new_shape(vertices, self.G)
			self.parent.UncertaintiesMatrix.new_shape(vertices, self.G)
			
			# place the frame in the gui
			sf.grid(row=0, column=len(self.G.shape_order))

			# write success message
			msg = "Shape with %s vertices added successfully." % vertices if int(vertices) > 2 else "Circle added successfully."
			self.parent.success_msg(msg)

			
			# add shape cells fro editing
			self._add_cells('shape')
			return

		# If vertices is not numeric or _empty raisie error
		self.parent.error_msg("Error: vertices must be 0 for a circle or a number greater than 2.")
		return
	
	def add_color(self, color_picked = None):
		'''
		Add a color in the gui
		'''
		if color_picked == None:
			color_picked = GUI_MyColorPicker().get() # get the color string
			self.parent.focus_force()
			
			if(color_picked is None):
				return
			if color_picked in self.G.color_order:
				self.parent.error_msg("Error: color %s already present." % color_picked)
				return
		# create the frame
		cf = CTkFrame(self.df)
		
		# add probability label and cell
		pl = CTkLabel(cf, text="p",width=1)
		pl.grid(row=0, column=0, padx=5)
		p = CTkEntry(cf,state = "normal",fg_color="#000", width=80, justify="center")
		p.bind('<FocusIn>', lambda evt: self._set_last_value(p.get()))
		p.grid(row=0, column=1, padx=5)
		CellHelper.bind_cell(p, self._update_cell)
		
		# add color label
		e = CTkLabel(cf, fg_color=color_picked, text_color=ColorHelper.getTextColor(color_picked), text=color_picked,width=80)
		e.grid(row=0, column=2)
		
		# add switch to lock/unlock the color row
		s = CTkSwitch(cf, text='', progress_color="darkred", width=1, command=lambda: self.lock_color(color_picked))
		s.grid(row=0, column=3)
			
		# append color data to array
		self.colors.append({ 
			's': s, 			# switch
			'p': p,	 			# color probability.
			'c': color_picked 	# the color
		})
		# add color in GUI_interface
		self.G.new_color(color_picked)
		self.parent.MultivariateDistributionMatrix.new_color(color_picked, self.G)
		self.parent.UncertaintiesMatrix.new_color(color_picked, self.G)
			
		# place the frame in the gui
		cf.grid(row=len(self.G.color_order), column=0)

			
		# write success message
		msg = "Color %s added successfully." % color_picked
		self.parent.success_msg(msg)
		
		# add shape cells fro editing
		self._add_cells('color')
		return

	def save(self, db_name):
		#print (self.G.prob_shape)
		#print (self.G.prob_color)
		#print (self.G.probability_matrix)
		try:
			self.G.save_data(self.csv)
		except:
			msg = "Unable to save %s" % self.csv
			self.parent.error_msg(msg)
			return False
			
		return True

	def load(self, path):
		sac = pd.read_csv('%s/%s' % (path, self.csv))

		shapes = (list(sac.columns)[1:])
		colors = (list(map(ColorHelper.rgbToHEX,sac[sac.columns[0]])))

		for shape in shapes:
			self.add_shape(shape)
		for color in colors:
			self.add_color(color)

		probabilities = sac.to_numpy()[:,1:]
		
		for pci,color in enumerate(colors):
			for psi, shape in enumerate(shapes):
				self.G.modify_cell(shape, color, probabilities[pci][psi])
		
		self.G.unlock_all()

		self._refresh_values()
				
	def reset(self):
		self.G.reset()

		for i in self.shapes:
			i['s'].deselect()
			CellHelper.lock_cell(i['p'], i['s'])
		
		for j in self.colors:
			j['s'].deselect()
			CellHelper.lock_cell(j['p'], j['s'])
		
		rows, cols = self.G.probability_matrix.shape
		for i in range(rows):
			for j in range(cols):
				self.cells[i][j]['s'].deselect()
				CellHelper.lock_cell(self.cells[i][j]['e'], self.cells[i][j]['s'])

		self._refresh_values()
		self._set_last_value(self.shapes[0]['p'].get())

	def lock_shape(self, vertices):
		'''
		lock/unlock a shape column data
		'''
		if self._empty():
			return
		
		self.G.lock_shape(str(vertices))
		self._refresh_values()

	def lock_color(self, color):
		'''
		lock/unlock a color row data
		'''
		if self._empty():
			return

		self.G.lock_color(color)
		self._refresh_values()

	def _switch_event(self, row, col):
		'''
		the switch event lock/unlock a cell and check all switch
		for linked ones to switch/unswitch
		'''
		i = row-1
		j = col-1
		
		shape = self.G.shape_order[j]
		color = self.G.color_order[i]
		
		self.G.lock_cell(shape,color)
		self._refresh_values()
		return

	def _empty(self):
		return len(self.colors) == 0 or len(self.shapes) == 0

	def _refresh_button_state(self):
		#self.undo.configure(state='normal' if len(self.shapes) > 0 or len(self.colors) > 0 else 'disabled') 
		self.reset.configure(state='normal')

	def _add_cells(self,what):
		'''
		Add a row/column of cells
		'''
		if self._empty():
			return

		row, col = len(self.colors), len(self.shapes)
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
		
		# refresh all p values
		self._refresh_values()
		
	def _add_cell(self, row, col):
		'''
		add a single cell in gui
		'''
		f= CTkFrame(self.df)
		f.grid(row=row,column=col)
		e = CTkEntry(f,state = "normal",fg_color="#000", width=80, justify="center")
		e.grid(row=0, column=0)
		e.bind('<FocusIn>', lambda evt: self._set_last_value(e.get()))
		CellHelper.bind_cell(e, self._update_cell)
		s = CTkSwitch(f, text='', progress_color="darkred", width=1)
		s.grid(row=0, column=1)
		s.configure(command=lambda: self._switch_event(row, col))
		return { 'e': e, 's': s }

	def _set_last_value(self, value):
		self._last_value = float(value)

	def _update_cell(self, e):
		'''
		update a single cell value (either p or matrix)
		'''
		info = e.master.grid_info()
		i = info['row'] - 1
		j = info['column'] - 1

		try:
			# only float accepted
			value = float(e.get())
		except:
			self.parent.error_msg("Error: p values must be numeric (float number with dot instead of comma).")
			EntryHelper.update_value(e, self._last_value)
			return
		if self._last_value is None or value == self._last_value:
			return

		self._last_value = value

		if info['row'] == 0:
			# updating shape p
			shape = self.G.shape_order[j]
			self.G.modify_shape(str(shape), value)
			self._refresh_values()
			return

		if info['column'] == 0:
			# updating color p
			color = self.G.color_order[i]
			self.G.modify_color(color, value)
			self._refresh_values()
			return
		
		# updating cell p
		shape = self.G.shape_order[j]
		color = self.G.color_order[i]
		self.G.modify_cell(str(shape), color, value)

		#refresh p values
		self._refresh_values()
		return	
			

	def _refresh_values(self, lock_cells = True):
		'''
		refresh all values in cell matrix and probabilities
		'''

		rows, cols = self.G.probability_matrix.shape
		
		if rows > 0 and cols > 0:
			self._refresh_button_state()
		# uodate color probabilities and lock them in case
		for i in range(rows):
			if np.sum(self.G.lock_matrix[i,:]) == cols:
				EntryHelper.update_value(self.colors[i]['p'], self.G.prob_color[i])
				if lock_cells:
					self.colors[i]['s'].select()
					CellHelper.lock_cell(self.colors[i]['p'], self.colors[i]['s'])
			else:
				if lock_cells:
					self.colors[i]['s'].deselect()
					CellHelper.lock_cell(self.colors[i]['p'], self.colors[i]['s'])
				EntryHelper.update_value(self.colors[i]['p'], self.G.prob_color[i])
				
		# uodate shape probabilities and lock them in case
		for j in range(cols):
			if np.sum(self.G.lock_matrix[:,j]) == rows:
				EntryHelper.update_value(self.shapes[j]['p'], self.G.prob_shape[j])
				if lock_cells:
					self.shapes[j]['s'].select()
					CellHelper.lock_cell(self.shapes[j]['p'], self.shapes[j]['s'])
			else:
				if lock_cells:
					self.shapes[j]['s'].deselect()
					CellHelper.lock_cell(self.shapes[j]['p'], self.shapes[j]['s'])
				EntryHelper.update_value(self.shapes[j]['p'], self.G.prob_shape[j])
		
		for i in range(rows):
			for j in range(cols):
				EntryHelper.update_value(self.cells[i][j]['e'], self.G.probability_matrix[i][j])
				
				if lock_cells:
					if self.G.lock_matrix[i][j]:
						self.cells[i][j]['s'].select()
					else:
						self.cells[i][j]['s'].deselect()
					CellHelper.lock_cell(self.cells[i][j]['e'], self.cells[i][j]['s'])

		
		# debug
		#print (self.G.prob_shape)	
		#print (self.G.prob_color)	
		#print (self.G.probability_matrix)	

		#update other matrix
		self.parent.SamplerPropertiesMatrix.update(self.G)
		self.parent.MultivariateDistributionMatrix.update(self.G)
		self.parent.UncertaintiesMatrix.update(self.G)