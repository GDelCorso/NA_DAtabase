# python import
import re
import numpy as np
import pandas as pd
from os import sys

# custom tkinter import
from customtkinter import *

#internal import
from GUI_helper import *
from GUI_MyColorPicker import *

class SamplerPropertiesMatrix():
	def __init__(self, parent):

		self.csv = 'sampler_properties_matrix.csv'

		self.parent = parent

		self.data = {
			'dataset_size': StringVar(),
			'sampling_strategy': StringVar(),
			'random_seed': StringVar(),
			'resolution': StringVar(),
			'background_color': StringVar(),
			'out_of_border': StringVar(),
			'correlation': StringVar(),
			'cc': {
				'shape': StringVar(),
				'color': StringVar(),
				'textbox': StringVar()
			}
		}

		self.cbox = {
			'shape': [],
			'color': []
		}
		
		self.list1 = []
		
		tab = parent.tabview.tab('Sampler Properties')
		
		tab.grid_columnconfigure((0,1), weight=1)
		tab.grid_rowconfigure((0,1), weight=1)
		
		lf = CTkFrame(tab)
		lf.grid_columnconfigure((0,1), weight=1)
		lf.grid_rowconfigure((0,1,2,3,4,5,6,7), weight=1)
		lf.grid(row=0, column=0, pady=10, rowspan=2, sticky="nswe")
		
		'''
		CTkLabel(lf, text='Dataset size:', anchor="e").grid(row=0, column=0, padx=5, pady=5, sticky='we')
		e = CTkEntry(lf,state = "normal",fg_color="#000", justify="right", textvariable=self.data['dataset_size'])
		e.configure(validate='focusout', validatecommand=lambda: self.v_int(self.data['dataset_size'].get(), e))
		e.grid(row=0, column=1, padx=5, sticky='we')
		'''

		self.add_entry(lf, 'Dataset size', 0, self.data['dataset_size'], default=100)

		CTkLabel(lf, text='Sampling strategy:', anchor="e").grid(row=1, column=0, padx=5, pady=5, sticky='we')
		self.lds = CTkComboBox(lf, values=["MC","LHC","LDS"], variable=self.data['sampling_strategy'], state="readonly")
		self.lds.set("MC")
		self.lds.grid(row=1, column=1, padx=5, sticky='we')
		self.mc();
		
		self.add_entry(lf, 'Random seed', 2, self.data['random_seed'], default=1520)
		self.add_entry(lf, 'Resolution', 3, self.data['resolution'], default=128)
		
		'''
		CTkLabel(lf, text='Random Seed:', anchor="e").grid(row=2, column=0, padx=5, pady=5, sticky='we')
		CTkEntry(lf,state = "normal",fg_color="#000", justify="right", textvariable=self.data['random_seed'],validate='focusout', validatecommand=lambda: self.v_int(self.data['random_seed'].get())).grid(row=2, column=1, padx=5, sticky='we')
		
		CTkLabel(lf, text='Resolution:', anchor="e").grid(row=3, column=0, padx=5, pady=5, sticky='we')
		CTkEntry(lf,state = "normal",fg_color="#000", justify="right", textvariable=self.data['resolution'], validate='focusout', validatecommand=lambda: self.v_int(self.data['resolution'].get())).grid(row=3, column=1, padx=5, sticky='we')
		'''
		
		CTkLabel(lf, text='Background Color:', anchor="e").grid(row=4, column=0, padx=5, pady=5, sticky='we')
		
		self.background_color_value = CTkButton(lf, text='#000000', fg_color='#000000', hover=False)
		self.background_color_value.configure(command=lambda: self.pick_color(self.background_color_value))
		self.background_color_value.grid(row=4, column=1, padx=5, sticky='we')
		black = str(ColorHelper.hexToRGB("#000000"))
		self.data['background_color'].set(black)

		CTkLabel(lf, text='Allow Out Of Border:', anchor="e").grid(row=5, column=0, padx=5, pady=5, sticky='we')
		self.c = CTkCheckBox(lf, text="", variable=self.data['out_of_border'],command=lambda: self.mc())
		self.c.deselect()
		self.c.grid(row=5, column=1, padx=5, sticky='we')
		
		CTkLabel(lf, text='Correlation:', anchor="e").grid(row=6, column=0, padx=5, pady=5, sticky='we')
		cor = CTkComboBox(lf, values=["Spearman"], variable=self.data['correlation'], state="readonly")
		cor.set("Spearman")
		cor.grid(row=6, column=1, padx=5, sticky='we')

		rf = CTkFrame(tab)
		rf.grid_columnconfigure((0,1,2,3), weight=1)
		rf.grid_rowconfigure((0,1,2,3,4,5,6,7), weight=1)
		rf.grid(row=0, column=1, pady=0, rowspan=2, sticky="nswe")
		
		CTkLabel(rf, text='Correct classes:').grid(row=0, column=0, columnspan="4", padx=5, pady=0, sticky='we')
		
		self.add_cbox(rf, 'shape',1,0)
		self.add_cbox(rf, 'color',2,0)

		CTkButton(rf, text='+', command=self.add_cc).grid(row=3, column=1, padx=5, pady=5,sticky='we')
		
		CTkButton(rf, text='Reset', command=self.reset_cc).grid(row=4, column=1, padx=5, pady=5,sticky='we')
		

		self.cc_textbox = CTkTextbox(rf, wrap="word", bg_color="#333", state="disabled")
		self.cc_textbox.grid(row=1, column=2, pady=5, rowspan="4",padx=5,sticky='nswe')
		
		
		'''
		CTkLabel(rfb, text='Probability:', anchor="e").grid(row=3, column=0, padx=5, pady=5, sticky='we')
		CTkEntry(rfb,state = "normal",fg_color="#000", justify="right", textvariable=self.data['cn']['probability'],validate='focusout', validatecommand=lambda: self.v_float(self.data['cn']['probability'].get())).grid(row=3, column=1, padx=5, sticky='we')
		'''
		
	def mc(self):
		if self.data['out_of_border'].get() == '1':
			self.lds.configure(state="readonly")
		else:
			self.lds.set("MC")
			self.lds.configure(state="disabled")

	def v_int(self, e):
		value = e.get().strip()
		if len(value) == 0:
			return
		if value.isnumeric() == False:
			self.parent.error_msg("Error, value must be numeric.")
			EntryHelper.update_value(e, '0')
			return False
		else:
			self.parent.success_msg("%s inserted successfully." % value)
			return True

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
		
	def add_cbox(self, master, what, row, column):
		CTkLabel(master, text="%s:" % what.title(), anchor="e").grid(row=row, column=column, padx=5, pady=5, sticky='we')
		what = re.sub(r'\d+', '', what)
		s = CTkComboBox(master, values=[''], state="readonly")
		if what=='color':
			s.configure(command=lambda e:self.update_bg_color(s))
		s.grid(row=row, column=(column+1), padx=5, sticky='we')
		self.cbox[what].append(s)
		return s

	def update_bg_color(self, c):
		color = c.get()
		if color == '':
			color='#343638'
		c.configure(fg_color=color, text_color=ColorHelper.getTextColor(color))

	def update(self, G):
		for s in self.cbox['shape']:
			s.configure(values=[''] + G.shape_order)
		for c in self.cbox['color']:
			c.configure(values=[''] + G.color_order)    
    
		return

	def add_cc(self):
		cc_shape, cc_color = self.cbox['shape'][0].get(),self.cbox['color'][0].get()
		if (len(cc_shape)+len(cc_color) == 0):
			return self.parent.error_msg("Error: one of Shape/Color combobox in Correct Classes must be valued.")
		if len(cc_shape) == 0:
			value = str(ColorHelper.hexToRGB(cc_color))
		elif len(cc_color) == 0:
			value = cc_shape
		else:
			value = "%s/%s" % (cc_shape, str(ColorHelper.hexToRGB(cc_color)))

		
		if value not in self.list1:
			self.list1.append(value)
		else:
			self.parent.error_msg('%s already present in list.' % value)
		
		self.update_textbox()	
		return

	def update_textbox(self):
		# print (self.list1)
		TextboxHelper.update_value(self.cc_textbox, "[%s]" % TextboxHelper.SEPARATOR.join(self.list1))
		
	def reset_cc(self):
		self.list1 = [];
		TextboxHelper.update_value(self.cc_textbox, "")
		self.parent.success_msg("Correct classes resetted.")
		
	def reduce(self,array):
		array = array.split(";")
		return "%s;%s" % (array[0],array[1])

	def pick_color(self, btn, color = None):
		if color == None:
			color = GUI_MyColorPicker().get()
			if color == None:
				return
		btn.configure(text=color, fg_color=color, text_color=ColorHelper.getTextColor(color))
		self.data['background_color'].set(ColorHelper.hexToRGB(color))

	def save(self, db_name):
		rows = len(self.list1)
		if rows == 0:
			rows = 1
		if (
			len(self.data['dataset_size'].get()) == 0 or 
			len(self.data['random_seed'].get()) == 0 or 
			len(self.data['resolution'].get()) == 0 
			):
			return True

		try:
			bg_color = str(self.data['background_color'].get())
			csvdata = np.empty((rows,9), dtype=object)
			csvdata[0][0] = int(self.data['dataset_size'].get())
			csvdata[0][1] = self.data['sampling_strategy'].get()
			csvdata[0][2] = int(self.data['random_seed'].get())
			csvdata[0][3] = int(self.data['resolution'].get())
			csvdata[0][4] = int(self.data['resolution'].get())
			csvdata[0][5] = self.list1[0] if len(self.list1) else None
			csvdata[0][6] = bg_color
			csvdata[0][7] = True if self.data['out_of_border'].get() == '1' else False 
			
			csvdata[0][8] = self.data['correlation'].get()
			
			for row in range(1,rows):
				csvdata[row][5] = self.list1[row] if row < len(self.list1) else None
			
			head = ['dataset_size', 'sampling_strategy', 'random_seed', 'pixel_resolution_x', 'pixel_resolution_y', 'correct_classes', 'background_color', 'out_of_border', 'correlation']

			path_data = PathHelper.get_db_path()
			path_data = os.path.join(path_data, db_name)
			
			filename = os.path.join(path_data, self.csv)
			pd.DataFrame(csvdata, columns=head).to_csv(filename, index=False)

		except:
			msg = "Unable to save %s" % self.csv
			self.parent.error_msg(msg)
			return False

		return True

	def load(self, path):
		sp = (pd.read_csv(os.path.join(path, self.csv))).to_dict()
		print(sp)
		for x in sp:
			if x == 'pixel_resolution_x' or x == 'pixel_resolution_y':
				self.data['resolution'].set(sp[x][0])
			elif x == 'correct_classes':
				# TODO
				if(type(sp[x][0]) == str):
					print ("QUI?")
					self.list1 = list(sp[x].values())
					self.update_textbox()
			elif x == 'out_of_border' and sp[x][0]:
				self.data[x].set(1)
				self.c.select()
				self.mc()
			else:
				self.data[x].set(sp[x][0])
				if x == 'background_color':
					self.pick_color(self.background_color_value, ColorHelper.rgbToHEX(sp[x][0]))

		

