# python import
import re
import numpy as np
import pandas as pd
from os import sys

# custom tkinter import
from customtkinter import *

#internal import
from GUI_helper import *

class UncertantiesMatrixTop(CTkToplevel):
	def __init__(self, parent):
		super().__init__()
		self.parent = parent

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
			'color': [],
			'cv': []
		}

		self.title("Uncertanties")
		self.protocol("WM_DELETE_WINDOW", self._close)

		self.infinity = "∞"

		# last cell value focused on
		self._last_value = None
			
		
		self.entries = np.empty((len(self.parent.continuous_variables), len(self.parent.index)), dtype=object)

		self.cbox_lock_values = {
			'constant' : ['lower_bound', 'sigma', 'upper_bound'],
			'uniform': ['mean', 'sigma'],
			'gaussian': ['lower_bound', 'upper_bound'],
			'truncated_gaussian': []
		}


		self.default_values = {
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
			'additive_noise_regression': {
				'cbox': 'constant',
				'mean': 0
			},
			'multiplicative_noise_regression': {
				'cbox': 'constant',
				'mean': 0
			},
		}

		self.withdraw()

		rf = CTkFrame(self)
		rf.grid_columnconfigure(0, weight=1)
		rf.grid(row=0, column=0, pady=0, rowspan=2, sticky="nswe")
		
		rfb = CTkFrame(rf)
		rfb.grid_columnconfigure((1,2,3,4,5,6), weight=1)
		rfb.grid(row=0, column=0, pady=5, padx=5,sticky="nswe")
		
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
		'''
		CTkLabel(rfb, text='Probability:', anchor="e").grid(row=3, column=0, padx=5, pady=5, sticky='we')
		CTkEntry(rfb,state = "normal",fg_color="#000", justify="right", textvariable=self.cn['probability'],validate='focusout', validatecommand=lambda: self.v_float(self.cn['probability'].get())).grid(row=3, column=1, padx=5, sticky='we')
		'''

		CTkButton(rfb, text='+', command=self.add_cn).grid(row=3, column=4, padx=5, pady=5,sticky='we')
		
		CTkButton(rfb, text='Reset', command=self.reset_cn).grid(row=4, column=4, padx=5, pady=5,sticky='we')
		
		self.cn_textbox = CTkTextbox(rfb, wrap="word", width=300, bg_color="#333", state="disabled")
		self.cn_textbox.grid(row=1, column=6, pady=5, rowspan="4",padx=5,sticky='nswe')


		f = CTkFrame(rf)
		f.grid_columnconfigure((0,1,2,3,4,5), weight=1)
		f.grid(row=1, column=0, pady=10, rowspan=2, sticky="nswe")

		c=2
		for i in self.parent.index:
			CTkLabel(f, text=i.title().replace("_", " ")).grid(row=0, column=c, padx=5, pady=5, sticky='we')
			c = c + 1
		
		r=1
		for i in self.parent.continuous_variables:
			self.cbox['cv'].append(self.add_row(f, i, r))
			r = r + 1
	
		CTkButton(f, text='Reset', command=self.reset_distribution).grid(row=r, column=5, padx=5, pady=5,sticky='we')
		
		CTkButton(self, text="Update", font=(None,16), command=self.save).grid(row=11, column=0, columnspan=11,pady=10,padx=10,sticky="e")
		CTkLabel(self, text="Remember to hit enter key,tab key or change focus by clicking on other cells to update the values.").grid(row=12, column=0, columnspan=11,pady=10,padx=10,sticky="ew")
		
		self.warning = CTkLabel(self, text="", bg_color="#222", text_color="#f66")
		self.warning.grid(row=13, column=0, columnspan=11,pady=10,padx=10,sticky="ew")

	def _close(self):
		self.grab_release()
		self.withdraw()
	
	def add_row(self, master, what, row):
		CTkLabel(master, text='%s:' % what.title().replace("_", " "), anchor="e").grid(row=row, column=0, padx=5, pady=5, sticky='we')
		values = list(self.cbox_lock_values.keys())
		cb = CTkComboBox(master, values=values, state="readonly", command=lambda v:self.lock(v, row-1))
		cb.grid(row=row, column=1, padx=5, sticky='we')
		
		for i in range(0,len(self.parent.index)):
			self.entries[row-1][i] = self._entry(master, row, i, what)

		'''
		if what in self.default_values:
			print(what, self.default_values[what]['cbox'])
			cb.set(self.default_values[what]['cbox'])
			self.lock(self.default_values[what]['cbox'], row-1)
		'''
		return cb

	def add_cbox(self, master, what, row, column):
		CTkLabel(master, text="%s:" % what.title(), anchor="e").grid(row=row, column=column, padx=5, pady=5, sticky='we')
		what = re.sub(r'\d+', '', what)
		s = CTkComboBox(master, values=[''], state="readonly")
		if what=='color':
			s.configure(command=lambda e:self.update_bg_color(s))
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
		

		value_to_check = '[%s/%s];[%s/%s]' % (cc_shape1, str(ColorHelper.hexToRGB(cc_color1)), cc_shape2, str(ColorHelper.hexToRGB(cc_color2)))

		value = '%s;%s' % (value_to_check, p.get())

		if value_to_check not in map(self.reduce,self.stuff['list']):
			self.stuff['list'].append(value)
		else:
			self.parent.error_msg('%s already present in list.' % value_to_check)

		TextboxHelper.update_value(self.cn_textbox, '\n'.join(self.stuff['list']))
		
		return 

	def reset_cn(self):

		EntryHelper.update_value(self.p, None)

		self.c1.deselect()
		self.c2.deselect()

		for key in self.cbox:
			for cb in self.cbox[key]:
				if key != 'cv':
					cb.set('')
					self.update_bg_color(cb)

		TextboxHelper.update_value(self.cn_textbox, '')

	def reset_distribution(self):
		self._update_values(self.parent._empty_cell_matrix())

		self.parent.success_msg("All Reset to default.")

	def reduce(self,array):
		array = array.split(";")
		return "%s;%s" % (array[0],array[1])


	def update_bg_color(self, c):
		color = c.get()
		if color == '':
			color='#343638'
		c.configure(fg_color=color, text_color=ColorHelper.getTextColor(color))

	def modify(self, stuff, row, col, multiple=False):
		self.reset_cn()
		self.reset_distribution()
		print(stuff)

		self.stuff = stuff
		self.row = row
		self.col = col
		self.multiple = multiple
		
		self._update_values(stuff);
		TextboxHelper.update_value(self.cn_textbox, '\n'.join(stuff['list']))

		self.deiconify()
		self.focus_force()
		self.grab_release()
		self.grab_set()
		
	def _update_values(self,stuff):
		for i in range(len(self.cbox['cv'])):
			self.cbox['cv'][i].set(stuff['cbox'][i])
			self.lock(stuff['cbox'][i], i)

		for i in range(stuff['entries'].shape[0]):
			for j in range (stuff['entries'].shape[1]):
				EntryHelper.update_value(self.entries[i][j], stuff['entries'][i][j])

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
			self.parent.error_msg("Error: %s can't be %s." % (self.parent.index[i], value))
			EntryHelper.update_value(e,self._last_value)
			return

		try:
			value = float(value)
			
			if value < min_value:
				self.parent.error_msg("Error: %s must be greater than %s." % (self.parent.index[i], str(min_value)))
				EntryHelper.update_value(e,self._last_value)
				return
		
			if value > max_value:
				self.parent.error_msg("Error: %s must be lower than %s." % (self.parent.index[i], str(max_value)))
				EntryHelper.update_value(e,self._last_value)
				return

			l = self.entries[row][0]
			l_value = l.get().strip()
				
			u = self.entries[row][3]			
			u_value = u.get().strip()

			if i==2 and value <= 0:
				self.parent.error_msg("Error: %s must be greater than zero." % self.parent.index[i])
				EntryHelper.update_value(e,self._last_value)
				return
			
			if len(l_value) > 0 and len(u_value) > 0:
				if float(l_value) >= float(u_value):
					self.parent.error_msg("Error: %s must be greater than %s." % (self.parent.index[3],self.parent.index[0]))
					EntryHelper.update_value(e,self._last_value)
					return False

			self.parent.success_msg("%s inserted successfully." % self.parent.index[i])
			return True
		except ValueError:
			self.parent.error_msg("Error: %s must be a numeric value." % self.parent.index[i])
			EntryHelper.update_value(e,self._last_value)
			return False



	def lock(self,v, r):
		to_locked = self.cbox_lock_values[v]
		for i in range(0,len(self.parent.index)):
			s = IntVar()
			cell = self.entries[r][i]
			if self.parent.index[i] in to_locked:
				s.set(1)
			else:
				s.set(0)
			
			CellHelper.lock_cell(cell, s, delete=True)
		
	def save(self):
		matrix = self.stuff['entries']

		errors = False

		for i in range(matrix.shape[0]):
			self.stuff['cbox'][i] = self.cbox['cv'][i].get()
			for j in range(matrix.shape[1]):
				cell_value = self.entries[i][j].get().strip()
				self.stuff['entries'][i][j] = cell_value
				if self.entries[i][j].cget('state') == 'normal' and cell_value == '':
					errors = True

		if errors:
			self.parent.error_msg("Warning, All unlocked distribution values must be filled.")
			return

		self._close()
		
	
	def _entry(self, master, row, i, what):
		e = CTkEntry(master, fg_color="#000", justify="center")
		e.bind('<FocusIn>', lambda evt: self._set_last_value(e.get()))
		e.grid(row=row, column=i+2, padx=5, sticky='we')
		if i == 0:
			EntryHelper.update_value(e,"-%s" % self.infinity)
		if i == 3:
			EntryHelper.update_value(e,"+%s" % self.infinity)

		if what in self.default_values:
			if self.parent.index[i] in self.default_values[what]:
				#f_value = 0 if self.default_values[what][self.parent.index[i]] == 0 else "%.2f" % self.default_values[what][self.parent.index[i]]
				f_value = self.default_values[what][self.parent.index[i]]
				EntryHelper.update_value(e, f_value)

		min_value = 0
		max_value = 100

		CellHelper.bind_cell(e, lambda evt: self.check(row-1,i, min_value, max_value))
		return e	

	def _set_last_value(self, value):
		self._last_value = value

	def update(self, G):
		for s in self.cbox['shape']:
			s.configure(values=[''] + G.shape_order)
		for c in self.cbox['color']:
			c.configure(values=[''] + G.color_order)    
		return