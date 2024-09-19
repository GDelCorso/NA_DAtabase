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

		self.excluded_cbox_values = {
			'deformation' : ['gaussian']
		}

		self._coords = (0, 0)
		
		self.protocol("WM_DELETE_WINDOW", self._close)

		# last cell value focused on
		self._last_value = None
			
		self.multiple = False

		self.entries = np.empty((len(self.parent.continuous_variables), len(ContinuousVariableHelper.index)), dtype=object)

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
		for i in ContinuousVariableHelper.index:
			CTkLabel(f, text=ContinuousVariableHelper.title(i)).grid(row=0, column=c, padx=5, pady=5, sticky='we')
			c = c + 1
		
		r=1
		for i in self.parent.continuous_variables:
			self.cbox['cv'].append(self.add_row(f, i, r))
			r = r + 1
	
		CTkButton(f, text='-%s' % ContinuousVariableHelper.infinity, command=self._add_negative_inf).grid(row=r, column=2, padx=5, pady=5,sticky='we')
		
		CTkButton(f, text='+%s' % ContinuousVariableHelper.infinity, command=self._add_positive_inf).grid(row=r, column=5, padx=5, pady=5,sticky='we')
		
		r = r + 1
		CTkButton(f, text='Reset', command=self.reset_distribution).grid(row=r, column=5, padx=5, pady=5,sticky='we')
		
		CTkButton(self, text="Update", font=(None,16), command=self.save).grid(row=11, column=0, columnspan=11,pady=10,padx=10,sticky="e")
		CTkLabel(self, text="Remember to hit enter key,tab key or change focus by clicking on other cells to update the values.").grid(row=12, column=0, columnspan=11,pady=10,padx=10,sticky="ew")
		
		self.warning = CTkLabel(self, text="", bg_color="#222", text_color="#f66")
		self.warning.grid(row=13, column=0, columnspan=11,pady=10,padx=10,sticky="ew")

	def _add_negative_inf(self):
		x,y = self._coords
		EntryHelper.update_value(self.entries[x][y], '-%s' % ContinuousVariableHelper.infinity)
		return;
	
	def _add_positive_inf(self):
		x,y = self._coords
		EntryHelper.update_value(self.entries[x][y], '+%s' % ContinuousVariableHelper.infinity)
		return;

	def _close(self):
		self.grab_release()
		self.withdraw()
	
	def add_row(self, master, what, row):
		CTkLabel(master, text='%s:' % ContinuousVariableHelper.title(what), anchor="e").grid(row=row, column=0, padx=5, pady=5, sticky='we')
		values = list(ContinuousVariableHelper.cbox_lock_values.keys())

		if what in self.excluded_cbox_values:
			values = [x for x in values if x not in self.excluded_cbox_values[what]]
		
		cb = CTkComboBox(master, values=values, state="readonly", command=lambda v:self.lock(v, row-1))
		cb.grid(row=row, column=1, padx=5, sticky='we')
		
		for i in range(0,len(ContinuousVariableHelper.index)):
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
				self.parent.parent.error_msg("Error: probability must be a float number between 0 and 1).")
				EntryHelper.update_value(e, '0')
				return False
			else:
				self.parent.parent.success_msg("%s inserted successfully." % value)
				return True
				
		except ValueError:
			self.parent.parent.error_msg("Error: probability must be a float number between 0 and 1).")
			EntryHelper.update_value(e, '0')
			return False

	def add_cn(self):
		cc_shape1, cc_shape2, cc_color1, cc_color2, p = self.cbox['shape'][0].get(),self.cbox['shape'][1].get(),self.cbox['color'][0].get(),self.cbox['color'][1].get(), self.p
		
		if len(p.get()) == 0:
			self.parent.parent.error_msg("Error: Probability in Classification Noise cannot be empty.")
			return

		if (self.v_float(p) == False):
			return;
		
		if self.cn['cb_shape'].get() == 'disabled':
			cc_shape2 = "*"

		if self.cn['cb_color'].get() == 'disabled':
			cc_color2 = "*"
		

		value_to_check = '[%s/%s]%s[%s/%s]' % (cc_shape1, str(ColorHelper.hexToRGB(cc_color1)), TextboxHelper.SEPARATOR, cc_shape2, str(ColorHelper.hexToRGB(cc_color2)))

		value = '%s%s%s' % (value_to_check, TextboxHelper.SEPARATOR, p.get())

		if value_to_check not in map(self.reduce,self.stuff['list']):
			self.stuff['list'].append(value)
		else:
			self.parent.parent.error_msg('%s already present in list.' % value_to_check)

		TextboxHelper.update_value(self.cn_textbox, '\n'.join(self.stuff['list']))
		
		return 

	def reset_cn(self):
		if hasattr(self, 'stuff'):
			self.stuff['list'] = []

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

		self.parent.parent.success_msg("All Reset to default.")

	def reduce(self,array):
		array = array.split(";")
		return "%s;%s" % (array[0],array[1])


	def update_bg_color(self, c):
		color = c.get()
		if color == '':
			color='#343638'
		c.configure(fg_color=color, text_color=ColorHelper.getTextColor(color))

	def modify(self, stuff, row, col, multiple=False):
		if multiple:
			self.title("Uncertanties (%s, %s)" % ("*","*"))
		else:
			sides = self.parent.G.shape_order[col]
			shape = "%s sides poly" % sides if int(sides) > 2 else "circle" 
			self.title("Uncertanties (%s, %s)" % (shape,self.parent.G.color_order[row]))

		self.reset_cn()
		self.reset_distribution()
		
		self.stuff = stuff.copy()
		
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

		if self.multiple:
			self.warning.configure(text="Warning, you're editing more multiple matrix at once. Be careful.")
		else:
			self.warning.configure(text="")

	def check(self, row, i, min_value, max_value):
		
		e = self.entries[row][i]
		value = e.get().strip()
		
		if len(value) == 0:
			return
		
		if i==0 and value == "-%s" % ContinuousVariableHelper.infinity:
			return

		if i==3 and value == "+%s" % ContinuousVariableHelper.infinity:
			return

		if i==0 and value == "+%s" % ContinuousVariableHelper.infinity or i==3 and value == "-%s" % ContinuousVariableHelper.infinity:
			self.parent.parent.error_msg("Error: %s can't be %s." % (ContinuousVariableHelper.index[i], value))
			EntryHelper.update_value(e,self._last_value)
			return

		try:
			value = float(value)
			
			if value < min_value:
				self.parent.parent.error_msg("Error: %s must be greater than %s." % (ContinuousVariableHelper.title(ContinuousVariableHelper.index[i]), str(min_value)))
				EntryHelper.update_value(e,self._last_value)
				return
		
			if max_value > 0 and value > max_value:
				self.parent.parent.error_msg("Error: %s must be lower than %s." % (ContinuousVariableHelper.title(ContinuousVariableHelper.index[i]), str(max_value)))
				EntryHelper.update_value(e,self._last_value)
				return

			l = self.entries[row][0]
			l_value = l.get().strip()
				
			u = self.entries[row][3]			
			u_value = u.get().strip()

			if i==2 and value <= 0:
				self.parent.parent.error_msg("Error: %s must be greater than zero." % ContinuousVariableHelper.title(ContinuousVariableHelper.index[i]))
				EntryHelper.update_value(e,self._last_value)
				return
			
			if len(l_value) > 0 and len(u_value) > 0:
				if float(l_value) >= float(u_value):
					self.parent.parent.error_msg("Error: %s must be greater than %s." % (ContinuousVariableHelper.title(ContinuousVariableHelper.index[3]),ContinuousVariableHelper.title(ContinuousVariableHelper.index[0])))
					EntryHelper.update_value(e,self._last_value)
					return False

			self.parent.parent.success_msg("%s inserted successfully." % ContinuousVariableHelper.title(ContinuousVariableHelper.index[i]))
			return True
		except ValueError:
			self.parent.parent.error_msg("Error: %s must be a numeric value." % ContinuousVariableHelper.title(ContinuousVariableHelper.index[i]))
			EntryHelper.update_value(e,self._last_value)
			return False



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
			self.parent.parent.error_msg("Warning, All unlocked distribution values must be filled.")
			return

		min_value = 0
		max_value = 100 if i < 4 else -1

		i,j = self._coords
		e = self.entries[i][j]
		if e.cget('state') == 'normal' and self.check(i,j,min_value,max_value) == False:
			return

		if self.multiple:
			self.parent.update_multiple_matrix(self.stuff, self.row, self.col)
		
		self.parent.parent.success_msg("Matrix updated successfully.")
		del self.stuff
		self._close()
	
	def _entry(self, master, row, i, what):
		e = CTkEntry(master, fg_color="#000", justify="center")
		e.bind('<FocusIn>', lambda evt: self._set_last_value(e.get(),row-1, i))
		e.grid(row=row, column=i+2, padx=5, sticky='we')
		if i == 0:
			EntryHelper.update_value(e,"-%s" % ContinuousVariableHelper.infinity)
		if i == 3:
			EntryHelper.update_value(e,"+%s" % ContinuousVariableHelper.infinity)

		if what in self.default_values:
			if ContinuousVariableHelper.index[i] in self.default_values[what]:
				#f_value = 0 if self.default_values[what][ContinuousVariableHelper.index[i]] == 0 else "%.2f" % self.default_values[what][ContinuousVariableHelper.index[i]]
				f_value = self.default_values[what][ContinuousVariableHelper.index[i]]
				EntryHelper.update_value(e, f_value)

		min_value = 0
		max_value = 100 if row < 5 else -1
		
		CellHelper.bind_cell(e, lambda evt: self.check(row-1,i, min_value, max_value))
		return e	

	def _set_last_value(self, value, row, col):
		self._last_value = value
		self._coords = (row,col)
		
	def update(self, G):
		for s in self.cbox['shape']:
			s.configure(values=[''] + G.shape_order)
		for c in self.cbox['color']:
			c.configure(values=[''] + G.color_order)    
		return