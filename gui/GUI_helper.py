# custom tkinter import
from customtkinter import END

class ColorHelper():
	'''
	calculates text color based on background one 	
	'''
	def getTextColor(color, lightColor='#ffffff', darkColor='#000000'):
		r = int(color[1:3], 16); # hexToR
		g = int(color[3:5], 16); # hexToG
		b = int(color[5:7], 16); # hexToB
		uicolors = [r / 255, g / 255, b / 255];
		c = list(map(ColorHelper._col, uicolors))
		L = (0.2126 * c[0]) + (0.7152 * c[1]) + (0.0722 * c[2]);
		if L > 0.179:
			return darkColor
		return lightColor

	def _col(col):
		if col <= 0.03928:
			return col / 12.92
		return pow((col + 0.055) / 1.055, 2.4)

	def hexToRGB(h):
		if len(h) != 7:
			return h if h == '*' else ''
		h = h.lstrip("#")
		return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

class EntryHelper():

	def update_value(e, txt):
		'''
		update a probability value in Textbox entry
		'''
		e.delete(0,END)
		if txt != None:
			e.insert(0,txt)

class TextboxHelper():

	def update_value(e, txt):
		'''
		update a probability value in Textbox entry
		'''
		e.configure(state="normal")
		e.delete("0.0",END)
		e.insert("0.0",txt)
		e.configure(state="disabled")

class CellHelper():

	def bind_cell(e, callback):
		'''
		bind callback to Return Key and focus out
		'''
		e.bind('<Return>', lambda evt:callback(e))
		e.bind('<FocusOut>', lambda evt:callback(e))
		
	def lock_cell(e, s, delete=False):
		'''
		lock an entry cell
		making it read only
		'''
		new_state = "disabled" if s.get() == 1 else "normal"
		color = "darkred" if s.get() == 1 else "black"
		
		if new_state == "disabled" and delete:
			EntryHelper.update_value(e,"")

		e.configure( 
			state = new_state, 
			fg_color = color 
		)

class ContinuousVariableHelper():
	index = [
		'lower_bound', 
		'mean', 
		'sigma', 
		'upper_bound'
	]

	infinity = "∞"

	cbox_lock_values = {
		'constant' : ['lower_bound', 'sigma', 'upper_bound'],
		'uniform': ['mean', 'sigma'],
		'gaussian': ['lower_bound', 'upper_bound'],
		'truncated_gaussian': []
	}

	def title(txt):
		return txt.title().replace("_", " ")

	def P_INF():
		return '+%s' % ContinuousVariableHelper.infinity
	
	def N_INF():
		return '-%s' % ContinuousVariableHelper.infinity
				


