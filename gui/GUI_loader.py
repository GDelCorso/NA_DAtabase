# custom tkinter import
from customtkinter import *

class LoaderWindow(CTkToplevel):
	'''
	Genertate the "Choose a Preset" window
	'''
	def __init__(self, parent):
		super().__init__()
		self.parent = parent
		self.geometry("400x500")
		self.title("Let's start!")
		self.protocol("WM_DELETE_WINDOW", lambda: sys.exit(0))
		self.grid_columnconfigure((0), weight=1)
		self.focus_force()
		self.grab_set()
		
		
		CTkButton(self, text='New database', command=self.new_db).grid(row=1, column=0, padx=5, pady=15)
	
		label = CTkLabel(self, text="or load exixting one:", font=(None,16))
		label.grid(row=2, column=0, pady=20)
		
	def new_db(self):
		self.parent.ask_db_name()
		self.back()

	def back(self):
		self.parent.deiconify()
		self.parent.focus_force()
		self.destroy()
		


