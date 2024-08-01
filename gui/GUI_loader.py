# custom tkinter import
from customtkinter import *
from GUI_helper import *
import os
import time

class LoaderWindow(CTkToplevel):
	'''
	Genertate the "Choose a Preset" window
	'''
	def __init__(self, parent):
		super().__init__()
		self.parent = parent

		self.selected_db = None
		self.entries = []

		self.geometry("400x500")
		self.title("NA-Database | Let's start!")
		self.protocol("WM_DELETE_WINDOW", lambda: sys.exit(0))
		self.grid_columnconfigure((0), weight=1)
		self.grid_rowconfigure((2,3,4,5,6,7,8), weight=1)
		self.focus_force()
		self.grab_set()
		
		
		self.new_db = CTkButton(self, text='New database', command=self.new_db)
		self.new_db.grid(row=0, column=0, padx=5, pady=15)
	
		label = CTkLabel(self, text="Saved databases:")
		label.grid(row=1, column=0, pady=20)
		
		saved_db = [ os.path.basename(f.path) for f in os.scandir(PathHelper.get_db_path()) if f.is_dir() ]
		saved_db.sort()

		sf = CTkScrollableFrame(self, bg_color="black")
		sf.grid(row=2, column=0, rowspan=7, sticky="nswe", padx=15)
		sf.grid_columnconfigure((0), weight=1)
		
		for si, s in enumerate(saved_db):
			self.entries.append(self.add_entry(sf, si, s))

		self.loadBtn = CTkButton(self, text='Load', command=self.load_db, state="disabled")
		self.loadBtn.grid(row=9, column=0, padx=5, pady=15)
	
	def add_entry(self, sf, si, s):
		b = CTkButton(sf, text=s, border_width=1,border_color="black", fg_color="#444", hover_color="#800")
		b.configure(command=lambda: self.select(b))
		b.grid(row=si, column=0, padx=5, pady=5, sticky="nswe")
		return b

	def select(self, e):
		[x.configure(fg_color="#444", hover_color="#800") for x in self.entries]
		e.configure(fg_color="#060", hover_color="#060")
		self.selected_db = e.cget("text")
		self.loadBtn.configure(state="normal")


	def new_db(self):
		if self.parent.ask_db_name():
			self.back()

	def load_db(self):
		self.new_db.configure(state="disabled")
		[x.configure(state="disabled") for x in self.entries]
		self.loadBtn.configure(state="disabled")
		self.loadBtn.configure(text="Loading %s" % self.selected_db, fg_color="#060")
		self.update()
		self.parent.load(self.selected_db)
		self.back()

	def back(self):
		self.parent.deiconify()
		self.parent.focus_force()
		self.destroy()
		


