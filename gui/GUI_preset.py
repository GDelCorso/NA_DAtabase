# custom tkinter import
from customtkinter import *

class Preset(CTkFrame):
	'''
	Add a Preset Frame
	'''
	col = 0

	def __init__(self, master, title, description, onchoose):
		super().__init__(master)
		self.grid(row=1, column=self.col, padx=5, pady=5, sticky="ew")
		self.grid_columnconfigure(0, weight=1)
		label = CTkLabel(self, text=title)
		label.grid(row=0, column=0, sticky="ew")
		textbox = CTkTextbox(self, wrap="word")
		textbox.insert("0.0", description)  # insert at line 0 character 0
		textbox.configure(state="disabled")
		textbox.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
		button = CTkButton(self, text="Choose", command=onchoose)
		button.grid(row=2, column=0, padx=5, pady=10)
		Preset.col += 1

class PresetWindow(CTkToplevel):
	'''
	Genertate the "Choose a Preset" window
	'''
	def __init__(self, parent):
		super().__init__()
		self.parent = parent
		self.geometry("800x500")
		self.title("Let's start!")
		self.protocol("WM_DELETE_WINDOW", lambda: sys.exit(0))
		self.grid_columnconfigure((0,1,2,3), weight=1)
		self.focus_force()
		self.grab_set()
		
		label = CTkLabel(self, text="Choose a preset", font=(None,16))
		label.grid(row=0, column=0, pady=20, columnspan=4, sticky="ew")
		
		Preset(master=self, title="Preset 1", description="Preset 1 description", onchoose=self.click)
		Preset(master=self, title="Preset 2", description="Preset 2 description", onchoose=self.click)
		Preset(master=self, title="Preset 3", description="Preset 3 description", onchoose=self.click)
		Preset(master=self, title="Preset 4", description="Preset $ description", onchoose=self.click)
		
		label = CTkLabel(self, text="or", font=(None,16))
		label.grid(row=2, column=0, pady=20, columnspan=4, sticky="ew")
		
		button = CTkButton(self, text="Start from scratch", font=(None,16), command=self.start)
		button.grid(row=3, column=1, columnspan=2)
		
	def start(self):
		self.destroy()
		self.parent.deiconify()

	def click(self):
		print ("click")

