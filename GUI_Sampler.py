#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

#%% Libraries
import customtkinter
import tkinter
from tkinter import filedialog
import NA_DAtabase as NA_D
import time


class GUI_generate():
    def __init__(self):
        self.deactivate_generator = True
        
        customtkinter.set_appearance_mode("dark")
        app = customtkinter.CTk()
        app.geometry("400x300")
        
        self.description = customtkinter.CTkLabel(app, text="Define the path to the folder containing the datasets.")
        self.description.place(relx=0.5, rely=0.2, anchor=tkinter.CENTER)
            
        self.import_button = customtkinter.CTkButton(app, text="Select dataset path", command=self.import_click_event)
        self.import_button.place(relx=0.5, rely=0.3, anchor=tkinter.CENTER)

        self.label = customtkinter.CTkLabel(app, text="[no path selected]")
        self.label.place(relx=0.5, rely=0.4, anchor=tkinter.CENTER)

        self.generate_button = customtkinter.CTkButton(app, text="Generate images", command=self.generate_click_event)
        self.generate_button.place(relx=0.5, rely=0.7, anchor=tkinter.CENTER)

        self.description2 = customtkinter.CTkLabel(app, text="Generate the images once the path is defined.")
        self.description2.place(relx=0.5, rely=0.8, anchor=tkinter.CENTER)
        self.description3 = customtkinter.CTkLabel(app, text="It requires several minutes.")
        self.description3.place(relx=0.5, rely=0.9, anchor=tkinter.CENTER)
 
        app.mainloop()
        
    def import_click_event(self):
        self.dirname = filedialog.askdirectory(mustexist=True)
        self.label.configure(text = self.dirname)
        self.deactivate_generator = False
        
    def generate_click_event(self):
        if self.deactivate_generator:
            print("Warning: no path selected")
        else:
            self.deactivate_generator = True
            NA_D.random_sampler(give_full_path=self.dirname).auto_process()
            self.generate_button.configure(text = "Done")

#%% Start the GUI:
gui = GUI_generate()
