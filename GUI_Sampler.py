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

class GUI_generate(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        
        self.deactivate_generator = True
        
        self.geometry("400x300")
        
        self.description = customtkinter.CTkLabel(self, text="Define the path to the folder containing the datasets.")
        self.description.place(relx=0.5, rely=0.2, anchor=tkinter.CENTER)
            
        self.import_button = customtkinter.CTkButton(self, text="Select dataset path", command=self.import_click_event)
        self.import_button.place(relx=0.5, rely=0.3, anchor=tkinter.CENTER)

        self.label = customtkinter.CTkLabel(self, text="[no path selected]")
        self.label.place(relx=0.5, rely=0.4, anchor=tkinter.CENTER)

        self.generate_button = customtkinter.CTkButton(self, text="Generate images", command=self.generate_click_event)
        self.generate_button.place(relx=0.5, rely=0.5, anchor=tkinter.CENTER)

        self.description2 = customtkinter.CTkLabel(self, text="Generate the images once the path is defined.")
        self.description2.place(relx=0.5, rely=0.6, anchor=tkinter.CENTER)
        self.description3 = customtkinter.CTkLabel(self, text="It may require several minutes.")
        self.description3.place(relx=0.5, rely=0.7, anchor=tkinter.CENTER)
        
        self.pb = customtkinter.CTkProgressBar(self, fg_color="gray", progress_color="gray")
        self.pb.place(relx=0.5, rely=0.8, anchor=tkinter.CENTER)
        self.pb.set(0)
        
        self.pbLabel = customtkinter.CTkLabel(self, text="")
        self.pbLabel.place(relx=0.5, rely=0.89, anchor=tkinter.CENTER)
        self.mainloop()

    def import_click_event(self):
        self.dirname = filedialog.askdirectory(mustexist=True)
        self.label.configure(text = self.dirname)
        self.deactivate_generator = False
        
    def generate_click_event(self):
        if self.deactivate_generator:
            print("Warning: no path selected")
        else:
            self.deactivate_generator = True
            NA_D.random_sampler(give_full_path=self.dirname, gui=self).auto_process()
            self.generate_button.configure(text = "Done")

#%% Start the GUI:
customtkinter.set_appearance_mode("dark")
gui = GUI_generate()
