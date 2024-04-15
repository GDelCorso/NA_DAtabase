#!/bin/bash
rm -R build
rm -R dist
pyinstaller --add-data=/home/federico/.local/lib/python3.8/site-packages/CTkColorPicker:CTkColorPicker --hidden-import PIL._tkinter_finder -F ./gui.py