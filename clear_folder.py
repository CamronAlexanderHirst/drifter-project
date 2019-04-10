#!/usr/bin/env python3

import os, shutil, sys


#folder = '/path/to/folder'
def clear_figure_folder():
    folder =  './figures' #Clear the figures folder (used to create gifs)
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

if __name__ == '__main__':
    # service.py executed as script
    # do something
    clear_folder()
