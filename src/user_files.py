# this script cleans the folder of user files and moves the uploaded files to data/user_files/images
# in this directory the models reads the image and makes the classification

import os
import streamlit as st
import tempfile
import shutil

def update_folder(uploaded_files,user_folder):

    if len (uploaded_files) != 0:

        # clean the bin

        for file in os.listdir (user_folder):
            try:
                file_path = os.path.join (user_folder, file)
                os.remove (file_path)
            except OSError as e:
                st.write (e.strerror)

        # move uploaded images to the user folder. streamlit loads an object, so need to tempfile

        for file in uploaded_files:
            filename = file.name
            with tempfile.NamedTemporaryFile (delete=False) as temp:
                temp.write (file.read ())
            try:
                shutil.move (temp.name, os.path.join (user_folder, filename))
            except Exception as e:
                st.error (f"Failed to move {filename}: {e}")
        st.success ("Imágenes copiadas y analizando... estos son los resultados:")