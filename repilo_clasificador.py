import streamlit as st
import os
from src.user_files import update_folder
from src.classifier import classify
from PIL import Image



##############################################################
### PAGE SETTINGS  ###########################################
##############################################################

st.set_page_config (page_title="Clasificador de repilo", layout="wide")
st.header ('Clasificador de hojas de olivo para detectar repilo')
st.write('##### Clasificador binario basado en una Red Neuronal Convolucional')
st.write('La app permite subir imágenes de una hoja de olivo y detectar si tiene repilo o no.')



# hide menu and footer from streamlit
hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
st.markdown (hide_menu_style, unsafe_allow_html=True)



##############################################################
### UPLOAD FILES   ###########################################
##############################################################

st.write ('### Sube los ficheros ')
user_folder = 'data/user_files/images'

uploaded_files = st.file_uploader ('Sube las imágenes', type=["jpg", "png"], accept_multiple_files=True)


if len(uploaded_files) != 0:
    update_folder(uploaded_files,user_folder)
else: st.stop()

##############################################################
### CLASSIFY FILES   #########################################
##############################################################

resultado_df = classify ()

##############################################################
### SHOW LOADED IMAGES ON SCREEN       #######################
##############################################################

num_rows = resultado_df.shape[0]
index = 1
col1, col2 = st.columns([1,1])


for row in range(num_rows):

    caption = resultado_df.at[row, 'Imagen analizada']
    prob_repilo = round(resultado_df.at[row, 'Hoja con repilo'],2)
    prob_no_repilo = round(resultado_df.at [row, 'Hoja sin repilo'],2)

    file = os.path.join (user_folder, caption)
    img = Image.open (file)
    width, height = img.size
    new_width = 500
    new_height = int (height / (width / new_width))
    resized_image = img.resize ((new_width, new_height), Image.ANTIALIAS)

    if (index % 2) == 1:

        with col1:
            st.image (resized_image,
                      width=new_width,
                      use_column_width=False,
                      caption=(f'{caption}  --- Probabilidad de repilo: {str(round(prob_repilo*100,2))} %'))

    else:

        with col2:
            st.image (resized_image,
                      width=new_width,
                      use_column_width=False,
                      caption=(f'{caption}  --- Probabilidad de repilo: {str(round(prob_repilo*100,2))} %'))


    index += 1


