# this script reads the files uploaded by user and applies the model for the classification

from keras.models import load_model
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
import os

# format the user files

def classify ():

    images = []
    user_folder = 'data/user_files/images'
    for file in os.listdir (user_folder):
        images.append (file)


    image_generator_submission = ImageDataGenerator (rescale=1 / 255)

    submission = image_generator_submission.flow_from_directory (
        directory='data/user_files',    # important, keras is expecting for subfolders, if images were in user_files --> error
        batch_size=1,
        shuffle=False,
        target_size=(224,224),
        class_mode=None)

    model = load_model ('model/olive_repilo-model.h5')

    submission_df = pd.DataFrame (images, columns=['Imagen analizada'])
    submission_df [['Hoja con repilo', 'Hoja sin repilo']] = model.predict (submission)
    return submission_df

