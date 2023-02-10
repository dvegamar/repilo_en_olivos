# this script test de model and gives statistics

from keras.models import load_model
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



def fill_pred_df(test_folder):

    images = []
    for file in os.listdir (test_folder):
        images.append (file)

    image_generator_test = ImageDataGenerator (rescale=1 / 255)

    test_set = image_generator_test.flow_from_directory (
        directory=test_folder,
        # important, keras is expecting for subfolders, if images were in user_files --> error; or USE:  classes=['.']
        batch_size=1,
        shuffle=False,
        target_size=(224, 224),
        classes=['.'],
        class_mode=None,
    )

    model = load_model ('../model/olive_repilo-model.h5')

    prediction_df_partial = pd.DataFrame()
    prediction_df_partial [['Hoja con repilo', 'Hoja sin repilo']] = model.predict (test_set)
    return prediction_df_partial



prediction_df=pd.DataFrame()
# 0 is a healthy leaf, 1 means it has repilo

test_folder = '../data/olives/test/sanas'
prediction_df_partial = fill_pred_df(test_folder)
prediction_df_partial['Carpeta']='Sanas'
prediction_df_partial['y_real']=0
prediction_df = pd.concat([prediction_df,prediction_df_partial],axis=0)

test_folder = '../data/olives/test/repilo'
prediction_df_partial = fill_pred_df(test_folder)
prediction_df_partial['Carpeta']='Repilo'
prediction_df_partial['y_real']=1
prediction_df = pd.concat([prediction_df,prediction_df_partial],axis=0)

prediction_df['y_pred'] = [1 if x > 0.5 else 0 for x in prediction_df['Hoja con repilo']]


print(prediction_df.head())
print(prediction_df.tail())
print(prediction_df.shape)


# Calculate confusion matrix
conf_matrix = pd.crosstab(prediction_df['y_real'], prediction_df['y_pred'], rownames=['Actual'], colnames=['Predicted'])
print('Matriz de confusion', conf_matrix)
conf_matrix_perc = pd.crosstab(prediction_df['y_real'], prediction_df['y_pred'], rownames=['Actual'], colnames=['Predicted'], normalize=True)*100
print('Matriz de confusion en porcentajes',conf_matrix_perc)

# Calculate MAE
mae = mean_absolute_error(prediction_df['y_real'], prediction_df['y_pred'])
print("Mean Absolute Error: ", mae)

# Calculate MSE
mse = mean_squared_error(prediction_df['y_real'], prediction_df['y_pred'])
print("Mean Squared Error: ", mse)

# Calculate R-Squared
r2 = r2_score(prediction_df['y_real'], prediction_df['y_pred'])
print("R-Squared: ", r2)