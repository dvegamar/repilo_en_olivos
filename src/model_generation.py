# this script creates de model from the dataset, can be run independently of streamlit

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras import callbacks
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense

################# TRAIN AND VALIDATION DATASETS, USING AUGMENTED GENERATOR:   #################
# this method splits de train dataset for validation automatically
# Data augmentation generator

batch_size = 32
image_generator = ImageDataGenerator (
    rescale=1 / 255,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.2, 1.2],
    validation_split=0.2, )

# Train & Validation Split
train_dataset = image_generator.flow_from_directory (batch_size=batch_size,
                                                     directory='../data/olives/train',
                                                     shuffle=True,
                                                     target_size=(224,224),
                                                     subset="training",
                                                     class_mode='categorical')

validation_dataset = image_generator.flow_from_directory (batch_size=batch_size,
                                                          directory='../data/olives/train',
                                                          shuffle=True,
                                                          target_size=(224,224),
                                                          subset="validation",
                                                          class_mode='categorical')

################# MODEL DEFINITIONS   #################

#esto es el olive_repilo-model.h5
model = Sequential ()
model.add (Conv2D (32, (3, 3), activation='relu', input_shape=[224, 224, 3])),
model.add (MaxPooling2D ()),
model.add (Conv2D (64, (2, 2), activation='relu')),
model.add (MaxPooling2D ()),
model.add (Conv2D (64, (2, 2), activation='relu')),
model.add (Flatten ()),
model.add (Dense (50, activation='relu')),
model.add (Dense (2, activation='softmax'))



################# MODEL COMPILATION     ###############

model.compile (optimizer='adam',
               loss='binary_crossentropy',
               metrics=['accuracy'])

callback = callbacks.EarlyStopping (monitor='val_loss', patience=3,restore_best_weights=True)

######################### MODEL FIT   ###################

history = model.fit(
    train_dataset,
    epochs=15,
    steps_per_epoch= train_dataset.samples // batch_size,
    validation_data=validation_dataset,
    validation_steps= validation_dataset.samples // batch_size,
    callbacks=callback
    )

######################### MODEL SAVE   #################


model.save ('../model/olive_repilo-model.h5', save_format="h5")

