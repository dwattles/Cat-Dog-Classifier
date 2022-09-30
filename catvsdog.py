# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 13:40:48 2022

@author: dwatt
"""

import os
import zipfile

from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

"""loc_zip = 'dogs-vs-cats.zip'
zip_ref = zipfile.ZipFile(loc_zip,'r')
zip_ref.extractall()
zip_ref.close()

loc_zip2 = 'test1.zip'
zip_ref = zipfile.ZipFile(loc_zip2,'r')
zip_ref.extractall()
zip_ref.close()"""

base_dir ='sample_data'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'validation')

img_input=layers.Input(shape=(150,150,3))

x = layers.Conv2D(16,3,activation='relu')(img_input)
x = layers.MaxPooling2D(2)(x)

x = layers.Conv2D(32,3,activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

x = layers.Conv2D(64,3,activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

x = layers.Conv2D(128,3,activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

x = layers.Flatten()(x)
x = layers.Dense(1024,activation='relu')(x)
x = layers.Dropout(0.3)(x)

output = layers.Dense(1,activation='sigmoid')(x)

model = Model(img_input,output)

#model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),#0.001RMSprop
              metrics=['acc'])

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(150,150),
                                                    batch_size=100,#50
                                                    class_mode='binary')
val_generator =  val_datagen.flow_from_directory(val_dir,
                                                 target_size=(150,150),
                                                 batch_size=40,#40
                                                 class_mode='binary')

# steps*batch_size=num_ims
history = model.fit(train_generator,
                    steps_per_epoch=100,#120
                    epochs=55,#30 for orig
                    validation_data=val_generator,
                    validation_steps=50,
                    verbose=2)


import matplotlib.pyplot as plt

# Retrieve a list of accuracy results on training and validation data
# sets for each training epoch
acc = history.history['acc']
val_acc = history.history['val_acc']

# Retrieve a list of list results on training and validation data
# sets for each training epoch
loss = history.history['loss']
val_loss = history.history['val_loss']

# Get number of epochs
epochs = range(len(acc))

# Plot training and validation accuracy per epoch
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and validation accuracy')

plt.figure()

# Plot training and validation loss per epoch
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Training and validation loss')

"""import signal
os.kill(os.getpid(), signal.SIGINT)"""

#model.predict()

#visualize features network learns???