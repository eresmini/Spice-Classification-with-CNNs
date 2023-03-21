# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 21:20:27 2022

@author: Emma Resmini
"""

######################
# Emma Resmini
# Semester Project
######################

# packages for this program
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#%%
### LOADING IMAGES AND PROCESSING

# this is the file path where my images are
# change this as necessary depending on where your images are on your computer!!
# NOTE: the 31 spice folders MUST be within the folder "train" - for example, if you were to navigate to the Alum Folder, the path would be G:\Spices\train\Alum
data_dir = "D:\Spice Dataset"

# function obtained from the Keras documentation page: https://keras.io/api/preprocessing/image/
# as long as the file path is correct, image_dataset_from_directory() will return a dataset with batches of images from the subdirectories
# indication "training" (and later "validation") in the subset parameter indicates to the functiont to split the dataset; validation_split=0.2 means 20% of the data will be reserved for validation
# data is shuffled as it's read in; seed is set to obtain same dataset across runs
train = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  labels = "inferred",
  label_mode="int",
  validation_split= 0.2,
  subset="training",
  seed = 1,
  image_size=(256, 256), # the images are already 256x256, so indicating that in the image_size parameter prevents the function from resizing the images any further
  batch_size = 1) # batch = 1 reads in all images

# same method to obtain validation set
validation = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  labels = "inferred",
  label_mode="int",
  validation_split = 0.2,
  subset="validation",
  seed = 1,
  image_size=(256, 256),
  batch_size = 1)

#%%
### BUILDING THE CNN MODEL
# referenced the class materials and a Tensorflow tutorials (https://www.tensorflow.org/) for initial start
# use the Keras Sequential class to build the layers of the model
# played around with different numbers of layers, but overall adding or dropping did not seem to make much of a difference in model fitting, so I opted for fewer layers
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1. / 255), # normalize rbg values
    tf.keras.layers.Conv2D(32, 2, (3,3), padding = "same", activation = "relu", input_shape = (256,256,3)), # increase filter
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Conv2D(64, 2, (3,3), padding = "same", activation = "relu"), # increase filter
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(31, activation = 'softmax')
])

#%%

### COMPILE AND RUN THE MODEL

# compile model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# Fitting the model takes a while (an hour)...so heads up :)
model_hist = model.fit(train, validation_data = validation, epochs = 30)

#%%
### PLOT METRICS

# Loss graph
# plot the training loss
plt.plot(model_hist.history['loss'], label='train loss')
# add validation loss to plot
plt.plot(model_hist.history['val_loss'], label='val loss')
plt.xlabel("epochs") # x-axis label
plt.ylabel("validation loss") # y-axis label
plt.legend() # legend to clarify which line is training and which is validation
plt.show()

# Accuracy Graph
# plot training accuracy
plt.plot(model_hist.history['accuracy'], label='train acc')
# add validation accuracy to plot
plt.plot(model_hist.history['val_accuracy'], label='val acc')
plt.xlabel("epochs")
plt.ylabel("validation accuracy")
plt.legend()
plt.show()

#%%
### MAKE PREDICTION ON ONE SPICE

### Try these images below...
# Cinnamon: Cinnamon/IMG_8015.jpg_num_102.jpg
# Coriander: Coriander/IMG_8099.jpg_num_104.jpg
# Cream of Tartar: Cream of Tartar/IMG_8526.jpg_num_026.jpg
# Oregano: Oregano/IMG_8396.jpg_num_123.jpg
# Salt: Iodized Salt/IMG_7825.jpg_num_066.jpg


# give keras the image file path
img = keras.preprocessing.image.load_img('G:/Spice Dataset/train/Oregano/IMG_8396.jpg_num_123.jpg',
                                         target_size = (256, 256, 3),
                                         interpolation= 'nearest')
# save image to array
img_array = keras.preprocessing.image.img_to_array(img)
# create batch axis
img_array = tf.expand_dims(img_array, 0)  
# run model predictions on the image
img_predictions = model.predict(img_array)

# this returns the value that model predicts
predicted_classes = np.argmax(img_predictions, axis=1)
# array of all class names
class_names = validation.class_names

# show image
plt.imshow(img)
# print prediction
print('\n', class_names[int(predicted_classes)])