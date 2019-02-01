# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 14:14:17 2018

@author: Tamer
"""

# Part 1 - Building the CNN

# Importing the required Keras libraries and packages

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initializing the CNN

classifier = Sequential()

# Step 1 - Convolution
# Number of filters begin with 32 or 64, number of rows and number of columns
# input_shape = (first dimention, second dimention, color shape) 3 means 3 channel for colored image
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation= 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding a second convolutional layer
# We removed input shape as we want only the number of feature detectors, the dimention of these
# feature detector and the activation function as it's the second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation= 'relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full Connection
# Hidden layer thet is full connection
classifier.add(Dense(units=128, activation='relu'))
# Output layer
classifier.add(Dense(activation='sigmoid', units=1))

# compile the CNN
# Choose the same stochastic gradient desent as in ANN adam
classifier.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])


# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                    steps_per_epoch=8000,
                    epochs=5,
                    validation_data=test_set,
                    validation_steps=2000)



import numpy as np

from keras.preprocessing import image

test_image = image.load_img('dataset/compet/dog3.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

result = classifier.predict(test_image)

training_set.class_indices

if result[0][0] == 0:
    prediction = 'Cat'
else:
    prediction = 'Dog'
    
print(prediction)







