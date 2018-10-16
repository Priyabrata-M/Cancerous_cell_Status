import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras import backend as K
from keras.layers import Input
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout, Add
from keras.models import load_model
from keras.layers import concatenate
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
 

vgg16_model = VGG16()
vgg16_model.summary()

model = Sequential()
for layer in vgg16_model.layers:
    model.add(layer)
    
model.layers.pop()

for layer in model.layers:
    layer.trainabel = False

model.add(Dense(2, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam')
train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('ExperimentDataSet/TrainingData', # Path to the target directory. It should contain one subdirectory per class.
target_size = (64, 64),
batch_size = 20,
class_mode = 'binary')


test_set = test_datagen.flow_from_directory('ExperimentDataSet/ValidationData',
target_size = (64, 64),
batch_size = 25, #if the size of the dataset is not divisible by the batch size. The generator is expected to loop over its data indefinitely. 
class_mode = 'binary')


model.fit_generator(training_set,
steps_per_epoch = 1744,#Total number of steps (batches of samples) to yield from generator before declaring one epoch finished and starting the next epoch
epochs = 25,
validation_data = test_set,
validation_steps = 155)




score = model.evaluate_generator(test_set, 155, workers=1)
scores = model.predict_generator(test_set, 155, workers=1)
correct = 0


model.save('VGGTrained_.h5') 