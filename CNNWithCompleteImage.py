#CNN with image as an extra input
#https://github.com/keras-team/keras/issues/1913



import numpy as np
from keras.datasets import mnist
from keras.layers import Dense, Flatten, Input, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, BatchNormalization
from keras.models import Model
from PIL import Image
from keras.utils import to_categorical
#from keras.models import Graph
from .legacy.models import Graph
from keras.models import load_model
import numpy as np
from keras.preprocessing import image
import cv2
from keras.preprocessing.image import ImageDataGenerator
np.set_printoptions(threshold='nan')


input_shape=(64, 64,3 )

input_layer = Input(input_shape)
#model = Model(input=input_layer, output=x)

#x = Conv2D(32, (3,3), activation='relu')(input_layer)

print 1
classifier=Graph()


classifier.add(Conv2D(32,10, input_shape = (64, 64, 3),activation = 'relu'))
classifier.add(Conv2D(32,10, activation = 'relu'))
classifier.add(Flatten())
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

print 2

classifier.summary()

print 3

classifier.compile(loss='binary_crossentropy', optimizer='adam')
train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)
classifier.save("normal.h5");

print 4

test_datagen = ImageDataGenerator(rescale = 1./255)

print 5

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


model.save('_PreTrainedCNN.h5') 


