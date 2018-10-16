from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import load_model
import numpy as np
from keras.preprocessing import image
import cv2
np.set_printoptions(threshold='nan')
from keras.preprocessing.image import ImageDataGenerator




#img = cv2.imread('Dataset/Training/dummy/CT1.2.840.113619.2.55.3.1871697879.431.1220356059.802.1.jpg')
#print img.shape

print (1)
# Initialising the CNN
classifier = Sequential()
# Step 1 - Convolution
classifier.add(Conv2D(32,10, input_shape = (64, 64, 3),activation = 'relu'))
# Step 2 - Pooling
#classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Adding a second convolutional layer
classifier.add(Conv2D(32,10, activation = 'relu'))
#classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Adding a third convolutional layer
#classifier.add(Conv2D(32, 10, activation = 'relu'))
#classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Adding a fourth convolutional layer
#classifier.add(Conv2D(32,10, activation = 'relu'))
#classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Step 3 - Flattening
classifier.add(Flatten())
# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))
# Compiling the CNN

print (2)

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

print (3)

train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)

print (4)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('ExperimentDataSet/TrainingData', # Path to the target directory. It should contain one subdirectory per class.
target_size = (64, 64),
batch_size = 20,
class_mode = 'binary')

print (5)
test_set = test_datagen.flow_from_directory('ExperimentDataSet/ValidationData',
target_size = (64, 64),
batch_size = 25, #if the size of the dataset is not divisible by the batch size. The generator is expected to loop over its data indefinitely. 
class_mode = 'binary')


#print training_set[0][0][0][0]



#
#
classifier.fit_generator(training_set,
steps_per_epoch = 1744,#Total number of steps (batches of samples) to yield from generator before declaring one epoch finished and starting the next epoch
epochs = 2,
validation_data = test_set,
validation_steps = 155)

print (6)

score = classifier.evaluate_generator(test_set, 155, workers=1)
scores = classifier.predict_generator(test_set, 155, workers=1)
correct = 0
#
#print (score)
#print (scores)


classifier.save('_PreTrainedCNN.h5') 


#print (7)
#for i, n in enumerate(test_set.filenames):
#    print (i,n,"=====")
#    if n.startswith("positive") and scores[i][0] <= 0.5:
#        correct += 1
#    if n.startswith("negative") and scores[i][0] > 0.5:
#        correct += 1
#        
#print (8)
#print("Correct:", correct, " Total: ", len(test_set.filenames))
#print("Loss: ", score[0], "Accuracy: ", score[1])
#
#



#
#from keras.models import load_model
#
#model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
#del model  # deletes the existing model
#
## returns a compiled model
## identical to the previous one
#model = load_model('my_model.h5')
