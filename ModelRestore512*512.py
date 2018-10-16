#Model Restoring
#from keras.models import Sequential
#from keras.layers import Conv2D
#from keras.layers import MaxPooling2D
#from keras.layers import Flatten
#from keras.layers import Dense
from keras.models import load_model
import numpy as np
#from keras.preprocessing import image
#from keras.preprocessing.image import ImageDataGenerator
import cv2
import pandas as pd
import os
import shutil# import copytree
np.set_printoptions(threshold='nan')

df = pd.read_csv("Test.csv")
#print df.iloc[0,0]
#print df.iloc[0,1]
#print df.iloc[0,2]
#print df['Patient_ID'].count()

print "******loading started*****"
#test_datagen = ImageDataGenerator(rescale = 1./255)
classifier = load_model('VGGTrained_.h5')
i=0
#for i in range(0,df['Patient_ID'].count()):
    
print "Write stated***********"
#f= open("guru99.txt","a")
for i in range(0,int(df['Patient_ID'].count())):
    print str(i)+"===>"
    Parent_DIR='test_test'
    DIR='test_test'+"/"+"Case_"+ str(df.iloc[i,0])
    shutil.copytree("Test_PNG"+"/"+"Case_"+ str(df.iloc[i,0]), DIR)
    noOffiles= len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
#    print noOffiles
    test_set = test_datagen.flow_from_directory(Parent_DIR,
    target_size = (224, 224),
    batch_size = 1,
    class_mode = 'binary') 
    score = classifier.evaluate_generator(test_set, noOffiles, workers=1)
    predictedScore = classifier.predict_generator(test_set, noOffiles, workers=1)
#    print predictedScore
    countgreater = sum(i > .5 for i in predictedScore)
    countSmaller = sum(i < .5 for i in predictedScore)
    if(countgreater>countSmaller):
        print "1"
        f.write("1")
        f.write("\n")
#        df.iloc[i,0]="1"
    else:
        f.write("0")
        f.write("\n")
        print "0"
#        df.iloc[i,0]="0"
    shutil.rmtree(DIR) 

