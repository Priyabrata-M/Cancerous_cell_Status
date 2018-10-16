#Copy files
import os
import shutil

sourceDir = 'SampleTrainingData/x'
distDir = 'SampleTrainingData/x_final'



for casefolder in os.listdir(sourceDir):
    print casefolder
    for f in os.listdir(sourceDir+"/"+casefolder):
#        dstdir =  os.path.join(distDir, os.path.dirname(f))
        distDir = f
        shutil.copyfile(sourceDir+"/"+casefolder+"/"+f, distDir)
    