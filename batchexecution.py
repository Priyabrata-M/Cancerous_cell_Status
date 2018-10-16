import os
import shutil


path='DataPreProcessing/512_Folderstr/Training_PNG'
for case in  os.listdir(path):
    print case
    length= len(os.listdir(path+'/'+case))
    for j in  range(0,len(os.listdir(path+'/'+case))):
        shutil.copy2(path+'/'+case+'/'+, '/Sample/1')