import numpy as np
import cv2
import os
import csv
from image_processing import func

path = r"C:\Users\FPCC\hand_gesture_project\dataset"
a=[]

# "a" is list of number in range [0, 9215] that represents number of features that
# we will train on
# i pick maximum number from my dataset and it was 183552 but you can take first 9215
# this number is enough for training
for i in range(9216):
    a.append("pixel"+str(i))
    

#outputLine = a.tolist()

# create a csv file (excel file)
with open(r'C:\Users\FPCC\hand_gesture_project\dataset\train60.csv', 'w') as csvfile:
    # initializing writer object that take file name and list of columns name 
    writer = csv.DictWriter(csvfile, fieldnames = a)
    writer.writeheader()
    
with open(r'C:\Users\FPCC\hand_gesture_project\dataset\train40.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames = a)
    writer.writeheader()
label=0

with open(r'C:\Users\FPCC\hand_gesture_project\dataset\train60.csv', 'a') as csvfile:
    # initialize witer object for writing in excel file (train60.csv)
    spamwriter = csv.writer(csvfile)
    with open(r'C:\Users\FPCC\hand_gesture_project\dataset\train40.csv', 'a') as csvf:
        writer=csv.writer(csvf)
        # os.walk return an iterator that yield full dirpath for every folder in path 
        # that you path it to walk() 
        # and dirnames that is list of names of folders inside path
        # and file names that is list of names of files inside each folder
        for (dirpath,dirnames,filenames) in os.walk(path):
            for dirname in dirnames:
                print(dirname)
                for(direcpath,direcnames,files) in os.walk(path+"\\"+dirname):
                    # num variable for take 60% from dataset for every class for training
                    num=0.6*len(files)
                    i=0
                    for file in files:
                        # actual_path is path for different image for every iteration
                        actual_path=path+"\\\\"+dirname+"\\\\"+file
                        print(actual_path)
                        # this func() is that we defined in sift_image_processing.py
                        bw_image=func(actual_path)
                        # flatten() for convert bw_image to vector with one dimension
                        # ex: (639, 128) => (639*128) = (81792,)
                        flattened_sign_image=bw_image.flatten()
                        # label variable will increase by 1 when the loop move to next class (folder name)
                        # to represent it as a column in excel file
                        outputLine = [label] + np.array(flattened_sign_image).tolist()
                        # if i less than num (number of images for training => write the list of
                        #                       descriptor for training excel file)
                        if i<num:
                            spamwriter.writerow(outputLine)
                        # if i greater than num (number of images for training => write the list of
                        #                       descriptor for testing excel file)
                        else:
                            writer.writerow(outputLine)
                        i=i+1
                        
                label=label+1






