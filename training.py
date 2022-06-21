import os
import cv2
from pathlib import Path
from visualize_submissions import savefig
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from data_manipulation import DataManipulation
from ml_algorithms import MLAlgorithms

project_path = r"C:\Users\FPCC\New folder"
dataset_path = os.path.join(project_path, "dataset")

label = 0
img_descs = []
y = []


# this function take path for image and return descriptors list of that image
def extract_sift(path):
    frame = cv2.imread(path)

    frame = cv2.resize(frame, (128, 128))

    # convert image from RGB to gray  (8 bit for every pixel)
    converted2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # convert image from RGB to HSV  (HSV is a color model that
    #                                less sensitive to shadow in the image)
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # lowerBoundary and upperBoundary are values that have been experimented by the author
    # to determine  H, S and V (HSV values) to mask the object that we intrested in
    # (the hand in this project)
    lowerBoundary = np.array([0, 40, 30], dtype="uint8")
    upperBoundary = np.array([43, 255, 254], dtype="uint8")

    # after determine the boundary we will use inRange method from opencv library
    # for applying these values to mask the hand
    # the mask mean convert the hand to white and background to black
    skinMask = cv2.inRange(converted, lowerBoundary, upperBoundary)
    # addWeight method is for overlaying one image over another image
    # see https://www.geeksforgeeks.org/opencv-alpha-blending-and-masking-of-images/
    skinMask = cv2.addWeighted(skinMask, 0.5, skinMask, 0.5, 0.0)

    # medianBlur method for applying filter with size 5*5 on image to eliminating the noise
    skinMask = cv2.medianBlur(skinMask, 5)

    # this line for apply the mask that we created for make the hand white, background black
    skin = cv2.bitwise_and(converted2, converted2, mask=skinMask)

    # Canny method for edge detection and 60 is hyperparameter
    # see https://www.geeksforgeeks.org/python-opencv-canny-function/
    img2 = cv2.Canny(skin, 60, 60)

    # initializing sift algorithm that extract descriptors that we will use them for training
    # classification model
    sift = cv2.xfeatures2d.SIFT_create()
    # resizing
    img2 = cv2.resize(img2, (256, 256))

    # useing detectAndCompute method for apply the algorithm on the image
    # it return des variable (descriptors) list of points that describe the feature of image
    # and kp variable (keypoits) that will show on image as points
    # for more info see https://www.youtube.com/watch?v=DZtUt4bKtmY
    kp, des = sift.detectAndCompute(img2, None)

    # drawKeypoints method for draw the points (des) on the image
    img2 = cv2.drawKeypoints(img2, kp, None, (0, 0, 255), 4)

    return des


# creating desc for each file with label
for train_test in ["train", "test"]:
    for (dirpath, dirnames, filenames) in os.walk(os.path.join(dataset_path, train_test)):
        for dirname in dirnames:
            print(dirname)
            for(direcpath, direcnames, files) in os.walk(os.path.join(dataset_path, train_test, dirname)):
                for file in files:
                    actual_path = os.path.join(
                        dataset_path, train_test, dirname, file)
                    des = extract_sift(actual_path)
                    img_descs.append(des)
                    y.append(label)
            label = label+1

# finding indexes of test train and validate
y = np.array(y)
data = DataManipulation(0.2,0, project_path)
data.train_test_val_split_idxs(len(img_descs))

# creating histogram using kmeans minibatch cluster model
model = MiniBatchKMeans(batch_size=1024, n_clusters=150)
X = data.cluster_features(img_descs, model)

# splitting data into test, train, validate using the indexes
X_train, X_test, X_val, y_train, y_test, y_val = data.perform_data_split(X, y)

algorithm = MLAlgorithms(project_path, 32, 30)
# using classification methods
algorithm.predict_knn(X_train, X_test, y_train, y_test)
algorithm.predict_mlp(X_train, X_test, y_train, y_test)
algorithm.predict_svm(X_train, X_test, y_train, y_test)

algorithm.predict_lr(X_train, X_test, y_train, y_test)
algorithm.predict_nb(X_train, X_test, y_train, y_test)

algorithm.predict_cnn(project_path)

# save confusion matrix figures for test data 
true_false_files = Path(f"{project_path}\\true_false_files") 
results_path = f"{project_path}\\results"

savefig(true_false_files, results_path)

# this code below for print the maximum number of features in dataset
# path = f"{project_path}\\dataset"
# max = 0
# for (dirpath,dirnames,filenames) in os.walk(path):
#     for dirname in dirnames:
#         print(dirname)
#         for(direcpath,direcnames,files) in os.walk(path+"\\"+dirname):
#             # num variable for take 60% from dataset for every class for training
#             num=0.6*len(files)
#             i=0
#             for file in files:
#                 # actual_path is path for different image for every iteration
#                 actual_path=path+"\\\\"+dirname+"\\\\"+file
#                 # print(actual_path)
#                 des = func(actual_path)
#                 if des.flatten().shape[0]>max:
#                     max = des.flatten().shape[0]

# print(max)
