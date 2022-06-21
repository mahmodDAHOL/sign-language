import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling2D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model

class CNN:
    def __init__(self, project_path):
        self.project_path = project_path
        self.test_path = f"{self.project_path}\\dataset\\test"
        self.train_path = f"{self.project_path}\\dataset\\train"
        self.input_shape = (128, 128, 1)
        self.n_classes = 24

    # Loading images
    # Converting images to an size of (128,128)
    def load_images(self, folder):
        data = []
        for label in os.listdir(folder):
            path = folder+'\\'+label
            for img in os.listdir(path):
                img = cv2.imread(path+'\\'+img, cv2.IMREAD_GRAYSCALE)
                new_img = cv2.resize(img, (128, 128))
                if new_img is not None:
                    data.append([new_img, label])
        return data

    def create_model(self):
        self.model = Sequential()
        # The first two layers with 32 filters of window size 3x3
        self.model.add(Conv2D(32, (3, 3), padding='same',
                         activation='relu', input_shape=self.input_shape))
        self.model.add(Conv2D(32, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.n_classes, activation='softmax'))
        return self.model

    def preprocess(self, data):

        # Seperating features and labels
        test_images = []
        test_labels = []

        for feature, label in data:
            test_images.append(feature)
            test_labels.append(label)

        # Converting images list to numpy array
        test_images = np.array(test_images)
        test_images = test_images.reshape((-1, 128, 128, 1))
        # Changing the datatype and Normalizing the data
        test_images = test_images.astype('float32')

        test_images = test_images/255.0

        # Encoding the label values
        le = LabelEncoder()
        le.fit_transform(test_labels)
        test_labels_label_encoded = le.transform(test_labels)

        # return test_labels_label_encoded, test_images
        return test_labels_label_encoded, test_images

    def predict_cnn(self, test_path, submission_cnn_path):
        # load the data from test_path
        test_data = self.load_images(test_path)
        test_labels, test_images = self.preprocess(test_data)

        # load the model from disk
        file_path = f"{self.project_path}\\models\\CNN.h5"
        cnn_model = load_model(file_path)
        y_pred = cnn_model.predict(test_images)
        y_pred = np.argmax(y_pred, axis=1)

        np.savetxt(submission_cnn_path, np.c_[range(1, len(
            test_labels)+1), y_pred, test_labels],
            delimiter=',', header='ImageId,Label,TrueLabel', comments='', fmt='%d')

    def plot_accuracy_and_loss(self, history, cnn):

        # Visualizing loss
        plt.figure(figsize=[8, 6])
        plt.plot(history.history['loss'], 'r', linewidth=2.0)
        plt.plot(history.history['val_loss'], 'b', linewidth=2.0)
        plt.legend(['Training loss', 'Validation Loss'], fontsize=15)
        plt.xlabel('Epochs ', fontsize=16)
        plt.ylabel('Loss', fontsize=16)
        plt.title('Loss Curves', fontsize=16)
        plt.savefig(
            f"{cnn.project_path}\\results\\cnn_training_result\\loss.png")

        # Visualizing accuracy
        plt.figure(figsize=[8, 6])
        plt.plot(history.history['accuracy'], 'r', linewidth=2.0)
        plt.plot(history.history['val_accuracy'], 'b', linewidth=2.0)
        plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=15)
        plt.xlabel('Epochs ', fontsize=16)
        plt.ylabel('Accuracy', fontsize=16)
        plt.title('Accuracy Curves', fontsize=16)
        plt.savefig(
            f"{cnn.project_path}\\results\\cnn_training_result\\acc.png")
