import pickle

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as sm
from sklearn.linear_model import LogisticRegression as lr
from sklearn.naive_bayes import GaussianNB as nb
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.neural_network import MLPClassifier as mlp
from sklearn.svm import SVC

from cnn import CNN
from data_manipulation import DataManipulation


class MLAlgorithms:

    def __init__(self, project_path, batch_size, epochs):
        self.project_path = project_path

        self.batch_size = batch_size
        self.epochs = epochs

    def predict_svm(self, X_train, X_test, y_train, y_test):
        svc = SVC(kernel='linear')
        print("svm started")
        svc.fit(X_train, y_train)
        y_pred = svc.predict(X_test)
        self.calc_accuracy("SVM", y_test, y_pred)

        # save the model to file
        file_name = f"{self.project_path}\\models\\SVM"
        outfile = open(file_name, 'wb')
        pickle.dump(svc, outfile)
        outfile.close()

        np.savetxt(f'{self.project_path}\\true_false_files\\submission_sift_svm.csv', np.c_[range(1, len(
            y_test)+1), y_pred, y_test], delimiter=',', header='ImageId,Label,TrueLabel', comments='', fmt='%d')

    def predict_lr(self, X_train, X_test, y_train, y_test):
        clf = lr()
        print("lr started")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        self.calc_accuracy("Logistic regression", y_test, y_pred)
        np.savetxt(f'{self.project_path}\\true_false_files\\submission_sift_lr.csv', np.c_[range(1, len(
            y_test)+1), y_pred, y_test], delimiter=',', header='ImageId,Label,TrueLabel', comments='', fmt='%d')

        # save the model to file
        file_name = f"{self.project_path}\\models\\LOGISTIC_REGRESSION"
        outfile = open(file_name, 'wb')
        pickle.dump(clf, outfile)
        outfile.close()

    def predict_nb(self, X_train, X_test, y_train, y_test):
        clf = nb()
        print("nb started")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        self.calc_accuracy("Naive Bayes", y_test, y_pred)
        np.savetxt(f'{self.project_path}\\true_false_files\\submission_sift_nb.csv', np.c_[range(1, len(
            y_test)+1), y_pred, y_test], delimiter=',', header='ImageId,Label,TrueLabel', comments='', fmt='%d')

        # save the model to file
        file_name = f"{self.project_path}\\models\\NAIVE_BAYES"
        outfile = open(file_name, 'wb')
        pickle.dump(clf, outfile)
        outfile.close()

    def predict_knn(self, X_train, X_test, y_train, y_test):
        clf = knn(n_neighbors=3)
        print("knn started")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        self.calc_accuracy("K nearest neighbours", y_test, y_pred)
        np.savetxt(f'{self.project_path}\\true_false_files\\submission_sift_knn.csv', np.c_[range(1, len(
            y_test)+1), y_pred, y_test], delimiter=',', header='ImageId,Label,TrueLabel', comments='', fmt='%d')

        # save the model to file
        file_name = f"{self.project_path}\\models\\KNN"
        outfile = open(file_name, 'wb')
        pickle.dump(clf, outfile)
        outfile.close()

    def predict_mlp(self, X_train, X_test, y_train, y_test):
        clf = mlp()
        print("mlp started")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        self.calc_accuracy("MLP classifier", y_test, y_pred)
        np.savetxt(f'{self.project_path}\\true_false_files\\submission_sift_mlp.csv', np.c_[range(1, len(
            y_test)+1), y_pred, y_test], delimiter=',', header='ImageId,Label,TrueLabel', comments='', fmt='%d')


        # save the model to file
        file_name = f"{self.project_path}\\models\\ML_PERSEPTRON"
        outfile = open(file_name, 'wb')
        pickle.dump(clf, outfile)
        outfile.close()

    def predict_cnn(self, project_path):
        cnn = CNN(self.project_path)
        train_images, train_labels_one_hot, test_images, test_labels_one_hot = \
            DataManipulation.cnn_preprocess(self.project_path)
        model = cnn.create_model()

        model.compile(optimizer='adam', loss='categorical_crossentropy',
                      metrics=['accuracy'])

        model.summary()

        # training the model
        history = model.fit(train_images, train_labels_one_hot, batch_size=self.batch_size,
                            epochs=self.epochs, verbose=1, validation_data=(
                                test_images, test_labels_one_hot))

        file_name = f"{self.project_path}\\models\\CNN.h5"
        model.save(file_name)

        test_path = f"{self.project_path}\\dataset\\test"
        submission_cnn_path = f"{self.project_path}\\true_false_files\\submission_cnn.csv"
        cnn.predict_cnn(test_path, submission_cnn_path)
        cnn.plot_accuracy_and_loss(history, cnn)

    def calc_accuracy(self, method, label_test, pred):
        print("accuracy score for ", method,
              sm.accuracy_score(label_test, pred))
        print("precision_score for ", method, sm.precision_score(
            label_test, pred, average='micro'))
        print("f1 score for ", method, sm.f1_score(
            label_test, pred, average='micro'))
        print("recall score for ", method, sm.recall_score(
            label_test, pred, average='micro'))
