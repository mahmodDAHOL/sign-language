import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
import itertools
from pathlib import Path


def savefig(true_false_files: Path, results_path: str):
    class_names=['a','b','c','d','e','f','g','h','i','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y']
    
    for true_false_file in true_false_files.glob("*"):
        sub = pd.read_csv(true_false_file)

        y_pred = np.array(sub.pop('Label'))
        y_test = np.array(sub.pop('TrueLabel'))

        # Compute confusion matrix
        cnf_matrix = confusion_matrix(y_test, y_pred)
        # np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plt.figure()
        score = f1_score(y_test, y_pred, average="micro")
        plot_confusion_matrix(cnf_matrix, classes=class_names,
                            title=f'f1score = {score}')

        print(f"{results_path}\\{true_false_file.name.split('.')[0]}.png")
        plt.savefig(f"{results_path}\\{true_false_file.name.split('.')[0]}.png")
    

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)


    for i in cm:
        a=0
        for j in i:
            a=a+j

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

