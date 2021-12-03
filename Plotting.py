#!/usr/bin/env python
# coding: utf-8

# In[2]:


import librosa
import numpy as np
import pandas as pd
import librosa.display
import matplotlib.pyplot as plt
import keras
import soundfile as sf
import csv


# In[4]:


# Function to plot error-accuracy
def results(history, val_data, model, title):
    acc_history = history.history['accuracy']
    val_acc_history = history.history['val_accuracy']

    plt.plot(range(1, len(acc_history) + 1), acc_history)
    plt.plot(range(1, len(val_acc_history) + 1), val_acc_history)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['training', 'validation'])
    plt.title(title, " accuracy vs epochs")
    plt.show()

    loss_history = history.history['loss']
    val_loss_history = history.history['val_loss']

    plt.plot(range(1, len(loss_history) + 1), loss_history)
    plt.plot(range(1, len(val_loss_history) + 1), val_loss_history)
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.legend(['training', 'validation'])
    plt.title(title, " loss vs epochs")
    plt.show()
    return


# In[1]:


from sklearn.metrics import confusion_matrix
import itertools
import numpy as np
# https://github.com/Rukshani/Medium/blob/main/ConfusionMatrix/MediumConfusionMatrix.ipynb
def plot_confusion_matrix(predicted_labels_list, y_test_list):
    cnf_matrix = confusion_matrix(y_test_list, predicted_labels_list)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    generate_confusion_matrix(cnf_matrix, classes=[0,1,2,3,4,5,6,7,8], title='Confusion matrix, without normalization')
    plt.show()
    
    # Plot normalized confusion matrix
    plt.figure()
    generate_confusion_matrix(cnf_matrix, classes=[0,1,2,3,4,5,6,7,8], normalize=True, title='Normalized confusion matrix')
    plt.show()
    return


# In[4]:


def generate_confusion_matrix(cnf_matrix, classes, normalize=False, title='Confusion matrix'):
    if normalize:
        cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cnf_matrix.max() / 2.

    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, format(cnf_matrix[i, j], fmt), horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return cnf_matrix


# In[ ]:


def accuracies (folds, accuracies):
    for i in range(folds):
        plt.plot(accuracies[i].history['accuracy'], label='Fold '+ str(i+1))
    plt.title('Accuracies vs Epochs')
    plt.legend()
    plt.show()
    return


# In[ ]:


def errors (folds, accuracies):
    for i in range(folds):
        plt.plot(accuracies[i].history['loss'], label='Fold '+ str(i+1))
    plt.title('Loss vs Epochs')
    plt.legend()
    plt.show()
    return


# In[ ]:


def avg_acc(folds, values):
    fig = plt.figure(figsize = (7, 5))
 
    # creating the bar plot
    plt.bar(folds, values, color ='Blue',
        width = 0.4)

    # average
    plt.axhline(y=np.mean(values),linewidth=3, color='red', label = "Average Accuracy")

    plt.xlabel("Fold")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Fold")
    plt.legend()
    plt.show()
    return

