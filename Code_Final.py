#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import main libraries
import librosa
import numpy as np
import pandas as pd
import librosa.display
import matplotlib.pyplot as plt
import keras
import soundfile as sf
import csv


# In[1]:


# Set seed for reproducible results
from numpy.random import seed
seed(1)
import tensorflow
tensorflow.random.set_seed(2)


# In[2]:


# import plotting functions from Plotting.py
from Plotting9 import *


# In[21]:


# import metadata csv with all file names and related species
metadata=pd.read_csv("C:\\Users\\dario\\OneDrive\\Documenti\\Thesis_PC\\metadata.csv")
metadata.head()


# In[4]:


# Extend audio if duration < 5 seconds
path = "C:\\Users\\dario\\OneDrive\\Documenti\\Thesis_PC\\orthoptera\\all_recordings\\"

for aud in metadata["id"]:
    dur = librosa.get_duration(filename=path+aud)
    y, sr = librosa.load(path+aud, sr = None)
    if dur >= 5:
        sf.write('resized16\\'+aud, y , sr, 'PCM_16')
    else:
        new = np.tile(y,5)
        sf.write('resized16\\'+aud, new , sr, 'PCM_16')


# In[22]:


# Encode species column to 0-8
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
metadata['species'] = encoder.fit_transform(metadata['species'])
metadata.head()


# In[6]:


# create dataset with duration = 5 to get same size

D = [] # Dataset

for row in metadata.itertuples():
    y, sr = librosa.load("C:\\Users\\dario\\OneDrive\\Documenti\\Thesis_PC\\resized16\\" + row.id, duration = 5, sr = 44100)  
    ps = librosa.feature.melspectrogram(y=y, sr=sr) ####
    db = librosa.power_to_db(ps, ref=np.max)
    if ps.shape != (128, 431): continue
    D.append( (db, row.species) )


# In[7]:


print("number of wav: ", len(D))
print("sample rate of wav: ", sr)
print("shape of CNN input: ", db.shape)


# In[8]:


# Example of audio
y, sr = librosa.load("C:\\Users\\dario\\OneDrive\\Documenti\\Thesis_PC\\resized16\\Oecanthus pellucens (8).wav", sr= 44100, duration = 5)
plt.figure()
librosa.display.waveplot(y, sr, color='blue')


# In[9]:


# Example of mel spectrogram

ps = librosa.feature.melspectrogram(y=y, sr=sr)
db = librosa.power_to_db(ps, ref=np.max)
plt.figure()
librosa.display.specshow(db)
plt.colorbar()
print(sr)


# In[8]:


# Prepare data for cross validation

import random
dataset = D
random.Random(100).shuffle(dataset)

train = dataset

X, y = zip(*train)

# Reshape for CNN input
X = np.array([x.reshape( (128, 431, 1) ) for x in X])

# One-Hot encoding for classes
y = np.array(keras.utils.np_utils.to_categorical(y, 9))


# In[9]:


len(D)


# ### Baseline Model

# In[24]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
from tensorflow.keras import layers
from keras.regularizers import l2
from tensorflow.keras.layers import BatchNormalization
from sklearn.model_selection import KFold
from keras.layers import Lambda
import keras.backend as K


# In[3]:


def build_clf(optimizer = "adam", kernel_regularizer=l2(0.001)):
  # creating the layers of the NN
    model = Sequential()
    model.add(layers.Input((128,431, 1)))
    model.add(layers.Conv2D(2, (3, 3), kernel_regularizer=kernel_regularizer, activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv2D(4, (3, 3), kernel_regularizer=kernel_regularizer, activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv2D(8, (3, 3), kernel_regularizer=kernel_regularizer, activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(16, kernel_regularizer=kernel_regularizer))
    model.add(layers.Dropout(0.5))
    model.add(Dense(9, activation='softmax'))
    
    model.compile(optimizer=optimizer,loss="categorical_crossentropy",metrics=["accuracy"])

    return model


# In[ ]:


from keras.wrappers.scikit_learn import KerasClassifier
batch_size = 5
epochs = 30
modello = KerasClassifier(build_fn=build_clf, epochs=epochs, batch_size=batch_size, verbose=1)


# In[ ]:


from sklearn.model_selection import GridSearchCV

optimizer= ["adam","RMsprop"]
kernel_regularizer = [l2(0.001), l2(0.01)]
params = dict(optimizer=optimizer, kernel_regularizer = kernel_regularizer)
gs=GridSearchCV(estimator=modello, param_grid=params, cv=5)
# now fit the dataset to the GridSearchCV object. 
gs = gs.fit(X, y)


# In[ ]:


best_params=gs.best_params_
accuracy=gs.best_score_
print(best_params)
print(accuracy)


# In[11]:


# Define the K-fold Cross Validator
kfold =  KFold(n_splits = 5, shuffle=True, random_state = 15)
acc_per_fold = []
loss_per_fold = []
accuracy_fold = []
predicted_targets = np.array([])
actual_targets = []

# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(X,y):

  # Define the model architecture
    model = Sequential()
    model.add(layers.Input((128,431, 1)))
    model.add(layers.Conv2D(2, (3, 3), kernel_regularizer=l2(0.001), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv2D(4, (3, 3), kernel_regularizer=l2(0.001), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv2D(8, (3, 3), kernel_regularizer=l2(0.001), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(16, kernel_regularizer=l2(0.001)))
    model.add(layers.Dropout(0.5))
    model.add(Dense(16, kernel_regularizer=l2(0.001)))
    model.add(Dense(9, activation='softmax'))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
  # Compile the model
    model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])


  # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

  # Fit data to model
    history = model.fit(X[train], y[train],
              batch_size=5,
              epochs=30, # deve essere 30
              verbose=1)
   
    accuracy_fold.append(history)
    
  # Generate generalization metrics
    scores = model.evaluate(X[test], y[test], verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    
    print(len(X[train]), len(y[train]))
    print(len(X[test]), len(y[test]))

    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])
    
    predict = model.predict(X[test])
    predicted_labels = predict.argmax(axis=1)
    predicted_targets = np.append(predicted_targets, predicted_labels)
    
    actual_labels = y[test].argmax(axis=1)
    actual_targets = np.append(actual_targets, actual_labels)
    
  # Increase fold number
    fold_no = fold_no + 1


# In[12]:


# Accuracy with standard deviation of cross validation
print("%.2f%% (+/- %.2f%%)" % (np.mean(acc_per_fold), np.std(acc_per_fold)))


# In[15]:


plot_confusion_matrix(predicted_targets, actual_targets)


# In[16]:


accuracies (5, accuracy_fold)


# In[17]:


## function from plotting.py
errors (folds = 5, accuracies = accuracy_fold)


# In[18]:


# function from plotting.py
avg_acc([1,2,3,4,5], acc_per_fold)


# ### Data Augmentation

# In[95]:


# array to append all wav names and species and to store it in a csv
data = []


# In[96]:


# copy audio in augmentation folder

for row in metadata.itertuples():
    y, sr = librosa.load("C:\\Users\\dario\\OneDrive\\Documenti\\Thesis_PC\\resized16\\" + row.id, duration = 5, sr = None) 
    sf.write("C:\\Users\\dario\\OneDrive\\Documenti\\Thesis_PC\\augmentation\\" + row.id, y, sr)
    data.append([row.id,row.species])


# In[97]:


# time_stretch: If rate > 1, then the signal is sped up. If rate < 1, then the signal is slowed down

rate = [0.80, 0.90]
for r in rate:
    for row in metadata.itertuples():
        y, sr = librosa.load("C:\\Users\\dario\\OneDrive\\Documenti\\Thesis_PC\\resized16\\" + row.id, duration = 5, sr = None) 
        y_changed = librosa.effects.time_stretch(y, rate=r)
        sf.write("C:\\Users\\dario\\OneDrive\\Documenti\\Thesis_PC\\augmentation\\" + "Aug_" + row.id, y_changed, sr)
        data.append( ["Aug_" + row.id, row.species])


# In[98]:


# Pitch shift
n_steps = [-1, 1, 2]
for s in n_steps:
    for row in metadata.itertuples():
        y, sr = librosa.load("C:\\Users\\dario\\OneDrive\\Documenti\\Thesis_PC\\resized16\\" + row.id, duration = 5, sr = None) 
        y_changed = librosa.effects.pitch_shift(y, sr, n_steps=s)
        sf.write("C:\\Users\\dario\\OneDrive\\Documenti\\Thesis_PC\\augmentation\\" + "Aug_7" + row.id, y_changed, sr)
        data.append( ["Aug_7" + row.id, row.species])


# In[99]:


# Time shift
shift_max = 0.2

for row in metadata.itertuples():
    y, sr = librosa.load("C:\\Users\\dario\\OneDrive\\Documenti\\Thesis_PC\\resized16\\" + row.id, duration = 5, sr = None) 
    shift = np.random.randint(sr * shift_max)
    shift = -shift
    y_changed = np.roll(y, shift)
    # Set to silence for heading/ tailing
    if shift > 0:
        y_changed[:shift] = 0
    else:
        y_changed[shift:] = 0
    sf.write("C:\\Users\\dario\\OneDrive\\Documenti\\Thesis_PC\\augmentation\\" + "Aug_4" + row.id, y_changed, sr)
    data.append( ["Aug_4" + row.id, row.species])


# #### Add background noise + data augmentation

# In[101]:


# set up Environmental background noise
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, AddBackgroundNoise

augment = Compose([
    AddBackgroundNoise(sounds_path ="C:\\Users\\dario\\OneDrive\\Documenti\\Thesis_PC\\audio\\", 
        min_snr_in_db=3,
        max_snr_in_db=40, p = 1)
])


# In[102]:


# Add Background Noise to original wav

data_noise = []
for row in metadata.itertuples():
    y, sr = librosa.load("C:\\Users\\dario\\OneDrive\\Documenti\\Thesis_PC\\resized16\\" + row.id, duration = 5, sr = 44100) 
    y_changed = augment(samples=y, sample_rate=44100)
    sf.write("C:\\Users\\dario\\OneDrive\\Documenti\\Thesis_PC\\augmentation\\" + "Aug_6" + row.id, y_changed, sr)
    data_noise.append( ["Aug_6" + row.id, row.species])
    data.append( ["Aug_6" + row.id, row.species])


# In[103]:


# save augmented to metadata_augmented csv

header = ["id", "species"]
with open(r"C:\Users\dario\OneDrive\Documenti\Thesis_PC\orthoptera\metadata_noise.csv", 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(data_noise)


# In[104]:


# import metadata csv with all file names and related species

noise=pd.read_csv("C:\\Users\\dario\\OneDrive\\Documenti\\Thesis_PC\\orthoptera\\metadata_noise.csv")
noise.tail()


# In[ ]:


# Time stretch on noised data

rate = 0.80

for row in noise.itertuples():
    y, sr = librosa.load("C:\\Users\\dario\\OneDrive\\Documenti\\Thesis_PC\\augmentation\\" + row.id, duration = 5, sr = None) 
    y_changed = librosa.effects.time_stretch(y, rate=rate)
    sf.write("C:\\Users\\dario\\OneDrive\\Documenti\\Thesis_PC\\augmentation\\" + "Aug_5" + row.id, y_changed, sr)
    data.append( ["Aug_5" + row.id, row.species])


# In[ ]:


len(data)


# In[33]:


# save augmented to metadata_augmented csv

header = ["id", "species"]
with open(r"C:\Users\dario\OneDrive\Documenti\Thesis_PC\orthoptera\metadata_augmented.csv", 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(data)


# In[34]:


# Read augmentation csv

metadata_aug=pd.read_csv("C:\\Users\\dario\\OneDrive\\Documenti\\Thesis_PC\\orthoptera\\metadata_augmented.csv")
metadata_aug.tail()


# In[35]:


# create dataset with duration = 5 to get same size

D = [] # Dataset

for row in metadata_aug.itertuples():
    y, sr = librosa.load("C:\\Users\\dario\\OneDrive\\Documenti\\Thesis_PC\\augmentation\\" + row.id, duration = 5, sr = 44100)  
    ps = librosa.feature.melspectrogram(y=y, sr=sr)
    db = librosa.power_to_db(ps, ref=np.max)
    if db.shape != (128, 431): continue 
    D.append( (db, row.species) )


# In[36]:


print("number of wav: ", len(D))
print("sample rate of wav: ", sr)
print("shape of CNN input: ", db.shape)


# In[37]:


# Prepare data for cross validation

import random
dataset = D
random.Random(100).shuffle(dataset)

train = dataset

X, y = zip(*train)

# Reshape for CNN input
X = np.array([x.reshape( (128, 431, 1) ) for x in X])

# One-Hot encoding for classes
y = np.array(keras.utils.np_utils.to_categorical(y, 9))


# In[ ]:


def build_clf(optimizer = "adam", kernel_regularizer=l2(0.001)):
  # creating the layers of the NN
    model = Sequential()
    model.add(layers.Input((128, 431, 1)))
    model.add(layers.Conv2D(filters=8, kernel_size=(3, 3), kernel_regularizer=kernel_regularizer, activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), kernel_regularizer=kernel_regularizer, activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), kernel_regularizer=kernel_regularizer, activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), kernel_regularizer=kernel_regularizer, activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Dense(9, activation='softmax'))
    model.compile(optimizer=optimizer, loss="categorical_crossentropy",metrics=["accuracy"])
    return model


# In[ ]:


from keras.wrappers.scikit_learn import KerasClassifier
batch_size = 5
epochs = 10
modello = KerasClassifier(build_fn=build_clf, epochs=epochs, batch_size=batch_size, verbose=1)


# In[ ]:


from sklearn.model_selection import GridSearchCV
kernel_regularizer=[l2(0.1),l2(0.01),l2(0.001)]
optimizer= ["adam","RMsprop"]
params = dict(optimizer=optimizer, kernel_regularizer=kernel_regularizer)
gs=GridSearchCV(estimator=modello, param_grid=params, cv=5)
# now fit the dataset to the GridSearchCV object. 
gs = gs.fit(X, y)


# In[ ]:


best_params=gs.best_params_
accuracy=gs.best_score_
print(best_params)
print(accuracy)


# In[38]:


# Define the K-fold Cross Validator
kfold =  KFold(n_splits=5, shuffle=True)
acc_per_fold = []
loss_per_fold = []
accuracy_fold = []
predicted_targets = np.array([])
actual_targets = []

# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(X,y):

  # Define the model architecture
    model = Sequential()
    model.add(layers.Input((128, 431, 1)))
    model.add(layers.Conv2D(8, (3, 3), kernel_regularizer=l2(0.001), activation='relu', name='Layer_1'))
    model.add(layers.MaxPooling2D((2, 2), name='MaxPool_1'))

    model.add(layers.Conv2D(16, (3, 3), kernel_regularizer=l2(0.001), activation='relu', name='Layer_2'))
    model.add(layers.MaxPooling2D((2, 2), name='MaxPool_2'))

    model.add(layers.Conv2D(32, (3, 3), kernel_regularizer=l2(0.001), activation='relu', name='Layer_3'))
    model.add(layers.MaxPooling2D((2, 2), name='MaxPool_3'))

    model.add(layers.Conv2D(64, (3, 3), kernel_regularizer=l2(0.001), activation='relu', name='Layer_4'))
    model.add(layers.MaxPooling2D((2, 2), name='MaxPool_4'))

    model.add(Flatten(name='Flat'))
    model.add(Dense(64, name='Layer_5'))
    model.add(Dense(9, activation='softmax', name='Output'))
    
  # Compile the model
    model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])


  # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

  # Fit data to model
    history = model.fit(X[train], y[train],
              batch_size=5,
              epochs=10,
              verbose=1)
   
    accuracy_fold.append(history)
    
  # Generate generalization metrics
    scores = model.evaluate(X[test], y[test], verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])
    
    predict = model.predict(X[test])
    predicted_labels = predict.argmax(axis=1)
    predicted_targets = np.append(predicted_targets, predicted_labels)
    
    actual_labels = y[test].argmax(axis=1)
    actual_targets = np.append(actual_targets, actual_labels)

  # Increase fold number
    fold_no = fold_no + 1


# In[39]:


# Accuracy with standard deviation of cross validation
print("%.2f%% (+/- %.2f%%)" % (np.mean(acc_per_fold), np.std(acc_per_fold)))


# In[40]:


# Plot of k-fold accuracies
errors (5, accuracy_fold)
accuracies (5, accuracy_fold)


# In[41]:


# COnfusion matrix
plot_confusion_matrix(predicted_targets, actual_targets)


# ### Tranfer Learning 158

# In[3]:


# import metadata csv with all file names and related species
metadata=pd.read_csv("C:\\Users\\dario\\OneDrive\\Documenti\\Thesis_PC\\metadata.csv")
metadata.tail()


# In[8]:


# Duration = 5 seconds
path = "C:\\Users\\dario\\OneDrive\\Documenti\\Thesis_PC\\resized16\\"

for aud in metadata["id"]:
    y, sr = librosa.load(path+aud, duration = 5, sr = None)
    sf.write('orthoptera\\transfer_learning\\'+aud, y , sr)


# In[4]:


import os

from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio


# In[6]:


# Import YAMNet model from url

yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)


# In[7]:


# Load a WAV file, convert it to a float tensor, resample to 44.1 kHz single-channel audio.

def load_wav_44k_mono(filename):   
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(
          file_contents,
          desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=44100)
    return wav


# In[8]:


# Categories in Audioset data

class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
class_names =list(pd.read_csv(class_map_path)['display_name'])

for name in class_names[:20]:
    print(name)
print('...')


# In[9]:


# Load Insects data

metadata_tl= "C:\\Users\\dario\\OneDrive\\Documenti\\Thesis_PC\\metadata.csv" # orthoptera\\metadata_augmented.csv
base_data_path = "C:\\Users\\dario\\OneDrive\\Documenti\\Thesis_PC\\orthoptera\\transfer_learning\\" # \\Thesis_PC\\augmentation\\

metadata = pd.read_csv(metadata_tl)
metadata.tail()


# In[10]:


# Encode species column to 0-8
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
metadata['species'] = encoder.fit_transform(metadata['species'])
metadata.tail()


# In[11]:


# Classes of insects data
my_classes = ["Chorthippus biguttulus",
              "Chorthippus brunneus",
              "Gryllus campestris",
              "Nemobius sylvestris",
              "Oecanthus pellucens",
              "Pholidoptera griseoaptera",
              "Pseudochorthippus parallelus" ,
              "Roeseliana roeselii",
              "Tettigonia viridissima"]


# In[12]:


# Add recordings path to csv metadata

full_path = metadata['id'].apply(lambda row: os.path.join(base_data_path, row))
metadata = metadata.assign(id=full_path)

metadata.tail()


# In[13]:


# Plit wav from species

filenames = metadata['id']
targets = metadata['species']

main_ds = tf.data.Dataset.from_tensor_slices((filenames, targets))
main_ds.element_spec


# In[14]:


def load_wav_for_map(filename, label):
    return load_wav_44k_mono(filename), label

main_ds = main_ds.map(load_wav_for_map)
main_ds.element_spec


# In[15]:


# run YAMNet to extract embedding from the wav data

def extract_embedding(wav_data, label):
    scores, embeddings, spectrogram = yamnet_model(wav_data)
    num_embeddings = tf.shape(embeddings)[0]
    return (embeddings,
            tf.repeat(label, num_embeddings))

# extract embedding
main_ds = main_ds.map(extract_embedding).unbatch()
main_ds.element_spec


# In[16]:


# Shuffle dataset
main_ds = main_ds.cache().shuffle(1000)

# Split dataset in train, validation, test
train_ds = main_ds.shard(num_shards=3, index=0)
val_ds = main_ds.shard(num_shards=3, index=1)
test_ds = main_ds.shard(num_shards=3, index=2)

train_ds = train_ds.cache().shuffle(15).batch(5).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.cache().batch(5).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.cache().batch(5).prefetch(tf.data.AUTOTUNE)


# In[17]:


# Add Dense layers on top of Yamnet

my_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1024), dtype=tf.float32, name='input_embedding'),
    tf.keras.layers.Dense(512, activation = "relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(9, activation = "softmax")
], name='my_model')

my_model.summary()


# In[18]:


my_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])


# In[19]:


# Model fit

history = my_model.fit(train_ds,
                       epochs=30,
                       validation_data=val_ds)


# In[20]:


# Evaluate TL model on test
loss, accuracy = my_model.evaluate(test_ds)


# In[21]:


results(history, val_ds, my_model)


# In[22]:


predicted_targets = my_model.predict(test_ds)


# In[23]:


len(predicted_targets)


# In[24]:


# Extract true labels from tensor set (test set in this case)
def get_labels_from_tfdataset(tfdataset, batched=False):

    labels = list(map(lambda x: x[1], tfdataset)) # Get labels 

    if not batched:
        return tf.concat(labels, axis=0) # concat the list of batched labels

    return labels


# In[25]:


predicted_targets = np.argmax(predicted_targets, axis =1)


# In[26]:


# Confusion matrix
plot_confusion_matrix(predicted_targets, get_labels_from_tfdataset(test_ds, batched=False))


# ### Transfer Learning & Augmentation Combined

# In[161]:


# Import YAMNet model from url

yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)


# In[162]:


# Load Insects data

metadata_aug = "C:\\Users\\dario\\OneDrive\\Documenti\\Thesis_PC\\orthoptera\\metadata_augmented.csv" 
base_data_path = "C:\\Users\\dario\\OneDrive\\Documenti\\Thesis_PC\\augmentation\\" 

metadata = pd.read_csv(metadata_aug)
metadata.head()


# In[163]:


# Add recordings path to csv metadata

full_path = metadata['id'].apply(lambda row: os.path.join(base_data_path, row))
metadata = metadata.assign(id=full_path)

metadata.head(100)


# In[164]:


len(metadata)


# In[165]:


# Plit wav from species

filenames = metadata['id']
targets = metadata['species']

main_ds = tf.data.Dataset.from_tensor_slices((filenames, targets))
main_ds.element_spec


# In[166]:


def load_wav_for_map(filename, label):
    return load_wav_44k_mono(filename), label

main_ds = main_ds.map(load_wav_for_map)
main_ds.element_spec


# In[167]:


# run YAMNet to extract embedding from the wav data

def extract_embedding(wav_data, label):
    scores, embeddings, spectrogram = yamnet_model(wav_data)
    num_embeddings = tf.shape(embeddings)[0]
    return (embeddings,
            tf.repeat(label, num_embeddings))

# extract embedding
main_ds = main_ds.map(extract_embedding).unbatch()
main_ds.element_spec


# In[168]:


# Shuffle dataset
main_ds = main_ds.cache().shuffle(1000)

# Split dataset in train, validation, test
train_ds = main_ds.shard(num_shards=3, index=0)
val_ds = main_ds.shard(num_shards=3, index=1)
test_ds = main_ds.shard(num_shards=3, index=2)

train_ds = train_ds.cache().shuffle(15).batch(5).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.cache().batch(5).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.cache().batch(5).prefetch(tf.data.AUTOTUNE)


# In[169]:


# Add Dense layers on top of Yamnet

my_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1024), dtype=tf.float32, name='input_embedding'),
    tf.keras.layers.Dense(512, activation = "relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(9, activation = "softmax")
], name='my_model')

my_model.summary()


# In[170]:


my_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])


# In[171]:


# Model fit

history = my_model.fit(train_ds,
                       epochs=30,
                       validation_data=val_ds)


# In[172]:


# Evaluate TL model on test
loss, accuracy = my_model.evaluate(test_ds)


# In[173]:


results(history, val_ds, my_model)


# In[174]:


predicted_targets = my_model.predict(test_ds)


# In[175]:


len(predicted_targets)


# In[176]:


predicted_targets = np.argmax(predicted_targets, axis =1)


# In[177]:


# Confusion matrix
plot_confusion_matrix(predicted_targets, get_labels_from_tfdataset(test_ds, batched=False))


# ### Raw Waveforms 158

# In[10]:


# import metadata csv with all file names and related species
metadata=pd.read_csv("C:\\Users\\dario\\OneDrive\\Documenti\\Thesis_PC\\metadata.csv")
metadata.tail()


# In[11]:


# Encode species column to 0-8
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
metadata['species'] = encoder.fit_transform(metadata['species'])
metadata.head()


# In[12]:


# create dataset with duration = 5 to get same size

D = [] # Dataset

for row in metadata.itertuples():
    y, sr = librosa.load("C:\\Users\\dario\\OneDrive\\Documenti\\Thesis_PC\\orthoptera\\transfer_learning\\" + row.id, duration = 5, sr = 44100)  
    D.append( (y, row.species) )


# In[13]:


len(D)


# In[14]:


import random
dataset = D
random.Random(100).shuffle(dataset)

train = dataset

X, y = zip(*train)

# Reshape for CNN input
X = np.array([x.reshape( (220500, 1) ) for x in X])

# One-Hot encoding for classes
y = np.array(keras.utils.np_utils.to_categorical(y, 9))


# In[184]:


def build_clf(optimizer = "adam", unit = 8):
  # creating the layers of the NN
    model = Sequential()
    model.add(layers.Input((220500, 1)))
    model.add(layers.Conv1D(filters=unit, kernel_size=5)) #, kernel_regularizer=l2(0.001)))
    model.add(Activation('relu'))

    model.add(layers.Conv1D(filters=unit, kernel_size=5)) #, kernel_regularizer=l2(0.01)))
    model.add(Activation('relu'))

    model.add(layers.Conv1D(filters=unit, kernel_size=5)) #, kernel_regularizer=l2(0.01)))
    model.add(Activation('relu'))

    model.add(layers.MaxPooling1D(2))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.MaxPooling1D(2))

    model.add(Dense(8))
    model.add(Lambda(lambda x: K.mean(x, axis=1))) # maybe change to 0
    model.add(Dense(9, activation='softmax'))
     
    model.compile(optimizer=optimizer,loss="categorical_crossentropy",metrics=["accuracy"])
    return model


# In[185]:


from keras.wrappers.scikit_learn import KerasClassifier
batch_size = 5
epochs = 100
modello = KerasClassifier(build_fn=build_clf, epochs=epochs, batch_size=batch_size, verbose=1)


# In[ ]:


from sklearn.model_selection import GridSearchCV
units = [8,16,32]
params = dict(optimizer=optimizer unit = units)
gs=GridSearchCV(estimator=modello, param_grid=params, cv=5)
# now fit the dataset to the GridSearchCV object. 
gs = gs.fit(X, y)


# In[ ]:


best_params=gs.best_params_
accuracy=gs.best_score_
print(best_params)
print(accuracy)


# In[15]:


import random
from keras import utils as np_utils

dataset = D
random.Random(100).shuffle(dataset)


train = dataset[:128]
valid = dataset[128:143]
test = dataset[143:]

X_train, y_train = zip(*train)
X_valid, y_valid = zip(*valid)
X_test, y_test = zip(*test)

# Reshape for CNN input
X_train = np.array([x.reshape( (220500, 1) ) for x in X_train])
X_valid = np.array([x.reshape( (220500, 1) ) for x in X_valid])
X_test = np.array([x.reshape( (220500, 1) ) for x in X_test])

# One-Hot encoding for classes
y_train = np.array(keras.utils.np_utils.to_categorical(y_train, 9))
y_valid = np.array(keras.utils.np_utils.to_categorical(y_valid, 9))
y_test = np.array(keras.utils.np_utils.to_categorical(y_test, 9))


# In[16]:


# model
model = Sequential()

model.add(layers.Input((220500, 1)))
model.add(layers.Conv1D(8, kernel_size=5)) #, kernel_regularizer=l2(0.001)))
model.add(Activation('relu'))
          
model.add(layers.Conv1D(16, kernel_size=5)) #, kernel_regularizer=l2(0.01)))
model.add(Activation('relu'))

model.add(layers.MaxPooling1D(2))
model.add(layers.MaxPooling1D(2))
model.add(layers.MaxPooling1D(2))

model.add(Dense(16))
model.add(Lambda(lambda x: K.mean(x, axis=1))) # maybe change to 0
model.add(Dense(9, activation='softmax'))

model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
model.summary()


# In[209]:


history = model.fit(X_train, y_train, batch_size=10, epochs=400, validation_data=(X_valid, y_valid), verbose=1)


# In[210]:


results(history, y_valid, model)


# In[211]:


pred_baseline = model.evaluate(X_test,y_test)


# In[212]:


pred = model.predict(X_test)


# In[213]:


pred_label = pred.argmax(axis=1)
actual_label = y_test.argmax(axis=1)


# In[214]:


plot_confusion_matrix(pred_label, actual_label)


# In[ ]:





# In[ ]:





# ### Raw Waveforms Combined Augmentation

# In[17]:


metadata=pd.read_csv("C:\\Users\\dario\\OneDrive\\Documenti\\Thesis_PC\\orthoptera\\metadata_augmented.csv")
metadata.head()


# In[18]:


# create dataset with duration = 5 to get same size

D = [] # Dataset

for row in metadata.itertuples():
    y, sr = librosa.load("C:\\Users\\dario\\OneDrive\\Documenti\\Thesis_PC\\augmentation\\" + row.id, duration = 5, sr = 44100)  
    D.append( (y, row.species) )


# In[19]:


import random
from keras import utils as np_utils

dataset = D
random.Random(100).shuffle(dataset)

train = dataset[:1132]
valid = dataset[1132:1277]
test = dataset[1277:]

X_train, y_train = zip(*train)
X_valid, y_valid = zip(*valid)
X_test, y_test = zip(*test)

# Reshape for CNN input
X_train = np.array([x.reshape( (220500, 1) ) for x in X_train])
X_valid = np.array([x.reshape( (220500, 1) ) for x in X_valid])
X_test = np.array([x.reshape( (220500, 1) ) for x in X_test])

# One-Hot encoding for classes
y_train = np.array(keras.utils.np_utils.to_categorical(y_train, 9))
y_valid = np.array(keras.utils.np_utils.to_categorical(y_valid, 9))
y_test = np.array(keras.utils.np_utils.to_categorical(y_test, 9))


# In[20]:


# model
model = Sequential()

model.add(layers.Input((220500, 1)))
model.add(layers.Conv1D(8, kernel_size=5)) #, kernel_regularizer=l2(0.001)))
model.add(Activation('relu'))
          
model.add(layers.Conv1D(16, kernel_size=5)) #, kernel_regularizer=l2(0.01)))
model.add(Activation('relu'))

model.add(layers.MaxPooling1D(2))
model.add(layers.MaxPooling1D(2))
model.add(layers.MaxPooling1D(2))

model.add(Dense(16)) #8
model.add(Lambda(lambda x: K.mean(x, axis=1))) # maybe change to 0
model.add(Dense(9, activation='softmax'))

#opt = tf.keras.optimizers.Adam(learning_rate=0.01)      
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
model.summary()


# In[67]:


history = model.fit(X_train, y_train, batch_size=10, epochs=100, validation_data=(X_valid, y_valid), verbose=1)


# In[90]:


results(history, y_valid, model)


# In[69]:


pred_baseline = model.evaluate(X_test,y_test)


# In[76]:


pred = model.predict(X_test)


# In[85]:


pred_label = pred.argmax(axis=1)
actual_label = y_test.argmax(axis=1)


# In[87]:


plot_confusion_matrix(pred_label, actual_label)


# In[ ]:





# In[ ]:




