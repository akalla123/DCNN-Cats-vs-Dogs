
# coding: utf-8

# In[1]:


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle

x = pickle.load(open("X.pickle","rb"))

y = pickle.load(open("y.pickle","rb"))

x = x/255

cnn = Sequential()


cnn.add(Conv2D(64, (3, 3), input_shape = (50,50,1), activation = 'relu'))
cnn.add(MaxPooling2D())
cnn.add(Conv2D(64, (3, 3), input_shape = (50,50,1), activation = 'relu'))
cnn.add(MaxPooling2D())
# Flatten is used to convert the output of the CNN part into a 1D feature vector
cnn.add(Flatten())

cnn.add(Dense(128, activation = 'relu'))
cnn.add(Dense(128, activation = 'relu'))
cnn.add(Dense(1, activation='sigmoid'))


cnn.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

cnn.fit(x,y,batch_size=100,epochs = 5, validation_split=0.1)

