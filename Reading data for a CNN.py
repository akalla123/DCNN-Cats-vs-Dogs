
# coding: utf-8

# In[1]:


import numpy as np
import os
import matplotlib.pyplot as plt
import cv2



# In[2]:


data = "/home/ayush/Python/Data/PetImages"
categories = ['Dog','Cat']

for category in categories:
    #set path as the image path
    path = os.path.join(data,category)
    #Read the images in the path
    for image in os.listdir(path):
    #Read the images into an array
    #Convert image to gray scale
        image_array = cv2.imread(os.path.join(path,image),cv2.IMREAD_GRAYSCALE)
        plt.imshow(image_array,cmap='gray')
        plt.show()
        break
    break


# In[3]:


size = 50

reshaped_array = cv2.resize(image_array, (size,size))

plt.imshow(reshaped_array,cmap="gray")
plt.show()


# In[4]:


training_data = []

def create_training_data():
    for category in categories:
        #set path as the image path
        path = os.path.join(data,category)
        #Coverting string to number
        num = categories.index(category)
        #Read the images in the path
        for image in os.listdir(path):
        #Read the images into an array
        #Convert image to gray scale
            try:
                image_array = cv2.imread(os.path.join(path,image),cv2.IMREAD_GRAYSCALE)
                reshaped_array = cv2.resize(image_array, (size,size))
                training_data.append([reshaped_array,num])
            except:
                pass
create_training_data()


# In[7]:


#Shuffle the data for better accuracy
import random
random.shuffle(training_data)


# In[8]:


x = []
y = []
for item in training_data:
    x.append(item[0])
    y.append(item[1])


# In[16]:


#1 means a grayscale image
x = np.array(x).reshape(-1,size,size,1)


# In[19]:


# Saving the data into pickles
import pickle
out = open("X.pickle","wb")
pickle.dump(x, out)
out.close()
y_out = open("y.pickle","wb")
pickle.dump(y, y_out)
y_out.close()

