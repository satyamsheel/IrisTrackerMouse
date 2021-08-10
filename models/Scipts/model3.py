#!/usr/bin/env python
# coding: utf-8

# In[521]:


import pandas as pd
import numpy as np
import json
import os


# In[522]:


def data_preprocessing(data) :
    Y = data[["x_coord", "y_coord"]].values
    X = data[["left_ear", "right_ear", "left_pupil", "right_pupil"]]
    for i in range(0, 6) :
        right_name = "right_pupil_" + str(i + 1)
        left_name = "left_pupil_" + str(i + 1)
        X[right_name] = X["right_pupil"].apply(lambda x : x[i])
        X[left_name] = X["left_pupil"].apply(lambda x : x[i])
    X.drop(["left_pupil", "right_pupil"], axis = 1, inplace = True)
    ret_X = X.values
    return X, X.columns, ret_X, Y


# In[523]:


def read_data(root_path) :
    data = None
    for folder in os.listdir(root_path) :
        if folder.find("Images") != -1 :
            df = pd.read_json(root_path + folder + "/data.json")
            if data is None :
                data = df
            else :
                data = pd.concat([data, df])
    return data


# In[524]:


from keras.layers import Dense, Input
from keras.optimizers import Adam
from keras.models import Sequential
from keras.regularizers import l2


# In[525]:


model = Sequential()
model.add(Input(shape=(14,)))
dense1 = Dense(14, activation = "relu", kernel_regularizer = l2(0.05))
dense2 = Dense(128, activation = "relu", kernel_regularizer = l2(0.03))
dense3 = Dense(64, activation = "relu", kernel_regularizer = l2(0.05))
dense4 = Dense(32, activation = "relu", kernel_regularizer = l2(0.03))
dense5 = Dense(16, activation = "relu", kernel_regularizer = l2(0.05))
dense6 = Dense(2, activation = "relu", kernel_regularizer = l2(0.03))
model.add(dense1)
model.add(dense2)
model.add(dense3)
model.add(dense4)
model.add(dense5)
model.add(dense6)
model.summary()


# In[526]:


adam = Adam(lr=0.03)
model.compile(optimizer= "adam", loss="mse", metrics=["mse"])


# In[527]:


#train data
data = read_data("Preprocessed Output/")
data, train_columns, X_train, Y_train = data_preprocessing(data)
# #test data
# test_columns, X_test, Y_test = data_preprocessing("Test/Preprocessed Output/")


# In[528]:


model.fit(X_train, Y_train, batch_size=25, epochs =100)


# In[529]:


data


# In[530]:


# dense3.get_weights()


# In[531]:


# preds = model.predict(X_test)
# for i in range(preds.shape[0]):
#     print(preds[i])
#     print(Y_test[i])


# In[532]:


model.save("models/model4.h5")


# In[533]:


metadata = {}
metadata["columns"] = train_columns.tolist()


file = open("models/model4.data", "+w")
file.write(json.dumps(metadata))
file.close()

