#!/usr/bin/env python
# coding: utf-8

# In[50]:


#import library 
from sklearn.preprocessing import OneHotEncoder
from sklearn import model_selection
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import sklearn.preprocessing


# In[51]:



import warnings
warnings.filterwarnings('ignore')


# In[52]:


###loading data
train_data = pd.read_csv('titanic_train.csv')
test_data = pd.read_csv('titanic_test.csv')


# In[53]:


####understanding the features of the data
train_data.head()


# In[54]:


### overview of the data : 
train_data.info()


# In[55]:


### there exists missing values . so its time to handle them .
train_data.isnull().sum() ### summation of the missing values in training data :


# In[56]:


#### we can see for the training data , 177 missing values in the age attribute , 687 missing values in the cabin attribute , and 2 missing values in the embark feature exists
### as age and fate are numeric values we replace the missing values with the mean of the entire feature 
####as embarked is the string so we can replace the missing values with mode
train_data['Age'].fillna(train_data['Age'].median(),inplace=True)
test_data['Age'].fillna(train_data['Age'].median(),inplace=True)

train_data['Embarked'].fillna(train_data['Embarked'].mode(),inplace=True)
test_data['Fare'].fillna(test_data['Fare'].median(), inplace = True) 


# In[57]:


train_data.isnull().sum()
##### no more missing values in the data set
test_data.isnull().sum()


# In[58]:


#### sex and embarked features are string . for the betterment of our computation we change the string data , to categorical data 
train_data['Sex']=train_data["Sex"].astype('category')
test_data['Sex']=test_data["Sex"].astype('category')
train_data['Embarked']=train_data["Embarked"].astype('category')
test_data['Embarked']=test_data["Embarked"].astype('category')


# In[59]:


###cat.codes is a processor which numerizes the categories as values and saves as a int type
train_data["Sex"]= train_data["Sex"].cat.codes
test_data["Sex"]= test_data["Sex"].cat.codes
train_data['Embarked']=train_data['Embarked'].cat.codes
test_data['Embarked']=test_data['Embarked'].cat.codes


# In[60]:


train_data.info()
test_data.info()


# In[61]:


#### as our main concern for this experiment is to understand the behavior of the metrics in training and valid data ,
###we are not focusing on the the feature engineering . so we drop the cabin ticket, passengerid and name column . 
### it has been seen that ticket and name column has some influence for getting a better accuracy but for our experiment , it is not necessary
train_data=train_data.drop(['Cabin','Ticket','PassengerId'],axis = 1)
test_data=test_data.drop(['Cabin','Ticket','PassengerId'],axis = 1)


# In[62]:


train_data=train_data.drop('Name',axis=1)
test_data=test_data.drop('Name',axis=1)


# In[64]:


#### all the input variables are transformed into float / int
train_data.info()


# In[65]:


#### determining the class labels in the training and test data
X_train = train_data.drop('Survived',axis = 1)
Y_train = train_data['Survived']
dum_Y=tf.keras.utils.to_categorical(Y_train)


# In[66]:


print(X_train)


# In[69]:


### we normalize our input data so that we can feed the network as an 1D array input
from sklearn.preprocessing import Normalizer
array = X_train.values
X = array[:,0:7]####input features , first 6 columns
scaler = Normalizer().fit(X)
X_norm = scaler.transform(X)


# In[70]:


print(X_norm)


# In[83]:


#### training and validation split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_norm ,dum_Y)


# In[249]:


### building up the model : 
model = tf.keras.models.Sequential()#### we want to have a feed forward network
model.add(tf.keras.layers.Flatten())### flattening the data helps to compress the input in such a which allows the network perform better
### we would want to have 2 hidden layers with 128 neurons and each have rectified linear function as the activation function 
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
### as the labels are 0 to 9 so 10 output nodes are needed and the activation function is softmax function 
model.add(tf.keras.layers.Dense(2,activation = tf.nn.sigmoid))


# In[257]:


model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])


# In[270]:


history = model.fit(x_train, y_train, epochs=12,batch_size=128, validation_split=0.1)


# In[271]:


##### save the results of the model's valid vs training accuracy + loss and save them and plot them in such a way that we can
#### tune the hyper parameters for our validation set 
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)


# In[272]:


plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'g', label='Validation acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.title('Training and validation accuracy')
plt.legend()
fig = plt.figure()



plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'g', label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and validation loss')

plt.legend()
plt.show()


# In[ ]:


###conclusion : 
#### for the titanic data set our neural network has managed to perform 77 percent accuracy in the validation data set .
### we first ran the network for 50 epoch and then using the elbow rule we found that 12 epoch is enough for the network not to overfit 

