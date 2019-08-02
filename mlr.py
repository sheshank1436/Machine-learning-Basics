#!/usr/bin/env python
# coding: utf-8

# In[82]:


#import the libraries
import pandas as pd #to read the file
import numpy as np #to perform array operations
import matplotlib.pyplot as plt  #to visualize the data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split #to split the data into training and testing
from sklearn.linear_model import LinearRegression #importing the linear regression to perform regression


# In[83]:


a=pd.read_csv('C:\\Users\\shkatta\\Desktop\\mlr.csv')#reading the csv file(dataset) from its specified location
a.head()


# In[84]:


x=a.iloc[:,0:2].values #here x is the inputs in the dataframe
y=a.iloc[:,2].values #here y is the outputs for the respective inputs column in the dataframe


# In[95]:


le=LabelEncoder()
a['Gender']=le.fit_transform(a['Gender'])
x[:,0]=a['Gender']

#onehe=OneHotEncoder(categorical_features=[2])
#a=onehe.fit_transform(a).toarray()
#a


# In[96]:


#splitting the data into training and testing here 
#here test_size=0.3 represents that 30% of data is used for testing and 70% for training the model
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
x_test


# In[100]:


mlr=linear_model.LinearRegression()
mlr.fit(x_train,y_train) #to train the model with the given inputs and respective output
predict=mlr.predict(x_test)# in order to predict we need to use this


# In[108]:


y_test[60]


# In[109]:


predict[60]


# In[104]:


#to find the coefficient of the model
mlr.coef_


# In[105]:


#to find the intercept of the model
mlr.intercept_


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




