#!/usr/bin/env python
# coding: utf-8

# In[99]:


#import the libraries
import pandas as pd #to read the file
import numpy as np #to perform array operations
import matplotlib.pyplot as plt  #to visualize the data
from sklearn.model_selection import train_test_split #to split the data into training and testing
from sklearn.linear_model import LinearRegression #importing the linear regression to perform regression


# In[100]:


a=pd.read_csv('C:\\Users\\shkatta\\Desktop\\salary.csv')#reading the csv file(dataset) from its specified location


# In[101]:


#a.head() #just to check the dataframe


# In[102]:


x=a.iloc[:,0].values #here x is the first column in the dataframe
y=a.iloc[:,1].values #here y is the second column in the dataframe
x=x.reshape(-1,1)#in order to convert input x to 2d form
y=y.reshape(-1,1)#in order to convert input y to 2d form


# In[103]:


#splitting the data into training and testing here 
#here test_size=0.3 represents that 30% of data is used for testing and 70% for training the model
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[104]:


#x_train


# In[105]:



slr=linear_model.LinearRegression()
slr.fit(x_train,y_train) #to train the model with the given inputs and respective output
predict=slr.predict(x_test)# in order to predict we need to use this


# In[106]:


y_test[3] #checking the test actual value 


# In[114]:


predict[3] #this is used to predict the  value based on the model


# In[ ]:


#y=b0+b1*x


# In[115]:


#to find the coefficient of the model
slr.coef_


# In[116]:


#to find the intercept of the model
slr.intercept_


# In[ ]:





# In[108]:


#%matplotlib inline
plt.scatter(x_train,y_train,color='red',marker='*')#plotting the training input x with respective output y
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.plot(x_train,slr.predict(x_train),color='blue')#plotting the regression line based on given inputs


# In[109]:


#now plotting for the test data
plt.scatter(x_test,y_test,color='red',marker='*')#plotting the testing input x with respective output y
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.plot(x_train,slr.predict(x_train),color='blue')#plotting the regression line based on given inputs


# In[ ]:





# In[ ]:





# In[ ]:




