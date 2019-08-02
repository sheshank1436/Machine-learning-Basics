#!/usr/bin/env python
# coding: utf-8

# In[6]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor


# In[13]:


# Importing the dataset
data = pd.read_csv('C:\\Users\\shkatta\\Desktop\\P_Salaries.csv')
data


# In[ ]:


x = data.iloc[:,1:2].values #here x is the inputs in the dataframe
y=data.iloc[:,2].values #here y is the outputs for the respective inputs column in the dataframe


# In[8]:


d=DecisionTreeRegressor(random_state=0)
d.fit(x,y)


# In[11]:


y_pred=d.predict([[6]])


# In[12]:


y_pred


# In[24]:


x_grid=np.arange(min(x),max(x),1)
x_grid=x_grid.reshape((len(x_grid),1))
plt.scatter(x,y,color='green')
plt.plot(x,d.predict(x_grid),color='blue')
plt.title('Decision Tree Regressor',color='red')
plt.xlabel('years or position',color='blue')
plt.ylabel('salary',color='blue')
plt.show()


# In[ ]:




