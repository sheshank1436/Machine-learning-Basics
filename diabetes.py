
# coding: utf-8

# In[7]:


import pandas as pd
#col_names=['Pregnancies','Glucose','BloodPressure','Skin','Insulin','Bmi','Pedigree','Age','Outcome']



# In[12]:


file=pd.read_csv('D:\\Users\\shkatta\\Desktop\\diabetes.csv')
file.head()
file.columns.values


# In[11]:


feature_cols=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
x=file[feature_cols]
y=file.Outcome


# In[26]:


from sklearn import model_selection
X_train,X_test,Y_train,Y_test=model_selection.train_test_split(x,y,test_size=0.2,random_state=0)


# In[24]:


from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(X_train,Y_train)
Y_pred=logreg.predict(X_test)


# In[27]:


from sklearn import metrics
cnf_matrix=metrics.confusion_matrix(Y_test,Y_pred)
cnf_matrix


# In[28]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[30]:


class_names=[0,1]
fig,ax=plt.subplots()
tick_marks=np.arange(len(class_names))
plt.xticks(tick_marks,class_names)
plt.yticks(tick_marks,class_names)
sns.heatmap(pd.DataFrame(cnf_matrix),annot=True,cmap="YlGnBu",fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('confusion matrix',y=1.1)
plt.ylabel('acutal label')
plt.xlabel('predicted label')

