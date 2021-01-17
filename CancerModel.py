#!/usr/bin/env python
# coding: utf-8

# In[50]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier          
from sklearn.model_selection import train_test_split
from sklearn import metrics 
import joblib 
from sklearn import tree
from os import system
from graphviz import Source
import matplotlib.pyplot as plt

df=pd.read_csv("cancert.csv")

df['Patient Id']=df['Patient Id'].str.slice(1)

X=df.drop(columns=['Level']) 

y=df['Level'] 

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

model= DecisionTreeClassifier(criterion='entropy')

model.fit(X_train,y_train)             #input output dataset in the model enables it to get trained 

tree.export_graphviz(model,out_file='cancer.dot',max_depth=None,feature_names=['Patient Id','Age','Gender','Air Pollution','Alcohol','Dust Allergy','OccuPational Hazards','Genetic Risk','chronic Lung Disease','Balanced Diet','Obesity','Smoking','Passive Smoker','Chest Pain','Coughing of Blood','Fatigue','Weight Loss','Shortness of Breath','Wheezing','Swallowing Difficulty','Clubbing of Finger Nails','Frequent Cold','Dry Cough','Snoring'],class_names=sorted(y.unique()),label='all',rounded=True,filled=True)

tree.plot_tree(model)

joblib.dump(model, 'cancert.dot')

pre=model.predict(X_test) 

score=metrics.accuracy_score(y_test,pre)
score


# In[48]:


get_ipython().system('dot -Tpng cancert.dot -o tree.png')


# In[53]:


tree.plot_tree(model,filled=True)
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (5,5), dpi=300)
plt.figure(figsize=(10,10))


# In[22]:


df.head()


# In[23]:


df.describe()


# In[ ]:




