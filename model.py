#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


dataset=pd.read_excel('Air Pollution Quality Index.xlsx')


# In[4]:


dataset=dataset.dropna()


# In[5]:


y=dataset['PM 2.5']


# In[6]:


x=dataset[['T','SLP','H','VV','V','VM']]


# In[7]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x=scaler.fit_transform(x)


# In[8]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[9]:


from sklearn.ensemble import RandomForestRegressor


# In[10]:


model=RandomForestRegressor(n_estimators=500, min_samples_split=2, min_samples_leaf= 1,max_features= 'sqrt',max_depth=15)


# In[11]:


model.fit(x_train,y_train)


# In[12]:


y_pred=model.predict(x_test)


# In[13]:


from sklearn.metrics import r2_score


# In[14]:


print(r2_score(y_test,y_pred))


# In[15]:


input_par=[25,1012,68,0.6,4,8.2]


# In[19]:


model.predict([np.array(input_par)])


# In[20]:


import pickle


# In[21]:


file=open('model.pkl','wb')
pickle.dump(model,file)


# In[ ]:




