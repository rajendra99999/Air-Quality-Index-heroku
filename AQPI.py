#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


dataset=pd.read_excel('Air Pollution Quality Index.xlsx')


# In[3]:


dataset=dataset.dropna()


# In[4]:


dataset.isnull().sum()


# In[15]:


sns.pairplot(dataset)


# In[5]:


plt.figure(figsize=(15,10))
sns.heatmap(dataset.corr(),cmap='RdYlGn',annot=True)


# In[6]:


dataset.head()


# In[8]:


y=dataset['PM 2.5']


# In[10]:


x=dataset[['T','SLP','H','VV','V','VM']]


# In[14]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x=scaler.fit_transform(x)


# In[16]:


from sklearn.model_selection import cross_val_score


# In[17]:


from sklearn.linear_model import Lasso,LinearRegression,Ridge


# In[19]:


linear_regre_model=LinearRegression()
score=cross_val_score(linear_regre_model,x,y,cv=10)
score.mean()


# In[20]:


lasso_model=Lasso()
score_lasso=cross_val_score(lasso_model,x,y,cv=10)
score_lasso.mean()


# In[21]:


ridge_model=Ridge()
score_ridge=cross_val_score(ridge_model,x,y,cv=10)
score_ridge.mean()


# In[22]:


from sklearn.tree import DecisionTreeRegressor
dt_model=DecisionTreeRegressor()
score_dt=cross_val_score(dt_model,x,y,cv=10)
score_dt.mean()


# In[23]:


from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor


# In[25]:


rf_model=RandomForestRegressor(n_estimators=12)
score_rf=cross_val_score(rf_model,x,y,cv=10)
score_rf.mean()


# In[27]:


scores_rf=[]
for i in range(12,50):
    rf_model=RandomForestRegressor(n_estimators=i)
    score_rf=cross_val_score(rf_model,x,y,cv=10)
    scores_rf.append(score_rf.mean())
    


# In[30]:


plt.figure(figsize=(15,12))
plt.plot(range(12,50),scores_rf)
for i in range(12,50):
    plt.text(i, np.round(scores_rf[i-12],2), (i, np.round(scores_rf[i-12],2)))
plt.xticks([i for i in range(12, 50)])
plt.xlabel('Number of estimator (n)')
plt.ylabel('Scores')
plt.title('cv scores for different n values')


# In[38]:


ab_model=AdaBoostRegressor(n_estimators=12)
score_ab=cross_val_score(ab_model,x,y,cv=10)
score_ab.mean()


# In[42]:


gb_model=GradientBoostingRegressor(n_estimators=35)
score_gb=cross_val_score(gb_model,x,y,cv=10)
score_gb.mean()


# In[45]:


from sklearn.neighbors import KNeighborsRegressor
knn_model=KNeighborsRegressor()
score_knn=cross_val_score(knn_model,x,y,cv=10)
score_knn.mean()


# In[46]:


from xgboost import XGBRegressor
xgb_model=XGBRegressor()
score_xgb=cross_val_score(xgb_model,x,y,cv=10)
score_xgb.mean()


# In[48]:



#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]
# Method of selecting samples for training each tree
# bootstrap = [True, False]


# In[49]:



# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)


# In[50]:


from sklearn.model_selection import RandomizedSearchCV


# In[51]:


model=RandomForestRegressor()


# In[54]:


search=RandomizedSearchCV(estimator=model,param_distributions=random_grid,scoring='neg_mean_squared_error', n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = 1)


# In[53]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[55]:


search.fit(x_train,y_train)


# In[56]:


search.best_estimator_


# In[57]:


search.best_params_


# In[58]:


search.best_score_


# In[59]:


y_pred=search.predict(x_test)


# In[60]:


from sklearn.metrics import accuracy_score,r2_score


# In[61]:


sns.displot(y_test-y_pred)


# In[62]:


plt.scatter(y_test,y_pred)


# In[66]:


print(r2_score(y_test,y_pred))


# In[74]:


input_par=[25,1012,68,0.6,4,8.2]
scl_input=scaler.transform([input_par])


# In[76]:


search.predict([input_par])


# In[75]:


search.predict(scl_input)


# In[73]:


scl_input


# In[77]:


import pickle


# In[78]:


file=open('Random Forest Model.pkl','wb')
pickle.dump(search,file)


# In[ ]:




