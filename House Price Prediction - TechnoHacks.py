#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data= pd.read_csv(r"C:\Users\Mohammed Yusuf\OneDrive\Desktop\Data Science Projects\Housing.csv")


# In[3]:


data


# # EXPLORATORY DATA ANALYSIS:

# In[40]:


data.shape


# In[41]:


data.describe()


# # To remove the scientific notation, we need to use the 
# ###"pd.set_option('display.float_format', lambda x: '%.3f' % x)"

# In[43]:


pd.set_option('display.float_format', lambda x: '%.3f' % x)


# In[44]:


data.describe()


# In[45]:


data.info()


# #Data Manipulation. This data is fully organized so we can use this without
# #manipulation. 

# #No missing values. 

# #To check the outliers using the boxplot. 

# In[49]:


sns.boxplot(x='bedrooms', y='price', data=data)


# # Train our Model 

# #Splitting our data for training and testing!

# In[51]:


data.columns


# In[52]:


x=data[['bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated',
        'sqft_living15', 'sqft_lot15']]


# In[53]:


x


# In[54]:


y=data['price']


# In[55]:


y


# In[56]:


from sklearn.model_selection import train_test_split


# In[57]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25, random_state =101)


# In[58]:


x_train


# In[59]:


y_train


# In[60]:


x_test


# In[61]:


y_test


# # Normalization ; which is basically standarization.

# In[62]:


from sklearn.preprocessing import StandardScaler
std = StandardScaler()


# In[63]:


x_train_std=std.fit_transform(x_train)
x_test_std=std.transform(x_test)


# In[64]:


x_train_std


# In[65]:


x_test


# In[66]:


x_test_std


# # Model Training

# In[67]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()


# In[68]:


lr.fit(x_train_std,y_train)


# In[69]:


y_pred=lr.predict(x_test_std)


# In[70]:


y_pred


# In[71]:


y_test


# In[72]:


from sklearn.metrics import mean_absolute_error,r2_score


# In[73]:


mean_absolute_error(y_test,y_pred)


# In[74]:


x_test


# In[75]:


x_test.loc[7148]


# In[76]:


r2_score(y_test,y_pred)


# # LETS PREDICT FOR SINGLE HOUSE

# In[77]:


new_house = [[3,1,1520,5500,1,0,0,5,7,1520,0,1936,0,2310,5500]]


# In[78]:


new_house_std=std.transform(new_house)


# In[79]:


new_house_std


# In[80]:


int(lr.predict(new_house_std))


# In[ ]:




