#!/usr/bin/env python
# coding: utf-8

# # TASK 1 - Prediction using Supervised Machine Learning
# In this task it is required to predict the percentage of a student on the basis of number of hours studied using the Linear Regression supervised machine learning algorithm.
# 
# Steps:
# Step 1 - Importing the dataset
# Step 2 - Visualizing the dataset
# Step 3 - Data preparation
# Step 4 - Training the algorithm
# Step 5 - Visualizing the model
# Step 6 - Making predcitions
# Step 7 - Evaluating the model
# Author: Aniket Tanpure
# 

# In[1]:


# import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# import the data in the pandas dataframe
data = pd.read_csv('student_marks.csv')


# In[3]:


# check first 5 rows of data
data.head()


# In[4]:


# check last 5 rows of data
data.tail()


# In[6]:


# find the number of rows and columns
data.shape


# In[7]:


# find more info of data
data.info()


# In[9]:


data.describe


# In[10]:


# checking the null values in the data
data.isnull().sum()


# # STEP 2 - Visualizing the dataset
# In this we will plot the dataset to check whether we can observe any relation between the two variables or not

# In[15]:


plt.figure(figsize=(16,9))
data.plot(x='Hours',y='Scores',style='.',color='red')
plt.title('Hours vs Percentage')
plt.xlabel('Hours')
plt.ylabel('Percentage')
plt.show()


# # From the graph above, we can observe that there is a linear relationship between "hours studied" and "percentage score". So, we can use the linear regression supervised machine model on it to predict further values.

# In[16]:


# we can also use the correlation between the variables
data.corr()


# # STEP 3 - Data preparation in this step we will divide the data into "features" (inputs) and "labels" (outputs). 
# After that we will split the whole dataset into 2 parts - testing data and training data.
# 

# In[17]:


data.head()


# In[18]:


# using the iloc funcion we can divide the data
x = data.iloc[:,:1].values
y = data.iloc[:,1:].values


# In[19]:


x


# In[20]:


y


# In[21]:


# split the data into training & testing data
from sklearn.model_selection import train_test_split


# In[22]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


# # STEP 4 - Training the Algorithm
# We have splited our data into training and testing sets, and now we will train our Model.

# In[24]:


from sklearn.linear_model import LinearRegression


# In[25]:


model = LinearRegression()


# In[26]:


model.fit(x_train,y_train)


# # STEP 5 - Visualizing the model
# After training the model, now its time to visualize it.

# In[27]:


line = model.coef_*x + model.intercept_

# plotting for the training data
plt.figure(figsize=(16,9))
plt.scatter(x_train,y_train,color='violet')
plt.plot(x,line,color='red')
plt.xlabel('Hours')
plt.ylabel('Score')
plt.show()


# In[29]:


# plot for the testing data
plt.figure(figsize=(16,8))
plt.scatter(x_test,y_test,color='black')
plt.plot(x,line,color='red')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()


# # STEP 6 - Making Predictions Now that we have trained our algorithm, it's time to make some predictions.

# In[30]:


print(x_test) # Testing data in hours
y_pred = model.predict(x_test) # predict the scores


# In[31]:


y_test


# In[32]:


y_pred


# In[33]:


# comparing the predicted vs the actual output
comp = pd.DataFrame({'Actual':[y_test],'predicted':[y_pred]})
comp


# In[36]:


# testing with the own data
hours = 9.25
own_pred = model.predict([[hours]])
print("The predicted score if a person studies for",hours,"hours is",own_pred[0])


# # STEP 7 - Evaluating the model
# In the last step, we are going to evaluate our trained model by calculating mean absolute error

# In[39]:


from sklearn import metrics
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,y_pred))


# In[ ]:




