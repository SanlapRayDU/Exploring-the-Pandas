#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
data=pd.DataFrame(pd.read_csv("C:/Users/admin/Downloads/loan_data_set - loan_data_set.csv"))
data.head()


# In[2]:


data.info()


# In[4]:


data.shape


# In[5]:


data1=data.drop('Loan_ID',axis=1)


# In[6]:


data1.describe()


# In[7]:


data1.isnull().sum()#checking the missing values


# In[8]:


#replacing the missing values
for i in range(data1.shape[1]):
    data1.iloc[:,i] = data1.iloc[:,i].fillna(data1.iloc[:,i].mode()[0])


# In[9]:


data1.isnull().sum()


# In[10]:


#Univariate EDA
from pandas_profiling import ProfileReport
profile=ProfileReport(data1,'Data Profile')  #Creates a summarised profile of the dataset
profile.to_notebook_iframe()


# In[28]:


t= [i for i in range(data1.shape[1]) if data1.iloc[:,i].dtype=='O'] #bivariate eda
i=0
j=0
fig, axs = plt.subplots(2,3, figsize=(30,20))
for k in t:
    ct=pd.crosstab(data1.iloc[:,k],data1['Loan_Status'])
    ct.plot(kind='bar',stacked=True, ax=axs[i][j])
    j+=1
    if j%3==0:
        i+=1
        j=0
plt.show()


# In[12]:


sns.distplot(data1.iloc[:,7],kde=True)


# In[13]:


#multivariate eda
from pandas.plotting import scatter_matrix
scatter_matrix(data1,figsize=(15,9))


# In[14]:


O= [i for i in range(data1.shape[1]) if data1.iloc[:,i].dtype=='O']
O


# In[15]:


In= [i for i in range(data1.shape[1]) if data1.iloc[:,i].dtype!='O']
In


# In[25]:


#Distplot
fig, axs = plt.subplots(1,3, figsize=(20,3))
i=0
for k in [5,6,7]:
    sns.distplot(data1.iloc[:,k],ax=axs[i])
    i+=1
plt.show()


# In[17]:


mean=[]
std=[]
i=0
for i in [5,6,7]:
    mean.append(data1.iloc[:,i].mean())
    std.append(data1.iloc[:,i].std())
print('mean of the dataset is', mean[1])
print('std. deviation is', std[1])


# In[18]:


k=0
outlier=[]
for j in [5,6,7]:
    z=[]
    z=(data1.iloc[:,j]-mean[k])/std[k]
    k=k+1
    for i in range(data1.shape[0]):
        if z[i] > 3.5:
            outlier.append(i)
outlier


# In[19]:


df=data1.drop(outlier,axis=0)
df.shape


# In[20]:


dataplot = sns.heatmap(df.corr(), cmap="YlGnBu", annot=True)


# In[21]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
for i in O:
    df.iloc[:,i]=le.fit_transform(df.iloc[:,i])
df.head()


# In[23]:


col=df.columns


# In[24]:


from scipy.stats import chi2_contingency
from scipy.stats import chi2
for k in O:
        table=pd.crosstab(df.iloc[:,k],df['Loan_Status'])
        stat, p, dof, expected = chi2_contingency(table)
        print("\n freature name: %s" % col[k])
        alpha = .05
        print('df=%.3f, significance=%.3f, p=%.3f' % (dof,alpha, p))
        if p <= alpha:
            print('Dependent')
        else:
            print('Independent')


# In[25]:


# First extract the target variable status
Y = df.Loan_Status.values
# Drop status from the dataframe and store in X
X=df.drop(['Loan_Status'],axis=1).values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)


# In[26]:


# building the model and fitting the data
logit_model = sm.Logit(Y_train, X_train).fit()


# In[27]:


logit_model.summary()


# In[28]:


yhat = logit_model.predict(X_test)
prediction = list(map(round, yhat))
print('Training Set Evaluation F1-Score: ',f1_score(Y_test,prediction))


# In[29]:


from sklearn.metrics import (confusion_matrix, accuracy_score)
  
# confusion matrix
cm = confusion_matrix(Y_test, prediction) 
print ("Confusion Matrix : \n", cm) 
  
# accuracy score of the model
print('Test accuracy = ', accuracy_score(Y_test, prediction))


# In[61]:


pd=1-yhat
pd


# In[59]:


y=[]
for i in 119:
    if pd[i]<0.4:
        y[i]='yes'
    else:
        y[i]='no'


# In[32]:


from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score

decisiontree = DecisionTreeClassifier(max_depth = 5, random_state = 33)
decisiontree.fit(X_train, Y_train)
y_pred = decisiontree.predict(X_test)
acc_decisiontree = round(accuracy_score(y_pred, Y_test), 2)
print(acc_decisiontree)
# ?\tree.plot_tree(decisiontree.fit(X_train, y_train)) 


# In[147]:


tree.plot_tree(decisiontree)


# In[33]:


y_pred


# In[34]:


prediction

