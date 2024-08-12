#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Loading the required libraries
import pandas as pd
import numpy as np


# In[135]:


#Loading the data
Insure_claim=pd.read_csv("E:\one drive\Desktop\Hackthon\Insurance_Claim_Info_data.csv")


# In[136]:


Insure_claim


# In[137]:


# reading the second dataset Insurance_Date_data
Insure_date=pd.read_csv("E:\one drive\Desktop\Hackthon\Insurance_Date_data.csv")


# In[138]:


Insure_date


# In[139]:


# reading the third dataset Insurance_result
Insure_result=pd.read_csv("E:\one drive\Desktop\Hackthon\Insurance_Result_data.csv")


# In[140]:


Insure_result


# In[141]:


#merging the first and second datasets
data1=pd.merge(Insure_claim,Insure_date,on=['Claim Number'], how='inner')


# In[142]:


data1


# In[143]:


# merging with third dataset
data=pd.merge(data1,Insure_result, on=['Claim Number'], how='inner')


# In[ ]:


# Understand the Exploratory Data Analysis (EDA)


# In[144]:


#checking first 5 columns
data.head()


# In[145]:


#checking last 5 columns
data.tail()


# In[146]:


#checking datatype
data.info()


# In[147]:


# checking unique values in dataset
data.nunique()


# In[148]:


# checking null values in the dataset
data.isnull().sum()


# In[ ]:


data.shape


# In[ ]:


# Statistic summary

data.describe()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(x='Disposition',data=data)


# In[ ]:





# In[ ]:





# In[ ]:





# In[149]:


data.drop(['Claim Number','City Code'],axis=1,inplace=True)


# In[150]:


data


# In[151]:


data['Incident Date',] = pd.to_datetime(data['Incident Date'])


# In[152]:


data['day']=data['Incident Date',].dt.day


# In[153]:


data['month']=data['Incident Date',].dt.month


# In[154]:


data['year']=data['Incident Date',].dt.year


# In[155]:


data.drop(columns='Incident Date', axis=1, inplace=True)


# In[156]:


data


# In[ ]:


train_df['Incident Date',] = pd.to_datetime(train_df['Incident Date'])

train_df['day']=train_df['Incident Date',].dt.day

train_df['month']=train_df['Incident Date',].dt.month

train_df['year']=train_df['Incident Date',].dt.year

train_df.drop(columns='Incident Date', axis=1, inplace=True)


# In[157]:


data.describe()


# In[191]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(x='Disposition',data=data)


# In[ ]:


# Data Preprocssing


# In[100]:


cat=data.select_dtypes(include=['object']).columns
num=data.select_dtypes(include=['float64']).columns


# In[101]:


cat


# In[102]:


data[cat]=data[cat].astype('category')


# In[103]:


cat=cat.drop('Disposition')


# In[104]:


cat


# In[105]:


num


# In[106]:


X=data.drop(['Disposition'],axis=1)
y=data['Disposition']


# In[107]:


from sklearn.model_selection import  train_test_split


# In[108]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=123)
len(X_train)


# In[109]:


len(y_train)


# In[110]:


len(X_test)


# In[111]:


len(y_test)


# In[112]:


from sklearn.preprocessing import StandardScaler ,OneHotEncoder ,LabelEncoder


# In[113]:


label=LabelEncoder()
label.fit(y_train)


# In[114]:


y_train=label.transform(y_train)
y_test=label.transform(y_test)


# In[115]:


scaler=StandardScaler()
scaler.fit(X_train[num])


# In[116]:


X_train_std=scaler.transform(X_train[num])
X_test_std=scaler.transform(X_test[num])


# In[117]:


onehot=OneHotEncoder(handle_unknown='ignore')
onehot.fit(X_train[cat])


# In[118]:


X_train_enc=onehot.transform(X_train[cat]).toarray()
X_test_enc=onehot.transform(X_test[cat]).toarray()


# In[119]:


X_train_sm=np.concatenate([X_train_std,X_train_enc],axis=1)
X_test_sm=np.concatenate([X_test_std,X_test_enc],axis=1)


# In[120]:


from sklearn.linear_model import LogisticRegression
logit=LogisticRegression()
logit.fit(X_train_sm,y_train)


# In[121]:


ypredtest=logit.predict(X_test_sm)
ypredtrain=logit.predict(X_train_sm)


# In[122]:


from sklearn.metrics import classification_report,accuracy_score


# In[123]:


print(accuracy_score(ypredtest,y_test))


# In[124]:


print(classification_report(ypredtest,y_test))


# In[125]:


print(accuracy_score(ypredtrain,y_train))


# In[126]:


print(classification_report(ypredtrain,y_train))


# In[127]:


from sklearn.tree import DecisionTreeClassifier


# In[128]:


tree=DecisionTreeClassifier()


# In[129]:


tree.fit(X_train_sm,y_train)


# In[131]:


ypredtrain=tree.predict(X_train_sm)
ypredtest=tree.predict(X_test_sm)


# In[132]:


print(accuracy_score(ypredtrain,y_train))


# In[133]:


print(classification_report(ypredtrain,y_train))


# In[130]:


print(classification_report(ypredtest,y_test))


# In[177]:


from sklearn.model_selection import GridSearchCV


# In[178]:


param_grid = {"criterion": ["gini", "entropy"],
              "min_samples_split": [2, 5],
              "max_depth": [None, 2],
              "min_samples_leaf": [1, 5]
             }


# In[180]:


dtclf_grid = GridSearchCV(tree, param_grid, cv=3)


# In[182]:


dtclf_grid.fit(X_train_sm, y_train)


# In[183]:


dtclf_grid.best_params_


# In[185]:


train_pred = dtclf_grid.predict(X_train_sm)
test_pred = dtclf_grid.predict(X_test_sm)


# In[189]:


print(classification_report(y_train,train_pred))
print(classification_report(y_test,test_pred))


# #### Test

# In[198]:


test_data=pd.read_excel("E:\\one drive\\Desktop\\Hackthon\\test_data-1663477366404.xlsx")


# In[199]:


test_data


# In[200]:


test_data.info()


# In[ ]:




