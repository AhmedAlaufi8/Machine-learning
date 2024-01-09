#!/usr/bin/env python
# coding: utf-8

# ## **Import Importnant Libraries**

# In[143]:


import numpy as np
import pandas as pd 

#For Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

#For Data modeling
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Metrics and helpful functions
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import roc_auc_score, roc_curve

import warnings
warnings.filterwarnings("ignore")


# ## **Read the Dataset**

# In[58]:


df = pd.read_csv('ifood_df.csv')


# In[59]:


df.head()


# From the first rows of data, we notice customer related data about income, education, family, purchases, channels and campaign response.

# ##  **Exploratory Data Analysis**

# In[60]:


df.info()


# In[61]:


df = df.reindex(columns=['Age','education_2n Cycle', 'education_Basic', 'education_Graduation', 'education_Master', 
                         'education_PhD', 'Income','marital_Single', 'marital_Married', 'marital_Together', 'marital_Divorced', 'marital_Widow',
                         'Kidhome', 'Teenhome', 'Recency', 'NumWebVisitsMonth', 'Customer_Days', 'Complain',
                         'MntTotal', 'MntRegularProds','MntWines', 'MntFruits','MntMeatProducts', 
                         'MntFishProducts', 'MntSweetProducts','MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
                         'NumCatalogPurchases', 'NumStorePurchases', 'AcceptedCmp1','AcceptedCmp2','AcceptedCmp3', 
                         'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmpOverall','Response', 'Z_CostContact', 'Z_Revenue' 
                        ]
               )


# In[62]:


df.describe()


# In[63]:


#check for missing values
df.isna().sum()


# In[64]:


#drop the missing values in Income and MntMeatProducts columns
df.dropna(subset = ["Income", "MntMeatProducts"], inplace = True)


# In[65]:


df.isna().sum()


# In[66]:


#fill the missing values in NumWebVisitsMonth column with mean
df["NumWebVisitsMonth"].fillna(df["NumWebVisitsMonth"].mean(), inplace = True)


# In[67]:


df.isna().sum()


# In[68]:


#check for outliers
df.duplicated().sum()


# In[69]:


df=df.drop_duplicates()
print(df.shape)


# In[70]:


#drop columns Z_CostContact and Z_Revenue     
df = df.drop(['Z_CostContact', 'Z_Revenue'], axis=1)
print(df.shape)


# In[71]:


#There were negative amounts of purchase
df[df['MntRegularProds'] < 0 ].head()


# In[82]:


#The Income column histogram
plt.figure(figsize=(5,3))
sns.histplot(x=df['Income'])
median = df['Income'].median()
plt.axvline(median, color='red', linestyle='--')
print('Median: {}'.format(median))


# In[83]:


#The Total Spent columns histogram
plt.figure(figsize=(5,3))
sns.histplot(x=df['MntTotal'], bins=range(0,2600,100))
median = df['MntTotal'].median()
plt.axvline(median, color='red', linestyle='--')
print('Median: {}'.format(median))


# ## **Feature Engineering**

# In[72]:


#Total number of purchases
df['NumPurchases'] = df['NumWebPurchases'] + df['NumCatalogPurchases'] + df['NumStorePurchases']
df['NumPurchases'].describe()


# In[73]:


#The Average amount per purchase
df['AvgMntPerPurchase'] = df['MntTotal']/df['NumPurchases']


# In[74]:


df.loc[df['AvgMntPerPurchase']==np.inf, 'AvgMntPerPurchase'] = 0
df['AvgMntPerPurchase'].describe()


# In[75]:


#The Average amount per regular purchase
df['AvgRegMntPerPurchase'] = df['MntRegularProds']/df['NumPurchases']


# In[76]:


df.loc[df['AvgRegMntPerPurchase']==np.inf, 'AvgRegMntPerPurchase'] = 0
df['AvgRegMntPerPurchase'].describe()


# In[77]:


#The influence of deals on number of purchases
df['DealPerPurchase'] = df['NumDealsPurchases']/df['NumPurchases']
df.loc[df['DealPerPurchase']==np.inf, 'DealPerPurchase'] = 1
df.loc[df['DealPerPurchase'].isna(), 'DealPerPurchase'] = 1


# In[78]:


df['DealPerPurchase'].describe()


# In[79]:


df[df['DealPerPurchase'] > 1].head()


# In[81]:


plt.figure(figsize=(6,3))
sns.histplot(x=df['DealPerPurchase'], bins=20)
median = df['DealPerPurchase'].median()
plt.axvline(median, color='red', linestyle='--')
print('Median: {}'.format(median))


# In[88]:


#Drop col added that didn't seem promising
df = df.drop(['NumPurchases', 'AvgRegMntPerPurchase'], axis=1)


# In[89]:


df.shape


# In[90]:


y = df['Response']
#features to use in model
X = df.drop('Response', axis=1)


# In[91]:


#Split the data into training set and testing set. Stratify to account for imbalance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, 
                                                    stratify=y, random_state=42)


# ##  **1- Decision Tree Model**

# In[126]:


decision_tree_model = DecisionTreeClassifier(random_state=0)
decision_tree_model = decision_tree_model.fit(X_train, y_train)


# In[127]:


y_pred = decision_tree_model.predict(X_test)


# In[115]:


print(classification_report(y_test, y_pred))


# In[116]:


y_pred_train = decision_tree_model.predict(X_train)


# In[117]:


print(classification_report(y_train, y_pred_train))


# In[120]:


decision_tree_cm = confusion_matrix(y_test, y_pred, 
                                    labels=decision_tree_model.classes_)
#display confusion matrix
tree_display = ConfusionMatrixDisplay(confusion_matrix=decision_tree_cm, 
                                  display_labels=decision_tree_model.classes_)
#Plot the confusion matrix
tree_display.plot(values_format='')
plt.show()


# In[129]:


from sklearn import tree
tree.plot_tree(decision_tree_model)


# ## **2- KNN Model**

# In[121]:


KNN_model = KNeighborsClassifier()
KNN_model.fit(X_train, y_train)


# In[122]:


y_pred = KNN_model.predict(X_test)


# In[123]:


print(classification_report(y_test, y_pred))


# In[124]:


KNN_cm = confusion_matrix(y_test, y_pred, 
                                    labels=KNN_model.classes_)
#display confusion matrix
KNN_display = ConfusionMatrixDisplay(confusion_matrix=KNN_cm, 
                                  display_labels=KNN_model.classes_)
#Plot the confusion matrix
KNN_display.plot(values_format='')
plt.show()

