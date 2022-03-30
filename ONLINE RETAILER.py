#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import required libaries for data set
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt


# In[3]:


# import required libraries for clustering
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# In[4]:


#use panadas's function to read the data which need to be analysed (in CSV file)
retail = pd.read_csv("C:/SHRUTI/2nd year/4 sem/Predictive Analytics Lab/CAT-2/OnlineRetail.csv",sep=",",encoding="ISO-8859-1")
retail.head()


# In[5]:


#shape of the dataframe (df)
retail.shape


# In[6]:


# info of the df
retail.info()


# In[7]:


# description of the df
retail.describe()


# In[8]:


# checking df's missing value's attribution in %
df_null = round(100*(retail.isnull().sum())/len(retail), 2)
df_null


# In[9]:


# checking df's missing value's lcoation
retail[retail.isnull().values==True]


# In[10]:


# drop the rows which have the missing value
retail = retail.dropna()
retail.shape


# In[11]:


# change customer ID data type, astype()--Cast a pandas object to a specified dtype dtype

retail['CustomerID'] = retail['CustomerID'].astype(str)


# In[12]:


# check the raw data after drop the missing value
df_null_after = round(100*(retail.isnull().sum())/len(retail), 2)
df_null_after


# In[13]:


# convert datetime
retail['InvoiceDate'] = pd.to_datetime(retail['InvoiceDate'],format='%d-%m-%Y %H:%M')


# In[14]:


# compute the last date/time of the period
max_date = max(retail['InvoiceDate'])
max_date


# In[15]:


# dir()shows the list of what objects can do
dir(max_date)


# In[16]:


# compute the recency
retail['Recency'] = max_date - retail['InvoiceDate']
retail.head()


# In[17]:


# calculate the last transition date
rfm_r = retail.groupby("CustomerID")["Recency"].min().reset_index()
rfm_r.head()


# In[18]:


# extract only the days
rfm_r["Recency"] = rfm_r["Recency"].dt.days
rfm_r.head()


# In[19]:


# extract each customerID's total Invoice Number & count in total 
rfm_f = retail.groupby('CustomerID')['InvoiceNo'].count().reset_index()


# In[20]:


# form new data set's columus
rfm_f.columns = ['CustomerID', 'Frequency']


# In[21]:


# function head()'s default parameter size is 5, so output will show 5
rfm_f.head()
retail["Amount"] = retail['Quantity']*retail['UnitPrice']
rfm_m = retail.groupby("CustomerID")["Amount"].sum().reset_index()
rfm_m.head()
rfm = pd.merge(rfm_r, rfm_f, on='CustomerID', how='inner')
rfm = pd.merge(rfm, rfm_m, on='CustomerID', how='inner')
rfm.head()
rfm.shape


# In[22]:


# use IQR method to detect outliers, tells any value which is beyond the range of -1.5 x IQR to 1.5 x IQR treated as outliers
def iqr_outliers(df, field):
    q1 = df[field].quantile(0.25)
    q3 = df[field].quantile(0.75)
    iqr = q3-q1
    lower_tail = q1 - 1.5 * iqr
    upper_tail = q3 + 1.5 * iqr
    
    df = df[(df[field] >= lower_tail) & (df[field] <= upper_tail)]    
    return df

rfm.head()

rfm_copy = iqr_outliers(rfm, 'Recency')
rfm_copy = iqr_outliers(rfm_copy, 'Frequency')
rfm_copy = iqr_outliers(rfm_copy, 'Amount')


# In[23]:


#rfm_copy.head()
# check columns name for extract columns
rfm_copy.columns


# In[24]:


# extract columns for rescaling the attributes
rfm_copy[['Amount','Frequency','Recency' ]]
rfm_rescale = rfm_copy[['Amount','Frequency','Recency' ]]
rfm_rescale.head()


# In[25]:


# rescaling the data frame by using Standardisation Scaling
rfm_df = rfm_rescale[['Amount', 'Frequency', 'Recency']]

scaler = StandardScaler()

rfm_rescale_rescale = scaler.fit_transform(rfm_rescale)
rfm_rescale_rescale.shape
rfm_rescale_rescale = pd.DataFrame(rfm_rescale_rescale)
rfm_rescale_rescale.columns = ['Amount', 'Frequency', 'Recency']
rfm_rescale_rescale.head()
range_n_clusters = [2, 3, 4, 5, 6, 7, 8,9,10]

for num_clusters in range_n_clusters:
    
    # intialise kmeans
    kmeans = KMeans(n_clusters=num_clusters, max_iter=80)
    kmeans.fit(rfm_rescale_rescale)
    
    cluster_labels = kmeans.labels_
    
    # silhouette score
    silhouette_avg = silhouette_score(rfm_rescale_rescale, cluster_labels)
    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))
    # the final K is 3, because it is the closest to 1
kmeans = KMeans(n_clusters=3, max_iter=80)
kmeans.fit(rfm_rescale_rescale)
kmeans.labels_
rfm_copy['Cluster_ID'] = kmeans.labels_

rfm_copy.head()


# In[26]:


# ClusterID vs Recency
sns.boxplot(x='Cluster_ID', y='Recency', data=rfm_copy)


# In[27]:


# ClusterID vs Frequency
sns.boxplot(x='Cluster_ID', y='Frequency', data=rfm_copy)


# In[28]:


# ClusterID vs Amount
sns.boxplot(x='Cluster_ID', y='Amount', data=rfm_copy)


# In[ ]:




