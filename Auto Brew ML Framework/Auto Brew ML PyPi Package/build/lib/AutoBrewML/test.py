# %%
def Acquisition_DataTypeConversion(filepath,col_dtype_dict):
    import pandas as pd
    import numpy as np
    df = pd.read_csv(filepath)
    print("Original datatypes:")
    print(df.dtypes)
    input_dataframe = df.astype(col_dtype_dict)
    print("Converted datatypes:")              
    print(input_dataframe.dtypes)
    input_dataframe['Index'] = np.arange(len(input_dataframe)) 
    print("Input dataframe:")
    print(input_dataframe.head())
    return(input_dataframe)
    
    
# %%
def impute(input_dataframe):
  # Replace NaNs with the median or mode of the column depending on the column type
  #print("Standard deviation of dataframe before imputation is:\n",input_dataframe.std(axis = 0, skipna = True))
  NumCol=[]
  CatCol=[]
  for column in input_dataframe.columns:
    if(input_dataframe[column].dtype) not in ["object"]:
      NumCol.append(column)
      input_dataframe[column].fillna(input_dataframe[column].median(), inplace=True)
      
    else:
      CatCol.append(column)
      most_frequent = input_dataframe[column].mode() 
      if len(most_frequent) > 0:
        input_dataframe[column].fillna(input_dataframe[column].mode()[0], inplace=True)
        
      else:
        input_dataframe[column].fillna(method='bfill', inplace=True)
       
        input_dataframe[column].fillna(method='ffill', inplace=True)
        
  print("\n","Imputation of Columns:")
  print("Imputation of Numerical Column done using Median:")
  print(*NumCol, sep = ", ") 
  print("Imputation of Categorical Column done using Mode/bfill/ffill:")
  print(*CatCol, sep = ", ") 
  

  return input_dataframe


# %%
# Function:Acquisition_DataTypeConversion
filepath=r"C:\Users\srde\Downloads\Titanic.csv"
col_dtype_dict={
                    'PassengerId':'int'
                    ,'Survived':'int'
                    ,'Pclass':'int'
                    ,'Name':'object'
                    ,'Sex':'object'
                    ,'Age':'float'
                    ,'SibSp':'int'
                    ,'Parch':'int'
                    ,'Ticket':'object'
                    ,'Fare':'float'
                    ,'Cabin':'object'
                    ,'Embarked':'object'
                }
input_dataframe=Acquisition_DataTypeConversion(filepath,col_dtype_dict)


# %%
input_dataframe












# %% TEST STRATIFY

#Imputation to avoid mixup in chi2 test with categorical values frequencies
input_dataframe=impute(input_dataframe)
#Deep=True (the default), a new object is produced with a copy of the calling object’s data and indices. Changes to the copy’s data or indices will not reflect the original object.
#df_fin=input_dataframe
df_fin = input_dataframe.copy(deep=True)

import pandas as pd
from collections import Counter

import pandas as pd
import numpy as np
import math


#Label encoder accepts only non nulls
for column in df_fin.columns.values:
      try:
          df_fin[column].fillna(df_fin[column].mean(), inplace=True)
      except TypeError:
          df_fin[column].fillna(df_fin[column].mode()[0], inplace=True)

#K means accept only numeric columns hence label encode
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in df_fin.columns:
  if (df_fin[col].dtype) not in ["int32","int64","float","float64"]:
    df_fin[col]=le.fit_transform(df_fin[col])

#Normalisation of data as K-Means clustering uses euclidean distance
column_list=list(df_fin.columns)
column_list_actual=column_list
column_list.remove('Index')
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
df_fin[column_list] = pd.DataFrame(mms.fit_transform(df_fin[column_list]),columns = column_list_actual )


#Getting best k for K-Means using Silhoute max score k    
from sklearn.cluster import KMeans
Dic = {}
for k in range(2,6):
  k_means = KMeans(n_clusters=k)
  model = k_means.fit(df_fin)
  y_hat = k_means.predict(df_fin)
  from sklearn import metrics
  labels = k_means.labels_
  Dic.update({k:metrics.silhouette_score(df_fin, labels, metric = 'euclidean')})
Keymax = max(Dic, key=Dic.get) 
#print(Keymax)

#K-Means clustering using optimal k
kmeans = KMeans(n_clusters=Keymax)
y = kmeans.fit_predict(df_fin)
df_fin['Cluster'] = y
#print(df_fin.head())

# summarize distribution actual
print('Data Summary before sampling: ')
y=df_fin['Cluster']
counter = Counter(y)
for k,v in counter.items():
    per = v / len(y) * 100
    print('Class=%d, n=%d (%.3f%%)' % (k, v, per))

#Stratifying to get sample
df_sample = pd.DataFrame()
count_records_per_cluster= list()
sample_records_per_cluster=list()
total_records=len(df_fin.index)

#Getting ideal sample size by Solven's Formula assuming Confidence Level=95% i.e n = N / (1 + Ne²)
total_records=len(df_fin.index)
sample_size= round(total_records / (1 + total_records* (1-0.95)*(1-0.95)))

#For chi2 test the two conditions are-
#     1.The distinct unique values in the original v/s sample must be same. So force fill missing values with frequency=0 (absent in sample w.r.t. original)
#     2.The sum of frequencies in both original & sample must be same. So scale up the sample frequency to match the sum of frequencies of original exactly- sample_count=sample_count*input_dataframe_count_sum/sample_count_sum. For the sum of frequencies to match exactly the scale up factor should be a whole number or this ratio of input_dataframe_count_sum/sample_count_sum should be a whole number. i.e. input_dataframe_count_sum should be a multiple of sample_count_sum (or sample_size)  
sample_size=int(total_records/math.floor(total_records/sample_size))


for i in range(Keymax):
  df_fin_test=df_fin
  count_records_per_cluster.append(df_fin_test['Cluster'].value_counts()[i])
  sample_records_per_cluster.append(count_records_per_cluster[i]/total_records * sample_size)
  df_sample_per_cluster=df_fin_test[df_fin_test.Cluster==i]
  df_sample_per_cluster=df_sample_per_cluster.sample(int(sample_records_per_cluster[i]),replace=True)   
  df_sample=df_sample.append(df_sample_per_cluster, ignore_index = True)
#df_sample.head()

# summarize distribution sampled
print('Data Summary after sampling: ')
y=df_sample['Cluster']
counter = Counter(y)
for k,v in counter.items():
    per = v / len(y) * 100
    print('Class=%d, n=%d (%.3f%%)' % (k, v, per))


# %%
df_sample

# %%
input_dataframe

# %%
#Remove Columns which are not present in input_dataframe
df_sample.drop(['Cluster'], axis=1, inplace=True)

# %%
#Getting back Sample in original data form
uniqueIndex = list(df_sample['Index'].unique())
print(uniqueIndex)

# %%

subsample= input_dataframe[input_dataframe.Index.isin(uniqueIndex)]
subsample


# %%
input_dataframe

# %%
