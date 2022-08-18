
# # %%
import warnings
warnings.filterwarnings('ignore')

# %%
# # DBTITLE 1,Pandas Version check:
def PandasVersion():
    import pandas as pd
    print("The in context Pandas version is:")
    print(pd.__version__)
#PandasVersion()




# %%
# # DBTITLE 1,Acquisition & DataTypeConversion:
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

# #Function Call:
# filepath=r"C:\Users\srde\Downloads\Titanic.csv"
# col_dtype_dict={
#                     'PassengerId':'int'
#                     ,'Survived':'int'
#                     ,'Pclass':'int'
#                     ,'Name':'string'
#                     ,'Sex':'string'
#                     ,'Age':'float'
#                     ,'SibSp':'int'
#                     ,'Parch':'int'
#                     ,'Ticket':'string'
#                     ,'Fare':'float'
#                     ,'Cabin':'string'
#                     ,'Embarked':'string'
#                 }
# input_dataframe=Acquisition_DataTypeConversion(filepath,col_dtype_dict)




# %%
# DBTITLE 1,Data Profiler
#pip install pandas-profiling
def BrewDataProfiler(input_dataframe):
  #import time 
  #ts = int(time.time())
  #ts= "%s" % str(ts)
  #filepath="adl://<Your ADLS Name>.azuredatalakestore.net/DEV/EDAProfile_" + ts +".html"
  import pandas_profiling as pp
  profile = pp.ProfileReport(input_dataframe) 
  p = profile.to_html() 
  #profile.to_file('/dbfs/FileStore/EDAProfile.html')
  #dbutils.fs.cp ("/FileStore/EDAProfile.html", filepath, True)
  #print("EDA Report can be downloaded from path: ",filepath)
  return(p)
  #displayHTML(p)
  #return(df.describe())

# #Function Call
# # p=Data_Profiling_viaPandasProfiling(input_dataframe)
# displayHTML(p)





# %%
# Data Cleanser
# # DBTITLE 1,Data Cleanser -Fix Categorical Columns
def fixCategoricalColumns(input_dataframe):
  import pandas as pd
  from sklearn.preprocessing import LabelEncoder 
  from sklearn.preprocessing import MinMaxScaler
  le = LabelEncoder()
  # Replace structural errors on categorical columns - Inconsistent capitalization and Label Encode Categorical Columns + MinMax Scaling
  print("\n","Categorical columns cleansing:","\n")
  print("* Fixing inconsistent capitalization and removing any white spaces.")
  for column in input_dataframe.columns.values:
    if str(input_dataframe[column].values.dtype) == 'object':
      for ind in input_dataframe.index:
        input_dataframe[column] = input_dataframe[column].astype(str).str.title()
        input_dataframe[column] = input_dataframe[column].str.replace(" ","")
    if str(input_dataframe[column].values.dtype) == 'object':
      for col in input_dataframe.columns:
        input_dataframe[col]=le.fit_transform(input_dataframe[col])
  print("* Label Encoding on categorical columns.")
  
  
  print("* MinMax scaling for Normalisation.")
  #scaler = MinMaxScaler()
  #print(scaler.fit(input_dataframe))
  #print(scaler.data_max_)
  #print(scaler.transform(input_dataframe))
  #Normalisation of data as K-Means clustering uses euclidean distance
  column_list=list(input_dataframe.columns)
  column_list_actual=column_list
  column_list.remove('Index')
  from sklearn.preprocessing import MinMaxScaler
  mms = MinMaxScaler()
  input_dataframe[column_list] = pd.DataFrame(mms.fit_transform(input_dataframe[column_list]),columns = column_list_actual )
  #print(scaler.transform([[2, 2]]))
  return input_dataframe                         



# DBTITLE 1,Data Cleanser -Imputation of whole df
def impute(input_dataframe):
  import pandas as pd

  # # Count number of missing values per column
  # missingCount = input_dataframe.isnull().sum()
  # print("* Displaying number of missing records per column:")
  # print(missingCount)


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
        
  print("\n","Imputation of Columns:","\n")
  print("* Imputation of Numerical Column done using Median:")
  print(*NumCol, sep = ", ") 
  print("* Imputation of Categorical Column done using Mode/bfill/ffill:")
  print(*CatCol, sep = ", ") 
  

  return input_dataframe

# DBTITLE 1,Data Cleanser -Impute col by col
def impute_col(input_dataframe,column):
  import pandas as pd

  # # Count number of missing values per column
  # missingCount = input_dataframe.isnull().sum()
  # print("* Displaying number of missing records per column:")
  # print(missingCount)


  # Replace NaNs with the median or mode of the column depending on the column type
  #print("Standard deviation of dataframe before imputation is:\n",input_dataframe.std(axis = 0, skipna = True))
  if(input_dataframe[column].dtype) not in ["object"]:
      input_dataframe[column].fillna(input_dataframe[column].median(), inplace=True)
      print("\n","Imputation of Columns:","\n")
      print("* Imputation of Numerical Column hence done using Median:")
      print(column) 

  else:
    most_frequent = input_dataframe[column].mode() 
    if len(most_frequent) > 0:
      input_dataframe[column].fillna(input_dataframe[column].mode()[0], inplace=True)
        
    else:
      input_dataframe[column].fillna(method='bfill', inplace=True)
      input_dataframe[column].fillna(method='ffill', inplace=True)
      print("* Imputation of Categorical Column hence done using Mode/bfill/ffill:")
      print(column) 
  

  return input_dataframe



# DBTITLE 1,Data Cleanser -Impute/Drop NULL Rows
def cleanMissingValues(input_dataframe):
  import numpy as np
  import pandas as pd
  print("\n","Impute/Drop NULL Rows:","\n")
  print("* Total rows in the Input dataframe:",len(input_dataframe.index)) #Total count of dataset
  totalCells = np.product(input_dataframe.shape)
  # Calculate total number of cells in dataframe
  print("* Total number of cells in the input dataframe:",totalCells)
  input_dataframe = input_dataframe.replace(r'^\s+$', np.nan, regex=True) #replace white spaces with NaN except spaces in middle
  # Count number of missing values per column
  missingCount = input_dataframe.isnull().sum()
  print("* Displaying number of missing records per column:")
  print(missingCount)
  # Calculate total number of missing values
  totalMissing = missingCount.sum()
  print("* Total no. of missing values=",totalMissing)
  # Calculate percentage of missing values
  print("* The dataset contains", round(((totalMissing/totalCells) * 100), 2), "%", "missing values.")
  cleaned_inputdf= input_dataframe   
  for col in input_dataframe.columns:
    if(missingCount[col]>0):
      print("* Percent missing data in '",col,"' column =", (round(((missingCount[col] / input_dataframe.shape[0]) * 100), 2)))
      if((round(((missingCount[col] / input_dataframe.shape[0]) * 100), 2))>50):
        print("* Dropping this column as it contains missing values in more than half of the dataset.")
        input_dataframe_CleanCols=input_dataframe.drop(col,axis=1)
        print("* Total Columns in original dataset: %d \n" % input_dataframe.shape[1])
        print("* Total Columns now with na's dropped: %d" % input_dataframe_CleanCols.shape[1])
        cleaned_inputdf=input_dataframe_CleanCols
        
      else:
        print("* As percent of missing values is less than half, imputing this column based on the data type.")
        input_dataframe_imputed=impute_col(input_dataframe,col)
        cleaned_inputdf=input_dataframe_imputed
        
  return cleaned_inputdf

# COMMAND ----------

# DBTITLE 1,Data Cleanser Master
def BrewDataCleanser(input_dataframe):
    import pandas as pd
    #inputdf_1=impute(input_dataframe)
    inputdf_2=cleanMissingValues(input_dataframe)
    inputdf_3=fixCategoricalColumns(inputdf_2)
    return inputdf_3












# %%
# DBTITLE 1,Sampling- Random Sampling:
def RandomSampling(input_dataframe):
  df_fin=input_dataframe
  import pandas as pd
  import numpy as np
  import math
  
  print("Random Sampling starting")

  #Imputation to avoid mixup in chi2 test with categorical values frequencies
  input_dataframe=impute(input_dataframe)

  #Getting ideal sample size by Solven's Formula assuming Confidence Level=95% i.e n = N / (1 + Ne²)
  total_records=len(df_fin.index)
  sample_size= round(total_records / (1 + total_records* (1-0.95)*(1-0.95)))

  #For chi2 test the two conditions are-
  #     1.The distinct unique values in the original v/s sample must be same. So force fill missing values with frequency=0 (absent in sample w.r.t. original)
  #     2.The sum of frequencies in both original & sample must be same. So scale up the sample frequency to match the sum of frequencies of original exactly- sample_count=sample_count*input_dataframe_count_sum/sample_count_sum. For the sum of frequencies to match exactly the scale up factor should be a whole number or this ratio of input_dataframe_count_sum/sample_count_sum should be a whole number. i.e. input_dataframe_count_sum should be a multiple of sample_count_sum (or sample_size)  
  sample_size=int(total_records/math.floor(total_records/sample_size))

  subsample=df_fin.sample(n=sample_size)
  
  #Hypothesis test to see if sample is accepted
  from scipy import stats
  pvalues = list()
  for col in subsample.columns:
    if (subsample[col].dtype) in ["int32","int64","float","float64"]: 
      # Numeric variable. Using Kolmogorov-Smirnov test
      pvalues.append(stats.ks_2samp(subsample[col],input_dataframe[col]))
        
    else:
      # Categorical variable. Using Pearson's Chi-square test
      from scipy import stats
      import pandas as pd
      #For chi2 test the two conditions are-
      # #     1.The distinct unique values in the original v/s sample must be same. So force fill missing values with frequency=0 (absent in sample w.r.t. original)
      sample_count=pd.DataFrame(subsample[col].value_counts())
      input_dataframe_count=pd.DataFrame(input_dataframe[col].value_counts())
      absent=input_dataframe_count-sample_count
      absent2=absent.loc[absent[col].isnull()]
      absent2[col]=0
      sample_count_final=pd.concat([absent2,sample_count]) 

      #     2.The sum of frequencies in both original & sample must be same. So scale up the sample frequency to match the sum of frequencies of original exactly- sample_count=sample_count*input_dataframe_count_sum/sample_count_sum. For the sum of frequencies to match exactly the scale up factor should be a whole number or this ratio of input_dataframe_count_sum/sample_count_sum should be a whole number. i.e. input_dataframe_count_sum should be a multiple of sample_count_sum (or sample_size)  
      sample_count_final_sum=sample_count_final.sum()
      input_dataframe_count_sum=input_dataframe_count.sum()
      sample_count_final=sample_count_final*input_dataframe_count_sum/sample_count_final_sum
      sample_count_final = sample_count_final.apply(np.ceil).astype(int)
      pvalues.append(stats.chisquare(sample_count_final.astype(int), input_dataframe_count.astype(int)))
  
  count=0
  length =  len(subsample.columns)
  pvalues_average=0
  for i in range(length): 
    if pvalues[i].pvalue >=0.05:
      count=count+1
      #print(pvalues[i].pvalue) 
      pvalues_average=pvalues_average+pvalues[i].pvalue 
  pvalues_average=pvalues_average/length
  #pvalues_average=pvalues_average[0]
  #atleast threashold% of columns pass the hypothesis then accept the sample else reject 
  threshold=0.5
  Len_Actualdataset=len(input_dataframe.index)
  Len_Sampleddataset=len(subsample.index)
  Sampling_Error=1/math.sqrt(Len_Sampleddataset) * 100
  print ("Volume of actual dataset = ",Len_Actualdataset)
  print ("Volume of Sampled dataset =",Len_Sampleddataset) 
  print('Sampling Error for actual_size={} sample_size={} is {:.3f}% '.format(Len_Actualdataset,Len_Sampleddataset,Sampling_Error))
    
  if count> threshold*len(input_dataframe.columns):
    print ("Random Sample accepted-via KS and Chi_square Null hypothesis")
    
    Len_Actualdataset= "%s" % str(Len_Actualdataset)
    Len_Sampleddataset= "%s" % str(Len_Sampleddataset)
    Sampling_Error= "%s" %  str(Sampling_Error)  
    
    
  else:
    print ("Random Sample rejected")
    
  #print(pvalues) 
  #P is the probability that there is no significant difference between sample and actual (Null hypothesis)
  print('Average p-value between sample and actual(the bigger the p-value the closer the two datasets are.)= ',pvalues_average)
  
  pvalues_average="%s" % str(pvalues_average)

    
  return(sample_size,pvalues_average,subsample)

# %%
# DBTITLE 1,Sampling- Systematic Sampling
def SystematicSampling(input_dataframe):
  df_fin=input_dataframe
  import pandas as pd
  import numpy as np
  import math
  
  #Imputation to avoid mixup in chi2 test with categorical values frequencies
  input_dataframe=impute(input_dataframe)
  
  #Getting ideal sample size by Solven's Formula assuming Confidence Level=95% i.e n = N / (1 + Ne²)
  total_records=len(df_fin.index)
  sample_size= round(total_records / (1 + total_records* (1-0.95)*(1-0.95)))

  #For chi2 test the two conditions are-
  #     1.The distinct unique values in the original v/s sample must be same. So force fill missing values with frequency=0 (absent in sample w.r.t. original)
  #     2.The sum of frequencies in both original & sample must be same. So scale up the sample frequency to match the sum of frequencies of original exactly- sample_count=sample_count*input_dataframe_count_sum/sample_count_sum. For the sum of frequencies to match exactly the scale up factor should be a whole number or this ratio of input_dataframe_count_sum/sample_count_sum should be a whole number. i.e. input_dataframe_count_sum should be a multiple of sample_count_sum (or sample_size)  
  sample_size=int(total_records/math.floor(total_records/sample_size))
 
  subsample = df_fin.loc[df_fin['Index'] % (round(total_records/sample_size)) ==0]
  
  #Hypothesis test to see if sample is accepted
  from scipy import stats
  pvalues = list()
  for col in subsample.columns:
    if (subsample[col].dtype) in ["int32","int64","float","float64"]: 
      # Numeric variable. Using Kolmogorov-Smirnov test
      pvalues.append(stats.ks_2samp(subsample[col],input_dataframe[col]))
        
    else:
      # Categorical variable. Using Pearson's Chi-square test
      from scipy import stats
      import pandas as pd
      #For chi2 test the two conditions are-
      # #     1.The distinct unique values in the original v/s sample must be same. So force fill missing values with frequency=0 (absent in sample w.r.t. original)
      sample_count=pd.DataFrame(subsample[col].value_counts())
      input_dataframe_count=pd.DataFrame(input_dataframe[col].value_counts())
      absent=input_dataframe_count-sample_count
      absent2=absent.loc[absent[col].isnull()]
      absent2[col]=0
      sample_count_final=pd.concat([absent2,sample_count]) 

      #     2.The sum of frequencies in both original & sample must be same. So scale up the sample frequency to match the sum of frequencies of original exactly- sample_count=sample_count*input_dataframe_count_sum/sample_count_sum. For the sum of frequencies to match exactly the scale up factor should be a whole number or this ratio of input_dataframe_count_sum/sample_count_sum should be a whole number. i.e. input_dataframe_count_sum should be a multiple of sample_count_sum (or sample_size)  
      sample_count_final_sum=sample_count_final.sum()
      input_dataframe_count_sum=input_dataframe_count.sum()
      sample_count_final=sample_count_final*input_dataframe_count_sum/sample_count_final_sum
      sample_count_final = sample_count_final.apply(np.ceil).astype(int)
      pvalues.append(stats.chisquare(sample_count_final.astype(int), input_dataframe_count.astype(int)))
  
  count=0
  length =  len(subsample.columns)
  pvalues_average=0
  for i in range(length): 
    if pvalues[i].pvalue >=0.05:
      count=count+1
      #print(pvalues[i].pvalue) 
      pvalues_average=pvalues_average+pvalues[i].pvalue 
  pvalues_average=pvalues_average/length
  #pvalues_average=pvalues_average[0]
  #atleast threashold% of columns pass the hypothesis then accept the sample else reject 
  threshold=0.5
  Len_Actualdataset=len(input_dataframe.index)
  Len_Sampleddataset=len(subsample.index)
  Sampling_Error=1/math.sqrt(Len_Sampleddataset) * 100
  print ("Volume of actual dataset = ",Len_Actualdataset)
  print ("Volume of Sampled dataset =",Len_Sampleddataset) 
  print('Sampling Error for actual_size={} sample_size={} is {:.3f}% '.format(Len_Actualdataset,Len_Sampleddataset,Sampling_Error))
    
  if count> threshold*len(input_dataframe.columns):
    print ("Systematic Sample accepted-via KS and Chi_square Null hypothesis")
    
    Len_Actualdataset= "%s" % str(Len_Actualdataset)
    Len_Sampleddataset= "%s" % str(Len_Sampleddataset)
    Sampling_Error= "%s" %  str(Sampling_Error)  
    
    
  else:
    print ("Systematic Sample rejected")
    
  #print(pvalues) 
  #P is the probability that there is no significant difference between sample and actual (Null hypothesis)
  print('Average p-value between sample and actual(the bigger the p-value the closer the two datasets are.)= ',pvalues_average)
  
  pvalues_average="%s" % str(pvalues_average)

    
  return(sample_size,pvalues_average,subsample)
 
    

# %%
# DBTITLE 1,Sampling- Stratified Sampling
def StratifiedSampling(input_dataframe_orig):
  
  #Imputation to avoid mixup in chi2 test with categorical values frequencies
  input_dataframe_orig=impute(input_dataframe_orig)
  #Deep=True (the default), a new object is produced with a copy of the calling object’s data and indices. Changes to the copy’s data or indices will not reflect the original object.
  df_fin=input_dataframe_orig.copy(deep=True)
  input_dataframe = input_dataframe_orig.copy(deep=True)
  
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
    
  #Remove Columns which are not present in input_dataframe
  df_sample.drop(['Cluster'], axis=1, inplace=True)
  subsample=df_sample

  

  #Hypothesis test to see if sample is accepted
  from scipy import stats
  pvalues = list()
  for col in subsample.columns:
    if (subsample[col].dtype) in ["int32","int64","float","float64"]: 
      # Numeric variable. Using Kolmogorov-Smirnov test
      pvalues.append(stats.ks_2samp(subsample[col],df_fin[col]))
        
    else:
      # Categorical variable. Using Pearson's Chi-square test
      from scipy import stats
      import pandas as pd
      #For chi2 test the two conditions are-
      # #     1.The distinct unique values in the original v/s sample must be same. So force fill missing values with frequency=0 (absent in sample w.r.t. original)
      sample_count=pd.DataFrame(subsample[col].value_counts())
      input_dataframe_count=pd.DataFrame(input_dataframe[col].value_counts())
      absent=input_dataframe_count-sample_count
      absent2=absent.loc[absent[col].isnull()]
      absent2[col]=0
      sample_count_final=pd.concat([absent2,sample_count]) 

      #     2.The sum of frequencies in both original & sample must be same. So scale up the sample frequency to match the sum of frequencies of original exactly- sample_count=sample_count*input_dataframe_count_sum/sample_count_sum. For the sum of frequencies to match exactly the scale up factor should be a whole number or this ratio of input_dataframe_count_sum/sample_count_sum should be a whole number. i.e. input_dataframe_count_sum should be a multiple of sample_count_sum (or sample_size)  
      sample_count_final_sum=sample_count_final.sum()
      input_dataframe_count_sum=input_dataframe_count.sum()
      sample_count_final=sample_count_final*input_dataframe_count_sum/sample_count_final_sum
      sample_count_final = sample_count_final.apply(np.ceil).astype(int)
      pvalues.append(stats.chisquare(sample_count_final.astype(int), input_dataframe_count.astype(int)))
  
  count=0
  length =  len(subsample.columns)
  pvalues_average=0
  for i in range(length): 
    if pvalues[i].pvalue >=0.05:
      count=count+1
      #print(pvalues[i].pvalue) 
      pvalues_average=pvalues_average+pvalues[i].pvalue 
  pvalues_average=pvalues_average/length
  #pvalues_average=pvalues_average[0]
  #atleast threashold% of columns pass the hypothesis then accept the sample else reject 
  threshold=0.5
  Len_Actualdataset=len(input_dataframe.index)
  Len_Sampleddataset=len(subsample.index)
  Sampling_Error=1/math.sqrt(Len_Sampleddataset) * 100
  print ("Volume of actual dataset = ",Len_Actualdataset)
  print ("Volume of Sampled dataset =",Len_Sampleddataset) 
  print('Sampling Error for actual_size={} sample_size={} is {:.3f}% '.format(Len_Actualdataset,Len_Sampleddataset,Sampling_Error))
    
  if count> threshold*len(input_dataframe.columns):
    print ("Stratified Sample accepted-via KS and Chi_square Null hypothesis")
    
    Len_Actualdataset= "%s" % str(Len_Actualdataset)
    Len_Sampleddataset= "%s" % str(Len_Sampleddataset)
    Sampling_Error= "%s" %  str(Sampling_Error)  
    
    
  else:
    print ("Stratified Sample rejected")
    
  #print(pvalues) 
  #P is the probability that there is no significant difference between sample and actual (Null hypothesis)
  print('Average p-value between sample and actual(the bigger the p-value the closer the two datasets are.)= ',pvalues_average)
  
  pvalues_average="%s" % str(pvalues_average)


  #Getting back Sample in original data form
  uniqueIndex = list(df_sample['Index'].unique())
  subsample_orig= input_dataframe_orig[input_dataframe_orig.Index.isin(uniqueIndex)]
    
  return(sample_size,pvalues_average,subsample_orig)


# %%
# DBTITLE 1,Sampling- Cluster Sampling (Oversampling SMOTE)
def ClusterSampling_Oversampling(input_dataframe_orig,cluster_col):
  #SMOTE only works for continuous features. So, what to do if you have mixed (categorical and continuous) features? In this case, we have another variation of SMOTE called SMOTE-NC (Nominal and Continuous)
  #Import the SMOTE-NC
  from imblearn.over_sampling import SMOTENC

  #Imputation to avoid mixup in chi2 test with categorical values frequencies
  input_dataframe_orig=impute(input_dataframe_orig)
  #Deep=True (the default), a new object is produced with a copy of the calling object’s data and indices. Changes to the copy’s data or indices will not reflect the original object.
  df_fin=input_dataframe_orig.copy(deep=True)
  input_dataframe = input_dataframe_orig.copy(deep=True)
  
  df_fin=input_dataframe
  df_fin['Y']=df_fin[cluster_col]
  print('Length of actual input data=', len(input_dataframe.index))
  
  from sklearn.preprocessing import LabelEncoder
  from collections import Counter
  import pandas as pd
  import numpy as np
  import math
  import imblearn
  
  print("Cluster sampling with SMOTE starting")
  
  
  # summarize distribution
  print('Data Summary before sampling')
  #y=df_fin['Y']
 
  y=df_fin['Y']
  counter = Counter(y)
  Classes = []
  for k,v in counter.items():
    Classes.append(k)
    per = v / len(y) * 100
    #print("k=",k, "v=",v,"per=",per)
    print('Class=',k,', n=%d (%.3f%%)' % (v, per))
  ### plot the distribution
  ##pyplot.bar(counter.keys(), counter.values())
  ##pyplot.show()
  #print("Classes:",Classes)
  
  #SMOTE 

  from imblearn.over_sampling import SMOTE
  # split into input and output elements
  y = df_fin.pop('Y')
  x = df_fin

  # transform the dataset
  #For SMOTE-NC we need to pinpoint the column position where is the categorical features are. If you have more than one categorical columns, just input all the columns position
  # dtype='o' means categorical/object type
  allCols = [col for col in input_dataframe.columns]
  catCols = [col for col in input_dataframe.columns if input_dataframe[col].dtype=="O"] 
  # print(allCols)
  # print(catCols)

  if not catCols: # If No Categorical columns then use simple SMOTE
    oversample = SMOTE()
    x, y = oversample.fit_resample(x, y)
      

  else:
    CatIndices = [allCols.index(i) for i in catCols]
    #print("The Match indices list is : ",CatIndices)
    smotenc = SMOTENC(CatIndices,random_state = 101)
    x, y = smotenc.fit_resample(x, y)

  # summarize distribution
  counter = Counter(y)
  for k,v in counter.items():
      per = v / len(y) * 100
      print('Class=',k,', n=%d (%.3f%%)' % (v, per))
  
  df_fin=x
  df_fin['Y']=y
  df_fin.cluster_col=y
  df_fin.head()                                                           
  #df_fin.count()
  #sample_size,pvalues_average,subsample=StratifiedSampling(df_fin)
  subsample=df_fin
  
  #Hypothesis test to see if sample is accepted
  subsample.drop(['Y'], axis=1, inplace=True)


  from scipy import stats
  pvalues = list()
  for col in subsample.columns:
    if (subsample[col].dtype) in ["int32","int64","float","float64"]: 
      # Numeric variable. Using Kolmogorov-Smirnov test
      pvalues.append(stats.ks_2samp(subsample[col],input_dataframe[col]))
        
    else:
      # Categorical variable. Using Pearson's Chi-square test
      from scipy import stats
      import pandas as pd
      #For chi2 test the two conditions are-
      # #     1.The distinct unique values in the original v/s sample must be same. So force fill missing values with frequency=0 (absent in sample w.r.t. original)
      
      #The sample is oversampled, compare only the input_df_original fed worth sample with the input_df_original
      subsample_dummy= subsample.copy(deep=True)
      input_df_len=len(input_dataframe.index)
      subsample_dummy=subsample_dummy.head(input_df_len)


      sample_count=pd.DataFrame(subsample_dummy[col].value_counts())
      input_dataframe_count=pd.DataFrame(input_dataframe[col].value_counts())
      absent=input_dataframe_count-sample_count
      absent2=absent.loc[absent[col].isnull()]
      absent2[col]=0
      sample_count_final=pd.concat([absent2,sample_count]) 

      #     2.The sum of frequencies in both original & sample must be same. So scale up the sample frequency to match the sum of frequencies of original exactly- sample_count=sample_count*input_dataframe_count_sum/sample_count_sum. For the sum of frequencies to match exactly the scale up factor should be a whole number or this ratio of input_dataframe_count_sum/sample_count_sum should be a whole number. i.e. input_dataframe_count_sum should be a multiple of sample_count_sum (or sample_size)  
      sample_count_final_sum=sample_count_final.sum()
      input_dataframe_count_sum=input_dataframe_count.sum()
      sample_count_final=sample_count_final*input_dataframe_count_sum/sample_count_final_sum
      sample_count_final = sample_count_final.apply(np.ceil).astype(int)
      pvalues.append(stats.chisquare(sample_count_final.astype(int), input_dataframe_count.astype(int)))
  
  count=0
  length =  len(subsample.columns)
  pvalues_average=0
  for i in range(length): 
    if pvalues[i].pvalue >=0.05:
      count=count+1
      #print(pvalues[i].pvalue) 
      pvalues_average=pvalues_average+pvalues[i].pvalue 
  pvalues_average=pvalues_average/length
  #pvalues_average=pvalues_average[0]
  #atleast threashold% of columns pass the hypothesis then accept the sample else reject 
  threshold=0.5
  Len_Actualdataset=len(input_dataframe.index)
  Len_Sampleddataset=len(subsample.index)
  Sampling_Error=1/math.sqrt(Len_Sampleddataset) * 100
  print ("Volume of actual dataset = ",Len_Actualdataset)
  print ("Volume of Sampled dataset =",Len_Sampleddataset) 
  print('Sampling Error for actual_size={} sample_size={} is {:.3f}% '.format(Len_Actualdataset,Len_Sampleddataset,Sampling_Error))
    
  if count> threshold*len(input_dataframe.columns):
    print ("ClusterSampling_Oversampling Sample accepted-via KS and Chi_square Null hypothesis")
    
    Len_Actualdataset= "%s" % str(Len_Actualdataset)
    Len_Sampleddataset= "%s" % str(Len_Sampleddataset)
    Sampling_Error= "%s" %  str(Sampling_Error)  
    
    
  else:
    print ("ClusterSampling_Oversampling Sample rejected")
    
  #print(pvalues) 
  #P is the probability that there is no significant difference between sample and actual (Null hypothesis)
  print('Average p-value between sample and actual(the bigger the p-value the closer the two datasets are.)= ',pvalues_average)
  
  pvalues_average="%s" % str(pvalues_average)

  
  return(Len_Sampleddataset,pvalues_average,subsample)

 

# %%
# DBTITLE 1,Best Sampling Estimator
def BrewDataSampling(input_dataframe,cluster_col):
  import pandas as pd
  subsample_StratifiedSampling = pd.DataFrame()
  subsample_RandomSampling = pd.DataFrame()
  subsample_SystematicSampling = pd.DataFrame()
  subsample_ClusterSampling_Oversampling = pd.DataFrame()

  input_dataframe_StratifiedSampling = input_dataframe
  input_dataframe_RandomSampling = input_dataframe
  input_dataframe_SystematicSampling = input_dataframe
  input_dataframe_ClusterSampling_Oversampling = input_dataframe
  
  print("\n","Stratified Sampling")
  sample_size_StratifiedSampling,pvalue_StratifiedSampling,subsample_StratifiedSampling= StratifiedSampling(input_dataframe_StratifiedSampling)
  
  
  print("\n","Random Sampling")
  sample_size_RandomSampling,pvalue_RandomSampling,subsample_RandomSampling= RandomSampling(input_dataframe_RandomSampling)
  
  
  print("\n","Systematic Sampling")
  sample_size_SystematicSampling,pvalue_SystematicSampling,subsample_SystematicSampling= SystematicSampling(input_dataframe_SystematicSampling)
  
  if cluster_col=="NULL":
    print("\n","No Cluster Sampling")
  else:
    print("\n","Cluster Sampling")
    sample_size_ClusterSampling_Oversampling,pvalue_ClusterSampling_Oversampling,subsample_ClusterSampling_Oversampling= ClusterSampling_Oversampling(input_dataframe_ClusterSampling_Oversampling,cluster_col)
  
  Dic = {}
  Dic.update({"StratifiedSampling":pvalue_StratifiedSampling})
  Dic.update({"RandomSampling":pvalue_RandomSampling})
  Dic.update({"SystematicSampling":pvalue_SystematicSampling})
  if cluster_col!="NULL":
    Dic.update({"ClusterSampling":pvalue_ClusterSampling_Oversampling})
  Keymax = max(Dic, key=Dic.get) 
  if Keymax=="StratifiedSampling":
    subsample_final=subsample_StratifiedSampling
  elif Keymax=="RandomSampling":
    subsample_final=subsample_RandomSampling
  elif Keymax=="SystematicSampling":
    subsample_final=subsample_SystematicSampling
  elif Keymax=="ClusterSampling":
    subsample_final=subsample_ClusterSampling_Oversampling
  print("\n","Best Suggested Sample is - ",Keymax)
  
  Keymax="%s" % Keymax

  
  return(subsample_final,subsample_StratifiedSampling,subsample_RandomSampling,subsample_SystematicSampling,subsample_ClusterSampling_Oversampling)





# %%
#DBTITLE 1, Feature Selection
def BrewFeatureSelection(input_dataframe,label_col,Y_discrete):
  import pandas as pd
  import numpy as np
  
  columns_list = input_dataframe.columns
  column = 'Index'
  if column in columns_list:
    input_dataframe.drop(['Index'], axis=1, inplace=True)
  
  #Doesn't accept nulls so impute
  input_dataframe=impute(input_dataframe)
  
  
  #Feature imp accept only numeric columns hence label encode
  allCols = [col for col in input_dataframe.columns]
  catCols = [col for col in input_dataframe.columns if input_dataframe[col].dtype=="O"] 
  # print(allCols)
  # print(catCols)
  if catCols: # If Categorical columns present (list not empty) then do label encode
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in input_dataframe.columns:
      if (input_dataframe[col].dtype) not in ["int32","int64","float","float64"]:
        input_dataframe[col]=le.fit_transform(input_dataframe[col])
  

  
  #X features,Y target or label
  Y = input_dataframe.pop(label_col)
  X = input_dataframe
  
  #If Target variable is continuous DO use this. Use these only in case of continuous/numerically growing Target variable.
  if (Y_discrete)  in ["Continuous"]:
    print("\n","*Fetures Selection based on Decision tree Regressor")
    #Decision tree for feature importance on a regression problem
    from sklearn.tree import DecisionTreeRegressor
    from matplotlib import pyplot
    # define the model
    model = DecisionTreeRegressor()
    # fit the model
    model.fit(X, Y)
    # get importance
    importances = model.feature_importances_
    feat_importances = pd.Series(importances, X.columns[0:len(X.columns)])
    sorted_feat_importances = feat_importances.sort_values(ascending= False)
    print("Bagged decision trees like Random Forest and Decision Trees can be used to estimate the importance of features. Larger score the more important the attribute.")
    print("Feature Importance values:","\n",sorted_feat_importances)
  
    #Fetures Selection based on Correlation matrix
    print("\n","*Fetures Selection based on Correlation matrix")
    Correlation_Thresh=0.9
    corr_matrix = X.corr()
    print(corr_matrix)
    #Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    print(upper)  
    #Find index of feature columns with correlation greater than 0.9
    to_drop = [column for column in upper.columns if any(upper[column] > Correlation_Thresh)]
    print("Threshold o Correlation=",Correlation_Thresh)
    print("Highly Correlated columns- drop one of them. Columns to drop=",to_drop) 
    
    #Fetures Selection based on Variance Threshold. 
    print("\n","*Fetures Selection based on Variance Threshold")
    from sklearn.feature_selection import VarianceThreshold
    Variance_threshold=0 #Default zero variance
    v_threshold = VarianceThreshold(threshold=Variance_threshold)
    v_threshold.fit(X) 
    v_threshold.get_support()
    BoolAccepted = pd.Series(v_threshold.get_support(), X.columns[0:len(X.columns)])
    print("Remove all features which variance doesn’t meet some threshold. We assume that features with a higher variance may contain more useful information, but note that we are not taking the relationship between feature variables or feature and target variables into account. True means that the variable does not have variance=",Variance_threshold)
    print(BoolAccepted)
  
  #If Target variable is continuous DO NOT use this. Use these only in case of boolean/class/multiclass Target variable.
  if (Y_discrete)  in ["Categorical"]:
    print("\n","*Fetures Selection based on Decision tree Regressor")
    #Decision tree for feature importance on a regression problem
    from sklearn.tree import DecisionTreeClassifier
    from matplotlib import pyplot
    # define the model
    model = DecisionTreeClassifier()
    # fit the model
    model.fit(X, Y)
    # get importance
    importances = model.feature_importances_
    feat_importances = pd.Series(importances, X.columns[0:len(X.columns)])
    sorted_feat_importances = feat_importances.sort_values(ascending= False)
    print("Bagged decision trees like Random Forest and Decision Trees can be used to estimate the importance of features. Larger score the more important the attribute.")
    print("Feature Importance values:","\n",sorted_feat_importances)
  
    #Fetures Selection based on Correlation matrix
    print("\n","*Fetures Selection based on Correlation matrix")
    Correlation_Thresh=0.9
    corr_matrix = X.corr()
    print(corr_matrix)
    #Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    print(upper)  
    #Find index of feature columns with correlation greater than 0.9
    to_drop = [column for column in upper.columns if any(upper[column] > Correlation_Thresh)]
    print("Threshold o Correlation=",Correlation_Thresh)
    print("Highly Correlated columns- drop one of them. Columns to drop=",to_drop) 
    
    #Fetures Selection based on Variance Threshold. 
    print("\n","*Fetures Selection based on Variance Threshold")
    from sklearn.feature_selection import VarianceThreshold
    Variance_threshold=0 #Default zero variance
    v_threshold = VarianceThreshold(threshold=Variance_threshold)
    v_threshold.fit(X) 
    v_threshold.get_support()
    BoolAccepted = pd.Series(v_threshold.get_support(), X.columns[0:len(X.columns)])
    print("Remove all features which variance doesn’t meet some threshold. We assume that features with a higher variance may contain more useful information, but note that we are not taking the relationship between feature variables or feature and target variables into account. True means that the variable does not have variance=",Variance_threshold)
    print(BoolAccepted)
  
    #Feature Imp based on Information gain & Entropy 
    print("\n","*Feature Imp based on Information Gain & Entropy")
    from sklearn.feature_selection import mutual_info_classif
    importances = mutual_info_classif(X, Y)
    feat_importances = pd.Series(importances, X.columns[0:len(X.columns)])
    sorted_feat_importances = feat_importances.sort_values(ascending= False)
    print("Information Gain indicates how much information a particular variable or feature gives us about the final outcome. Order of feture Importance. Ascending-Most important first.","\n",sorted_feat_importances)
	





# %%
# DBTITLE 1,AnomalyDetection
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.font_manager
from pyod.models.abod import ABOD
from pyod.models.knn import KNN
from pyod.utils.data import generate_data, get_outliers_inliers
from pyod.models.cblof import CBLOF
#from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from sklearn.preprocessing import MinMaxScaler
import warnings
from io import BytesIO
from pyspark.sql.functions import base64
from pyspark.sql.functions import unbase64
warnings.filterwarnings("ignore")
                                                                                                                                                                                                                                                                                                            
def BrewAnomalyDetection(df,target_variable,variables_to_analyze,outliers_fraction):
    #Doesn't accept nulls so impute, will impute only if nulls present
    df=impute(df)

    #Feature imp accept only numeric columns hence label encode
    allCols = [col for col in df.columns]
    catCols = [col for col in df.columns if df[col].dtype=="O"] 
    # print(allCols)
    # print(catCols)
    if catCols: # If Categorical columns present (list not empty) then do label encode
      from sklearn.preprocessing import LabelEncoder
      le = LabelEncoder()
      for col in df.columns:
        if (df[col].dtype) not in ["int32","int64","float","float64"]:
          df[col]=le.fit_transform(df[col])


    X1 = df[variables_to_analyze].values.reshape(-1,1)
    X2 = df[target_variable].values.reshape(-1,1)
    X = np.concatenate((X1,X2),axis=1)
    random_state = np.random.RandomState(42)
    #print(X)    # variables_to_analyze, target_variable two cols put together

    classifiers = {
        'Angle-based Outlier Detector (ABOD)': ABOD(contamination=outliers_fraction),
        'Cluster-based Local Outlier Factor (CBLOF)':CBLOF(contamination=outliers_fraction,check_estimator=False, random_state=random_state),
        #'Feature Bagging':FeatureBagging(LOF(n_neighbors=35),contamination=outliers_fraction,check_estimator=False,random_state=random_state),
        'Histogram-base Outlier Detection (HBOS)': HBOS(contamination=outliers_fraction),
        'Isolation Forest': IForest(contamination=outliers_fraction,random_state=random_state),
        'K Nearest Neighbors (KNN)': KNN(contamination=outliers_fraction),
        'Average KNN': KNN(method='mean',contamination=outliers_fraction)
    }
    xx , yy = np.meshgrid(np.linspace(0,1 , 200), np.linspace(0, 1, 200))
    for i, (clf_name, clf) in enumerate(classifiers.items()):
      #print(i, (clf_name, clf))
      clf.fit(X)
      # predict raw anomaly score
      scores_pred = clf.decision_function(X) * -1
      # prediction of a datapoint category outlier(1) or inlier(0)
      y_pred = clf.predict(X)
      n_inliers = len(y_pred) - np.count_nonzero(y_pred)
      n_outliers = np.count_nonzero(y_pred == 1)

      # copy of dataframe : dfy with Outlier(1), NotOutlier(0), AnomalyScore columns 
      dfx = df
      dfy = df

      clf_name_string="%s" % str(clf_name)
      outlier_clf="outlier_"+clf_name_string
      outlier_score_clf="outlier_score_"+clf_name_string
      dfx[outlier_clf] = y_pred.tolist()
      dfy[outlier_clf] = y_pred.tolist()
      dfy[outlier_score_clf] = scores_pred.tolist()
      dfy[target_variable] = df[target_variable]

      #display(dfy)


      n_outliers="%s" % str(n_outliers)
      n_inliers="%s" % str(n_inliers)


      # IX1 - inlier feature 1,  IX2 - inlier feature 2
      IX1 =  np.array(dfx[variables_to_analyze][dfx[outlier_clf] == 0]).reshape(-1,1)
      IX2 =  np.array(dfx[target_variable][dfx[outlier_clf] == 0]).reshape(-1,1)
      # OX1 - outlier feature 1, OX2 - outlier feature 2
      OX1 =  dfx[variables_to_analyze][dfx[outlier_clf] == 1].values.reshape(-1,1)
      OX2 =  dfx[target_variable][dfx[outlier_clf] == 1].values.reshape(-1,1) 
      print('OUTLIERS : ',n_outliers,', INLIERS : ',n_inliers, clf_name)


      # threshold value to consider a datapoint inlier or outlier
      threshold = stats.scoreatpercentile(scores_pred,100 * outliers_fraction)
      # decision function calculates the raw anomaly score for every point
      Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
      Z = Z.reshape(xx.shape)
      plt.figure(figsize=(10, 10))
      # fill blue map colormap from minimum anomaly score to threshold value
      plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7),cmap=plt.cm.Blues_r)
      # draw red contour line where anomaly score is equal to thresold
      a = plt.contour(xx, yy, Z, levels=[threshold],linewidths=2, colors='red')
      # fill orange contour lines where range of anomaly score is from threshold to maximum anomaly score
      plt.contourf(xx, yy, Z, levels=[threshold, Z.max()],colors='orange')
      b = plt.scatter(IX1,IX2, c='white',s=20, edgecolor='k')
      c = plt.scatter(OX1,OX2, c='black',s=20, edgecolor='k')
      plt.axis('tight')  
      # loc=2 is used for the top left corner 
      plt.legend(
          [a.collections[0], b,c],
          ['learned decision function', 'inliers','outliers'],
          prop=matplotlib.font_manager.FontProperties(size=20),
          loc=2)
      plt.xlim((0, 1))
      plt.ylim((0, 1))
      plt.title(clf_name)
      #tmpfile = BytesIO()
      #plt.savefig(tmpfile, format='png')
      #plt.savefig('/dbfs/FileStore/figure.png')
      plt.show() 																		
    return dfy







# ############################################# CALLING FUNCTIONS #############################################################################################################################################################################################################
# # %%
# # Function:Acquisition_DataTypeConversion
# filepath=r"C:\Users\srde\Downloads\Titanic.csv"
# col_dtype_dict={
#                     'PassengerId':'int'
#                     ,'Survived':'int'
#                     ,'Pclass':'int'
#                     ,'Name':'object'
#                     ,'Sex':'object'
#                     ,'Age':'float'
#                     ,'SibSp':'int'
#                     ,'Parch':'int'
#                     ,'Ticket':'object'
#                     ,'Fare':'float'
#                     ,'Cabin':'object'
#                     ,'Embarked':'object'
#                 }
# input_dataframe=Acquisition_DataTypeConversion(filepath,col_dtype_dict)

# # %%
# input_dataframe

# # %%
# # # Function:Exploratory Data Analysis
# p=BrewDataProfiler(input_dataframe)
# print(p)
# displayHTML(p)


# # %%
# # # Function:Data Cleanser
# Cleansed_data=BrewDataCleanser(input_dataframe)
# Cleansed_data.head()





#  %%
#  Function: Sampling
# sample_size,pvalues_average,subsample=RandomSampling(input_dataframe)
# sample_size,pvalues_average,subsample=SystematicSampling(input_dataframe)
# sample_size,pvalues_average,subsample=StratifiedSampling(input_dataframe)
# sample_size,pvalues_average,subsample=ClusterSampling_Oversampling(input_dataframe,'Sex')

#subsample_final,subsample_StratifiedSampling,subsample_RandomSampling,subsample_SystematicSampling,subsample_ClusterSampling_Oversampling=BrewDataSampling(input_dataframe,'Sex')









##################################################################AutoML

# %%
# DBTITLE 1, AutoML/Responsible AI
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from contextlib import contextmanager
import sys, os

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

def BrewTrainTestSplit(input_dataframe,label_col,splitratio):
  from sklearn.model_selection import train_test_split
  y_df = input_dataframe.pop(label_col)
  x_df = input_dataframe
  x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=splitratio, random_state=223)
  return x_train, x_test, y_train, y_test


# url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv'
# dataframe = pd.read_csv(url, header=None)
# label = 60
# x_train, x_test, y_train, y_test = TrainTestSplit(dataframe,label,0.2)





# %%
def BrewAutoML_Classifier(x_train,y_train,x_test,y_test):
  with suppress_stdout():
    
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import RepeatedStratifiedKFold
    from tpot import TPOTClassifier
    # split into input and output elements
    x_train_dummy = x_train.copy(deep=True)
    y_train_dummy = y_train.copy(deep=True)
    x_test_dummy  = x_test.copy(deep=True)
    y_test_dummy  = y_test.copy(deep=True)

    # define model evaluation
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

    # define search
    model = TPOTClassifier(generations=2, population_size=50, cv=cv, scoring='accuracy', verbosity=0, random_state=1, n_jobs=-1)
    # perform the search
    model.fit(x_train_dummy, y_train_dummy)
    y_predict=model.predict(x_test_dummy)


    from sklearn.metrics import confusion_matrix 
    from sklearn.metrics import accuracy_score 
    from sklearn.metrics import classification_report
    x_test_dummy['y_predict'] = y_predict
    x_test_dummy['y_actual'] = y_test_dummy
    y_actual = y_test_dummy.tolist()
    results = confusion_matrix(y_actual, y_predict) 
  print ('Confusion Matrix :')
  print(results) 
  print ('Accuracy Score :',accuracy_score(y_actual, y_predict))
  return(model,x_test_dummy)


# dataframe = Cleansed_data
# label = 'Survived'
# x_train, x_test, y_train, y_test = TrainTestSplit(dataframe,label,0.2)
# model=automl_classifier(x_train,y_train)

# # %%
# model


# %%

def BrewAutoML_Regressor(x_train,y_train,x_test,y_test):
  with suppress_stdout():
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import RepeatedKFold
    from tpot import TPOTRegressor
    # split into input and output elements
    x_train_dummy = x_train.copy(deep=True)
    y_train_dummy = y_train.copy(deep=True)
    x_test_dummy  = x_test.copy(deep=True)
    y_test_dummy  = y_test.copy(deep=True)

    #Label Encoder Doesn't accept nulls so impute
    x_train_dummy = impute(x_train_dummy)
    x_test_dummy  = impute(x_test_dummy)

    #Feature imp accept only numeric columns hence label encode
    allCols = [col for col in x_train_dummy.columns]
    catCols = [col for col in x_train_dummy.columns if x_train_dummy[col].dtype=="O"] 
    # print(allCols)
    # print(catCols)
    if catCols: # If Categorical columns present (list not empty) then do label encode
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        for col in x_train_dummy.columns:
            if (x_train_dummy[col].dtype) not in ["int32","int64","float","float64"]:
                le.fit(x_train_dummy[col])
                x_train_dummy[col]=le.transform(x_train_dummy[col])
                x_test_dummy[col]=le.transform(x_test_dummy[col])


    # define model evaluation
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

    # define search
    model = TPOTRegressor(generations=5, population_size=50, scoring='neg_mean_absolute_error', cv=cv, verbosity=2, random_state=1, n_jobs=-1)
    # perform the search
    model.fit(x_train_dummy, y_train_dummy)
    y_predict=model.predict(x_test_dummy)



    from sklearn.metrics import confusion_matrix 
    from sklearn.metrics import accuracy_score 
    from sklearn.metrics import classification_report
    x_test_dummy['y_predict'] = y_predict
    x_test_dummy['y_actual'] = y_test_dummy
    y_actual = y_test_dummy.tolist()

    sum_actuals = sum_errors = 0
    for actual_val, predict_val in zip(y_actual, y_predict):
        abs_error = abs(actual_val) - abs(predict_val)
        if abs_error < 0:
            abs_error = abs_error * -1

        sum_errors = sum_errors + abs_error
        sum_actuals = sum_actuals + actual_val
    mean_abs_percent_error = sum_errors / sum_actuals
    #print("Model Accuracy:")
    #print(1 - mean_abs_percent_error)
    Accuracy_score=1 - mean_abs_percent_error
    print('Accuracy Score :',Accuracy_score)

    return(model,x_test_dummy)


# model,x_test_dummy=BrewAutoML_Regressor(x_train,y_train,x_test,y_test)
# model



# %%
def BrewFairnessEvaluator(model,x_test,sensitivefeatures,y_test):
  import numpy as np
  from fairlearn.metrics import MetricFrame,mean_prediction,false_negative_rate
  from sklearn.metrics import accuracy_score
  x_test_dummy  = x_test.copy(deep=True)
  y_test_dummy  = y_test.copy(deep=True)


  df=x_test[sensitivefeatures]
  
  y_pred = model.predict(x_test_dummy).tolist()
  y_test2=y_test_dummy.astype(np.int32).tolist()
  
  gm = MetricFrame(metrics=accuracy_score, y_true=y_test2, y_pred=y_pred,sensitive_features=df)
  print("Overall Accuracy:",gm.overall)
  print("Group Wise Accuracy:",gm.by_group)
  return (gm.overall,gm.by_group)


# y_pred = FairnessEvaluator(model,x_test,'Sex',y_test)


  


# y_pred = FairnessEvaluator(model,x_test,'Sex',y_test)




def BrewDisparityMitigation(x_test,sensitivefeatures,y_test):
  import numpy as np
  from fairlearn.metrics import MetricFrame
  from sklearn.metrics import accuracy_score
  from fairlearn.reductions import ExponentiatedGradient, DemographicParity
  from sklearn.tree import DecisionTreeClassifier
  
  x_test_dummy  = x_test.copy(deep=True)
  y_test_dummy  = y_test.copy(deep=True)
  x_test_dummy['y_actual'] = y_test_dummy
  
  np.random.seed(0)
  sensitive=x_test_dummy[sensitivefeatures]
  sensitive2=sensitive.astype('int')
  sensitive3=sensitive2.astype('str')
  sensitive4=sensitive3.astype('category')
  x_test_dummy[sensitivefeatures]=x_test_dummy[sensitivefeatures].astype('int').astype('str')
  y_test2=y_test_dummy.astype(np.int32).tolist()
  constraint = DemographicParity()
  classifier = DecisionTreeClassifier()
  mitigator = ExponentiatedGradient(classifier, constraint)
  mitigator.fit(x_test_dummy, y_test2, sensitive_features=sensitive4)
  y_pred_mitigated = mitigator.predict(x_test_dummy)
  
  x_test_dummy['y_predict_mitigated'] = y_pred_mitigated
  
    
    
  gm = MetricFrame(metrics=accuracy_score, y_true=y_test2, y_pred=y_pred_mitigated,sensitive_features=sensitive4)
  print("Overall Accuracy:",gm.overall)
  print("Group Wise Accuracy:",gm.by_group)
  return (mitigator,x_test_dummy)

# y_pred = FairnessEvaluator(model,x_test,'Sex',y_test)
# y_pred


# # %%
# DisparityMitigation(y_pred,sex)



# %%
def BrewResponsibleAI(model,Final_df_actual_pred,x_test,sensitive_data_col,y_test):
  import numpy as np 
  overall_acc,grouped_acc=BrewFairnessEvaluator(model,x_test,sensitive_data_col,y_test)

  for index, value in grouped_acc.items():
    print(f"Cohort : {index}, Accuracy for the Cohort : {value}")

  #If on comparing the accuracies along all the cohorts, the variance of accuracy is >50% then the accuracies are biased
  #The maximum absolute difference in the series will always be the absolute difference between the minimum and the maximum element from the array.
  min_acc_ind=grouped_acc.astype(np.int32).argmin()
  max_acc_ind=grouped_acc.astype(np.int32).argmax()
  print("Minimum Accuracy amongst all the cohorts: ", grouped_acc[min_acc_ind])
  print("Maximum Accuracy amongst all the cohorts: ", grouped_acc[max_acc_ind])

  perc_diff_minmax_acc= ((grouped_acc[max_acc_ind]- grouped_acc[min_acc_ind])/grouped_acc[max_acc_ind]) * 100
  print("Percentage Difference b/w the min and max accuracies amongst all cohorts", perc_diff_minmax_acc)

  if (perc_diff_minmax_acc > 50):
    print("Triggerring Bias Mitigation")
    Mitigated_model,Mitigated_Pred=BrewDisparityMitigation(x_test,sensitive_data_col,y_test)
    return (Mitigated_model,Mitigated_Pred)
  else:
    print("Original Modelling Unbiased")
    return (model,Final_df_actual_pred)





# %%
def MagicBrewer(input_dataframe,label_col,sensitive_data_col):
  # user input: input_dataframe, label_col, sensitive_col
  # cleansed=cleansing(input_dataframe)
  # sampled=sampling(cleansed)
  # sampled_splitted=train test split(sampled)
  # autoclass(sampled_splitted)
  # model=fairness estimator(model)
  # fairness mitigator(model, threshold for bias)

  #   input_dataframe=input_dataframe
  #   label_col='Survived'
  #   sensitive_data_col='Sex'

  print("\n","******************** CLEANSING STARTING ********************")
  cleansed_df=BrewDataCleanser(input_dataframe)


  print("\n","******************** SAMPLING STARTING ********************")
  subsample_final,subsample_StratifiedSampling,subsample_RandomSampling,subsample_SystematicSampling,subsample_ClusterSampling_Oversampling=BrewDataSampling(cleansed_df,label_col)


  print("\n","******************** TRAIN-TEST SPLIT STARTING ********************")
  x_train, x_test, y_train, y_test = BrewTrainTestSplit(subsample_final,label_col,0.2)
  print("Train-Test Split completed in the ratio of Train:Test :: 80:20")

  print("\n","******************** MODELLING STARTING ********************")
  model,Final_df_actual_pred=BrewAutoML_Classifier(x_train,y_train,x_test,y_test)


  print("\n","******************** RESPONSIBLE-AI STARTING ********************")
  FinalModel,FinalPredictedData=BrewResponsibleAI(model,Final_df_actual_pred,x_test,sensitive_data_col,y_test)
  return (FinalModel,FinalPredictedData)