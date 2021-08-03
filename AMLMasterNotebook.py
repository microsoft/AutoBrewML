# Databricks notebook source
# MAGIC 
# MAGIC %md
# MAGIC <img src='https://github.com/retkowsky/images/blob/master/AzureMLservicebanniere.png?raw=true'>
# MAGIC # Auto Tune ML
# MAGIC ## Introduction:
# MAGIC 
# MAGIC Traditional machine learning model development is resource-intensive, requiring significant domain/statistical knowledge and time to produce and compare dozens of models. 
# MAGIC With automated machine learning, the time it takes to get production-ready ML models with great ease and efficiency highly accelerates. But the automated Machine learning still has miles to go in terms of data preparation and feature engineering. 
# MAGIC The Auto Tune ML framework tries to solve this problem at scale as well as simplifies the overall process for the user. It leverages the Azure Automated ML coupled with components like Data Profiler, Data Sampler, Data Cleanser, Anomaly Detector, Data Fairness Estimator which ensures quality data as a critical pre-step for building the ML model. This is powered with Telemetry, DevOps and Power BI integration, thus providing the users with a one-stop shop solution to productionize any ML model. 
# MAGIC The framework aims at ‘Democratizing’ AI all the while maintaining the vision of ‘Responsible’ AI.

# COMMAND ----------

# MAGIC %md
# MAGIC # Getting Started
# MAGIC ## Prerequisites:
# MAGIC 1. Azure Databricks
# MAGIC 2. Auto Tune Model Notebooks (Master, Trigger notebooks)
# MAGIC 3. Azure ML Services workspace
# MAGIC 4. Python cluster in Databricks with libraries installed as mentioned in step 'Prerequisite libraries'
# MAGIC 
# MAGIC If in PS-Data and Insights Team you can use the following resources handy in DEV:
# MAGIC 1. Azure ML Services workspace (subscription_id:'3ecb9b6a-cc42-4b0a-9fd1-6c08027eb201', resource_group:'psbidev' Contributor access, workspace_name:'psdatainsightsML')
# MAGIC 2. Python cluster in Databricks (AMLsrdecluster)

# COMMAND ----------

# DBTITLE 1,Prerequisite libraries
# MAGIC %pip install azureml-train-automl-runtime
# MAGIC %pip install azureml-automl-runtime
# MAGIC %pip install azureml-widgets
# MAGIC %pip install azureml-sdk[automl]
# MAGIC %pip install ipywidgets
# MAGIC %pip install pandas-profiling
# MAGIC %pip install pyod
# MAGIC %pip install azureml-sdk
# MAGIC %pip install azureml-explain-model
# MAGIC %pip install imbalanced-learn
# MAGIC %pip install pyod

# COMMAND ----------

# DBTITLE 1,Prerequisities - Version check
import warnings
warnings.filterwarnings('ignore')
#from __future__ import print_function
import scipy, numpy as np, matplotlib, pandas as pd, sklearn
from sklearn.preprocessing import LabelEncoder
import argparse
print("Versions:")
print('- scipy = {}'.format(scipy.__version__))
print('- numpy = {}'.format(np.__version__))
print('- matplotlib = {}'.format(matplotlib.__version__))
print('- pandas = {}'.format(pd.__version__))
print('- sklearn = {}'.format(sklearn.__version__))



# COMMAND ----------

# MAGIC %md
# MAGIC ## Anomaly Detection:
# MAGIC Anomaly detection aims to detect abnormal patterns deviating from the rest of the data, called anomalies or outliers. Handling Outliers and anomalies is critical to the machine learning process. Outliers can impact the results of our analysis and statistical modeling in a drastic way. Our tendency is to use straightforward methods like box plots, histograms and scatter-plots to detect outliers. But dedicated outlier detection algorithms are extremely valuable in fields which process large amounts of data and require a means to perform pattern recognition in larger datasets. The PyOD library can step in to bridge this gap, which is a comprehensive and scalable Python toolkit for detecting outlying objects in multivariate data. We will be using the algorithms within PyOD to detect and analyze the Outliers and indicate their presence in datasets.

# COMMAND ----------

# DBTITLE 1,AnomalyDetection
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
%matplotlib inline
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
outliers_fraction = 0.05
output_path = '/dbfs/FileStore/AnomalyDetection_HTML'
def AnomalyDetection(df,target_variable,variables_to_analyze,outliers_fraction,input_appname,task_type):
    import time
    from datetime import date
    today = date.today()
    ts = int(time.time())
    appname = input_appname
    appnamequotes = "'%s'" % appname
    tsquotes = "'%s'" % str(ts)
    task = "'%s'" % str(task_type)
    
    #Scale the data is required to create a explainable visualization (it will become way too stretched otherwise)
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[[target_variable,variables_to_analyze]] = scaler.fit_transform(df[[target_variable,variables_to_analyze]])
    X1 = df[variables_to_analyze].values.reshape(-1,1)
    X2 = df[target_variable].values.reshape(-1,1)
    X = np.concatenate((X1,X2),axis=1)
    random_state = np.random.RandomState(42)
    # Define seven outlier detection tools to be compared
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
        clf.fit(X)
        # predict raw anomaly score
        scores_pred = clf.decision_function(X) * -1
        # prediction of a datapoint category outlier or inlier
        y_pred = clf.predict(X)
        n_inliers = len(y_pred) - np.count_nonzero(y_pred)
        n_outliers = np.count_nonzero(y_pred == 1)
        X2
        # copy of dataframe
        dfx = df
        dfy=df
        dfx['outlier'] = y_pred.tolist()
        dfy['outlier'] = y_pred.tolist()
        dfy['scores_pred'] = scores_pred.tolist()
        dfy[target_variable] = df[target_variable]
        

        clf_name_string="%s" % str(clf_name)
        ts_string="%s" % str(ts)
        #OutputfileName="adl://<Your ADLS Name>.azuredatalakestore.net/DEV/AnomalyDetection_"+clf_name_string +".csv"
        #copydbfs = '/dbfs/FileStore/AnomalyDetection.csv'
        #dfy.to_csv(copydbfs, index=False)
        #dbutils.fs.cp ("/FileStore/AnomalyDetection.csv", OutputfileName, True) 
        n_outliers="%s" % str(n_outliers)
        n_inliers="%s" % str(n_inliers)
        rm_str3 = "Insert into TelemetryTable values (" + appnamequotes + ","+ task + ",'OUTLIERS :" + n_outliers + "  INLIERS :" + n_inliers  + "  :- " + clf_name+ "'," + tsquotes + ")"
        spark.sql(rm_str3)
        is_outlier =  dfy['outlier']==1
        Outlier_data = dfy[is_outlier]
        html_data = Outlier_data.to_html(classes='table table-striped')
        # IX1 - inlier feature 1,  IX2 - inlier feature 2
        IX1 =  np.array(dfx[variables_to_analyze][dfx['outlier'] == 0]).reshape(-1,1)
        IX2 =  np.array(dfx[target_variable][dfx['outlier'] == 0]).reshape(-1,1)
        # OX1 - outlier feature 1, OX2 - outlier feature 2
        OX1 =  dfx[variables_to_analyze][dfx['outlier'] == 1].values.reshape(-1,1)
        OX2 =  dfx[target_variable][dfx['outlier'] == 1].values.reshape(-1,1) 
        print('OUTLIERS : ',n_outliers,'INLIERS : ',n_inliers, clf_name)
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
       # encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
       # print("done2")
        #text = 'OUTLIERS : '+ str(n_outliers)+', INLIERS : '+str(n_inliers)
        #clf_text = clf_name
        #output_file = "adl://<Your ADLS Name>.azuredatalakestore.net/DEV/AnomalyDetection_chart" + clf_text + '.html'
        #html = '<h1 style="text-align: center;">{clf_text}</h1><h3 style="text-align: center;">{text}</h3><p><img style="display: block; margin-left: auto; margin-right: auto;" src="/FileStore/figure.png" alt="Plot" /></p>'
        #print(html)
        
        #print(html2)
        #html3 = html2+html_data
        #s = Template(html).safe_substitute(clf_text=clf_text)
        #t = Template(s).safe_substitute(text=text)
        #print(t)
        #dbutils.fs.put("/dbfs/FileStore/anamolydetection.html", "Contents of my file")
        #dbutils.fs.cp ("/dbfs/FileStore/anamolydetection.html", output_file, True)
        #print(html3)
        #with open(output_file,'w') as f:
        #    f.write(t)
        


# COMMAND ----------

# MAGIC %md
# MAGIC ## Sampling:
# MAGIC By Data Sampling, we can select, manipulate and analyze a representative subset of data points to identify patterns and trends in the larger dataset being examined. The dataset thus obtained is a weighted sample of the actual dataset, thus enabling a clear picture of the bigger dataset with best performance, retaining the overall data density and distribution. The following method is used to obtain samples of data from the original input data using different techniques and the best sample thus obtained is suggested to the user. The function ‘Sampling’ encompasses all the features of this as explained below.
# MAGIC <br/>
# MAGIC 1.	**Get the ideal sample size from the original input dataset using Solven’s formula**
# MAGIC <br/>n=N/((1+N^2 ) )
# MAGIC <br/>Here,
# MAGIC <br/>n=Number of Samples
# MAGIC <br/>N=Total Population
# MAGIC <br/>e=Error tolerance (level) = 1-Confidence Level in percentage (~ 95%)
# MAGIC <br/>
# MAGIC 2.	**Random Sampling**
# MAGIC <br/>
# MAGIC Pick (n) items from the whole actual dataset (N) randomly assuming every item has equal probability (1/N) of getting its place in the sample irrespective of its weightage in the actual dataset.
# MAGIC <br/>
# MAGIC 3.	**Systematic Sampling**
# MAGIC <br/>
# MAGIC This method allows to choose the sample members of a population at regular intervals. It requires the selection of a starting point for the sample and sample size that can be repeated at regular intervals. This type of sampling method has a predefined range, and hence this sampling technique is the least time-consuming.
# MAGIC Pick every kth item from the actual dataset where k = N/n 
# MAGIC <br/>
# MAGIC 4.	**Stratified Sampling**
# MAGIC <br/>
# MAGIC Clustering :- Classify input data into k clusters using K-means clustering and add an extra column to the data frame ‘Cluster’ to identify which record belongs to which cluster (0- to k-1). Get the ideal ‘k’ for a dataset using Silhouette score. The silhouette coefficient of a data measures how well data are grouped within a cluster and how far they are from other clusters. A silhouette close to 1 means the data points are in an appropriate cluster and a silhouette coefficient close to −1 implies that data is in the wrong cluster. i.e., get the scores for a range of values for k and choose the cluster k value which gives Highest Silhouette score.
# MAGIC Weighted Count :- Get the percentage count of records corresponding to each cluster in actual dataframe, create a weighted subsample of (n) records maintaining the same weighted distribution of records from each cluster. 
# MAGIC <br/>
# MAGIC 5.	**Clustered Sampling**
# MAGIC <br/>
# MAGIC If the input data is having a predefined distribution to different classes, check if the distribution is biased towards one or more classes. If yes, then apply SMOTE(Synthetic Minority Oversampling Technique) to level the distribution for each class. This approach for addressing imbalanced datasets is to oversample the minority class. This involves duplicating examples in the minority class, although these examples don’t add any new information to the model. Instead, new examples can be synthesized from the existing examples. Create a weighted subsample of (n) records maintaining the same weighted distribution of records from each cluster (after SMOTE).
# MAGIC <br/>
# MAGIC 6.	**Get the sampling error**
# MAGIC <br/>
# MAGIC The margin of error is 1/√n, where n is the size of the sample for each of the above techniques.
# MAGIC <br/>
# MAGIC 7.	**Getting the best Sample obtained**
# MAGIC <br/>
# MAGIC Using a Null Hypothesis for each column, calculate the p-value using Kolmogorov-Smirnov test (For Continuous columns) and Pearson's Chi-square test (for categorical columns). If the p-values are >=0.05 for more than a threshold number of columns (50% used here), the subsample created is accepted. P-value can be used to decide whether there is evidence of a statistical difference between the two population (Sample v/s the Original dataset) means. The smaller the p-value, the stronger the evidence is that the two populations have different means. The samples obtained above that has the highest average p-value is suggested to be the closest to the actual dataset. p-value is the probability of obtaining results at least as extreme as the observed results of a statistical hypothesis test, assuming that the null hypothesis is correct. A smaller p-value means that there is stronger evidence in favor of the alternative hypothesis.
# MAGIC <br/>

# COMMAND ----------

# DBTITLE 1,Sampling- Cluster Sampling
def ClusterSampling(input_dataframe,filepath,task_type,input_appname,cluster_col):
  #input_dataframe = pd.read_csv("/dbfs/FileStore/glass.csv", header='infer')
  #cluster_col='Y'
  #task_type='Sampling'
  #input_appname='RealEstate'
  #filepath="/dbfs/FileStore/glass.csv"
  
  df_fin=input_dataframe
  df_fin['Y']=df_fin[cluster_col]
  print('Length of actual input data=', len(input_dataframe.index))
  
  from sklearn.preprocessing import LabelEncoder
  from collections import Counter
  import pandas as pd
  import numpy as np
  import math
  import imblearn
  
  import time
  ts = int(time.time())
  appname = input_appname
  appnamequotes = "'%s'" % appname
  tsquotes = "'%s'" % str(ts)
  task = "'%s'" % str(task_type)
  print("Cluster sampling with SMOTE starting")
  rm_str = "Insert into TelemetryTable values (" + appnamequotes + ","+ task + ",'Cluster Sampling starting'," + tsquotes + ")"
  spark.sql(rm_str)
  
  # summarize distribution
  print('Data Summary before sampling')
  y=df_fin['Y']
  from sklearn.preprocessing import LabelEncoder
  le = LabelEncoder()
  df_fin['Y']=le.fit_transform(df_fin['Y'])
  counter = Counter(y)
  Classes = []
  for k,v in counter.items():
    Classes.append(k)
    per = v / len(y) * 100
    print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
  ### plot the distribution
  ##pyplot.bar(counter.keys(), counter.values())
  ##pyplot.show()
  #print("Classes:",Classes)
  
  #SMOTE 
  print('Data Summary after SMOTE')
  from imblearn.over_sampling import SMOTE
  # split into input and output elements
  y = df_fin.pop('Y')
  x = df_fin
  # transform the dataset
  oversample = SMOTE()
  x, y = oversample.fit_resample(x, y)
  # summarize distribution
  counter = Counter(y)
  for k,v in counter.items():
      per = v / len(y) * 100
      print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
  
  df_fin=x
  df_fin['Y']=y
  df_fin.head()    
  #Stratifying to get sample
  df_sample = pd.DataFrame()
  count_records_per_cluster= list()
  sample_records_per_cluster=list()
  total_records=len(df_fin.index)
  #Getting ideal sample size by Solven's Formula assuming Confidence Level=95% i.e n = N / (1 + Ne²)
  sample_size= round(total_records / (1 + total_records* (1-0.95)*(1-0.95)))
  for i in Classes:
    df_fin_test=df_fin
    count_records_per_cluster.append(df_fin_test['Y'].value_counts()[i])
    sample_records_per_cluster.append(count_records_per_cluster[i]/total_records * sample_size)
    df_sample_per_cluster=df_fin_test[df_fin_test.Y==i]
    df_sample_per_cluster=df_sample_per_cluster.sample(int(sample_records_per_cluster[i]),replace=True)   
    df_sample=df_sample.append(df_sample_per_cluster, ignore_index = True)
  
  # summarize distribution
  print('Data Summary after sampling')
  y=df_sample['Y']
  counter = Counter(y)
  for k,v in counter.items():
      per = v / len(y) * 100
      print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
  
  #Hypothesis test to see if sample is accepted
  input_dataframe = pd.read_csv(filepath, header='infer')
  from scipy import stats
  pvalues = list()
  subsample=df_sample
  for col in subsample.columns:
    if (subsample[col].dtype) in ["int32","int64","float"]: 
      # Numeric variable. Using Kolmogorov-Smirnov test
      pvalues.append(stats.ks_2samp(subsample[col],input_dataframe[col]))
        
    else:
      # Categorical variable. Using Pearson's Chi-square test
      from scipy import stats
      import pandas as pd
      sample_count=pd.DataFrame(subsample[col].value_counts())
      input_dataframe_count=pd.DataFrame(input_dataframe[col].value_counts())
      absent=input_dataframe_count-sample_count
      absent2=absent.loc[absent[col].isnull()]
      absent2[col]=0
      sample_count_final=pd.concat([absent2,sample_count]) 
      pvalues.append(stats.chisquare(sample_count_final, input_dataframe_count))
  
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
  Len_Actualdataset_SMOTE= len(df_fin.index)
  Sampling_Error=1/math.sqrt(Len_Sampleddataset) * 100
  print ("Volume of actual dataset = ",Len_Actualdataset)
  print ("Volume of actual dataset after oversampling = ",Len_Actualdataset_SMOTE)
  print ("Volume of Sampled dataset =",Len_Sampleddataset) 
  print('Sampling Error for actual_size={} sample_size={} is {:.3f}% '.format(Len_Actualdataset_SMOTE,Len_Sampleddataset,Sampling_Error))
  if count> threshold*len(input_dataframe.columns):
    print ("Cluster Sample accepted-via KS and Chi_square Null hypothesis")
    Len_Actualdataset= "%s" % str(Len_Actualdataset)
    Len_Sampleddataset= "%s" % str(Len_Sampleddataset)
    Sampling_Error= "%s" %  str(Sampling_Error)  
    rm_str1 = "Insert into TelemetryTable values (" + appnamequotes + ","+ task + ",'Volume of actual dataset = " + Len_Actualdataset + "'," + tsquotes + ")"
    spark.sql(rm_str1)
    rm_str2 = "Insert into TelemetryTable values (" + appnamequotes + ","+ task + ",'Volume of Sampled dataset =" + Len_Sampleddataset + "'," + tsquotes + ")"
    spark.sql(rm_str2)
    rm_str3 = "Insert into TelemetryTable values (" + appnamequotes + ","+ task + ",'Sampling Error = " + Sampling_Error + "'," + tsquotes + ")"
    spark.sql(rm_str3)
    rm_str4 = "Insert into TelemetryTable values (" + appnamequotes + ","+ task + ",'Cluster Sample accepted-via KS and Chi_square Null hypothesis'," + tsquotes + ")"
    spark.sql(rm_str4)
    
  else:
    print ("Cluster Sample rejected")
    rm_str5 = "Insert into TelemetryTable values (" + appnamequotes + ","+ task + ",'Cluster Sample rejected'," + tsquotes + ")"
    spark.sql(rm_str5)
  #print(pvalues) 
  #P is the probability that there is no significant difference between sample and actual (Null hypothesis)
  print('Average p-value between sample and actual(the bigger the p-value the closer the two datasets are.)= ',pvalues_average)
  
  pvalues_average="%s" % str(pvalues_average)
  rm_str6 = "Insert into TelemetryTable values (" + appnamequotes + ","+ task + ",'The bigger the p-value the closer the two datasets are, Average p-value between sample and actual- = " + pvalues_average + "'," + tsquotes + ")"
  spark.sql(rm_str6)
    
  return(sample_size,pvalues_average,subsample)

# COMMAND ----------

# DBTITLE 1,Sampling- Random Sampling
def RandomSampling(input_dataframe, filepath,task_type,input_appname):
  df_fin=input_dataframe
  import pandas as pd
  import numpy as np
  import math
  
  import time
  ts = int(time.time())
  appname = input_appname
  appnamequotes = "'%s'" % appname
  tsquotes = "'%s'" % str(ts)
  task = "'%s'" % str(task_type)
  print("Random Sampling starting")
  rm_str = "Insert into TelemetryTable values (" + appnamequotes + ","+ task + ",'Random Sampling starting'," + tsquotes + ")"
  spark.sql(rm_str)
  
  #Getting ideal sample size by Solven's Formula assuming Confidence Level=95% i.e n = N / (1 + Ne²)
  total_records=len(df_fin.index)
  sample_size= round(total_records / (1 + total_records* (1-0.95)*(1-0.95)))
  
  subsample=df_fin.sample(n=sample_size)
  
  #Hypothesis test to see if sample is accepted
  input_dataframe = pd.read_csv(filepath, header='infer')
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
      sample_count=pd.DataFrame(subsample[col].value_counts())
      input_dataframe_count=pd.DataFrame(input_dataframe[col].value_counts())
      absent=input_dataframe_count-sample_count
      absent2=absent.loc[absent[col].isnull()]
      absent2[col]=0
      sample_count_final=pd.concat([absent2,sample_count]) 
      pvalues.append(stats.chisquare(sample_count_final, input_dataframe_count))
  
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
    rm_str1 = "Insert into TelemetryTable values (" + appnamequotes + ","+ task + ",'Volume of actual dataset = " + Len_Actualdataset + "'," + tsquotes + ")"
    spark.sql(rm_str1)
    rm_str2 = "Insert into TelemetryTable values (" + appnamequotes + ","+ task + ",'Volume of Sampled dataset =" + Len_Sampleddataset + "'," + tsquotes + ")"
    spark.sql(rm_str2)
    rm_str3 = "Insert into TelemetryTable values (" + appnamequotes + ","+ task + ",'Sampling Error = " + Sampling_Error + "'," + tsquotes + ")"
    spark.sql(rm_str3)
    rm_str4 = "Insert into TelemetryTable values (" + appnamequotes + ","+ task + ",'Systematic Sample accepted-via KS and Chi_square Null hypothesis'," + tsquotes + ")"
    spark.sql(rm_str4)
    
  else:
    print ("Random Sample rejected")
    rm_str5 = "Insert into TelemetryTable values (" + appnamequotes + ","+ task + ",'Systematic Sample rejected'," + tsquotes + ")"
    spark.sql(rm_str5)
  #print(pvalues) 
  #P is the probability that there is no significant difference between sample and actual (Null hypothesis)
  print('Average p-value between sample and actual(the bigger the p-value the closer the two datasets are.)= ',pvalues_average)
  
  pvalues_average="%s" % str(pvalues_average)
  rm_str6 = "Insert into TelemetryTable values (" + appnamequotes + ","+ task + ",'The bigger the p-value the closer the two datasets are, Average p-value between sample and actual- = " + pvalues_average + "'," + tsquotes + ")"
  spark.sql(rm_str6)
    
  return(sample_size,pvalues_average,subsample)

# COMMAND ----------

# DBTITLE 1,Sampling- Systematic Sampling
def SystematicSampling(input_dataframe,filepath,task_type,input_appname):
  df_fin=input_dataframe
  import pandas as pd
  import numpy as np
  import math
  
  import time
  ts = int(time.time())
  appname = input_appname
  appnamequotes = "'%s'" % appname
  tsquotes = "'%s'" % str(ts)
  task = "'%s'" % str(task_type)
  print("Systematic Sampling starting")
  rm_str = "Insert into TelemetryTable values (" + appnamequotes + ","+ task + ",'Systematic Sampling starting'," + tsquotes + ")"
  spark.sql(rm_str)
  
  #Getting ideal sample size by Solven's Formula assuming Confidence Level=95% i.e n = N / (1 + Ne²)
  total_records=len(df_fin.index)
  sample_size= round(total_records / (1 + total_records* (1-0.95)*(1-0.95)))
  
  subsample = df_fin.loc[df_fin['Index'] % (round(total_records/sample_size)) ==0]
  
  #Hypothesis test to see if sample is accepted
  input_dataframe = pd.read_csv(filepath, header='infer')
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
      sample_count=pd.DataFrame(subsample[col].value_counts())
      input_dataframe_count=pd.DataFrame(input_dataframe[col].value_counts())
      absent=input_dataframe_count-sample_count
      absent2=absent.loc[absent[col].isnull()]
      absent2[col]=0
      sample_count_final=pd.concat([absent2,sample_count]) 
      pvalues.append(stats.chisquare(sample_count_final, input_dataframe_count))
  
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
    rm_str1 = "Insert into TelemetryTable values (" + appnamequotes + ","+ task + ",'Volume of actual dataset = " + Len_Actualdataset + "'," + tsquotes + ")"
    spark.sql(rm_str1)
    rm_str2 = "Insert into TelemetryTable values (" + appnamequotes + ","+ task + ",'Volume of Sampled dataset =" + Len_Sampleddataset + "'," + tsquotes + ")"
    spark.sql(rm_str2)
    rm_str3 = "Insert into TelemetryTable values (" + appnamequotes + ","+ task + ",'Sampling Error = " + Sampling_Error + "'," + tsquotes + ")"
    spark.sql(rm_str3)
    rm_str4 = "Insert into TelemetryTable values (" + appnamequotes + ","+ task + ",'Systematic Sample accepted-via KS and Chi_square Null hypothesis'," + tsquotes + ")"
    spark.sql(rm_str4)
    
  else:
    print ("Systematic Sample rejected")
    rm_str5 = "Insert into TelemetryTable values (" + appnamequotes + ","+ task + ",'Systematic Sample rejected'," + tsquotes + ")"
    spark.sql(rm_str5)
  #print(pvalues) 
  #P is the probability that there is no significant difference between sample and actual (Null hypothesis)
  print('Average p-value between sample and actual(the bigger the p-value the closer the two datasets are.)= ',pvalues_average)
  pvalues_average="%s" % str(pvalues_average)
  rm_str6 = "Insert into TelemetryTable values (" + appnamequotes + ","+ task + ",'The bigger the p-value the closer the two datasets are, Average p-value between sample and actual- = " + pvalues_average + "'," + tsquotes + ")"
  spark.sql(rm_str6)
    
  return(sample_size,pvalues_average,subsample)

# COMMAND ----------

# DBTITLE 1,Sampling- Stratified Sampling
def StratifiedSampling(input_dataframe,filepath,task_type,input_appname):
  df_fin=input_dataframe
  
  import pandas as pd
  from collections import Counter

  import pandas as pd
  import numpy as np
  import math
  
  import time
  ts = int(time.time())
  appname = input_appname
  appnamequotes = "'%s'" % appname
  tsquotes = "'%s'" % str(ts)
  task = "'%s'" % str(task_type)
  print("Stratified Sampling starting")
  rm_str = "Insert into TelemetryTable values (" + appnamequotes + ","+ task + ",'Stratified Sampling starting'," + tsquotes + ")"
  spark.sql(rm_str)
  
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
  sample_size= round(total_records / (1 + total_records* (1-0.95)*(1-0.95)))
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
  
  #Getting back Sample in original data form
  input_dataframe = pd.read_csv(filepath, header='infer')
  uniqueIndex = list(df_sample['Index'].unique())
  subsample= input_dataframe[input_dataframe.Index.isin(uniqueIndex)]
  
  #Hypothesis test to see if sample is accepted
  input_dataframe = pd.read_csv(filepath, header='infer')
  from scipy import stats
  pvalues = list()
  for col in subsample.columns:
    if (subsample[col].dtype) in ["int32","int64","float"]: 
      # Numeric variable. Using Kolmogorov-Smirnov test
      pvalues.append(stats.ks_2samp(subsample[col],input_dataframe[col]))
        
    else:
      # Categorical variable. Using Pearson's Chi-square test
      from scipy import stats
      import pandas as pd
      sample_count=pd.DataFrame(subsample[col].value_counts())
      input_dataframe_count=pd.DataFrame(input_dataframe[col].value_counts())
      absent=input_dataframe_count-sample_count
      absent2=absent.loc[absent[col].isnull()]
      absent2[col]=0
      sample_count_final=pd.concat([absent2,sample_count]) 
      pvalues.append(stats.chisquare(sample_count_final, input_dataframe_count))
  
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
    rm_str1 = "Insert into TelemetryTable values (" + appnamequotes + ","+ task + ",'Volume of actual dataset = " + Len_Actualdataset + "'," + tsquotes + ")"
    spark.sql(rm_str1)
    rm_str2 = "Insert into TelemetryTable values (" + appnamequotes + ","+ task + ",'Volume of Sampled dataset =" + Len_Sampleddataset + "'," + tsquotes + ")"
    spark.sql(rm_str2)
    rm_str3 = "Insert into TelemetryTable values (" + appnamequotes + ","+ task + ",'Sampling Error = " + Sampling_Error + "'," + tsquotes + ")"
    spark.sql(rm_str3)
    rm_str4 = "Insert into TelemetryTable values (" + appnamequotes + ","+ task + ",'Stratified Sample accepted-via KS and Chi_square Null hypothesis'," + tsquotes + ")"
    spark.sql(rm_str4)
    
  else:
    print ("Stratified Sample rejected")
    rm_str5 = "Insert into TelemetryTable values (" + appnamequotes + ","+ task + ",'Stratified Sample rejected'," + tsquotes + ")"
    spark.sql(rm_str5)
  #print(pvalues) 
  #P is the probability that there is no significant difference between sample and actual (Null hypothesis)
  print('Average p-value between sample and actual (the bigger the p-value the closer the two datasets are.)= ',pvalues_average)
  
  pvalues_average="%s" % str(pvalues_average)
  rm_str6 = "Insert into TelemetryTable values (" + appnamequotes + ","+ task + ",'The bigger the p-value the closer the two datasets are, Average p-value between sample and actual- = " + pvalues_average + "'," + tsquotes + ")"
  spark.sql(rm_str6)
    
  return(sample_size,pvalues_average,subsample)

# COMMAND ----------

# DBTITLE 1,Best Sampling Estimator
def Sampling(input_dataframe,filepath,task_type,input_appname,cluster_col):
  #filepath = '/dbfs/FileStore/glass.csv'
  input_dataframe.to_csv(filepath, index=False)
  task_type1=task_type
  input_appname1=input_appname
  task_type2=task_type
  input_appname2=input_appname
  task_type3=task_type
  input_appname3=input_appname
  task_type4=task_type
  input_appname4=input_appname
  subsample1 = pd.DataFrame()
  subsample2 = pd.DataFrame()
  subsample3 = pd.DataFrame()
  subsample4 = pd.DataFrame()
  
  print("\n","Stratified Sampling")
  input_dataframe = pd.read_csv(filepath, header='infer')
  sample_size1,pvalue1,subsample1= StratifiedSampling(input_dataframe,filepath,task_type1,input_appname1)
  
  
  print("\n","Random Sampling")
  input_dataframe = pd.read_csv(filepath, header='infer')
  sample_size2,pvalue2,subsample2= RandomSampling(input_dataframe,filepath,task_type2,input_appname2)
  
  
  print("\n","Systematic Sampling")
  input_dataframe = pd.read_csv(filepath, header='infer')
  sample_size3,pvalue3,subsample3= SystematicSampling(input_dataframe,filepath,task_type3,input_appname3)
  
  if cluster_col=="NULL":
    print("\n","No Cluster Sampling")
  else:
    print("\n","Cluster Sampling")
    input_dataframe = pd.read_csv(filepath, header='infer')
    sample_size4,pvalue4,subsample4= ClusterSampling(input_dataframe,filepath,task_type4,input_appname4,cluster_col)
  
  import time
  ts = int(time.time())
  appname = input_appname
  appnamequotes = "'%s'" % appname
  tsquotes = "'%s'" % str(ts)
  task = "'%s'" % str(task_type)
  Dic = {}
  Dic.update({"StratifiedSampling":pvalue1})
  Dic.update({"RandomSampling":pvalue2})
  Dic.update({"SystematicSampling":pvalue3})
  if cluster_col!="NULL":
    Dic.update({"ClusterSampling":pvalue4})
  Keymax = max(Dic, key=Dic.get) 
  if Keymax=="StratifiedSampling":
    subsample_final=subsample1
  elif Keymax=="RandomSampling":
    subsample_final=subsample2
  elif Keymax=="SystematicSampling":
    subsample_final=subsample3
  elif Keymax=="ClusterSampling":
    subsample_final=subsample4
  print("\n","Best Suggested Sample is - ",Keymax)
  
  Keymax="%s" % Keymax
  rm_str7 = "Insert into TelemetryTable values (" + appnamequotes + ","+ task + ",'Best Suggested Sample is - " + Keymax + "'," + tsquotes + ")"
  spark.sql(rm_str7)
  
  return(subsample_final,subsample1,subsample2,subsample3,subsample4)

# COMMAND ----------

# DBTITLE 1,Histogram Data distribution
def display_DataDistribution(input_dataframe,label_col):
  import matplotlib.pyplot as plot
  #col_list=list()
  #for col in sampled_dataframe.columns:
  #  col_list.append(col)
  input_dataframe.hist(column=label_col,bins=10)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Cleansing:
# MAGIC Before triggering the Azure Auto ML, our proposed framework helps improve the data quality of our input dataset using the Data Cleansing component.
# MAGIC Since data is considered the currency for any machine learning model, it is very critical for the success of Machine Learning applications. The algorithms that we may use can be powerful, but without the relevant or right data training, our system may fail to yield ideal results.
# MAGIC Data cleansing refers to identifying and correcting errors in the dataset that may negatively impact a predictive model. It refers to all kinds of tasks and activities to detect and repair errors in the data. This improves the quality of the training data for analytics and enables accurate decision-making.
# MAGIC The function ‘autodatacleaner’ encompasses all the underlying features of the data cleansing component that are outlined below.
# MAGIC <br/>
# MAGIC 
# MAGIC 1. **Handle Missing Values:**
# MAGIC <br/>
# MAGIC Data can have missing values for several reasons such as observations that were not recorded and data corruption. Handling missing data is important as many machine learning algorithms do not support data with missing values.
# MAGIC If data is missing, we can either indicate missing values by simply creating a Missing category if the data is categorical or flagging and filling with 0 if it is numerical or apply imputation to fill the missing values.
# MAGIC Hence, as part of the Data Cleansing component, we are applying imputation or dropping the columns in the dataset to fill all the missing values, which is decided based on a threshold of 50%. First, we replace all the white spaces or empty values with NaN except those in the middle. If more than half of the data in a column is NaN, we drop the column else we impute the missing values with median for numerical columns and mode for categorical columns. One limitation with dropping the columns is by dropping missing values, we drop information that may assist us in making better conclusions about the study. It may deny us the opportunity of benefiting from the possible insights that can be gotten from the fact that a particular value is missing. This can be handled by applying feature importance and understanding the significant columns in the dataset that can be useful for the predictive model which shouldn’t be dropped hence, treating this as an exception.
# MAGIC <br/>
# MAGIC 2. **Fix Structural Errors:**
# MAGIC <br/>
# MAGIC After removing unwanted observations and handling missing values, the next thing we make sure is that the wanted observations are well-structured. Structural errors may occur during data transfer due to a slight human mistake or incompetency of the data entry personnel. 
# MAGIC Some of the things we will look out for when fixing data structure include typographical errors, grammatical blunders, and so on. The data structure is mostly concerned with categorical data. 
# MAGIC We are fixing these structural errors by removing leading/trailing white spaces and solving inconsistent capitalization for categorical columns.
# MAGIC <br/>
# MAGIC 3. **Encoding of Categorical Columns:**
# MAGIC <br/>
# MAGIC In machine learning, we usually deal with datasets which contains multiple labels in one or more than one column. These labels can be in the form of words or numbers. The training data is often labeled in words to make it understandable or in human readable form.
# MAGIC Label Encoding refers to converting the labels into numeric form to convert it into the machine-readable form. Machine learning algorithms can then decide in a better way on how those labels must be operated. 
# MAGIC Hence, for Label encoding we are using the Label Encoder component of the python class sklearn preprocessing package.
# MAGIC from sklearn.preprocessing import LabelEncoder
# MAGIC Encode target labels with value between 0 and n_classes-1.
# MAGIC <br/>
# MAGIC 4. **Normalization:**
# MAGIC <br/>
# MAGIC As most of the datasets have multiple features spanning varying degrees of magnitude, range, and units. This can deviate the ML model to be biased towards the dominant scale and hence make it as an obstacle for the machine learning algorithms as they are highly sensitive to these features. Hence, we are tackling this problem using normalization. The goal of normalization is to change the values of numeric columns in the dataset to use a common scale, without distorting differences in the ranges of values or losing information.
# MAGIC We normalize our dataset using the MinMax scaling component of the python class sklearn preprocessing package:
# MAGIC from sklearn.preprocessing import MinMaxScaler
# MAGIC MinMax scaler transforms features by scaling each feature to a given range on the training set, e.g., between zero and one. It shifts and rescales the values so that they end up ranging between 0 and 1.
# MAGIC <br/>Here’s the formula for normalization:
# MAGIC <br/>X^'=  (X- X_min)/X_max⁡〖- X_min 〗  
# MAGIC <br/>Here, 
# MAGIC <br/>Xmax and Xmin are the maximum and the minimum values of the feature respectively.
# MAGIC <br/>	When the value of X is the minimum value in the column, the numerator will be 0, and hence X’ is 0
# MAGIC <br/><br/>	On the other hand, when the value of X is the maximum value in the column, the numerator is equal to the denominator and thus the value of X’ is 1
# MAGIC <br/>	If the value of X is between the minimum and the maximum value, then the value of X’ is between 0 and 1
# MAGIC <br/>
# MAGIC <br/>The transformation is given by:
# MAGIC <br/>X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# MAGIC <br/>X_scaled = X_std * (max - min) + min
# MAGIC <br/>where min, max = feature_range.

# COMMAND ----------

# DBTITLE 1,Data Cleanser -Fix Categorical Columns
def fixCategoricalColumns(input_dataframe,input_appname,task_type):
  from sklearn.preprocessing import LabelEncoder 
  from sklearn.preprocessing import MinMaxScaler
  le = LabelEncoder()
  import time
  from datetime import date
  today = date.today()
  ts = int(time.time())
  appname = input_appname
  appnamequotes = "'%s'" % appname
  tsquotes = "'%s'" % str(ts)
  task = "'%s'" % str(task_type)
  # Replace structural erros on categorical columns - Inconsistent capitalization and Label Encode Categorical Columns + MinMax Scaling
  print("\n","Categorical columns cleansing:")
  print("Fixing inconsistent capitalization and removing any white spaces.")
  for column in input_dataframe.columns.values:
    if str(input_dataframe[column].values.dtype) == 'object':
      for ind in input_dataframe.index:
        input_dataframe[column] = input_dataframe[column].astype(str).str.title()
        input_dataframe[column] = input_dataframe[column].str.replace(" ","")
    if str(input_dataframe[column].values.dtype) == 'object':
      for col in input_dataframe.columns:
        input_dataframe[col]=le.fit_transform(input_dataframe[col])
  print("Label Encoding on categorical columns.")
  rm_str = "Insert into TelemetryTable values (" + appnamequotes + "," + task + ",'Fixed structural errors for categorical columns'," + tsquotes + ")"
  spark.sql(rm_str)
  rm_str = "Insert into TelemetryTable values (" + appnamequotes + "," + task + ",'Applied Label Encoding for categorical columns'," + tsquotes + ")"
  spark.sql(rm_str)
  
  print("MinMax scaling for Normalisation.")
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
  rm_str = "Insert into TelemetryTable values (" + appnamequotes + "," + task + ",'Applied MinMax scaling for Normalisation'," + tsquotes + ")"
  spark.sql(rm_str)
  #print(scaler.transform([[2, 2]]))
  return input_dataframe                         

# COMMAND ----------

# DBTITLE 1,Data Cleanser -Imputation
def impute(input_dataframe,input_appname,task_type):
  # Replace NaNs with the median or mode of the column depending on the column type
  #print("Standard deviation of dataframe before imputation is:\n",input_dataframe.std(axis = 0, skipna = True))
  import time
  from datetime import date
  today = date.today()
  ts = int(time.time())
  appname = input_appname
  appnamequotes = "'%s'" % appname
  tsquotes = "'%s'" % str(ts)
  task = "'%s'" % str(task_type)
  NumCol=[]
  CatCol=[]
  for column in input_dataframe.columns:
    if(input_dataframe[column].dtype) not in ["object"]:
      NumCol.append(column)
      input_dataframe[column].fillna(input_dataframe[column].median(), inplace=True)
      rm_str = "Insert into TelemetryTable values (" + appnamequotes + ","+ task + ",'Imputation of Numerical Column done using Median= " + column + "'," + tsquotes + ")"
      spark.sql(rm_str)
    else:
      CatCol.append(column)
      most_frequent = input_dataframe[column].mode() 
      if len(most_frequent) > 0:
        input_dataframe[column].fillna(input_dataframe[column].mode()[0], inplace=True)
        rm_str = "Insert into TelemetryTable values (" + appnamequotes + ","+ task + ",'Imputation of Categorical Column done using Mode= " + column + "'," + tsquotes + ")"
        spark.sql(rm_str)
      else:
        input_dataframe[column].fillna(method='bfill', inplace=True)
        rm_str = "Insert into TelemetryTable values (" + appnamequotes + ","+ task + ",'Imputation of Categorical Column done using bfill= " + column + "'," + tsquotes + ")"
        spark.sql(rm_str)
        input_dataframe[column].fillna(method='ffill', inplace=True)
        rm_str = "Insert into TelemetryTable values (" + appnamequotes + ","+ task + ",'Imputation of Categorical Column done using ffill= " + column + "'," + tsquotes + ")"
        spark.sql(rm_str)
  print("\n","Imputation of Columns:")
  print("Imputation of Numerical Column done using Median:")
  print(*NumCol, sep = ", ") 
  print("Imputation of Categorical Column done using Mode/bfill/ffill:")
  print(*CatCol, sep = ", ") 
  

  return input_dataframe

# COMMAND ----------

# DBTITLE 1,Data Cleanser -Impute/Drop NULL Rows
def cleanMissingValues(input_dataframe,input_appname,task_type):
  import numpy as np
  import time
  from datetime import date
  today = date.today()
  ts = int(time.time())
  appname = input_appname
  appnamequotes = "'%s'" % appname
  tsquotes = "'%s'" % str(ts)
  task = "'%s'" % str(task_type)
  print("\n","Impute/Drop NULL Rows:")
  print("Total rows in the Input dataframe:",len(input_dataframe.index)) #Total count of dataset
  totalCells = np.product(input_dataframe.shape)
  # Calculate total number of cells in dataframe
  print("Total number of cells in the input dataframe:",totalCells)
  input_dataframe = input_dataframe.replace(r'^\s+$', np.nan, regex=True) #replace white spaces with NaN except spaces in middle
  # Count number of missing values per column
  missingCount = input_dataframe.isnull().sum()
  print("Displaying number of missing records per column:")
  print(missingCount)
  # Calculate total number of missing values
  totalMissing = missingCount.sum()
  print("Total no. of missing values=",totalMissing)
  # Calculate percentage of missing values
  print("The dataset contains", round(((totalMissing/totalCells) * 100), 2), "%", "missing values.")
  cleaned_inputdf= input_dataframe   
  for col in input_dataframe.columns:
    if(missingCount[col]>0):
      print("Percent missing data in",col,"column =", (round(((missingCount[col] / input_dataframe.shape[0]) * 100), 2)))
      if((round(((missingCount[col] / input_dataframe.shape[0]) * 100), 2))>50):
        print("Dropping this column as it contains missing values in more than half of the dataset...")
        input_dataframe_CleanCols=input_dataframe.drop(col,axis=1,inplace=True)
        rm_str = "Insert into TelemetryTable values (" + appnamequotes + ","+ task + ",'Null Column removed: " + col + "'," + tsquotes + ")"
        spark.sql(rm_str)
        print("Total Columns in original dataset: %d \n" % input_dataframe.shape[1])
        print("Total Columns now with na's dropped: %d" % input_dataframe_CleanCols.shape[1])
        cleaned_inputdf=input_dataframe_CleanCols
        
      else:
        print("As percent of missing values is less than half, imputing this column based on the data type.")
        input_dataframe_imputed=impute(input_dataframe,'Auto Tune Model','Data Cleansing',col)
        cleaned_inputdf=input_dataframe_imputed
        rm_str = "Insert into TelemetryTable values (" + appnamequotes + "," + task + ",'Imputation completed for all Columns'," + tsquotes + ")"
        spark.sql(rm_str)
  return cleaned_inputdf

# COMMAND ----------

# DBTITLE 1,Data Cleanser Master
def autodatacleaner(inputdf,input_appname,task_type):
  filepath = '/dbfs/FileStore/RealEstate_tobeCleansed.csv'
  inputdf.to_csv(filepath, index=False)
  task_type1=task_type
  input_appname1=input_appname
  task_type2=task_type
  input_appname2=input_appname
  
  inputdf_1=impute(inputdf,input_appname1,task_type1)
  inputdf_2=cleanMissingValues(inputdf_1,input_appname2,task_type2)
  inputdf_3=fixCategoricalColumns(inputdf_2,input_appname2,task_type2)
  return inputdf_3

# COMMAND ----------

# DBTITLE 1,Data Acquisition:
def DataTypeConversion(input_dataframe,cols_string,cols_int,cols_datetime,cols_Float):
  import pandas as pd
  import numpy as np
  from pyspark.sql.functions import col
  # Pandas to Spark
  #df = spark.createDataFrame(input_dataframe)
  df=input_dataframe
  for col_name in cols_datetime:
    df = df.withColumn(col_name, col(col_name).cast('timestamp'))      
  for col_name in cols_Float:
    df = df.withColumn(col_name, col(col_name).cast('float')) 
  for col_name in cols_string:
    df = df.withColumn(col_name, col(col_name).cast('string'))
  input_dataframe = df.toPandas()
  return input_dataframe

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exploratory Data Analysis (EDA):
# MAGIC Exploratory Data Analysis refers to the critical process of performing initial investigations on data to discover patterns, to spot anomalies, to test hypothesis and to check assumptions with the help of summary statistics and graphical representations. 

# COMMAND ----------

#pip install sweetviz

# COMMAND ----------

# DBTITLE 1,Data Profiler
#pip install pandas-profiling
def Data_Profiling_viaPandasProfiling(df):
  #import time 
  #ts = int(time.time())
  #ts= "%s" % str(ts)
  #filepath="adl://<Your ADLS Name>.azuredatalakestore.net/DEV/EDAProfile_" + ts +".html"
  import pandas_profiling as pp
  profile = pp.ProfileReport(df) 
  p = profile.to_html() 
  #profile.to_file('/dbfs/FileStore/EDAProfile.html')
  #dbutils.fs.cp ("/FileStore/EDAProfile.html", filepath, True)
  #print("EDA Report can be downloaded from path: ",filepath)
  return(p)
  #displayHTML(p)
  #return(df.describe())

# COMMAND ----------

# DBTITLE 1,Data Profiling - Heatmap Correlation
def Data_Profiling_Fin(input_dataframe):
  import seaborn as sns
  corr = input_dataframe.corr()
  # plot the heatmap
  sns.heatmap(corr, 
          xticklabels=corr.columns,
          yticklabels=corr.columns)
  


# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Selection:
# MAGIC <br/>
# MAGIC To be added

# COMMAND ----------

# DBTITLE 1,Feature Selection
def FeatureSelection(input_dataframe,label_col,Y_discrete,filepath,input_appname,task_type):
  input_dataframe = pd.read_csv(filepath, header='infer')
  input_dataframe.drop(['Index'], axis=1, inplace=True)
  
  #Feature imp accept only numeric columns hence label encode
  from sklearn.preprocessing import LabelEncoder
  le = LabelEncoder()
  for col in input_dataframe.columns:
    if (input_dataframe[col].dtype) not in ["int32","int64","float","float64"]:
      input_dataframe[col]=le.fit_transform(input_dataframe[col])
  
  #Doesn't accept nulls so impute
  #input_dataframe=impute(input_dataframe,input_appname,task_type)
  
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
	
# COMMAND ----------

# DBTITLE 1,Feature Selection- PCA
def PrinCompAnalysis(input_dataframe,input_appname,task_type):
  from datetime import date
  today = date.today()
  import time 
  ts = int(time.time())
  appname = input_appname 
  appnamequotes = "'%s'" % appname
  tsquotes = "'%s'" % str(ts)
  task = "'%s'" % str(task_type)
  
  from sklearn.decomposition import PCA
  #Label encoding already done in Sampling
  #from sklearn.preprocessing import LabelEncoder
  #le = LabelEncoder()
  #for col in df.columns:
  #  df[col]=le.fit_transform(df[col])
  #for col in df.columns:
  #  if (df[col].dtype) not in ["int32","int64","float"]: 
  #    df.drop([col], axis=1, inplace=True)
  
  #Drop Row index column, so that they are not considered while getting PC 
  #input_dataframe.pop('row_index')
  
  model = PCA(n_components=2).fit(input_dataframe)
  #X_pca = model.transform(input_dataframe) #No need to transform now, just getting PC, not feeding to AML

  # number of components
  n_pcs= model.components_.shape[0]
  
  # get the index of the most important feature on EACH component i.e. largest absolute value
  # using LIST COMPREHENSION HERE
  most_important = [abs(model.components_[i]).argmax() for i in range(n_pcs)]
  #print(model.components_)
  #print(model.components_[0].argmax())
  #print(most_important)
  initial_feature_names = list(input_dataframe.columns) 
  
  # get the names
  most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]
  
  # using LIST COMPREHENSION HERE AGAIN
  dic = {'PC{}'.format(i+1): most_important_names[i] for i in range(n_pcs)}
  
  # build the dataframe
  df_see = pd.DataFrame(sorted(dic.items()))
  
  rm_str = "Insert into TelemetryTable values (" + appnamequotes + "," + task + ",'PCA Done'," + tsquotes + ")"
  spark.sql(rm_str)
  
  df_see.columns =['Principal_Component','Most_Important_Feature_Along_PC']
  return(df_see)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Importance:
# MAGIC To improve the efficiency and effectiveness of a predictive model on the problem, we can apply Feature importance since it provides us with a basis for dimensionality reduction and feature selection. It refers to a class of techniques that assign a score to input features based on how useful they are at predicting a target variable. Feature importance scores provide insight into the dataset. The relative scores highlight which features may be the most relevant to the target, and the converse, which features are the least relevant. We are using ML Interpret and Exploration classes so that from a wide number of variables, we can pick those variables only that provide maximum variability along the prediction column of these classes. We can choose out of two flavors of feature importance implementations on need basis that is: Global (Based on whole dataset, aggregated value) and Local (Record to record basis). Local measures focus on the contribution of features for a specific prediction, whereas global measures take all predictions into account.
# MAGIC <br/>To generate an explanation for AutoML models, we also use the MimicWrapper class. 
# MAGIC <br/>We can initialize the MimicWrapper with these parameters:
# MAGIC <br/>•	The explainer setup object
# MAGIC <br/>•	Your workspace
# MAGIC <br/>•	A surrogate model to explain the fitted_model automated ML model
# MAGIC <br/>The MimicWrapper also takes the automl_run object where the engineered explanations will be uploaded.

# COMMAND ----------

# DBTITLE 1,Feature Importance
import logging
import os
import random
import time
import json

from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
import numpy as np
import pandas as pd

import azureml.core
from azureml.core.experiment import Experiment
from azureml.core.workspace import Workspace
from azureml.train.automl import AutoMLConfig
from azureml.train.automl.run import AutoMLRun  
import azureml.train.automl

from azureml.train.automl.automlconfig import AutoMLConfig
from azureml.core import Workspace

def Feature_Importance(df,subscription_id,resource_group,workspace_name,workspace_region,run_id,iteration,label_col,task,ImportanceType,local_df):
  ws = Workspace(workspace_name = workspace_name,
                 subscription_id = subscription_id,
                 resource_group = resource_group)
  experiment_name = 'Experiment'
  experiment = Experiment(ws, experiment_name)
  automl_classV2 = AutoMLRun(experiment = experiment, run_id = run_id)
  exp_run,exp_model = automl_classV2.get_output(iteration=iteration)
  
  from sklearn.model_selection import train_test_split
  y_df = df.pop(label_col)
  x_df = df
  x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=223)
  
  if ImportanceType=="global":
    from azureml.train.automl.runtime.automl_explain_utilities import automl_setup_model_explanations
  
    automl_explainer_setup_obj = automl_setup_model_explanations(exp_model, X=x_train, 
                                                                 X_test=x_test, y=y_train, 
                                                                 task=task)
    
    from azureml.explain.model.mimic_wrapper import MimicWrapper
    
    # Initialize the Mimic Explainer
    explainer = MimicWrapper(ws, automl_explainer_setup_obj.automl_estimator,
                             explainable_model=automl_explainer_setup_obj.surrogate_model, 
                             init_dataset=automl_explainer_setup_obj.X_transform, run=exp_run,
                             features=automl_explainer_setup_obj.engineered_feature_names, 
                             feature_maps=[automl_explainer_setup_obj.feature_map],
                             classes=automl_explainer_setup_obj.classes,
                             explainer_kwargs=automl_explainer_setup_obj.surrogate_model_params)
    
    
    engineered_explanations = explainer.explain(['global'], eval_dataset=automl_explainer_setup_obj.X_test_transform)
    print(engineered_explanations.get_feature_importance_dict())
  
  
  if ImportanceType=="local":
    model = exp_model.fit(x_train, y_train) 
    featurenames = x_df.columns
    from interpret.ext.blackbox import TabularExplainer
    explainer = TabularExplainer(model, 
                             x_train, 
                             features=featurenames
                            )
    local_explanation = explainer.explain_local(local_df)
    sorted_local_importance_names = local_explanation.get_ranked_local_names()
    sorted_local_importance_values = local_explanation.get_ranked_local_values()
    #print(dict(zip(sorted_local_importance_names, sorted_local_importance_values)))
    print(sorted_local_importance_names)
    print(sorted_local_importance_values)

# COMMAND ----------

# MAGIC %md
# MAGIC #Auto ML Trigger and obtaining Predicted v/s Actual Data:
# MAGIC <br/>
# MAGIC During training, Azure Machine Learning creates several pipelines in parallel that try different algorithms and parameters for you. The service iterates through ML algorithms paired with feature selections, where each iteration produces a model with a training score. The higher the score, the better the model is considered to "fit" your data. It will stop once it hits the exit criteria defined in the experiment. The function ‘AutoMLFunc’ encompasses all the features of this as explained below.
# MAGIC <br/>
# MAGIC Using Azure Machine Learning, you can design and run your automated ML training experiments with these steps:
# MAGIC 1. Identify the ML problem to be solved: classification or regression.
# MAGIC 2. Configure the automated machine learning parameters that determine how many iterations over different models, hyperparameter settings, advanced preprocessing/featurization, and what metrics to look at when determining the best model.
# MAGIC 3. Divide the input preprocessed data into train and test datasets.
# MAGIC 4. Submit the training run and extract the best model based on primary metrics. 
# MAGIC 5. Use this best model thus obtained to predict data and calculate accuracy scores on actual v/s predicted components of data. Mean Absolute Percentage Error (MAPE) is generally used to determine the performance of a model, but problems can occur when calculating the MAPE value with small denominators (Actual value =0 in denominator). A singularity problem of the form 'one divided by can occur. As an alternative, each actual value (A_(t )  of the series in the original formula can be replaced by the average of all actual values A_avg of that series. This is equivalent to dividing the sum of absolute differences by the sum of actual values and is sometimes referred to as WAPE (Weighted Absolute Percentage Error).
# MAGIC 6. The best model obtained can also be deployed and used using a REST API. The actual v/s predicted data can be reported and analyzed in Power BI along with the telemetry timestamps. 

# COMMAND ----------

# DBTITLE 1,Auto ML Run -PCA dimensionality reduction also 

def AutoMLFuncPCA(subscription_id,resource_group,workspace_name,input_dataframe,label_col,task_type,input_appname):
  initial_feature_names = list(input_dataframe.columns)
  
  import time
  ts = int(time.time())
  appname = input_appname
  appnamequotes = "'%s'" % appname
  tsquotes = "'%s'" % str(ts)
  task = "'%s'" % str(task_type)
  
  #AML workspace import 
  from azureml.core import Workspace
  ws = Workspace(subscription_id = subscription_id, resource_group = resource_group, workspace_name = workspace_name)
  #ws.write_config()
  
  #Train test split
  from sklearn.model_selection import train_test_split
  y_df = input_dataframe.pop(label_col)
  x_df = input_dataframe
  x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=223)
  
  # Applying PCA function on training and testing set of X component 
  from sklearn.decomposition import PCA 
  model = PCA(n_components=2).fit(input_dataframe)
  n_pcs= model.components_.shape[0]
  most_important = [abs(model.components_[i]).argmax() for i in range(n_pcs)] 
  most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]
  dic = {'PC{}'.format(i+1): most_important_names[i] for i in range(n_pcs)}
  df_see = pd.DataFrame(sorted(dic.items()))
  rm_str = "Insert into TelemetryTable values (" + appnamequotes + "," + task + ",'PCA Done,Transformed data to reduce Dimensionality'," + tsquotes + ")"
  spark.sql(rm_str)
  df_see.columns =['Principal_Component','Most_Important_Feature_Along_PC']
  
  pca = PCA(n_components = 2)   
  x_train = pca.fit_transform(x_train) 
  x_test = pca.transform(x_test)

  #AML config
  from azureml.train.automl import AutoMLConfig
  import logging
  #primary_metric_regression=["spearman_correlation","normalized_root_mean_squared_error"]
  #primary_metric_classification=["average_precision_score_weighted","AUC_weighted"]
  if task_type=="regression":
    automl_config = AutoMLConfig(task=task_type,
                             iteration_timeout_minutes= 10,
                             iterations= 2,
                             primary_metric='spearman_correlation',
                             preprocess= True,
                             verbosity= logging.INFO,
                             n_cross_validations= 5,
                             debug_log='automated_ml_errors.log',
                             X=x_train,
                             y=y_train)
    #AML run
    from azureml.core.experiment import Experiment
    experiment = Experiment(ws, "Experiment")
    local_run = experiment.submit(automl_config, show_output=True) 
    
    #best model extraction
    best_run, fitted_model = local_run.get_output()
    
    #accuracy calculation
    y_predict = fitted_model.predict(x_test)
    y_actual = y_test.tolist()
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
    Accuracy_score="'%s'" % str(1 - mean_abs_percent_error)
    rm_str1 = "Insert into TelemetryTable values (" + appnamequotes + "," + task + ",'Accuracy Score as per absolute Error Percentage (PCA) =' " + Accuracy_score + "," + tsquotes + ")"
    spark.sql(rm_str1)
    #return(1 - mean_abs_percent_error)
    print("Follow this URL for your Experiment run: ")  
    print(local_run.get_portal_url())
    return(df_see)
  if task_type=="classification":
    automl_config = AutoMLConfig(task=task_type,
                             iteration_timeout_minutes= 10,
                             iterations= 2,
                             primary_metric='AUC_weighted',
                             preprocess= True,
                             verbosity= logging.INFO,
                             n_cross_validations= 5,
                             debug_log='automated_ml_errors.log',
                             X=x_train,
                             y=y_train)
    #AML run
    from azureml.core.experiment import Experiment
    experiment = Experiment(ws, "Experiment")
    local_run = experiment.submit(automl_config, show_output=True)
    #best model extraction
    best_run, fitted_model = local_run.get_output()
    #accuracy calculation
    from sklearn.metrics import confusion_matrix 
    from sklearn.metrics import accuracy_score 
    from sklearn.metrics import classification_report
    y_predict = fitted_model.predict(x_test)
    y_actual = y_test.tolist()
    results = confusion_matrix(y_actual, y_predict) 
    #print ('Confusion Matrix :')
    #print(results) 
    #print ('Accuracy Score :',accuracy_score(y_actual, y_predict))
    #print ('Report : ')
    #print (classification_report(y_actual, y_predict)) 
    Accuracy_score="'%s'" % str(accuracy_score(y_actual, y_predict))
    rm_str1 = "Insert into TelemetryTable values (" + appnamequotes + "," + task + ",'Accuracy Score as per absolute Error Percentage (PCA) =' " + Accuracy_score + "," + tsquotes + ")"
    spark.sql(rm_str1)
    #return(accuracy_score(y_actual, y_predict))
    print("Follow this URL for your Experiment run: ")  
    print(local_run.get_portal_url())
    return(df_see)

# COMMAND ----------

# DBTITLE 1,Auto ML Run 

def AutoMLFunc(subscription_id,resource_group,workspace_name,input_dataframe,label_col,task_type,input_appname):
  import time
  ts = int(time.time())
  appname = input_appname
  appnamequotes = "'%s'" % appname
  tsquotes = "'%s'" % str(ts)
  task = "'%s'" % str(task_type)
  
  #AML workspace import 
  from azureml.core import Workspace
  ws = Workspace(subscription_id = subscription_id, resource_group = resource_group, workspace_name = workspace_name)
  #ws.write_config()
  
  #Train test split
  from sklearn.model_selection import train_test_split
  y_df = input_dataframe.pop(label_col)
  x_df = input_dataframe
  x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=223)
  
  #AML config
  from azureml.train.automl import AutoMLConfig
  import logging
  #primary_metric_regression=["spearman_correlation","normalized_root_mean_squared_error"]
  #primary_metric_classification=["average_precision_score_weighted","AUC_weighted"]
  if task_type=="regression":
    automl_config = AutoMLConfig(task=task_type,
                             iteration_timeout_minutes= 10,
                             iterations= 2,
                             primary_metric='spearman_correlation',
                             preprocess= True,
                             verbosity= logging.INFO,
                             n_cross_validations= 5,
                             debug_log='automated_ml_errors.log',
                             X=x_train,
                             y=y_train)
    #AML run
    from azureml.core.experiment import Experiment
    experiment = Experiment(ws, "Experiment")
    local_run = experiment.submit(automl_config, show_output=True) 
    
    #best model extraction
    best_run, fitted_model = local_run.get_output()
    
    #accuracy calculation
    y_predict = fitted_model.predict(x_test)
    x_test['y_predict'] = y_predict
    x_test['y_actual'] = y_test
    x_train['y_predict'] = y_train
    x_train['y_actual'] = y_train
    y_actual = y_test.tolist()
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
    Accuracy_score="'%s'" % str(1 - mean_abs_percent_error)
    rm_str1 = "Insert into TelemetryTable values (" + appnamequotes + "," + task + ",'Accuracy Score as per absolute Error Percentage=' " + Accuracy_score + "," + tsquotes + ")"
    spark.sql(rm_str1)
    #return(1 - mean_abs_percent_error)
    print("Follow this URL for your Experiment run: ")  
    print(local_run.get_portal_url())
    #ret=x_test.append(x_train, ignore_index = True)
    #return(ret)
    return(x_test)
  if task_type=="classification":
    automl_config = AutoMLConfig(task=task_type,
                             iteration_timeout_minutes= 10,
                             iterations= 2,
                             primary_metric='AUC_weighted',
                             preprocess= True,
                             verbosity= logging.INFO,
                             n_cross_validations= 5,
                             debug_log='automated_ml_errors.log',
                             X=x_train,
                             y=y_train)
    #AML run
    from azureml.core.experiment import Experiment
    experiment = Experiment(ws, "Experiment")
    local_run = experiment.submit(automl_config, show_output=True)
    #best model extraction
    best_run, fitted_model = local_run.get_output()
    #accuracy calculation
    from sklearn.metrics import confusion_matrix 
    from sklearn.metrics import accuracy_score 
    from sklearn.metrics import classification_report
    y_predict = fitted_model.predict(x_test)
    x_test['y_predict'] = y_predict
    x_test['y_actual'] = y_test
    x_train['y_predict'] = y_train
    x_train['y_actual'] = y_train
    y_actual = y_test.tolist()
    results = confusion_matrix(y_actual, y_predict) 
    #print ('Confusion Matrix :')
    #print(results) 
    #print ('Accuracy Score :',accuracy_score(y_actual, y_predict))
    #print ('Report : ')
    #print (classification_report(y_actual, y_predict)) 
    Accuracy_score="'%s'" % str(accuracy_score(y_actual, y_predict))
    rm_str1 = "Insert into TelemetryTable values (" + appnamequotes + "," + task + ",'Accuracy Score as per absolute Error Percentage=' " + Accuracy_score + "," + tsquotes + ")"
    spark.sql(rm_str1)
    #return(accuracy_score(y_actual, y_predict))
    print("Follow this URL for your Experiment run: ")  
    print(local_run.get_portal_url())
    #ret=x_test.append(x_train, ignore_index = True)
    return(x_test)

# COMMAND ----------

# DBTITLE 1,Auto ML Run -service principal auth (Bypass AML Authentication)

def AutoMLFuncSP(subscription_id,resource_group,workspace_name,svc_pr_password,tenant_id,service_principal_id,input_dataframe,label_col,task_type,input_appname):
  ts = int(time.time())
  appname = input_appname 
  appnamequotes = "'%s'" % appname
  tsquotes = "'%s'" % str(ts)
  task = "'%s'" % str(task_type)
  from azureml.core import Workspace
  from azureml.core.authentication import ServicePrincipalAuthentication
  import os
  #AML workspace import
  svc_pr = ServicePrincipalAuthentication(
    tenant_id=tenant_id,
    service_principal_id=service_principal_id,
    service_principal_password=svc_pr_password)
   
  
  ws = Workspace(subscription_id = subscription_id, resource_group = resource_group, workspace_name = workspace_name,auth=svc_pr)
  #ws.write_config()
  
  #Train test split
  from sklearn.model_selection import train_test_split
  y_df = input_dataframe.pop(label_col)
  x_df = input_dataframe
  x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=223)
  
  #AML config
  from azureml.train.automl import AutoMLConfig
  import logging
  #primary_metric_regression=["spearman_correlation","normalized_root_mean_squared_error"]
  #primary_metric_classification=["average_precision_score_weighted","AUC_weighted"]
  if task_type=="regression":
    automl_config = AutoMLConfig(task=task_type,
                             iteration_timeout_minutes= 10,
                             iterations= 2,
                             primary_metric='spearman_correlation',
                             preprocess= True,
                             verbosity= logging.INFO,
                             n_cross_validations= 5,
                             debug_log='automated_ml_errors.log',
                             X=x_train,
                             y=y_train)
    #AML run
    from azureml.core.experiment import Experiment
    experiment = Experiment(ws, "Experiment")
    local_run = experiment.submit(automl_config, show_output=True) 
    
    #best model extraction
    best_run, fitted_model = local_run.get_output()
    
    #accuracy calculation
    y_predict = fitted_model.predict(x_test)
    y_actual = y_test.tolist()
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
    Accuracy_score="'%s'" % str(1 - mean_abs_percent_error)
    rm_str1 = "Insert into TelemetryTable values (" + appnamequotes + "," + task + ",'Accuracy Score as per absolute Error Percentage=' " + Accuracy_score + "," + tsquotes + ")"
    spark.sql(rm_str1)
    #return(1 - mean_abs_percent_error)

  if task_type=="classification":
    automl_config = AutoMLConfig(task=task_type,
                             iteration_timeout_minutes= 10,
                             iterations= 2,
                             primary_metric='AUC_weighted',
                             preprocess= True,
                             verbosity= logging.INFO,
                             n_cross_validations= 5,
                             debug_log='automated_ml_errors.log',
                             X=x_train,
                             y=y_train)
    #AML run
    from azureml.core.experiment import Experiment
    experiment = Experiment(ws, "Experiment")
    local_run = experiment.submit(automl_config, show_output=True)
    #best model extraction
    best_run, fitted_model = local_run.get_output()
    #accuracy calculation
    from sklearn.metrics import confusion_matrix 
    from sklearn.metrics import accuracy_score 
    from sklearn.metrics import classification_report
    y_predict = fitted_model.predict(x_test)
    y_actual = y_test.tolist()
    results = confusion_matrix(y_actual, y_predict) 
    #print ('Confusion Matrix :')
    #print(results) 
    #print ('Accuracy Score :',accuracy_score(y_actual, y_predict))
    #print ('Report : ')
    #print (classification_report(y_actual, y_predict)) 
    Accuracy_score="'%s'" % str(accuracy_score(y_actual, y_predict))
    rm_str1 = "Insert into TelemetryTable values (" + appnamequotes + "," + task + ",'Accuracy Score as per absolute Error Percentage=' " + Accuracy_score + "," + tsquotes + ")"
    spark.sql(rm_str1)
    #return(accuracy_score(y_actual, y_predict))
  print("Follow this URL for your Experiment run: ")  
  print(local_run.get_portal_url())


# COMMAND ----------

# DBTITLE 1,Telemetry Table Creation in gen2 dev lake
#%sql
#Create database
#%sql
#CREATE TABLE  TelemetryTable
#(MLKey String,
#Step String,
#Results String,
#TimeGenerated String
#)
#
#Using delta
#    LOCATION 'abfss://<Your ADLS Name>.dfs.core.windows.net/dev/TelemetryTable'
