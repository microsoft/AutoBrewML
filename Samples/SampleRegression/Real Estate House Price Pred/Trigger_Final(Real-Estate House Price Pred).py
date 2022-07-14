# Databricks notebook source
# MAGIC %md
# MAGIC **PROBLEM STATEMENT**
# MAGIC <br/>Predict the House price depending on depending on the date of purchase, distance from local institutes, location etc.
# MAGIC <br/>Get Sample data from Source- https://archive.ics.uci.edu/ml/datasets/Real+estate+valuation+data+set
# MAGIC <br/>
# MAGIC <br/>**COLUMN DEFINITION**
# MAGIC <br/>X1=the transaction date (for example, 2013.250=2013 March, 2013.500=2013 June, etc.)
# MAGIC <br/>X2=the house age (unit: year)
# MAGIC <br/>X3=the distance to the nearest MRT station (unit: meter)
# MAGIC <br/>X4=the number of convenience stores in the living circle on foot (integer)
# MAGIC <br/>X5=the geographic coordinate, latitude. (unit: degree)
# MAGIC <br/>X6=the geographic coordinate, longitude. (unit: degree)
# MAGIC <br/>The output is as follow:
# MAGIC <br/>Y= house price of unit area (10000 New Taiwan Dollar/Ping, where Ping is a local unit, 1 Ping = 3.3 meter squared)
# MAGIC <br/>
# MAGIC <br/>
# MAGIC <br/>**STEPS IN MODELLING**
# MAGIC <br/>1.Data Acquisation
# MAGIC <br/>2.Data understanding
# MAGIC <br/>3.Data visualisation/EDA
# MAGIC <br/>4.Data cleaning/missing imputation/typecasting
# MAGIC <br/>5.Sampling/ bias removal
# MAGIC <br/>6.Anomaly detection
# MAGIC <br/>7.Feature selection/importance
# MAGIC <br/>8.Azure ML Model trigger
# MAGIC <br/>9.Model Interpretation
# MAGIC <br/>10.Telemetry
# MAGIC <br/>
# MAGIC <br/>
# MAGIC <br/>**FEATURE ENGINEERING**
# MAGIC <br/>1. Lat long coordinates have a problem that they are 2 features that represent a three dimensional space. This means that the long coordinate goes all around, which means the two most extreme values are actually very close together. I've dealt with this problem a few times and what I do in this case is map them to x, y and z coordinates. This means close points in these 3 dimensions are also close in reality. Depending on the use case you can disregard the changes in height and map them to a perfect sphere. These features can then be standardized properly.
# MAGIC <br/>x = RadiusOfEarth * cos(lat) * cos(lon)
# MAGIC <br/>y = RadiusOfEarth * cos(lat) * sin(lon), 
# MAGIC <br/>z = RadiusOfEarth * sin(lat) 
# MAGIC <br/>Once we have the x,y,z coordinates, we can use x2 + y2 + z2 to get the distance from (0,0,0) which brings all locations to a standard scale. We can ignore the Radius of earth as we are anyway going to standardize the col.
# MAGIC <br/>2. Year column extraction from a date column to add attribute of the year of purchase.
# MAGIC <br/>3. Cluster sampling by 'X1 transaction year' as the data plots look skewed towards the year 2013>2012

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import functions from Master Notebook:
# MAGIC Import the Functions and dependencies from the Master notebook to be used in the Trigger Notebook

# COMMAND ----------

# DBTITLE 1,Import Functions
# MAGIC %run /Users/.../AMLMasterNotebook

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.Data Acquisition
# MAGIC 1.Acquisition of data from datasource ADLS path in CSV/Parquet/JSON etc format.
# MAGIC <br/>2.Logical Transformations in data. 
# MAGIC <br/>3.Transforming columns into required datatypes, converting to pandas df, persisiting actual dataset, intoducing a column 'Index' to assign a unique identifier to each dataset row so that this canm be used to retrieve back the original form after any data manupulations. 

# COMMAND ----------

# DBTITLE 1,Read Data from Lake
# MAGIC %scala
# MAGIC //<USER INPUT FILEPATH PARQUET OR CSV>
# MAGIC 
# MAGIC val filepath= "adl://<Your ADLS Name>.azuredatalakestore.net/.../RealEstateData.csv"
# MAGIC var df=spark.read.format("csv").option("header", "true").option("delimiter", ",").load(filepath)
# MAGIC //val filepath ="abfss:/.../.parquet"
# MAGIC //var df = spark.read.parquet(filepath)
# MAGIC df.createOrReplaceTempView("vw")

# COMMAND ----------

# DBTITLE 1,Logical transformation of data
df= spark.sql("""select * , ROUND((x*x + y*y + z*z),4) AS Distance from
(select 
`Index`
--,`X1 transaction date`
,`X2 house age`
,`X3 distance to the nearest MRT station`
,`X4 number of convenience stores`
--,`X5 latitude`
--,`X6 longitude`
,`Y house price of unit area`
,LEFT(`X1 transaction date`,4) AS `X1 transaction year`
,ROUND(ABS(cos(`X5 latitude`) * cos(`X6 longitude`)),4) AS x
,ROUND(ABS(cos(`X5 latitude`) * sin(`X6 longitude`)),4) AS y
,ROUND(ABS(sin(`X5 latitude`)),4) AS z
 from vw)""")

# COMMAND ----------

# DBTITLE 1,Data columns and structural transformation
import pandas as pd
import numpy as np
from pyspark.sql.functions import col
##df.select(*(col(c).cast("float").alias(c) for c in df.columns))
#cols=df.columns
#cols.remove('Index')
#for col_name in cols:
#    df = df.withColumn(col_name, col(col_name).cast('float'))
#for col_name in ['Index']:
#    df = df.withColumn(col_name, col(col_name).cast('Int'))   

# <USER INPUT COLUMN NAMES WITH DATAYPES IN RESPECTIVE BUCKET>
cols_all=[
'Index'
,'X2 house age'
,'X3 distance to the nearest MRT station'
,'X4 number of convenience stores'
,'Y house price of unit area'
,'X1 transaction year'
,'x'
,'y'
,'z' 
,'Distance'
]
cols_string=['X1 transaction year']
cols_int=['Index']
cols_bool=[]
cols_Float=[
'X2 house age'
,'X3 distance to the nearest MRT station'
,'X4 number of convenience stores'
,'Y house price of unit area'
,'x'
,'y'
,'z' 
,'Distance'
]
for col_name in cols_int:
    df = df.withColumn(col_name, col(col_name).cast('Int'))  
for col_name in cols_Float:
    df = df.withColumn(col_name, col(col_name).cast('float')) 
for col_name in cols_bool:
    df = df.withColumn(col_name, col(col_name).cast('bool')) 
    
input_dataframe = df.toPandas()
#input_dataframe['Index'] = np.arange(len(input_dataframe))
outdir = '/dbfs/FileStore/RealEstate.csv'
input_dataframe.to_csv(outdir, index=False)
#input_dataframe = pd.read_csv("/dbfs/FileStore/Dataframe.csv", header='infer')


# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.Data Exploration
# MAGIC 1.Exploratory Data Analysis (EDA)- To understand the overall data at hand, analysing each feature independently for its' statistics, the correlation and interraction between variables, data sample etc. 
# MAGIC <br/>2.Data Profiling Plots- To analyse the Categorical and Numerical columns separately for any trend in data, biasness in data etc.

# COMMAND ----------

# DBTITLE 1,EDA
input_dataframe = pd.read_csv("/dbfs/FileStore/RealEstate.csv", header='infer')

p=Data_Profiling_viaPandasProfiling(input_dataframe,'RealEstate','EDA')
displayHTML(p)


# COMMAND ----------

# DBTITLE 1,Data Profiling Plots
input_dataframe = pd.read_csv("/dbfs/FileStore/RealEstate.csv", header='infer')

#User Inputs
cols_all=[
'Index'
,'X2 house age'
,'X3 distance to the nearest MRT station'
,'X4 number of convenience stores'
,'Y house price of unit area'
,'X1 transaction year'
,'x'
,'y'
,'z' 
,'Distance'
]
Categorical_cols=['X1 transaction year'
]
Numeric_cols=['X2 house age'
,'X3 distance to the nearest MRT station'
,'X4 number of convenience stores'
,'X1 transaction year'
,'x'
,'y'
,'z' 
,'Distance'
]
Label_col='Y house price of unit area'

#Data_Profiling_Plots(input_dataframe,Categorical_cols,Numeric_cols,Label_col)
Data_Profiling_Plots(input_dataframe,Categorical_cols,Numeric_cols,Label_col)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.Sampling
# MAGIC Perform Stratified, Systematic, Random, Cluster sampling over data and compare the so obtained sampled dataset with the original data using a NULL Hypothesis, and suggest the best sample obtained thus. Compare the data densities of sampled datasets with that of the original input dataset to validate that our sample matches the data trend of original set.

# COMMAND ----------

# DBTITLE 1,3.Sampling 
input_dataframe = pd.read_csv("/dbfs/FileStore/RealEstate.csv", header='infer')
subsample_final = pd.DataFrame()
subsample1 = pd.DataFrame()
subsample2 = pd.DataFrame()
subsample3 = pd.DataFrame()
subsample4 = pd.DataFrame()

#Sampling(input_dataframe,task_type,input_appname,cluster_classified_col_ifany(Supervised))
subsample_final,subsample1,subsample2,subsample3,subsample4=Sampling(input_dataframe,"/dbfs/FileStore/RealEstate.csv",'Sampling','RealEstate','X1 transaction year')

#persist sampled data sets 
filepath1 = '/dbfs/FileStore/StratifiedSampled_RealEstate.csv'
subsample1.to_csv(filepath1, index=False)
filepath2 = '/dbfs/FileStore/RandomSampled_RealEstate.csv'
subsample2.to_csv(filepath2, index=False)
filepath3 = '/dbfs/FileStore/SystematicSampled_RealEstate.csv'
subsample3.to_csv(filepath3, index=False)
filepath4 = '/dbfs/FileStore/ClusterSampled_RealEstate.csv'
subsample4.to_csv(filepath4, index=False)
filepath = '/dbfs/FileStore/subsample_final_RealEstate.csv'
subsample_final.to_csv(filepath, index=False)

# COMMAND ----------

# DBTITLE 1,Data Distribution (Histogram)- Input dataset
#input_dataframe = pd.read_csv("/dbfs/FileStore/RealEstate.csv", header='infer')

display(display_DataDistribution(input_dataframe,'Y house price of unit area'))

# COMMAND ----------

# DBTITLE 1,Data Distribution (Histogram)- Stratified Sampled dataset
#subsample1 = pd.read_csv("/dbfs/FileStore/StratifiedSampled_RealEstate.csv", header='infer')

display(display_DataDistribution(subsample1,'Y house price of unit area'))

# COMMAND ----------

# DBTITLE 1,Data Distribution (Histogram)- Random Sampled dataset
#subsample2 = pd.read_csv("/dbfs/FileStore/RandomSampled_RealEstate.csv", header='infer')

display(display_DataDistribution(subsample2,'Y house price of unit area'))

# COMMAND ----------

# DBTITLE 1,Data Distribution (Histogram)- Systematic Sampled dataset
#subsample3 = pd.read_csv("/dbfs/FileStore/SystematicSampled_RealEstate.csv", header='infer')

display(display_DataDistribution(subsample3,'Y house price of unit area'))

# COMMAND ----------

# DBTITLE 1,Data Distribution (Histogram)-  Cluster Sampled dataset
#subsample4 = pd.read_csv("/dbfs/FileStore/ClusterSampled_RealEstate.csv", header='infer')

display(display_DataDistribution(subsample4,'Y house price of unit area'))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.Cleansing
# MAGIC To clean the data from NULL values, fix structural errors in columns, drop empty columns, encode the categorical values, normalise the data to bring to the same scale. We also check the Data Distribution via Correlation heatmap of original input dataset v/s the Cleansed dataset to validate whether or not the transformations hampered the original data trend/density.

# COMMAND ----------

# DBTITLE 1,Auto Cleanser 
subsample_final = pd.read_csv("/dbfs/FileStore/subsample_final_RealEstate.csv", header='infer')
#subsample_final=subsample_final.drop(['Index'], axis = 1) # Index is highest variability column hence always imp along PC but has no business value. You can append columns to be dropped by your choice here in the list

inputdf_new=autodatacleaner(subsample_final,"/dbfs/FileStore/subsample_final_RealEstate.csv","RealEstate","Data Cleanser")
print("Total rows in the new pandas dataframe:",len(inputdf_new.index))

#persist cleansed data sets 
filepath1 = '/dbfs/FileStore/Cleansed_RealEstate.csv'
inputdf_new.to_csv(filepath1, index=False)


# COMMAND ----------

# DBTITLE 1,Data profiling(Heatmap correlation)- User input dataframe
subsample_final = pd.read_csv("/dbfs/FileStore/subsample_final_RealEstate.csv", header='infer')

display(Data_Profiling_Fin(subsample_final))

# COMMAND ----------

# DBTITLE 1,Data profiling(Heatmap correlation)- Cleansed dataframe
Cleansed=pd.read_csv("/dbfs/FileStore/Cleansed_RealEstate.csv", header='infer')

display(Data_Profiling_Fin(Cleansed))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.Anomaly Detection
# MAGIC Iterate data over various Anomaly-detection techniques and estimate the number of Inliers and Outliers for each.

# COMMAND ----------

# DBTITLE 0,Anomaly Detection
#Calling the Anamoly Detection Function for identifying outliers 
outliers_fraction = 0.05
df =pd.read_csv("/dbfs/FileStore/Cleansed_RealEstate.csv", header='infer')
target_variable = 'Y house price of unit area'
variables_to_analyze='X3 distance to the nearest MRT station'

AnomalyDetection(df,target_variable,variables_to_analyze,outliers_fraction,'anomaly_test','test')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.Feature Selection
# MAGIC Perform feature selection on the basis of Feature Importance ranking, correlation values, variance within the column.
# MAGIC Choose features with High Importance value score, drop one of the two highly correlated features, drop features which offer zero variability to data and thus do not increase the entropy of dataset.

# COMMAND ----------

df =pd.read_csv("/dbfs/FileStore/Cleansed_RealEstate.csv", header='infer')
FeatureSelection(df,'Y house price of unit area','Continuous',"/dbfs/FileStore/Cleansed_RealEstate.csv",'RealEstate','FeatureSelection')

# COMMAND ----------

#%pip install ruamel.yaml==0.16.10
#%pip install azure-core==1.8.0
#%pip install liac-arff==2.4.0
#%pip install msal==1.4.3
#%pip install msrest==0.6.18
#%pip install ruamel.yaml.clib==0.2.0
#%pip install tqdm==4.49.0
#%pip install zipp==3.2.0
#%pip install interpret-community==0.15.0
#%pip install azure-identity==1.4.0
#%pip install dotnetcore2==2.1.16
#%pip install jinja2==2.11.2
#%pip install azure-core==1.15.0
#%pip install azure-mgmt-containerregistry==8.0.0
#%pip install azure-mgmt-core==1.2.2
#%pip install distro==1.5.0
#%pip install google-api-core==1.30.0
#%pip install google-auth==1.32.1
#%pip install importlib-metadata==4.6.0
#%pip install msal==1.12.0
#%pip install packaging==20.9
#%pip install pathspec==0.8.1
#%pip install requests==2.25.1
#%pip install ruamel.yaml.clib==0.2.4
#%pip install tqdm==4.61.1
#%pip install zipp==3.4.1
#%pip install scipy==1.5.2
%pip install charset-normalizer==2.0.3
%pip install websocket-client==1.1.0
%pip install scikit-learn==0.22.1
%pip install interpret-community==0.19.0

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7.Auto ML Trigger - after preprocessing
# MAGIC Trigger Azure auto ML, pick the best model so obtained and use it to predict the label column. Calculate the Weighted Absolute Accuracy amd push to telemetry. also obtain the data back in original format by using the unique identifier of each row 'Index' and report Actual v/s Predicted Columns. We also provide the direct link to the azure Portal Run for the current experiment for users to follow.

# COMMAND ----------

# DBTITLE 1,7.Auto ML Trigger - after preprocessing
import pandas as pd
dfclean = pd.read_csv("/dbfs/FileStore/Cleansed_RealEstate.csv", header='infer')

#AutoMLFunc(subscription_id,resource_group,workspace_name,input_dataframe,label_col,task_type,input_appname)
df=AutoMLFunc(<subscription_id>,<resource_group>,<workspace_name>,dfclean,'Y house price of unit area','regression','RealEstate')


# COMMAND ----------

# DBTITLE 1,8.Obtain back data in original format after modelling
##df has just index,y actual, y predicted cols, as rest all cols are encoded after manipulation
for col in df.columns:
  if col not in ["y_predict","y_actual","Index"]: 
    df.drop([col], axis=1, inplace=True)
    
#dataframe is the actual input dataset     
dataframe = pd.read_csv("/dbfs/FileStore/RealEstate.csv", header='infer')

#Merging Actual Input dataframe with AML output df using Index column
dataframe_fin = pd.merge(left=dataframe, right=df, left_on='Index', right_on='Index')

#De-coding the label columns using scaling with actual label input
dataframe_fin['Y house price of unit area_Actual'] = dataframe_fin['Y house price of unit area']
dataframe_fin['Y house price of unit area_Predicted'] = (dataframe_fin['Y house price of unit area'] / dataframe_fin.y_actual)* dataframe_fin.y_predict

#div by zero in error above eqn raises Nans so replace all Nans with 0
dataframe_fin['Y house price of unit area_Predicted'].fillna(0, inplace=True)

# deleting unwanted intermediate columns 
for col in dataframe_fin.columns:
  if col in ["y_predict","y_actual"]: 
    dataframe_fin.drop([col], axis=1, inplace=True)
    
dataframe_fin

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9.Model Interpretation, Feature Importance
# MAGIC We can explore the model by splitting the Model metrics over various cohorts and analyse the data and model performance for each subclass.We can also get Global & Local feature Importance values for the Model.

# COMMAND ----------

# DBTITLE 1,9.1.Feature importance_Global
#Use dfclean i.e. preprocessed as input 
dfclean =pd.read_csv("/dbfs/FileStore/Cleansed_RealEstate.csv", header='infer')
df1 = pd.DataFrame() # in case of local importance pass values

#Choose global/local and if local feed df1 as sample data above or x_test[0:n] range, feed run-id from AML iteration in above cell 
#Feature_Importance(df,subscription_id,resource_group,workspace_name,workspace_region,run_id,iteration,label_col,task,ImportanceType,local_df)
Feature_Importance(dfclean,"3ecb9b6a-cc42-4b0a-9fd1-6c08027eb201","psbidev","psdatainsightsML","Central US",'AutoML_1889dc17-ee61-4f14-b2be-841fdcf3e4c0',1,'Y house price of unit area','regression','global',df1)


# COMMAND ----------

# DBTITLE 1,9.2.Feature importance_Local 
#For Local only
#use brute AML trigger as The Preprocessing cannot be done to the 'data' and hence the fitted model data input and 'Data' sample here not of same type

#Use preprocessed data as input 
df =pd.read_csv("/dbfs/FileStore/Cleansed_RealEstate.csv", header='infer')

#Local df input only X i.e. without the label column
import pandas as pd 
data = [[1000, 2012.917,	32,	84.87882,	10,	24.98298,	121.54024]] 
df1 = pd.DataFrame(data, columns = ['Index', 'X1 transaction date','X2 house age','X3 distance to the nearest MRT station','X4 number of convenience stores','X5 latitude','X6 longitude']) 

#For Local do not sample data or add row_id, or remove Index/Cluster intermediate columns
#Choose global/local and if local feed df1 as sample data above or x_test[0:n] range, feed run-id from AML iteration in above cell 
Feature_Importance(df,"3ecb9b6a-cc42-4b0a-9fd1-6c08027eb201","psbidev","psdatainsightsML","Central US",'AutoML_1889dc17-ee61-4f14-b2be-841fdcf3e4c0',1,'Y house price of unit area','regression','local',df1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10.Telemetry
# MAGIC We can get comparative analysis of experiments via the telemetry captured for each and every step of the model run. 

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from TelemetryTable where Step='regression' and MLKey='RealEstate' order by TimeGenerated desc
