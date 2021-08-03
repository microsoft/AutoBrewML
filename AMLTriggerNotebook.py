# Databricks notebook source
# DBTITLE 1,Import Functions
# MAGIC %run .../.../AMLMasterNotebook


# COMMAND ----------

# DBTITLE 1,1.Acquisition - from datasource ADLS path
# MAGIC %scala
# MAGIC //<USER INPUT FILEPATH PARQUET OR CSV>
# MAGIC 
# MAGIC val filepath= "adl://<Your ADLS Name>.azuredatalakestore.net/Temp/ML-PJC/RealEstateData.csv"
# MAGIC var df=spark.read.format("csv").option("header", "true").option("delimiter", ",").load(filepath)
# MAGIC //val filepath ="abfss:/.../.parquet"
# MAGIC //var df = spark.read.parquet(filepath)
# MAGIC df.createOrReplaceTempView("vw")

# COMMAND ----------

# DBTITLE 1,Acquisition - by transforming columns into required datatypes, introducing an Index identifier, converting to pandas df, persisiting actual dataset
import pandas as pd
import numpy as np
from pyspark.sql.functions import col

input_dataframe= spark.sql("""select * FROM vw""")

cols_string=[]
cols_int=['Index']
cols_datetime=[]
cols_Float=['X1 transaction date','X2 house age','X3 distance to the nearest MRT station','X4 number of convenience stores','X5 latitude','X6 longitude','Y house price of unit area']

#Function call: DataTypeConversion(input_dataframe,cols_string,cols_int,cols_datetime,cols_Float)
input_dataframe = DataTypeConversion(input_dataframe,cols_string,cols_int,cols_datetime,cols_Float)

##To assign an Index unique identifier of original record from after data massaging
#input_dataframe['Index'] = np.arange(len(input_dataframe)) 

#Saving data acquired in dbfs for future use
outdir = '/dbfs/FileStore/RealEstate.csv'
input_dataframe.to_csv(outdir, index=False)
#input_dataframe = pd.read_csv("/dbfs/FileStore/Dataframe.csv", header='infer')


# COMMAND ----------

# DBTITLE 1,2.Data Profiling 
input_dataframe = pd.read_csv("/dbfs/FileStore/RealEstate.csv", header='infer')

#Function Call: Data_Profiling_viaPandasProfiling(input_dataframe)
p=Data_Profiling_viaPandasProfiling(input_dataframe)
displayHTML(p)


# COMMAND ----------

# DBTITLE 1,3.Sampling 
input_dataframe = pd.read_csv("/dbfs/FileStore/RealEstate.csv", header='infer')
filepath="/dbfs/FileStore/RealEstate.csv"
subsample_final = pd.DataFrame()
subsample1 = pd.DataFrame()
subsample2 = pd.DataFrame()
subsample3 = pd.DataFrame()
subsample4 = pd.DataFrame()

#Function Call: Sampling(input_dataframe,filepath,task_type,input_appname,cluster_classified_col_ifany(Supervised))
subsample_final,subsample1,subsample2,subsample3,subsample4=Sampling(input_dataframe,filepath,'Sampling','RealEstate','NULL')

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

# DBTITLE 1,Sampling -With Cluster Sampling
# MAGIC %scala
# MAGIC //<USER INPUT FILEPATH PARQUET OR CSV>
# MAGIC 
# MAGIC val filepath= "adl://<Your ADLS Name>.azuredatalakestore.net/Temp/ML-PJC/glass.csv"
# MAGIC var df=spark.read.format("csv").option("header", "true").option("delimiter", ",").load(filepath)
# MAGIC //val filepath ="abfss:/.../.parquet"
# MAGIC //var df = spark.read.parquet(filepath)
# MAGIC df.createOrReplaceTempView("vw2")

# COMMAND ----------

import pandas as pd
import numpy as np
from pyspark.sql.functions import col

input_dataframe2= spark.sql("""select * FROM vw2""")

cols_string=[]
cols_int=['Y']
cols_datetime=[]
cols_Float=['X1','X2','X3','X4','X5','X6']

#Function call: DataTypeConversion(input_dataframe,cols_string,cols_int,cols_datetime,cols_Float)
input_dataframe2 = DataTypeConversion(input_dataframe2,cols_string,cols_int,cols_datetime,cols_Float)

##To assign an Index unique identifier of original record from after data massaging
input_dataframe2['Index'] = np.arange(len(input_dataframe2)) 

#Saving data acquired in dbfs for future use
outdir = '/dbfs/FileStore/glass.csv'
input_dataframe2.to_csv(outdir, index=False)
#input_dataframe = pd.read_csv("/dbfs/FileStore/Dataframe.csv", header='infer')


# COMMAND ----------

input_dataframe = pd.read_csv("/dbfs/FileStore/glass.csv", header='infer')
filepath="/dbfs/FileStore/glass.csv"
subsample_final = pd.DataFrame()
subsample1 = pd.DataFrame()
subsample2 = pd.DataFrame()
subsample3 = pd.DataFrame()
subsample4 = pd.DataFrame()

#Sampling(input_dataframe,task_type,input_appname,cluster_classified_col_ifany(Supervised))
subsample_final,subsample1,subsample2,subsample3,subsample4=Sampling(input_dataframe,filepath,'Sampling','ChemicalGlass','Y')

#persist sampled data sets 
filepath1 = '/dbfs/FileStore/StratifiedSampled_glass.csv'
subsample1.to_csv(filepath1, index=False)
filepath2 = '/dbfs/FileStore/RandomSampled_glass.csv'
subsample2.to_csv(filepath2, index=False)
filepath3 = '/dbfs/FileStore/SystematicSampled_glass.csv'
subsample3.to_csv(filepath3, index=False)
filepath4 = '/dbfs/FileStore/ClusterSampled_glass.csv'
subsample4.to_csv(filepath4, index=False)
filepath = '/dbfs/FileStore/subsample_final_glass.csv'
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

# DBTITLE 1,Telemetry- Sampling 
# MAGIC %sql
# MAGIC select * from amltelemetry order by timegenerated ASC

# COMMAND ----------

# DBTITLE 1,4.Auto Cleanser 
subsample_final = pd.read_csv("/dbfs/FileStore/subsample_final_RealEstate.csv", header='infer')
#subsample_final=subsample_final.drop(['Index'], axis = 1) # Index is highest variability column hence always imp along PC but has no business value. You can append columns to be dropped by your choice here in the list


inputdf_new=autodatacleaner(subsample_final,"RealEstate","Data Cleanser")
print("Total rows in the new pandas dataframe:",len(inputdf_new.index))

#persist cleansed data sets 
filepath1 = '/dbfs/FileStore/Cleansed_RealEstate.csv'
inputdf_new.to_csv(filepath1, index=False)


inputdf_new=autodatacleaner(subsample_final,"RealEstate","Data Cleanser")
print("Total rows in the new pandas dataframe:",len(inputdf_new.index))

# COMMAND ----------

# DBTITLE 1,Telemetry- Cleansing
# MAGIC %sql
# MAGIC select * from amltelemetry where TimeGenerated>'1601711630'

# COMMAND ----------

# DBTITLE 1,Data profiling(Heatmap correlation)- User input dataframe
subsample_final = pd.read_csv("/dbfs/FileStore/subsample_final_RealEstate.csv", header='infer')

display(Data_Profiling_Fin(subsample_final))

# COMMAND ----------

# DBTITLE 1,Data profiling(Heatmap correlation)- Cleansed dataframe
Cleansed=pd.read_csv("/dbfs/FileStore/Cleansed_RealEstate.csv", header='infer')

display(Data_Profiling_Fin(Cleansed))

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from amltelemetry order by TimeGenerated desc



# COMMAND ----------

# DBTITLE 1,5.Anomaly Detection
#Calling the Anamoly Detection Function for identifying outliers 
outliers_fraction = 0.05
df =pd.read_csv("/dbfs/FileStore/Cleansed_RealEstate.csv", header='infer')
target_variable = 'Y house price of unit area'
variables_to_analyze='X3 distance to the nearest MRT station'

AnomalyDetection(df,target_variable,variables_to_analyze,outliers_fraction,'anomaly_test','test')

# COMMAND ----------

# DBTITLE 1,6.PCA - Dimensionality deduction only
df = pd.read_csv("/dbfs/FileStore/Cleansed_RealEstate.csv", header='infer')

#PrinCompAnalysis(df,InputAppName,Task)
dataframe=PrinCompAnalysis(df,"RealEstate","PCA")
dataframe

# COMMAND ----------

#%pip install ruamel.yaml==0.16.10
#%pip install azure-core==1.8.0
#%pip install interpret-community==0.14.3
#%pip install liac-arff==2.4.0
#%pip install msal==1.4.3
#%pip install msrest==0.6.18
#%pip install ruamel.yaml.clib==0.2.0
#%pip install tqdm==4.49.0
#%pip install zipp==3.2.0
#%pip install interpret-community==0.15.0
#%pip install azure-identity==1.4.0
#%pip install dotnetcore2==2.1.16

# COMMAND ----------

# DBTITLE 1,7.Auto ML Trigger - after preprocessing
dfclean = pd.read_csv("/dbfs/FileStore/Cleansed_RealEstate.csv", header='infer')

#AutoMLFunc(subscription_id,resource_group,workspace_name,input_dataframe,label_col,task_type,input_appname)
df=AutoMLFunc('<Enter your Azure ML subscription_id>','<Enter your Azure ML resource_group>','<Enter your Azure ML workspace_name>',dfclean,'Y house price of unit area','regression','RealEstate')


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

# DBTITLE 1,9.1.Feature importance_Global
#Use dfclean i.e. preprocessed as input 
dfclean =pd.read_csv("/dbfs/FileStore/Cleansed_RealEstate.csv", header='infer')
df1 = pd.DataFrame() # in case of local importance pass values

#Choose global/local and if local feed df1 as sample data above or x_test[0:n] range, feed run-id from AML iteration in above cell 
#Feature_Importance(df,subscription_id,resource_group,workspace_name,workspace_region,run_id,iteration,label_col,task,ImportanceType,local_df)
Feature_Importance(dfclean,'<Enter your Azure ML subscription_id>','<Enter your Azure ML resource_group>','<Enter your Azure ML workspace_name>',"<Enter the location of your Azure ML ws>",'<Enter your Azure Auto ML Run ID>',1,'Y house price of unit area','regression','global',df1)

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
Feature_Importance(dfclean,'<Enter your Azure ML subscription_id>','<Enter your Azure ML resource_group>','<Enter your Azure ML workspace_name>',"<Enter the location of your Azure ML ws>",'<Enter your Azure Auto ML Run ID>',1,'Y house price of unit area','regression','local',df1)

# COMMAND ----------

# DBTITLE 1,***10.Auto ML Trigger - brute without preprocessing
df = pd.read_csv("/dbfs/FileStore/RealEstate.csv", header='infer')

#AutoMLFunc(subscription_id,resource_group,workspace_name,input_dataframe,label_col,task_type,input_appname)
df=AutoMLFunc('<Enter your Azure ML subscription_id>','<Enter your Azure ML resource_group>','<Enter your Azure ML workspace_name>',df,'Y house price of unit area','regression','RealEstate')

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from amltelemetry where Step='regression' and MLKey='RealEstate' --order by TimeGenerated desc

# COMMAND ----------

# DBTITLE 1,***11.Manual ML Trigger with Auto-Correction to AML 
dfManual = pd.read_csv("/dbfs/FileStore/Cleansed_RealEstate.csv", header='infer')
Accuracy_score=Manual(dfManual,'Y house price of unit area','regression','RealEstate')

#MANUAL ACCURACY
df1 = pd.DataFrame({"Accuracy_score":[Accuracy_score]}) 
val1 = str(df1['Accuracy_score'].values[0])
#print(val1)

#ACCURACY LAST TRIGGER
df = sqlContext.sql("""SELECT split(Results, '=')[1] as Accuracy_score
From amltelemetry 
WHERE TimeGenerated!=(SELECT TimeGenerated 
                 FROM amltelemetry
                 where MLKey='RealEstate' and Step='regression'
                 ORDER BY TimeGenerated DESC 
                 LIMIT 1)
and MLKey='RealEstate' and Step='regression'                 
ORDER BY TimeGenerated DESC 
LIMIT 1""")
df2 = df.toPandas()
val2 = str(df2['Accuracy_score'].values[0])
#print(val2)

#IF LAST RUN WAS BETTER THAN THIS RUN THEN RETRIGGER AML
if val1 < val2:
  dfAutoCorrect = pd.read_csv("/dbfs/FileStore/Cleansed_RealEstate.csv", header='infer') 
  #print(pandas_df)
  df=AutoMLFunc('<Enter your Azure ML subscription_id>','<Enter your Azure ML resource_group>','<Enter your Azure ML workspace_name>',dfAutoCorrect,'Y house price of unit area','regression','RealEstate')
else:
  pass 


# COMMAND ----------

# MAGIC %sql
# MAGIC select * from amltelemetry where Step='regression' order by TimeGenerated asc

# COMMAND ----------

# DBTITLE 1,***12.Auto ML Trigger-PCA_DimensionReduction(Cannot follow beyond step 6 as data changed after PCA Transform)
#AutoMLFunc(subscription_id,resource_group,workspace_name,input_dataframe,label_col,task_type,input_appname)
dfPCA = pd.read_csv("/dbfs/FileStore/Cleansed_RealEstate.csv", header='infer')

df=AutoMLFuncPCA('<Enter your Azure ML subscription_id>','<Enter your Azure ML resource_group>','<Enter your Azure ML workspace_name>',dfPCA,'Y house price of unit area','regression','RealEstate')
df

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from amltelemetry where Step='regression' order by TimeGenerated asc

# COMMAND ----------

# DBTITLE 1,***13.Auto ML Trigger -service principal auth (Bypass AML Authentication)
#AML TRIGGER WITHOUT BYPASSING AUTHENTICATION
dfclean = pd.read_csv("/dbfs/FileStore/Cleansed_RealEstate.csv", header='infer')

#AML TRIGGER BYPASSING AUTHENTICATION
#AutoMLFunc(subscription_id,resource_group,workspace_name,svc_pr_password,tenant_id,service_principal_id,input_dataframe,label_col,task_type,input_appname)

subscription_id= "<Enter your Azure ML subscription_id>"
resource_group= "<Enter your Azure ML resource_group>"
workspace_name= "<Enter your Azure ML workspace_name>"
service_principal_password= "<Enter your Service Principal Pwd>"
tenant_id= "<Enter your Azure Tenant ID>"
service_principal_id= "<Enter your Service Principal ID>"

df=AutoMLFuncSP(subscription_id,resource_group,workspace_name,service_principal_password,tenant_id,service_principal_id,dfclean,'Y house price of unit area','regression','RealEstate')

# DBTITLE 1,***13.Feature Selection
input_dataframe = pd.read_csv("/dbfs/FileStore/RealEstate.csv", header='infer')
label_col='Y house price of unit area'
filepath="/dbfs/FileStore/RealEstate.csv"
input_appname='RealEstate'
task_type='FeatureSelectionCleansing'
Y_discrete='Continuous'
FeatureSelection(input_dataframe,label_col,Y_discrete,filepath,input_appname,task_type)

# COMMAND ----------

# DBTITLE 1,6.Logging information
# MAGIC %sql 
# MAGIC select * from  amltelemetry where TimeGenerated > '1596353701' order by TimeGenerated desc 
