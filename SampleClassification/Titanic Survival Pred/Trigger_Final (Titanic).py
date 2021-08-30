# Databricks notebook source
# MAGIC %md
# MAGIC **PROBLEM STATEMENT**
# MAGIC <br/>Predict the Survival of people from Titanic based on the gender, class, age etc.
# MAGIC <br/>Get Sample data from Source- https://data.world/nrippner/titanic-disaster-dataset
# MAGIC <br/>
# MAGIC <br/>**COLUMN DEFINITION**
# MAGIC <br/>survival - Survival (0 = No; 1 = Yes)
# MAGIC <br/>class - Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
# MAGIC <br/>name - Name
# MAGIC <br/>sex - Sex (Male, Female the dataset is Imbalanced towards Males)
# MAGIC <br/>age - Age
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
# MAGIC <br/>9.Model Interpretation & Error Analysis
# MAGIC <br/>10.Telemetry
# MAGIC <br/>
# MAGIC <br/>**FEATURE ENGINEERING**
# MAGIC <br/>1. Data is Imbalanced with more Males, so Cluster Oversample by 'Sex' Column and then model. This imbalance can be identified via the Data Plots.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import functions from Master Notebook:
# MAGIC Import the Functions and dependencies from the Master notebook to be used in the Trigger Notebook

# COMMAND ----------

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
# MAGIC val filepath= "adl://<Your Datalake storage>.azuredatalakestore.net/Temp/ML-PJC/Titanic.csv"
# MAGIC var df=spark.read.format("csv").option("header", "true").option("delimiter", ",").load(filepath)
# MAGIC //val filepath ="abfss:/.../.parquet"
# MAGIC //var df = spark.read.parquet(filepath)
# MAGIC df.createOrReplaceTempView("vw")

# COMMAND ----------

# DBTITLE 1,Logical transformation of data
# MAGIC %sql
# MAGIC select * from vw

# COMMAND ----------

# DBTITLE 1,Data columns and structural transformation
import pandas as pd
import numpy as np
from pyspark.sql.functions import col

input_dataframe= spark.sql("""select * FROM vw""")
#input_dataframe = pd.read_csv("/dbfs/FileStore/Titanic.csv", header='infer')

cols_string=['Name','PClass','Sex']
cols_int=['Age','Survived']
cols_datetime=[]
cols_Float=[]				

#Function call: DataTypeConversion(input_dataframe,cols_string,cols_int,cols_datetime,cols_Float)
input_dataframe = DataTypeConversion(input_dataframe,cols_string,cols_int,cols_datetime,cols_Float)

##To assign an Index unique identifier of original record from after data massaging
input_dataframe['Index'] = np.arange(len(input_dataframe)) 

#Saving data acquired in dbfs for future use
outdir = '/dbfs/FileStore/Titanic.csv'
input_dataframe.to_csv(outdir, index=False)
#input_dataframe = pd.read_csv("/dbfs/FileStore/Dataframe.csv", header='infer')


# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.Data Exploration
# MAGIC 1.Exploratory Data Analysis (EDA)- To understand the overall data at hand, analysing each feature independently for its' statistics, the correlation and interraction between variables, data sample etc. 
# MAGIC <br/>2.Data Profiling Plots- To analyse the Categorical and Numerical columns separately for any trend in data, biasness in data etc.

# COMMAND ----------

# DBTITLE 1,EDA
input_dataframe = pd.read_csv("/dbfs/FileStore/Titanic.csv", header='infer')

#Function Call: Data_Profiling_viaPandasProfiling(input_dataframe)
p=Data_Profiling_viaPandasProfiling(input_dataframe)
displayHTML(p)


# COMMAND ----------

# DBTITLE 1,Data Profiling Plots
input_dataframe = pd.read_csv("/dbfs/FileStore/Titanic.csv", header='infer')

#User Inputs
cols_all=['Name','PClass','Sex','Age','Survived']
Categorical_cols=['Name','PClass','Sex']
Numeric_cols=['Age','Survived']
Label_col='Survived'

#Data_Profiling_Plots(input_dataframe,Categorical_cols,Numeric_cols,Label_col)
Data_Profiling_Plots(input_dataframe,Categorical_cols,Numeric_cols,Label_col)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.Cleansing
# MAGIC To clean the data from NULL values, fix structural errors in columns, drop empty columns, encode the categorical values, normalise the data to bring to the same scale. We also check the Data Distribution via Correlation heatmap of original input dataset v/s the Cleansed dataset to validate whether or not the transformations hampered the original data trend/density.

# COMMAND ----------

# DBTITLE 1,3.Auto Cleanser 
subsample_final = pd.read_csv("/dbfs/FileStore/Titanic.csv", header='infer')
filepath="/dbfs/FileStore/Titanic.csv"
#subsample_final=subsample_final.drop(['Index'], axis = 1) # Index is highest variability column hence always imp along PC but has no business value. You can append columns to be dropped by your choice here in the list


inputdf_new=autodatacleaner(subsample_final,filepath,"Titanic","Data Cleanser")
print("Total rows in the new pandas dataframe:",len(inputdf_new.index))

#persist cleansed data sets 
filepath1 = '/dbfs/FileStore/Cleansed_Titanic.csv'
inputdf_new.to_csv(filepath1, index=False)


# COMMAND ----------

# DBTITLE 1,Data profiling(Heatmap correlation)- User input dataframe
original = pd.read_csv("/dbfs/FileStore/Titanic.csv", header='infer')
display(Data_Profiling_Fin(original))

# COMMAND ----------

# DBTITLE 1,Data profiling(Heatmap correlation)- Cleansed dataframe
Cleansed=pd.read_csv("/dbfs/FileStore/Cleansed_Titanic.csv", header='infer')

display(Data_Profiling_Fin(Cleansed))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.Sampling
# MAGIC Perform Stratified, Systematic, Random, Cluster sampling over data and compare the so obtained sampled dataset with the original data using a NULL Hypothesis, and suggest the best sample obtained thus. Compare the data densities of sampled datasets with that of the original input dataset to validate that our sample matches the data trend of original set.

# COMMAND ----------

# DBTITLE 1,Sampling 
input_dataframe = pd.read_csv("/dbfs/FileStore/Cleansed_Titanic.csv", header='infer') ## Sample after cleansing so that all categorical cols converted to num and hence no chi test. chi test requires the total of observed and tot of original sample to be same in frequency. 
filepath="/dbfs/FileStore/Cleansed_Titanic.csv"
subsample_final = pd.DataFrame()
subsample1 = pd.DataFrame()
subsample2 = pd.DataFrame()
subsample3 = pd.DataFrame()
subsample4 = pd.DataFrame()

#Function Call: Sampling(input_dataframe,filepath,task_type,input_appname,cluster_classified_col_ifany(Supervised))
subsample_final,subsample1,subsample2,subsample3,subsample4=Sampling(input_dataframe,filepath,'Sampling','Titanic','Sex')

#persist sampled data sets 
filepath1 = '/dbfs/FileStore/StratifiedSampled_Titanic.csv'
subsample1.to_csv(filepath1, index=False)
filepath2 = '/dbfs/FileStore/RandomSampled_Titanic.csv'
subsample2.to_csv(filepath2, index=False)
filepath3 = '/dbfs/FileStore/SystematicSampled_Titanic.csv'
subsample3.to_csv(filepath3, index=False)
filepath4 = '/dbfs/FileStore/ClusterSampled_Titanic.csv'
subsample4.to_csv(filepath4, index=False)
filepath = '/dbfs/FileStore/subsample_final_Titanic.csv'
subsample_final.to_csv(filepath, index=False)

# COMMAND ----------

# DBTITLE 1,Data Distribution (Histogram)- Input dataset
original = pd.read_csv("/dbfs/FileStore/Titanic.csv", header='infer')

display(display_DataDistribution(original,'Survived'))

# COMMAND ----------

# DBTITLE 1,Data Distribution (Histogram)- Stratified Sampled dataset
subsample1 = pd.read_csv("/dbfs/FileStore/StratifiedSampled_Titanic.csv", header='infer')

display(display_DataDistribution(subsample1,'Survived'))

# COMMAND ----------

# DBTITLE 1,Data Distribution (Histogram)- Random Sampled dataset
subsample2 = pd.read_csv("/dbfs/FileStore/RandomSampled_Titanic.csv", header='infer')

display(display_DataDistribution(subsample2,'Survived'))

# COMMAND ----------

# DBTITLE 1,Data Distribution (Histogram)- Systematic Sampled dataset
subsample3 = pd.read_csv("/dbfs/FileStore/SystematicSampled_Titanic.csv", header='infer')

display(display_DataDistribution(subsample3,'Survived'))

# COMMAND ----------

# DBTITLE 1,Data Distribution (Histogram)- Cluster Sampled dataset
subsample4 = pd.read_csv("/dbfs/FileStore/ClusterSampled_Titanic.csv", header='infer')

display(display_DataDistribution(subsample4,'Survived'))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.Anomaly Detection
# MAGIC Iterate data over various Anomaly-detection techniques and estimate the number of Inliers and Outliers for each.

# COMMAND ----------

# DBTITLE 0,5.Anomaly Detection
#Calling the Anamoly Detection Function for identifying outliers  
outliers_fraction = 0.05
#df =pd.read_csv("/dbfs/FileStore/subsample_final_Titanic.csv", header='infer')
df =pd.read_csv("/dbfs/FileStore/ClusterSampled_Titanic.csv", header='infer')
target_variable = 'Survived'
variables_to_analyze='Sex'

AnomalyDetection(df,target_variable,variables_to_analyze,outliers_fraction,'anomaly_test','Titanic')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.Feature Selection
# MAGIC Perform feature selection on the basis of Feature Importance ranking, correlation values, variance within the column.
# MAGIC Choose features with High Importance value score, drop one of the two highly correlated features, drop features which offer zero variability to data and thus do not increase the entropy of dataset.

# COMMAND ----------

# DBTITLE 0,Feature Selection
import pandas as pd
import numpy as np
#input_dataframe = pd.read_csv("/dbfs/FileStore/RealEstate.csv", header='infer')
#label_col='Y house price of unit area'
#filepath="/dbfs/FileStore/RealEstate.csv"
#input_appname='RealEstate'
#task_type='FeatureSelectionCleansing'
#Y_discrete='Continuous'

input_dataframe = pd.read_csv("/dbfs/FileStore/Cleansed_Titanic.csv", header='infer')
label_col='Survived'
filepath='/dbfs/FileStore/Cleansed_Titanic.csv'
input_appname='Titanic'
task_type='FeatureSelectionCleansing'
Y_discrete='Categorical'

FeatureSelection(input_dataframe,label_col,Y_discrete,filepath,input_appname,task_type)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7.Auto ML Trigger - after preprocessing
# MAGIC Trigger Azure auto ML, pick the best model so obtained and use it to predict the label column. Calculate the Weighted Absolute Accuracy amd push to telemetry. also obtain the data back in original format by using the unique identifier of each row 'Index' and report Actual v/s Predicted Columns. We also provide the direct link to the azure Portal Run for the current experiment for users to follow.

# COMMAND ----------

# DBTITLE 1,7.Auto ML Trigger - after preprocessing
import pandas as pd
dfclean = pd.read_csv("/dbfs/FileStore/Cleansed_Titanic.csv", header='infer')

#AutoMLFunc(subscription_id,resource_group,workspace_name,input_dataframe,label_col,task_type,input_appname)
df=AutoMLFunc('3ecb9b6a-cc42-4b0a-9fd1-6c08027eb201','psbidev','psdatainsightsML',dfclean,'Survived','classification','Titanic')


# COMMAND ----------

# DBTITLE 1,8.Obtain back data in original format after modelling-Classification
##df has just index,y actual, y predicted cols, as rest all cols are encoded after manipulation
for col in df.columns:
  if col not in ["y_predict","y_actual","Index"]: 
    df.drop([col], axis=1, inplace=True)
    
#dataframe is the actual input dataset     
dataframe = pd.read_csv("/dbfs/FileStore/Titanic.csv", header='infer')

#Merging Actual Input dataframe with AML output df using Index column
dataframe_fin = pd.merge(left=dataframe, right=df, left_on='Index', right_on='Index')
dataframe_fin

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9.Model Interpretation, Feature Importance, Error Analysis
# MAGIC We can explore the model by splitting the Model metrics over various cohorts and analyse the data and model performance for each subclass.We can also get Global & Local feature Importance values for the Model.

# COMMAND ----------

# DBTITLE 1,Model interpretation
df = pd.read_csv("/dbfs/FileStore/Cleansed_Titanic.csv", header='infer')
label_col='Survived'
subscription_id='3ecb9b6a-cc42-4b0a-9fd1-6c08027eb201'
resource_group='psbidev'
workspace_name='psdatainsightsML'
run_id='AutoML_45a82620-d605-4643-8a1b-8055e32ffd9b'
iteration=1
task='classification'

ModelInterpret(df,label_col,subscription_id,resource_group,workspace_name,run_id,iteration,task)

# COMMAND ----------

# DBTITLE 1,Error Analysis
df = pd.read_csv("/dbfs/FileStore/Cleansed_Titanic.csv", header='infer')
label_col='Survived'
subscription_id='3ecb9b6a-cc42-4b0a-9fd1-6c08027eb201'
resource_group='psbidev'
workspace_name='psdatainsightsML'
run_id='AutoML_45a82620-d605-4643-8a1b-8055e32ffd9b'
iteration=1
task='classification'

ErrorAnalysisDashboard(df,label_col,subscription_id,resource_group,workspace_name,run_id,iteration,task)
