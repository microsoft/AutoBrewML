# Databricks notebook source
# DBTITLE 1,Problem Definition
# MAGIC %md
# MAGIC **PROBLEM STATEMENT**
# MAGIC <br/>Predict the Life Expectancy of countries depending on the social/economic/health factors taken between the years 2000-2015.
# MAGIC <br/>Get Sample data from Source- https://www.kaggle.com/kumarajarshi/life-expectancy-who
# MAGIC <br/>
# MAGIC <br/>**COLUMN DEFINITION**
# MAGIC <br/>'Country'-Country where record taken
# MAGIC <br/>,'Year'-Year when record taken 2000-2015 i.e. 15 yrs data. Each years data for all countries, ie 16 records per country
# MAGIC <br/>,'Status'-Developed or Developing status, skewed towards Developing
# MAGIC <br/>,'Life expectancy '-Life Expectancy in age or age till when people live, maximum people in bucket of 70-80 yrs age group
# MAGIC <br/>,'Adult Mortality'-Adult Mortality Rates of both sexes (probability of dying between 15 and 60 years) (out of per 1000 population)
# MAGIC <br/>,'infant deaths'-Number of Infant Deaths (out of per 1000 population)
# MAGIC <br/>,'Alcohol'-Alcohol consumption in litres recorded per capita (15+) consumption 
# MAGIC <br/>,'percentage expenditure'-Expenditure on health as a percentage of Gross Domestic Product per capita(%)-??? percentage then how come value in thousand ranges
# MAGIC <br/>,'Hepatitis B'-Hepatitis B immunization coverage among 1-year-olds (out of 100%)
# MAGIC <br/>,'Measles '-number of reported cases (out of per 1000 population)
# MAGIC <br/>,' BMI '-Average Body Mass Index of entire population
# MAGIC <br/>,'under-five deaths '-Number of under five yrs age deaths (out of per 1000 population)
# MAGIC <br/>,'Polio'-Polio (Pol3) immunization coverage among 1-year-olds (out of 100%)
# MAGIC <br/>,'Total expenditure'-General government expenditure on health as a percentage of total government expenditure (out of 100%)
# MAGIC <br/>,'Diphtheria '-DTP3 immunization coverage among 1-year-olds (out of 100%)
# MAGIC <br/>,' HIV/AIDS'-Deaths per 1000 live births (0-4 years) due to HIV (out of per 1000 population)
# MAGIC <br/>,'GDP'-Gross Domestic Product per capita (in USD)
# MAGIC <br/>,'Population'-Population of the country
# MAGIC <br/>,' thinness  1-19 years'-Prevalence of thinness among children and adolescents for Age 10 to 19 (out of 100%)
# MAGIC <br/>,' thinness 5-9 years'-Prevalence of thinness among children for Age 5 to 9 (out of 100%)
# MAGIC <br/>,'Income composition of resources'-Human Development Index in terms of income composition of resources (index ranging from 0 to 1)
# MAGIC <br/>,'Schooling'-Number of years of Schooling (years)
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
# MAGIC <br/>**LOGICAL SCALING OF COLS**
# MAGIC <br/>col1=column/TotalPopulation *1000---> Out of 1000 people data
# MAGIC <br/>col2=column/TotalPopulation *100---> Out of 100 people data or % data
# MAGIC <br/>Bring everything to same scale. i.e Percentage data so div col1/10
# MAGIC <br/>We should not bring data at scale as Absolute count out of total population because the total population is different for different countries so the comparison would not be fair.
# MAGIC <br/>
# MAGIC <br/>**FEATURE ENGINEERING**
# MAGIC <br/>Summation of features in % eg: Immunization='Hepatitis B'+ 'Polio'+ 'Diphtheria'
# MAGIC <br/>X%           Y%          Z%          ---% of tot populations
# MAGIC <br/>x/T*100      y/T*100     y/T*100     ---x,y,z is the number of people out of T total population
# MAGIC <br/>avg no. of people of total categories=(x+y+z)/3
# MAGIC <br/>% of avg no. of people               =((x+y+z)/3)/T *100
# MAGIC <br/>                                     =1/3 * ((x+y+z)/100)
# MAGIC <br/>                                     =1/3 * ((x/T + y/T + z/T)*100)
# MAGIC <br/>                                     =1/3 * (X% +Y% +Z%)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import functions from Master Notebook:
# MAGIC Import the Functions and dependencies from the Master notebook to be used in the Trigger Notebook

# COMMAND ----------

# DBTITLE 0,Import Functions
# MAGIC %run .../.../AMLMasterNotebook

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.Data Acquisition
# MAGIC 1.Acquisition of data from datasource ADLS path in CSV/Parquet/JSON etc format.
# MAGIC <br/>2.Logical Transformations in data- For the WHO data we get features in mixed scales of Percentage out of Total Population & out of 1000 people, so we make everything to a scale of out of 100% scale for uniformity in data features values. 
# MAGIC <br/>3.Transforming columns into required datatypes, converting to pandas df, persisiting actual dataset, intoducing a column 'Index' to assign a unique identifier to each dataset row so that this canm be used to retrieve back the original form after any data manupulations. 

# COMMAND ----------

# DBTITLE 1,Read Data from Lake
# MAGIC %scala
# MAGIC //<USER INPUT FILEPATH PARQUET OR CSV>
# MAGIC 
# MAGIC val filepath= "adl://<Your ADLS Name>.azuredatalakestore.net/.../WHO.csv"
# MAGIC var df=spark.read.format("csv").option("header", "true").option("delimiter", ",").load(filepath)
# MAGIC //val filepath ="abfss:/.../.parquet"
# MAGIC //var df = spark.read.parquet(filepath)
# MAGIC df.createOrReplaceTempView("vw")

# COMMAND ----------

# DBTITLE 1,Logical transformation of data
df= spark.sql("""With Original as
(select 
`Country` AS  Country
,`Year` AS  Year
,`Status` AS  Status
,`Life expectancy ` AS  LifeExpectancy
,`Adult Mortality` AS  AdultMortality
,`infant deaths` AS  InfantDeaths
,`Alcohol` AS  Alcohol
,`percentage expenditure` AS  PercentageExpenditure
,`Hepatitis B` AS  HepatitisB
,`Measles ` AS  Measles
,` BMI ` AS  Bmi
,`under-five deaths ` AS  UnderFiveDeaths
,`Polio` AS  Polio
,`Total expenditure` AS  TotalExpenditure
,`Diphtheria ` AS  Diphtheria
,` HIV/AIDS` AS  HivAids
,`GDP` AS  Gdp
,`Population` AS  Population
,` thinness  1-19 years` AS  Thinness1_19Years
,` thinness 5-9 years` AS  Thinness5_9Years
,`Income composition of resources` AS  IncomeCompositionOfResources
,`Schooling` AS  Schooling
From vw
)
Select 
Country
,Year
,Status
,LifeExpectancy
,ROUND(AdultMortality/10,2) AS AdultMortality
,ROUND(InfantDeaths/10,2) AS InfantDeaths
,Alcohol
,PercentageExpenditure
,HepatitisB
,ROUND(Measles/10,2) AS Measles
,Bmi
,ROUND(UnderFiveDeaths/10,2) AS UnderFiveDeaths
,Polio
,TotalExpenditure
,Diphtheria
,ROUND(HivAids/10,2) AS HivAids
,Gdp
,Population
,Thinness1_19Years
,Thinness5_9Years
,IncomeCompositionOfResources
,Schooling
,ROUND((HepatitisB + Polio + Diphtheria)/3,2) AS Immunization_perc
,ROUND((InfantDeaths/10 + UnderFiveDeaths/10 + HivAids/10 + AdultMortality/10)/3,2) AS Mortality_perc
,ROUND((PercentageExpenditure + TotalExpenditure)/3,2) AS EconomicInvestment_perc
FROM Original""")

# COMMAND ----------

# DBTITLE 1,Data columns and structural transformation
import pandas as pd
import numpy as np
from pyspark.sql.functions import col
# <USER INPUT COLUMN NAMES WITH DATAYPES IN RESPECTIVE BUCKET>

All_Cols=['Country'
,'Year'
,'Status'
,'LifeExpectancy'
,'AdultMortality'
,'InfantDeaths'
,'Alcohol'
,'PercentageExpenditure'
,'HepatitisB'
,'Measles'
,'Bmi'
,'UnderFiveDeaths'
,'Polio'
,'TotalExpenditure'
,'Diphtheria'
,'HivAids'
,'Gdp'
,'Population'
,'Thinness1_19Years'
,'Thinness5_9Years'
,'IncomeCompositionOfResources'
,'Schooling'
,'Immunization_perc'
,'Mortality_perc'
,'EconomicInvestment_perc'
]
cols_string=['Country'
,'Year'
,'Status'
]
cols_int=[]
cols_bool=[]
cols_Float=['LifeExpectancy'
,'AdultMortality'
,'InfantDeaths'
,'Alcohol'
,'PercentageExpenditure'
,'HepatitisB'
,'Measles'
,'Bmi'
,'UnderFiveDeaths'
,'Polio'
,'TotalExpenditure'
,'Diphtheria'
,'HivAids'
,'Gdp'
,'Population'
,'Thinness1_19Years'
,'Thinness5_9Years'
,'IncomeCompositionOfResources'
,'Schooling'
,'Immunization_perc'
,'Mortality_perc'
,'EconomicInvestment_perc'
 ]

for col_name in cols_int:
    df = df.withColumn(col_name, col(col_name).cast('Int'))  
for col_name in cols_Float:
    df = df.withColumn(col_name, col(col_name).cast('float')) 
for col_name in cols_bool:
    df = df.withColumn(col_name, col(col_name).cast('bool')) 
    

#Convert Spark df to pandas
input_dataframe = df.toPandas()

#Add a unique identifier to each row
input_dataframe['Index'] = np.arange(len(input_dataframe))


# Column names: remove white spaces and convert to Camel case
#input_dataframe.columns= input_dataframe.columns.str.strip().str.title().str.replace(' ', '')
#print(input_dataframe.columns)

outdir = '/dbfs/FileStore/who.csv'
input_dataframe.to_csv(outdir, index=False)
#input_dataframe = pd.read_csv("/dbfs/FileStore/Dataframe.csv", header='infer')
input_dataframe

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.Data Exploration
# MAGIC 1.Exploratory Data Analysis (EDA)- To understand the overall data at hand, analysing each feature independently for its' statistics, the correlation and interraction between variables, data sample etc. 
# MAGIC <br/>2.Data Profiling Plots- To analyse the Categorical and Numerical columns separately for any trend in data, biasness in data etc.

# COMMAND ----------

# DBTITLE 1,EDA
input_dataframe = pd.read_csv("/dbfs/FileStore/who.csv", header='infer')

#p=Data_Profiling_viaPandasProfiling(input_dataframe)
#displayHTML(p)

Data_Profiling_viaPandasProfiling(input_dataframe)

# COMMAND ----------

# DBTITLE 1,Data Profiling Plots
input_dataframe = pd.read_csv("/dbfs/FileStore/who.csv", header='infer')

#User Inputs
Categorical_cols=['Country'
,'Year'
,'Status']
Numeric_cols=['LifeExpectancy'
,'AdultMortality'
,'InfantDeaths'
,'Alcohol'
,'PercentageExpenditure'
,'HepatitisB'
,'Measles'
,'Bmi'
,'UnderFiveDeaths'
,'Polio'
,'TotalExpenditure'
,'Diphtheria'
,'HivAids'
,'Gdp'
,'Population'
,'Thinness1_19Years'
,'Thinness5_9Years'
,'IncomeCompositionOfResources'
,'Schooling'
,'Immunization_perc'
,'Mortality_perc'
,'EconomicInvestment_perc'
]
Label_col='LifeExpectancy'

#Data_Profiling_Plots(input_dataframe,Categorical_cols,Numeric_cols,Label_col)
Data_Profiling_Plots(input_dataframe,Categorical_cols,Numeric_cols,Label_col)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.Cleansing
# MAGIC To clean the data from NULL values, fix structural errors in columns, drop empty columns, encode the categorical values, normalise the data to bring to the same scale. We also check the Data Distribution via Correlation heatmap of original input dataset v/s the Cleansed dataset to validate whether or not the transformations hampered the original data trend/density.

# COMMAND ----------

df = pd.read_csv("/dbfs/FileStore/who.csv", header='infer')
#df=df.drop(['Index'], axis = 1) # Index is highest variability column hence always imp along PC but has no business value. You can append columns to be dropped by your choice here in the list

#autodatacleaner(inputdf,filepath,input_appname,task_type)
inputdf_new=autodatacleaner(df,"/dbfs/FileStore/who.csv","WHO","Data Cleanser")
print("Total rows in the new pandas dataframe:",len(inputdf_new.index))

#persist cleansed data sets 
filepath1 = '/dbfs/FileStore/Cleansed_WHO.csv'
inputdf_new.to_csv(filepath1, index=False)



# COMMAND ----------

# DBTITLE 1,Data Distribution (Heatmap)- Input dataset
Cleansed=pd.read_csv("/dbfs/FileStore/who.csv", header='infer')

display(Data_Profiling_Fin(Cleansed))

# COMMAND ----------

# DBTITLE 1,Data Distribution (Heatmap)- Cleansed dataset
Cleansed=pd.read_csv("/dbfs/FileStore/Cleansed_WHO.csv", header='infer')

display(Data_Profiling_Fin(Cleansed))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.Sampling
# MAGIC Perform Stratified, Systematic, Random, Cluster sampling over data and compare the so obtained sampled dataset with the original data using a NULL Hypothesis, and suggest the best sample obtained thus. Compare the data densities of sampled datasets with that of the original input dataset to validate that our sample matches the data trend of original set.

# COMMAND ----------

input_dataframe = pd.read_csv("/dbfs/FileStore/Cleansed_WHO.csv", header='infer')
subsample_final = pd.DataFrame()
subsample1 = pd.DataFrame()
subsample2 = pd.DataFrame()
subsample3 = pd.DataFrame()
subsample4 = pd.DataFrame()

#Sampling(input_dataframe,filepath,task_type,input_appname,cluster_classified_col_ifany(Supervised))
subsample_final,subsample1,subsample2,subsample3,subsample4=Sampling(input_dataframe,"/dbfs/FileStore/Cleansed_WHO.csv",'Sampling','WHO','Status')


#persist sampled data sets 
filepath1 = '/dbfs/FileStore/StratifiedSampled_who.csv'
subsample1.to_csv(filepath1, index=False)
filepath2 = '/dbfs/FileStore/RandomSampled_who.csv'
subsample2.to_csv(filepath2, index=False)
filepath3 = '/dbfs/FileStore/SystematicSampled_who.csv'
subsample3.to_csv(filepath3, index=False)
filepath4 = '/dbfs/FileStore/ClusterSampled_who.csv' #The oversampled data without sampling is in '/dbfs/FileStore/SMOTE.csv'
subsample4.to_csv(filepath4, index=False)
filepath = '/dbfs/FileStore/subsample_final_who.csv'
subsample_final.to_csv(filepath, index=False)

# COMMAND ----------

# DBTITLE 1,Data Distribution (Histogram)- Input dataset
input_dataframe = pd.read_csv("/dbfs/FileStore/who.csv", header='infer')

display(display_DataDistribution(input_dataframe,'LifeExpectancy'))

# COMMAND ----------

# DBTITLE 1,Data Distribution (Histogram)- Stratified Sampled dataset
subsample1 = pd.read_csv("/dbfs/FileStore/StratifiedSampled_who.csv", header='infer')

display(display_DataDistribution(subsample1,'LifeExpectancy'))

# COMMAND ----------

# DBTITLE 1,Data Distribution (Histogram)- Random Sampled dataset
subsample2 = pd.read_csv("/dbfs/FileStore/RandomSampled_who.csv", header='infer')

display(display_DataDistribution(subsample2,'LifeExpectancy'))

# COMMAND ----------

# DBTITLE 1,Data Distribution (Histogram)- Systematic Sampled dataset
subsample3 = pd.read_csv("/dbfs/FileStore/SystematicSampled_who.csv", header='infer')

display(display_DataDistribution(subsample3,'LifeExpectancy'))

# COMMAND ----------

# DBTITLE 1,Data Distribution (Histogram)- Clustered Sampled dataset
subsample4 = pd.read_csv("/dbfs/FileStore/ClusterSampled_who.csv", header='infer')

display(display_DataDistribution(subsample4,'LifeExpectancy'))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.Anomaly Detection
# MAGIC Iterate data over various Anomaly-detection techniques and estimate the number of Inliers and Outliers for each.

# COMMAND ----------

#Calling the Anamoly Detection Function for identifying outliers 
outliers_fraction = 0.05
df =pd.read_csv("/dbfs/FileStore/ClusterSampled_who.csv", header='infer')
target_variable = 'LifeExpectancy'
variables_to_analyze='Population'

AnomalyDetection(df,target_variable,variables_to_analyze,outliers_fraction,'anomaly_test','WHO')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.Feature Selection
# MAGIC Perform feature selection on the basis of Feature Importance ranking, correlation values, variance within the column.
# MAGIC Choose features with High Importance value score, drop one of the two highly correlated features, drop features which offer zero variability to data and thus do not increase the entropy of dataset.

# COMMAND ----------

df =pd.read_csv("/dbfs/FileStore/ClusterSampled_who.csv", header='infer')
FeatureSelection(df,'LifeExpectancy','Continuous',"/dbfs/FileStore/ClusterSampled_who.csv",'WHO','FeatureSelection')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.Auto ML Trigger - after preprocessing
# MAGIC Trigger Azure auto ML, pick the best model so obtained and use it to predict the label column. Calculate the Weighted Absolute Accuracy amd push to telemetry. also obtain the data back in original format by using the unique identifier of each row 'Index' and report Actual v/s Predicted Columns. We also provide the direct link to the azure Portal Run for the current experiment for users to follow.

# COMMAND ----------

# DBTITLE 1,Azure Auto ML Trigger
import pandas as pd
dfclean = pd.read_csv("/dbfs/FileStore/ClusterSampled_who.csv", header='infer')

#Drop Feature selection recommended features Unimportant/Highly Correlated
dfclean.drop(['UnderFiveDeaths', 'Thinness5_9Years', 'EconomicInvestment_perc'], axis=1, inplace=True) 

#AutoMLFunc(subscription_id,resource_group,workspace_name,input_dataframe,label_col,task_type,input_appname)
df=AutoMLFunc(<subscription_id>,<resource_group>,<workspace_name>,dfclean,'LifeExpectancy','regression','WHO')


# COMMAND ----------

# DBTITLE 1,Obtain back data in original format after modelling
##df has just index,y actual, y predicted cols, as rest all cols are encoded after manipulation
for col in df.columns:
  if col not in ["y_predict","y_actual","Index"]: 
    df.drop([col], axis=1, inplace=True)
    
#dataframe is the actual input dataset     
dataframe = pd.read_csv("/dbfs/FileStore/who.csv", header='infer')

#Merging Actual Input dataframe with AML output df using Index column
dataframe_fin = pd.merge(left=dataframe, right=df, left_on='Index', right_on='Index')

#De-coding the label columns using scaling with actual label input
dataframe_fin['LifeExpectancy_Actual'] = dataframe_fin['LifeExpectancy']
dataframe_fin['LifeExpectancy_Predicted'] = (dataframe_fin['LifeExpectancy'] / dataframe_fin.y_actual)* dataframe_fin.y_predict

#div by zero in error above eqn raises Nans so replace all Nans with 0
dataframe_fin['LifeExpectancy_Predicted'].fillna(0, inplace=True)

# deleting unwanted intermediate columns 
for col in dataframe_fin.columns:
  if col in ["y_predict","y_actual"]: 
    dataframe_fin.drop([col], axis=1, inplace=True)
    
dataframe_fin

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7.Model Interpretation, Feature Importance
# MAGIC We can explore the model by splitting the Model metrics over various cohorts and analyse the data and model performance for each subclass.We can also get Global & Local feature Importance values for the Model.

# COMMAND ----------

#featureset should match with what was passed as X as part of the model training experiment
df= pd.read_csv("/dbfs/FileStore/ClusterSampled_who.csv", header='infer')

ModelInterpret(df,'LifeExpectancy',<subscription_id>,<resource_group>,<workspace_name>,<run_id>,<iteration>,'regression')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8.Telemetry
# MAGIC We can get comparative analysis of experiments via the telemetry captured for each and every step of the model run. 

# COMMAND ----------

# MAGIC %sql 
# MAGIC select * from  TelemetryTable
# MAGIC where MLKey='WHO' and Step='regression' 
# MAGIC order by TimeGenerated desc 
# MAGIC 
# MAGIC --Experimentation Accuracies:
# MAGIC --1.Cleansing but no oversampling by 'Status' col by SMOTE=96%
# MAGIC --2.Cleansing+SMOTE+Logical scaling of Percentage and OutOfThousand columns=97.2%
# MAGIC --3.Cleansing+SMOTE+Logical scaling of Percentage and OutOfThousand columns+FeatureEngg=97.19% (accuracy dropped as we noticed high correlation variable (refer feature selection) EconomicInvestment_perc added as per feature engineered)
# MAGIC --4.Cleansing+SMOTE+Logical scaling of Percentage and OutOfThousand columns+FeatureEngg+FeatureSelection=97.17%
# MAGIC --5.Cleansing+SMOTE+Logical scaling of Percentage and OutOfThousand columns+FeatureEngg+FeatureSelection+AnomaliesHandling=
