# Databricks notebook source
# MAGIC %md
# MAGIC **PROBLEM STATEMENT**
# MAGIC <br/>Predict the Active Covid cases of 2021 depending on 2020 statewise data in India. 
# MAGIC <br/>Get Sample data from Source- https://www.kaggle.com/aritranandi23/covid-19-analysis-and-prediction/data
# MAGIC <br/>
# MAGIC <br/>**COLUMN DEFINITION**
# MAGIC <br/>Date:string
# MAGIC <br/>Time:string
# MAGIC <br/>State/UnionTerritory:string
# MAGIC <br/>ConfirmedIndianNational:string
# MAGIC <br/>ConfirmedForeignNational:string
# MAGIC <br/>Cured:integer
# MAGIC <br/>Deaths:integer
# MAGIC <br/>Confirmed:integer
# MAGIC 
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
# MAGIC <br/>1.Get Death rates, Discharge rates, Active cases rates from Confirmed, Cured, Death cases 
# MAGIC <br/>2.In TS Forecasting each group must provide atleast 3 datapoints to obtain frequency, remove the records where frequency<=3 for the train set i.e. data from the year 2020

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import functions from Master Notebook:
# MAGIC Import the Functions and dependencies from the Master notebook to be used in the Trigger Notebook

# COMMAND ----------

# DBTITLE 1,Import functions
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
# MAGIC val filepath1= "adl://psinsightsadlsdev01.azuredatalakestore.net/Temp/ML-PJC/covid_19_india.csv"
# MAGIC var df1=spark.read.format("com.databricks.spark.csv").option("inferSchema", "true").option("header", "true").option("delimiter", ",").load(filepath1)
# MAGIC df1.createOrReplaceTempView("CovidIndia")
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC val filepath2= "adl://psinsightsadlsdev01.azuredatalakestore.net/Temp/ML-PJC/covid_vaccine_statewise.csv"
# MAGIC var df2=spark.read.format("com.databricks.spark.csv").option("inferSchema", "true").option("header", "true").option("delimiter", ",").load(filepath2)
# MAGIC df2.createOrReplaceTempView("Vaccine")
# MAGIC 
# MAGIC val filepath3= "adl://psinsightsadlsdev01.azuredatalakestore.net/Temp/ML-PJC/StatewiseTestingDetails.csv"
# MAGIC var df3=spark.read.format("com.databricks.spark.csv").option("inferSchema", "true").option("header", "true").option("delimiter", ",").load(filepath3)
# MAGIC df3.createOrReplaceTempView("Testing")

# COMMAND ----------

# MAGIC %sql
# MAGIC with CTE1 as
# MAGIC (select
# MAGIC MAX(date_format(Date, 'MMMM')) AS Month
# MAGIC ,MAX(Year(Date)) AS Year
# MAGIC ,CONCAT(Year(Date),'-',RIGHT(CONCAT('00',MONTH(Date)),2),'-','01') AS Date
# MAGIC ,`State/UnionTerritory` as State
# MAGIC ,SUM(Cured) as Cured
# MAGIC ,SUM(Deaths) as Deaths
# MAGIC ,SUM(Confirmed) as Confirmed
# MAGIC ,((SUM(Confirmed)-SUM(Deaths)-SUM(Cured))/SUM(Confirmed) * 100) as ActiveCasesRate
# MAGIC ,(SUM(Cured)/SUM(Confirmed) * 100) AS DischargeRate
# MAGIC ,(SUM(Deaths)/SUM(Confirmed) * 100) as DeathsRate
# MAGIC from CovidIndia C
# MAGIC group by 
# MAGIC CONCAT(Year(Date),'-',RIGHT(CONCAT('00',MONTH(Date)),2),'-','01')
# MAGIC ,`State/UnionTerritory`
# MAGIC )
# MAGIC 
# MAGIC --select distinct state from CTE1 group by State having count(*)<=3
# MAGIC select * from CTE1 where State not in (select distinct state from CTE1 group by State having count(*)<=3)
# MAGIC and Year=2021

# COMMAND ----------

# DBTITLE 1,Logical transformation of data
df= spark.sql("""
with CTE1 as
(select
MAX(date_format(Date, 'MMMM')) AS Month
,MAX(Year(Date)) AS Year
,CONCAT(Year(Date),'-',RIGHT(CONCAT('00',MONTH(Date)),2),'-','01') AS Date
,`State/UnionTerritory` as State
,SUM(Cured) as Cured
,SUM(Deaths) as Deaths
,SUM(Confirmed) as Confirmed
,((SUM(Confirmed)-SUM(Deaths)-SUM(Cured))/SUM(Confirmed) * 100) as ActiveCasesRate
,(SUM(Cured)/SUM(Confirmed) * 100) AS DischargeRate
,(SUM(Deaths)/SUM(Confirmed) * 100) as DeathsRate
from CovidIndia C
group by 
CONCAT(Year(Date),'-',RIGHT(CONCAT('00',MONTH(Date)),2),'-','01')
,`State/UnionTerritory`
)

select * from CTE1 
where State not like 'Lakshadweep' and --Low Frequency for train as only one month dec in 2020 train
State not in (select distinct state from CTE1 group by State having count(*)<=3) --low frequency for train as atleast three months required """)

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
'Month'
,'Year'
,'Date'
,'State'
,'Cured'
,'Deaths'
,'Confirmed'
,'ActiveCasesRate'
,'DischargeRate'
,'DeathsRate'
]
cols_string=[
'Month'
,'Year'
,'Date'
,'State'
]
cols_int=[
'Cured'
,'Deaths'
,'Confirmed'
]
cols_bool=[]
cols_Float=[
'ActiveCasesRate'
,'DischargeRate'
,'DeathsRate'
]
for col_name in cols_int:
    df = df.withColumn(col_name, col(col_name).cast('Int'))  
for col_name in cols_Float:
    df = df.withColumn(col_name, col(col_name).cast('float')) 
for col_name in cols_bool:
    df = df.withColumn(col_name, col(col_name).cast('bool')) 
    
input_dataframe = df.toPandas()
input_dataframe['Index'] = np.arange(len(input_dataframe))
outdir = '/dbfs/FileStore/Covid.csv'
input_dataframe.to_csv(outdir, index=False)
#input_dataframe = pd.read_csv("/dbfs/FileStore/Dataframe.csv", header='infer')


# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.Data Exploration
# MAGIC 1.Exploratory Data Analysis (EDA)- To understand the overall data at hand, analysing each feature independently for its' statistics, the correlation and interraction between variables, data sample etc. 
# MAGIC <br/>2.Data Profiling Plots- To analyse the Categorical and Numerical columns separately for any trend in data, biasness in data etc.

# COMMAND ----------

# DBTITLE 1,EDA
input_dataframe = pd.read_csv("/dbfs/FileStore/Covid.csv", header='infer')

p=Data_Profiling_viaPandasProfiling(input_dataframe,'RealEstate','EDA')
displayHTML(p)


# COMMAND ----------

# DBTITLE 1,Data Profiling Plots
input_dataframe = pd.read_csv("/dbfs/FileStore/Covid.csv", header='infer')

#User Inputs
cols_all=[
'Month'
,'Year'
,'Date'
,'State'
,'Cured'
,'Deaths'
,'Confirmed'
,'ActiveCasesRate'
,'DischargeRate'
,'DeathsRate'
]
Categorical_cols=['Month'
,'Year'
,'Date'
,'State'
]
Numeric_cols=['Cured'
,'Deaths'
,'Confirmed'
,'ActiveCasesRate'
,'DischargeRate'
,'DeathsRate'
]
Label_col='ActiveCasesRate'

#Data_Profiling_Plots(input_dataframe,Categorical_cols,Numeric_cols,Label_col)
Data_Profiling_Plots(input_dataframe,Categorical_cols,Numeric_cols,Label_col)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.Cleansing
# MAGIC To clean the data from NULL values, fix structural errors in columns, drop empty columns, encode the categorical values, normalise the data to bring to the same scale. We also check the Data Distribution via Correlation heatmap of original input dataset v/s the Cleansed dataset to validate whether or not the transformations hampered the original data trend/density.

# COMMAND ----------

# DBTITLE 1,Auto Cleanser 
subsample_final = pd.read_csv("/dbfs/FileStore/Covid.csv", header='infer')
#subsample_final=subsample_final.drop(['Index'], axis = 1) # Index is highest variability column hence always imp along PC but has no business value. You can append columns to be dropped by your choice here in the list

inputdf_new=autodatacleaner(subsample_final,"/dbfs/FileStore/Covid.csv","Covid","Data Cleanser")
print("Total rows in the new pandas dataframe:",len(inputdf_new.index))

#persist cleansed data sets 
filepath1 = '/dbfs/FileStore/Cleansed_Covid.csv'
inputdf_new.to_csv(filepath1, index=False)


# COMMAND ----------

# DBTITLE 1,Data profiling(Heatmap correlation)- User input dataframe
subsample_final = pd.read_csv("/dbfs/FileStore/Covid.csv", header='infer')

display(Data_Profiling_Fin(subsample_final))

# COMMAND ----------

# DBTITLE 1,Data profiling(Heatmap correlation)- Cleansed dataframe
Cleansed=pd.read_csv("/dbfs/FileStore/Cleansed_Covid.csv", header='infer')

display(Data_Profiling_Fin(Cleansed))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.Anomaly Detection
# MAGIC Iterate data over various Anomaly-detection techniques and estimate the number of Inliers and Outliers for each.

# COMMAND ----------

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
#df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
#target_variable = 'SalePrice'
#variables_to_analyze = '1stFlrSF'
output_path = '/dbfs/FileStore/AnomalyDetection_HTML'
#df.plot.scatter('1stFlrSF','SalePrice')
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
        #OutputfileName="adl://psinsightsadlsdev01.azuredatalakestore.net/DEV/AnomalyDetection_"+clf_name_string +".csv"
        #copydbfs = '/dbfs/FileStore/AnomalyDetection.csv'
        #dfy.to_csv(copydbfs, index=False)
        #dbutils.fs.cp ("/FileStore/AnomalyDetection.csv", OutputfileName, True) 
        n_outliers="%s" % str(n_outliers)
        n_inliers="%s" % str(n_inliers)
        rm_str3 = "Insert into AutoTuneML.amltelemetry values (" + appnamequotes + ","+ task + ",'OUTLIERS :" + n_outliers + "  INLIERS :" + n_inliers  + "  :- " + clf_name+ "'," + tsquotes + ")"
        #spark.sql(rm_str3)
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
        #output_file = "adl://psinsightsadlsdev01.azuredatalakestore.net/DEV/AnomalyDetection_chart" + clf_text + '.html'
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
        
        #filepath="adl://psinsightsadlsdev01.azuredatalakestore.net/DEV/AnomalyDetection.html"
        ##plt.savefig(tmpfile, format='png')
        #plt.savefig('/dbfs/FileStore/AnomalyDetection.png')
        #dbutils.fs.cp ("/FileStore/AnomalyDetection.png", filepath, True)
        #print("Anomaly Detection Report can be downloaded from path: ",filepath)


# COMMAND ----------

# DBTITLE 0,Anomaly Detection
#Calling the Anamoly Detection Function for identifying outliers 
outliers_fraction = 0.05
df =pd.read_csv("/dbfs/FileStore/Cleansed_Covid.csv", header='infer')
target_variable = 'ActiveCasesRate'
variables_to_analyze='Confirmed'

AnomalyDetection(df,target_variable,variables_to_analyze,outliers_fraction,'anomaly_test','test')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.Feature Selection
# MAGIC Perform feature selection on the basis of Feature Importance ranking, correlation values, variance within the column.
# MAGIC Choose features with High Importance value score, drop one of the two highly correlated features, drop features which offer zero variability to data and thus do not increase the entropy of dataset.

# COMMAND ----------

df =pd.read_csv("/dbfs/FileStore/Cleansed_Covid.csv", header='infer')
FeatureSelection(df,'ActiveCasesRate','Continuous',"/dbfs/FileStore/Cleansed_Covid.csv",'Covid','FeatureSelection')

# COMMAND ----------

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
# MAGIC %pip install skfeature-chappers
# MAGIC %pip install raiwidgets 
# MAGIC 
# MAGIC %pip install ruamel.yaml==0.16.10
# MAGIC %pip install azure-core==1.8.0
# MAGIC %pip install liac-arff==2.4.0
# MAGIC %pip install msal==1.4.3
# MAGIC %pip install msrest==0.6.18
# MAGIC %pip install ruamel.yaml.clib==0.2.0
# MAGIC %pip install tqdm==4.49.0
# MAGIC %pip install zipp==3.2.0
# MAGIC %pip install interpret-community==0.15.0
# MAGIC %pip install azure-identity==1.4.0
# MAGIC %pip install dotnetcore2==2.1.16
# MAGIC %pip install jinja2==2.11.2
# MAGIC %pip install azure-core==1.15.0
# MAGIC %pip install azure-mgmt-containerregistry==8.0.0
# MAGIC %pip install azure-mgmt-core==1.2.2
# MAGIC %pip install distro==1.5.0
# MAGIC %pip install google-api-core==1.30.0
# MAGIC %pip install google-auth==1.32.1
# MAGIC %pip install importlib-metadata==4.6.0
# MAGIC %pip install msal==1.12.0
# MAGIC %pip install packaging==20.9
# MAGIC %pip install pathspec==0.8.1
# MAGIC %pip install requests==2.25.1
# MAGIC %pip install ruamel.yaml.clib==0.2.4
# MAGIC %pip install tqdm==4.61.1
# MAGIC %pip install zipp==3.4.1
# MAGIC %pip install scipy==1.5.2
# MAGIC %pip install charset-normalizer==2.0.3
# MAGIC %pip install websocket-client==1.1.0
# MAGIC %pip install scikit-learn==0.22.1
# MAGIC %pip install interpret-community==0.19.0
# MAGIC %pip install cryptography==3.4.7
# MAGIC %pip install llvmlite==0.36.0
# MAGIC %pip install numba==0.53.1

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7.Auto ML Trigger - after preprocessing
# MAGIC Trigger Azure auto ML, pick the best model so obtained and use it to predict the label column. Calculate the Weighted Absolute Accuracy amd push to telemetry. also obtain the data back in original format by using the unique identifier of each row 'Index' and report Actual v/s Predicted Columns. We also provide the direct link to the azure Portal Run for the current experiment for users to follow.

# COMMAND ----------

# DBTITLE 1,Train-Test Split
##df has just index,y actual, y predicted cols, as rest all cols are encoded after manipulation
import pandas as pd
df =pd.read_csv("/dbfs/FileStore/Cleansed_Covid.csv", header='infer')
for col in df.columns:
  if col not in ["Index"]: 
    df.drop([col], axis=1, inplace=True)
    
#dataframe is the actual input dataset     
dataframe = pd.read_csv("/dbfs/FileStore/Covid.csv", header='infer')

#Merging Actual Input dataframe with AML output df using Index column
dataframe_fin = pd.merge(left=dataframe, right=df, left_on='Index', right_on='Index')
#dataframe_fin

#train-test split
train_data=dataframe_fin[dataframe_fin['Year']==2020]
test_data=dataframe_fin[dataframe_fin['Year']==2021]
label='DeathsRate'#'ActiveCasesRate'
test_labels = test_data.pop(label).values
train_data

# COMMAND ----------

#train_date duplicate check
train_data[train_data.duplicated(['State','Date'])]


# COMMAND ----------

#frequency <=3 check
df_new= train_data.groupby(['State']).count()
freq = df_new[(df_new['Index'] <= 3)]
freq

# COMMAND ----------

##2 removed lakshdweep
#frequency <=3 check
df_new= train_data.groupby(['State']).count()
freq = df_new[(df_new['Index'] <= 3)]
freq

# COMMAND ----------

# DBTITLE 1,Auto ML Run
time_series_settings = {
    "time_column_name": "Date",
    "grain_column_names": ["State"],
    "max_horizon": 2,
    "target_lags": 2,
    "target_rolling_window_size": 2,
    "featurization": "auto",
    "short_series_handling_configuration":'auto',
    "freq": 'MS',
    "short_series_handling_config": "auto"
}

from azureml.core.workspace import Workspace
from azureml.core.experiment import Experiment
from azureml.train.automl import AutoMLConfig
import logging
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.experiment import Experiment
from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.dataset import Dataset
from azureml.widgets import RunDetails
from azureml.core import Dataset, Datastore
from azureml.data.datapath import DataPath
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
import os
import warnings
from sklearn.metrics import mean_squared_error
from math import sqrt
warnings.filterwarnings('ignore')

automl_config = AutoMLConfig(task='forecasting',
                             primary_metric='normalized_root_mean_squared_error',
                             iterations= 1,
                             experiment_timeout_minutes=15,
                             enable_early_stopping=True,
                             n_cross_validations=2,
                             training_data=train_data,
                             label_column_name=label,
                             enable_ensembling=False,
                             verbosity=logging.INFO,
                             **time_series_settings)


ws = Workspace(subscription_id = '3ecb9b6a-cc42-4b0a-9fd1-6c08027eb201', resource_group = 'psbidev', workspace_name = 'psdatainsightsML')
#ws = Workspace.from_config()

# Verify that cluster does not exist already
compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D12_V2',
                                                           max_nodes=100)
compute_target = ComputeTarget.create(ws, amlcompute_cluster_name, compute_config)
compute_target.wait_for_completion(show_output=True)
  
datastore = ws.get_default_datastore()
train_dataset = Dataset.Tabular.register_pandas_dataframe(train_data,datastore,'Covid')
test_dataset = Dataset.Tabular.register_pandas_dataframe(test_data,datastore,'Covid')
  
experiment = Experiment(ws, "TS_forecasting")
remote_run = experiment.submit(automl_config, show_output=True)
remote_run.wait_for_completion()
best_run, fitted_model = remote_run.get_output()

# COMMAND ----------

# DBTITLE 1,Forecasting for Test set
y_predictions, X_trans = fitted_model.forecast(test_data)
y_predictions

# COMMAND ----------

# DBTITLE 1,Data Actuals v/s Predicted
result = test_data
y_predictions, X_trans = fitted_model.forecast(test_data)
result['Values_pred']=y_predictions
result['Values_actual']=test_labels
result['Error']=result['Values_actual']-result['Values_pred']
result['Percentage_change'] = ((result['Values_actual']-result['Values_pred']) / result['Values_actual'] )* 100
result=result.reset_index(drop=True)
result

# COMMAND ----------

# DBTITLE 1,Accuracy Calculation
y_actual = test_labels
sum_actuals = sum_errors = 0
for actual_val, predict_val in zip(y_actual, y_predictions):
  abs_error = actual_val - predict_val
  if abs_error < 0:
    abs_error = abs_error * -1
    sum_errors = sum_errors + abs_error
    sum_actuals = sum_actuals + actual_val

mean_abs_percent_error = sum_errors / sum_actuals
Accuracy_score = 1 - mean_abs_percent_error
print(Accuracy_score)
