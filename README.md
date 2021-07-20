# Introduction
Traditional machine learning model development is resource-intensive, requiring significant domain/statistical knowledge and time to produce and compare dozens of models. 
With automated machine learning, the time it takes to get production-ready ML models with great ease and efficiency highly accelerates. However, the Automated Machine Learning does not yet provide much in terms of data preparation and feature engineering. 
The Auto Tune ML framework tries to solve this problem at scale as well as simplifies the overall process for the user. It leverages the Azure Automated ML coupled with components like Data Profiler, Data Sampler, Data Cleanser, Anomaly Detector which ensures quality data as a critical pre-step for building the ML model. This is powered with Telemetry, DevOps and Power BI integration, thus providing the users with a one-stop shop solution to productionize any ML model. The framework aims at ‘Democratizing’ AI all the while maintaining the vision of ‘Responsible’ AI.

- [WiKi](https://github.com/microsoft/AutoTuneML/blob/510595082a36d6d015f00dbc59d39ae367866e73/AUTO%20TUNE%20MODEL-%20Demo.pptx)

# Getting Started
## Prerequisites 
   1. Azure Databricks 
   2. Auto Tune Model Notebooks **(Master, Trigger notebooks)**
   3. Azure ML Services workspace 
   4. Python cluster in Databricks with configurations as mentioned in Installations link above (PyPi library azureml-sdk[automl],azureml-opendatasets, azureml-widgets in cluster) 

## Using the Notebooks
1. AMLMasterNotebook: Contains all the base functions used Data Acquisition, EDA, Sampling, Cleansing, Anomaly Detection, Azure AutoML Trigger, AutoML Trigger bypassing authentication to Azure ML(used for pipelining the notebook). 
2. AMLMasterNotebook- Trigger: Function calls in order to perform a pipeline of tasks. 

## Framework Components
1.	[Exploratory Data Analysis](#exploratory-data-analysis)
2.	[Data Sampling](#data-sampling) 
    1. Random Sampling
    2. Stratified Sampling
    3. Systematic Sampling
    4. Cluster Sampling (with SMOTE)
3.	[Data Cleansing](#data-cleansing)
4.	[Anomaly Detection](#anomaly-detection)
5.	[Azure Auto ML Trigger](#azure-auto-ml-trigger)
    (*Azure Component encapsulated with all cofigs and parameters)
7.	[Responsible AI Guidelines](#responsible-ai-guidelines) 
    1. Error Analysis
    2. Model Interpretation and Exploration
    3. Fairlearn to detect Fairness of the data and model 
    4. Identify & Remove Biasness in data
    5. SmartNoise to maintain PII data secrecy
8.	[Telemetry & DevOps Integration for Pipelining](#telemetry-and-devops-integration-for-pipelining)

![](https://github.com/microsoft/AutoTuneML/blob/0b7ba9c7526e00b7911de87f68ff0f387fbe6bf2/Pipeline.png)

# Exploratory Data Analysis
Exploratory Data Analysis refers to the critical process of performing initial investigations on data to discover patterns, to spot anomalies, to test hypothesis and to check assumptions with the help of summary statistics and graphical representations. 

# Data Sampling
By Data Sampling, we can select, manipulate and analyze a representative subset of data points to identify patterns and trends in the larger dataset being examined. The dataset thus obtained is a weighted sample of the actual dataset, thus enabling a clear picture of the bigger dataset with best performance, retaining the overall data density and distribution. The following method is used to obtain samples of data from the original input data using different techniques and the best sample thus obtained is suggested to the user. The function ‘Sampling’ encompasses all the features of this as explained below.
<br/>

![](https://github.com/microsoft/AutoTuneML/blob/0b7ba9c7526e00b7911de87f68ff0f387fbe6bf2/Sampling_Techniques_Explained.png)

1.	**Get the ideal sample size from the original input dataset using Solven’s formula**
<br/>n=N/((1+N^2 ) )
<br/>Here,
<br/>n=Number of Samples
<br/>N=Total Population
<br/>e=Error tolerance (level) = 1-Confidence Level in percentage (~ 95%)
<br/>

2.	**Random Sampling**
<br/>Pick (n) items from the whole actual dataset (N) randomly assuming every item has equal probability (1/N) of getting its place in the sample irrespective of its weightage in the actual dataset.
<br/>

3.	**Systematic Sampling**
<br/>This method allows to choose the sample members of a population at regular intervals. It requires the selection of a starting point for the sample and sample size that can be repeated at regular intervals. This type of sampling method has a predefined range, and hence this sampling technique is the least time-consuming.
Pick every kth item from the actual dataset where k = N/n 
<br/>

4.	**Stratified Sampling**
<br/>Clustering :- Classify input data into k clusters using K-means clustering and add an extra column to the data frame ‘Cluster’ to identify which record belongs to which cluster (0- to k-1). Get the ideal ‘k’ for a dataset using Silhouette score. The silhouette coefficient of a data measures how well data are grouped within a cluster and how far they are from other clusters. A silhouette close to 1 means the data points are in an appropriate cluster and a silhouette coefficient close to −1 implies that data is in the wrong cluster. i.e., get the scores for a range of values for k and choose the cluster k value which gives Highest Silhouette score.
Weighted Count :- Get the percentage count of records corresponding to each cluster in actual dataframe, create a weighted subsample of (n) records maintaining the same weighted distribution of records from each cluster. 
<br/>

5.	**Clustered Sampling**
<br/>If the input data is having a predefined distribution to different classes, check if the distribution is biased towards one or more classes. If yes, then apply SMOTE(Synthetic Minority Oversampling Technique) to level the distribution for each class. This approach for addressing imbalanced datasets is to oversample the minority class. This involves duplicating examples in the minority class, although these examples don’t add any new information to the model. Instead, new examples can be synthesized from the existing examples. Create a weighted subsample of (n) records maintaining the same weighted distribution of records from each cluster (after SMOTE).
<br/>

6.	**Get the sampling error**
<br/>The margin of error is 1/√n, where n is the size of the sample for each of the above techniques.
<br/>

7.	**Getting the best Sample obtained**
<br/>Using a Null Hypothesis for each column, calculate the p-value using Kolmogorov-Smirnov test (For Continuous columns) and Pearson's Chi-square test (for categorical columns). If the p-values are >=0.05 for more than a threshold number of columns (50% used here), the subsample created is accepted. P-value can be used to decide whether there is evidence of a statistical difference between the two population (Sample v/s the Original dataset) means. The smaller the p-value, the stronger the evidence is that the two populations have different means. The samples obtained above that has the highest average p-value is suggested to be the closest to the actual dataset. p-value is the probability of obtaining results at least as extreme as the observed results of a statistical hypothesis test, assuming that the null hypothesis is correct. A smaller p-value means that there is stronger evidence in favor of the alternative hypothesis.
<br/>

# Data Cleansing
Before triggering the Azure Auto ML, our proposed framework (Auto Tune Model) helps improve the data quality of our input dataset using the Data Cleansing component.
Since data is considered the currency for any machine learning model, it is very critical for the success of Machine Learning applications. The algorithms that we may use can be powerful, but without the relevant or right data training, our system may fail to yield ideal results.
Data cleansing refers to identifying and correcting errors in the dataset that may negatively impact a predictive model. It refers to all kinds of tasks and activities to detect and repair errors in the data. This improves the quality of the training data for analytics and enables accurate decision-making.
The function ‘autodatacleaner’ encompasses all the underlying features of the data cleansing component that are outlined below.
<br/>

1. **Handle Missing Values:**
<br/>Data can have missing values for several reasons such as observations that were not recorded and data corruption. Handling missing data is important as many machine learning algorithms do not support data with missing values.
If data is missing, we can either indicate missing values by simply creating a Missing category if the data is categorical or flagging and filling with 0 if it is numerical or apply imputation to fill the missing values.
Hence, as part of the Data Cleansing component, we are applying imputation or dropping the columns in the dataset to fill all the missing values, which is decided based on a threshold of 50%. First, we replace all the white spaces or empty values with NaN except those in the middle. If more than half of the data in a column is NaN, we drop the column else we impute the missing values with median for numerical columns and mode for categorical columns. One limitation with dropping the columns is by dropping missing values, we drop information that may assist us in making better conclusions about the study. It may deny us the opportunity of benefiting from the possible insights that can be gotten from the fact that a particular value is missing. This can be handled by applying feature importance and understanding the significant columns in the dataset that can be useful for the predictive model which shouldn’t be dropped hence, treating this as an exception.
<br/>

2. **Fix Structural Errors:**
<br/>After removing unwanted observations and handling missing values, the next thing we make sure is that the wanted observations are well-structured. Structural errors may occur during data transfer due to a slight human mistake or incompetency of the data entry personnel. 
Some of the things we will look out for when fixing data structure include typographical errors, grammatical blunders, and so on. The data structure is mostly concerned with categorical data. 
We are fixing these structural errors by removing leading/trailing white spaces and solving inconsistent capitalization for categorical columns.
<br/>

3. **Encoding of Categorical Columns:**
<br/>In machine learning, we usually deal with datasets which contains multiple labels in one or more than one column. These labels can be in the form of words or numbers. The training data is often labeled in words to make it understandable or in human readable form.
Label Encoding refers to converting the labels into numeric form to convert it into the machine-readable form. Machine learning algorithms can then decide in a better way on how those labels must be operated. 
Hence, for Label encoding we are using the Label Encoder component of the python class sklearn preprocessing package.
from sklearn.preprocessing import LabelEncoder
Encode target labels with value between 0 and n_classes-1.
<br/>

4. **Normalization:**
<br/>As most of the datasets have multiple features spanning varying degrees of magnitude, range, and units. This can deviate the ML model to be biased towards the dominant scale and hence make it as an obstacle for the machine learning algorithms as they are highly sensitive to these features. Hence, we are tackling this problem using normalization. The goal of normalization is to change the values of numeric columns in the dataset to use a common scale, without distorting differences in the ranges of values or losing information.
We normalize our dataset using the MinMax scaling component of the python class sklearn preprocessing package:
from sklearn.preprocessing import MinMaxScaler
MinMax scaler transforms features by scaling each feature to a given range on the training set, e.g., between zero and one. It shifts and rescales the values so that they end up ranging between 0 and 1.
<br/>Here’s the formula for normalization:
<br/>X^'=  (X- X_min)/(X_max⁡- X_min) 
<br/>Here, 
<br/>Xmax and Xmin are the maximum and the minimum values of the feature respectively.
<br/>	When the value of X is the minimum value in the column, the numerator will be 0, and hence X’ is 0
<br/><br/>	On the other hand, when the value of X is the maximum value in the column, the numerator is equal to the denominator and thus the value of X’ is 1
<br/>If the value of X is between the minimum and the maximum value, then the value of X’ is between 0 and 1
<br/>The transformation is given by:
<br/>X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
<br/>X_scaled = X_std * (max - min) + min
<br/>where min, max = feature_range.

# Anomaly Detection
Anomaly detection aims to detect abnormal patterns deviating from the rest of the data, called anomalies or outliers. Handling Outliers and anomalies is critical to the machine learning process. Outliers can impact the results of our analysis and statistical modeling in a drastic way. Our tendency is to use straightforward methods like box plots, histograms and scatter-plots to detect outliers. But dedicated outlier detection algorithms are extremely valuable in fields which process large amounts of data and require a means to perform pattern recognition in larger datasets. The PyOD library can step in to bridge this gap, which is a comprehensive and scalable Python toolkit for detecting outlying objects in multivariate data. We will be using the following algorithms within PyOD to detect and analyze the Outliers and indicate their presence in datasets.
<br/>

![](https://github.com/microsoft/AutoTuneML/blob/0b7ba9c7526e00b7911de87f68ff0f387fbe6bf2/AnomalyDetection_Techniques_Explained.png)

<br/>**1. Angle-Based Outlier Detection (ABOD)**
<br/>It considers the relationship between each point and its neighbor(s). It does not consider the relationships among these neighbors. The variance of its weighted cosine scores to all neighbors could be viewed as the outlying score. ABOD performs well on multi-dimensional data

<br/>**2. k-Nearest Neighbors Detector**
<br/>For any data point, the distance to its kth nearest neighbor could be viewed as the outlying score. PyOD supports three kNN detectors:
<br/>Largest: Uses the distance of the kth neighbor as the outlier score
<br/>Mean: Uses the average of all k neighbors as the outlier score
<br/>Median: Uses the median of the distance to k neighbors as the outlier score

<br/>**3. Isolation Forest**
<br/>It uses the scikit-learn library internally. In this method, data partitioning is done using a set of trees. Isolation Forest provides an anomaly score looking at how isolated the point is in the structure. The anomaly score is then used to identify outliers from normal observations.
<br/>Isolation Forest performs well on multi-dimensional data.

<br/>**4. Histogram-based Outlier Detection**
<br/>It is an efficient unsupervised method which assumes the feature independence and calculates the outlier score by building histograms. It is much faster than multivariate approaches, but at the cost of less precision

<br/>**5. Local Correlation Integral (LOCI)**
<br/>LOCI is very effective for detecting outliers and groups of outliers. It provides a LOCI plot for each point which summarizes a lot of the information about the data in the area around the point, determining clusters, micro-clusters, their diameters, and their inter-cluster distances. None of the existing outlier-detection methods can match this feature because they output only a single number for each point

<br/>**6. Feature Bagging**
<br/>A feature bagging detector fits a number of base detectors on various sub-samples of the dataset. It uses averaging or other combination methods to improve the prediction accuracy. By default, Local Outlier Factor (LOF) is used as the base estimator. However, any estimator could be used as the base estimator, such as kNN and ABOD. Feature bagging first constructs n sub-samples by randomly selecting a subset of features. This brings out the diversity of base estimators. Finally, the prediction score is generated by averaging or taking the maximum of all base detectors.

<br/>**7. Clustering Based Local Outlier Factor**
<br/>It classifies the data into small clusters and large clusters. The anomaly score is then calculated based on the size of the cluster the point belongs to, as well as the <br/>distance to the nearest large cluster.
<br/>Using each of the above algorithms we would estimate the number of outliers and inliers and assign the dataset points a Boolean value to identify them as inliers and <br/>outliers separately. We would allow user intervention to take the final call on which outliers to remove from the dataset and retrain in the model henceforth. 
<br/>
Anomalies are not always bad data, instead they can reveal data trends which play a key role in predictions sometimes. Hence it is important to analyze the anomalies thus  pointed but not get rid of them blindly.


# Azure Auto ML Trigger 
<br/>
During training, Azure Machine Learning creates several pipelines in parallel that try different algorithms and parameters for you. The service iterates through ML algorithms paired with feature selections, where each iteration produces a model with a training score. The higher the score, the better the model is considered to "fit" your data. It will stop once it hits the exit criteria defined in the experiment. The function ‘AutoMLFunc’ encompasses all the features of this as explained below.
<br/>
Using Azure Machine Learning, you can design and run your automated ML training experiments with these steps:
<br/>1. Identify the ML problem to be solved: classification or regression.
<br/>2. Configure the automated machine learning parameters that determine how many iterations over different models, hyperparameter settings, advanced preprocessing/featurization, and what metrics to look at when determining the best model.
<br/>3. Divide the input preprocessed data into train and test datasets.
<br/>4. Submit the training run and extract the best model based on primary metrics. 
<br/>5. Use this best model thus obtained to predict data and calculate accuracy scores on actual v/s predicted components of data. Mean Absolute Percentage Error (MAPE) is generally used to determine the performance of a model, but problems can occur when calculating the MAPE value with small denominators (Actual value =0 in denominator). A singularity problem of the form 'one divided by can occur. As an alternative, each actual value (A_(t )  of the series in the original formula can be replaced by the average of all actual values A_avg of that series. This is equivalent to dividing the sum of absolute differences by the sum of actual values and is sometimes referred to as WAPE (Weighted Absolute Percentage Error).
<br/>6. The best model obtained can also be deployed and used using a REST API. The actual v/s predicted data can be reported and analyzed in Power BI along with the telemetry timestamps. 

![](https://github.com/microsoft/AutoTuneML/blob/0b7ba9c7526e00b7911de87f68ff0f387fbe6bf2/AutoMLTrigger_steps.png)

# Responsible AI Guidelines
AI systems can cause a variety of fairness-related harms, including harms involving people’s individual experiences with AI systems or the ways that AI systems represent the groups to which they belong. Prioritizing fairness in AI systems is a sociotechnical challenge.
Responsible AI Guidelines suggest the best way to build fairness, interpretability, privacy, and security into these systems.
<br/>

![](https://github.com/microsoft/AutoTuneML/blob/0b7ba9c7526e00b7911de87f68ff0f387fbe6bf2/ResponsibleAI_Pipeline.png)
![](https://github.com/microsoft/AutoTuneML/blob/0b7ba9c7526e00b7911de87f68ff0f387fbe6bf2/ResponsibleAI_Explained.png)

# Telemetry and DevOps Integration for Pipelining
We would maintain two notebooks and a pipeline can be set to trigger the Trigger notebook from Azure Data Factory–
1. Auto Tune ML Master
2. Auto Tune ML Trigger
<br/>The Trigger notebook calls all the functions explained above using the desired dataset and specifying intermediate information like Dataset filepath in Azure Datalake, Key identifier of the experiment run to locate it in telemetry, location filepath to push telemetry data, Azure workspace and subscription details, Service Principle and Tenant ID of the subscription where the Workspace lies to bypass the Azure authentication process each time the Experiment is submitted as this needs to be an automated run on trigger of the pipeline. 
<br/>The Telemetry captured encompasses of the Key identifier of the experiment run, Accuracy scores for the experiment, Function call information, Results and time generated for each step. The telemetry along with actual v/s predicted data is written to the Azure Datalake and can be fetched into Power BI from the Datalake in a live connection. Each time the telemetry files are updated, they would reflect in the Power BI report with updated information and current run status as compared to previous runs. 
<br/>

# OLD
# Project

> This repo has been populated by an initial template to help get you started. Please make sure to update the content to build a great experience for community-building.

As the maintainer of this project, please make a few updates:

- Improving this README.MD file to provide a great experience
- Updating SUPPORT.MD with content about this project's support experience
- Understanding the security reporting process in SECURITY.MD
- Remove this section from the README

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
