Copyright (c) Microsoft Corporation
<br/>Licensed under the MIT License

# Overview
Traditional machine learning model development is resource-intensive, requiring significant domain/statistical knowledge and time to produce and compare dozens of models. 
With automated machine learning, the time it takes to get production-ready ML models with great ease and efficiency highly accelerates. However, the Automated Machine Learning does not yet provide much in terms of data preparation and feature engineering. 
The AutoBrewML  framework tries to solve this problem at scale as well as simplifies the overall process for the user. It leverages the Azure Automated ML coupled with components like Data Profiler, Data Sampler, Data Cleanser, Anomaly Detector which ensures quality data as a critical pre-step for building the ML model. This is powered with Telemetry, DevOps and Power BI integration, thus providing the users with a one-stop shop solution to productionize any ML model. The framework aims at ‘Democratizing’ AI all the while maintaining the vision of ‘Responsible’ AI.
<br/>
![](https://github.com/microsoft/AcceleratedML/blob/e02bbbe4f5d036607de5d1a494f872960f2f1fba/Resources/Pipeline.png)

## Our Inspiration Story
As we think about the future of technology, it resides in the notion of intelligence. At Microsoft, we have an approach that’s both ambitious and broad, an approach that seeks to Democratize Machine Learning & Artificial Intelligence, to take it from the high walls of ivory towers and make it accessible for all.

<br/>It is very rightly said that if you want something you never had, you have to do something you have never done! When we started on our ML journey we hoped to have a hand-holding which could accelerate our sail across the extensive nature of ML. We were stuck in the vicious loop of finding right algorithms and tools to achieve our target rather than focusing on the data at hand and fine tuning it with our Business Domain knowledge. Also traditional machine learning model development is resource-intensive, requiring significant domain/statistical knowledge and time to produce and compare dozens of models.  A team comprises of folks from various backgrounds and ML knowledge base, but driven by our mission at Microsoft 'To empower every person and every organization on the planet to achieve more' we want to open ways for each and everyone to have access over the wonders of ML & AI.

<br/>So by amalgamating all our pain points and covering the aspects of an end-to-end ML pipeline we came up with a Framework to get production-ready ML models with great ease and efficiency.

<br/>How we pursue this bold ambition to democratize AI for all via this Framework:
   1. Implement machine learning solutions without extensive programming knowledge
   2. Find the right dataset for modelling
   3. Save time and resources
   4. Leverage Data Science best practices & Responsible AI
   5. Provide agile problem-solving
   6. Provide visualizations to interpret data
   7. Capture telemetry throughout the process

# Getting Started
## Prerequisites 
   1. Azure Databricks 
   2. Auto Brew ML Notebooks **(Master, Trigger notebooks)**
   3. Azure ML Services workspace 
   4. Python cluster in Databricks with configurations as mentioned in Installations link above (PyPi library azureml-sdk[automl],azureml-opendatasets, azureml-widgets in cluster) 
   
## How to use it
1. AMLMasterNotebook: Contains all the base functions used Data Acquisition, EDA, Sampling, Cleansing, Anomaly Detection, Azure AutoML Trigger, AutoML Trigger bypassing authentication to Azure ML(used for pipelining the notebook). 
2. AMLMasterNotebook_Trigger: Function calls in order to perform a pipeline of tasks. 
3. For sample dataset to be used in notebook refer- [Real Estate Data](https://archive.ics.uci.edu/ml/datasets/Real+estate+valuation+data+set) 
4. For sample notebook run refer- [Real Estate House Price Pred](https://github.com/microsoft/AcceleratedML/blob/d8050f9bbd87b02ddfa6180f4a9aa4caf39dace5/SampleRegression/Real%20Estate%20House%20Price%20Pred/Trigger_Final(Real-Estate%20House%20Price%20Pred).ipynb)

## Contributing
This project welcomes contributions and suggestions. Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit [https://cla.opensource.microsoft.com](https://cla.opensource.microsoft.com).

This project has adopted the [Microsoft Open Source Code of Conduct](https://cla.opensource.microsoft.com/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com?) with any additional questions or comments.
<br/>

*Note: To know in detail of the workings of Auto Brew ML Framework, please visit [Auto Brew ML WiKi](https://github.com/microsoft/AutoBrewML/wiki).*
