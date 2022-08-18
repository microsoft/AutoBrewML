
# %%
def BrewTrainTestSplit(input_dataframe,label_col,splitratio):
  columns_list = input_dataframe.columns
  column = 'Index'
  if column in columns_list:
    input_dataframe.drop(['Index'], axis=1, inplace=True)

  #Label Encoder Doesn't accept nulls so impute
  x_train_dummy = impute(input_dataframe)
  
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
              le.fit(input_dataframe[col])
              input_dataframe[col]=le.transform(input_dataframe[col])
                
  from sklearn.model_selection import train_test_split
  y_df = input_dataframe.pop(label_col)
  x_df = input_dataframe
  x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=splitratio, random_state=223)
  return x_train, x_test, y_train, y_test


#Calling
import pandas as pd
input_dataframe = pd.read_csv(r"C:\Users\srde\Downloads\carprices.csv")
print(input_dataframe.head())

label = 'Sell Price($)'
x_train, x_test, y_train, y_test = BrewTrainTestSplit(input_dataframe,label,0.2)


# %%
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


# %%

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from contextlib import contextmanager
import sys, os
import tpot

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

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


    # #Label Encoder Doesn't accept nulls so impute
    # x_train_dummy = impute(x_train_dummy)
    # x_test_dummy  = impute(x_test_dummy)

    # #Feature imp accept only numeric columns hence label encode
    # allCols = [col for col in x_train_dummy.columns]
    # catCols = [col for col in x_train_dummy.columns if x_train_dummy[col].dtype=="O"] 
    # # print(allCols)
    # # print(catCols)
    # if catCols: # If Categorical columns present (list not empty) then do label encode
    #     from sklearn.preprocessing import LabelEncoder
    #     le = LabelEncoder()
    #     for col in x_train_dummy.columns:
    #         if (x_train_dummy[col].dtype) not in ["int32","int64","float","float64"]:
    #             le.fit(x_train_dummy[col])
    #             x_train_dummy[col]=le.transform(x_train_dummy[col])
    #             x_test_dummy[col]=le.transform(x_test_dummy[col])


    # define model evaluation
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

    # define search
    model = TPOTRegressor(generations=1, population_size=50, scoring='neg_mean_absolute_error', cv=cv, verbosity=2, random_state=1, n_jobs=-1)
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




# %%
model,x_test_dummy=BrewAutoML_Regressor(x_train,y_train,x_test,y_test)
model




# %%
def BrewFairnessEvaluator_Regression(model,x_test,sensitivefeatures,y_test):
  import numpy as np
  from fairlearn.metrics import MetricFrame,mean_prediction,false_negative_rate
  from sklearn.metrics import mean_absolute_error
  x_test_dummy  = x_test.copy(deep=True)
  y_test_dummy  = y_test.copy(deep=True)


  df=x_test[sensitivefeatures]
  
  y_pred = model.predict(x_test_dummy).tolist()
  y_test2=y_test_dummy.astype(np.int32).tolist()
  
  gm = MetricFrame(metrics=mean_absolute_error, y_true=y_test2, y_pred=y_pred,sensitive_features=df)
  print("Overall Accuracy:",gm.overall)
  print("Group Wise Accuracy:",gm.by_group)
  return (gm.overall,gm.by_group)


#Calling
y_pred = BrewFairnessEvaluator_Regression(model,x_test,'Car Model',y_test)





# # %%
# def BrewDisparityMitigation_Regression(x_test,sensitivefeatures,y_test):
#   import numpy as np
#   from fairlearn.metrics import MetricFrame
#   from sklearn.metrics import accuracy_score
#   from fairlearn.reductions import ErrorRateParity, ExponentiatedGradient, DemographicParity , BoundedGroupLoss
#   from sklearn.linear_model import LinearRegression
  
#   x_test_dummy  = x_test.copy(deep=True)
#   y_test_dummy  = y_test.copy(deep=True)
#   x_test_dummy['y_actual'] = y_test_dummy
  
#   np.random.seed(0)
#   sensitive=x_test_dummy[sensitivefeatures]
#   sensitive2=sensitive.astype('int')
#   sensitive3=sensitive2.astype('str')
#   sensitive4=sensitive3.astype('category')
#   x_test_dummy[sensitivefeatures]=x_test_dummy[sensitivefeatures].astype('int').astype('str')
#   y_test2=y_test_dummy.astype(np.int32).tolist()
#   constraint = BoundedGroupLoss(loss=0)
#   regr = LinearRegression()
#   mitigator = ExponentiatedGradient(regr, constraint)
#   mitigator.fit(x_test_dummy, y_test2, sensitive_features=sensitive4)
#   y_pred_mitigated = mitigator.predict(x_test_dummy)
  
#   x_test_dummy['y_predict_mitigated'] = y_pred_mitigated
  
    
    
#   gm = MetricFrame(metrics=accuracy_score, y_true=y_test2, y_pred=y_pred_mitigated,sensitive_features=sensitive4)
#   print("Overall Accuracy:",gm.overall)
#   print("Group Wise Accuracy:",gm.by_group)
#   return (mitigator,x_test_dummy)

# y_pred = BrewFairnessEvaluator_Regression(model,x_test,'Car Model',y_test)
# y_pred


# # %%
# Mitigated_model,Mitigated_Pred=BrewDisparityMitigation_Regression(x_test,'Car Model',y_test)

# # %%
# y_test

# # %%
