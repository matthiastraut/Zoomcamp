#!/usr/bin/env python
# coding: utf-8

# In[102]:


import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor


# In[103]:


data_results = pd.read_csv("https://raw.githubusercontent.com/tonmcg/US_County_Level_Election_Results_08-24/refs/heads/master/2024_US_County_Level_Presidential_Results.csv", dtype={'county_fips': str})


# In[104]:


data_results.head()


# In[105]:


# Due to duplicate county names and differenes in spelling, we identify counties by fips code
data_results.set_index('county_fips', inplace=True)
# We further need to 


# In[106]:


# To simplify the analysis, we focus on the two-party vote, ignoring third-party candidates
data_results['Trump'] = data_results['votes_gop'] / (data_results['votes_gop'] + data_results['votes_dem'])
data_results['Harris'] = data_results['votes_dem'] / (data_results['votes_gop'] + data_results['votes_dem'])


# In[107]:


data_results.head()


# In[108]:


# Now we are going to import the relevant Census data, obtained via the Census API https://www.census.gov/data/developers/data-sets.html
# Since this requires username & password, the dataset was produced using a different script but the data are saved as census_data_2024_final.pkl

data_census = pd.read_pickle("census_data_2024_final.pkl")


# In[109]:


data_census.head()


# In[110]:


# We will remove some superfluous variables
data_census.drop(columns=['state', 'county', 'county_name'], inplace=True)
census_cols = data_census.columns.to_list()


# In[111]:


# One variable that would be very useful to have is the unemployment rate by county, 
# which is not included in the Census data but can be found here: https://www.bls.gov/lau/tables.htm
data_urate = pd.read_excel("laucntycur14.xlsx", skiprows=2, dtype={'State FIPS Code': str, 'County FIPS Code': str})
data_urate = data_urate[data_urate['Period'] == 'Oct-24']
data_urate['fips'] = data_urate['State FIPS Code'] + data_urate['County FIPS Code']
data_urate.set_index('fips', inplace=True)
data_urate = data_urate.rename(columns={'Unemploy-ment Rate (%)': 'urate'})


# In[112]:


data_urate.head()


# In[113]:


# We will now merge the three dataframes & remove the counties with missing data
data = pd.concat([data_results, data_census.astype(float), data_urate['urate'].astype(float)], axis=1).dropna(how='any')
data_train_full, data_test = train_test_split(data, train_size=0.8)
data_train, data_val = train_test_split(data_train_full, train_size=0.75)


# In[ ]:


# We can now try to predict the votes for Trump vs Harris via the census variables
cols = census_cols + ['urate']
xgb = XGBRegressor(seed=1)
xgb.fit(X=data_train[cols], y=data_train['Trump'])


# In[118]:


pd.DataFrame(index=xgb.feature_names_in_, data=xgb.feature_importances_).sort_values(by=0, ascending=False).plot()


# In[ ]:


# We find that explanatory factors become much less potent below the ~0.05 threshold, so we cut off our variables just a little below 5%
expl_variables = pd.DataFrame(index=xgb.feature_names_in_, data=xgb.feature_importances_).sort_values(by=0, ascending=False)
expl_variables = expl_variables[expl_variables[0] >= 0.025].index.to_list()
expl_variables = expl_variables[:5]
expl_variables


# In[ ]:


# Now training the model with only the important explanatory variables
# xgb = XGBRegressor()
xgb.fit(X=data_train[expl_variables], y=data_train['Trump'])
y_pred = xgb.predict(data_val[expl_variables])

# Calculation of RMSE on the validation set
rmse = mean_squared_error(y_pred, data_val['Trump']) ** 1/2
rmse


# In[131]:


# We obtain a decent fit for our model but it's by no means perfect
plt.scatter(x=y_pred, y=data_val['Trump'])


# In[132]:


# Let's now apply the results to the test set
y_pred = xgb.predict(data_test[expl_variables])

# Calculation of RMSE on the validation set
rmse = mean_squared_error(y_pred, data_test['Trump']) ** 1/2
print(rmse)
plt.scatter(x=y_pred, y=data_test['Trump'])


# In[ ]:


# We'll write our own "vectorizer" that takes as input a dictionary and produces a DataFrame that can be fed into the model
input_data = {'pct_above_bachelors': 0.1,
 'race_pct_white': 0.9,
 'pct_bachelors_and_above': 0.3,
 'race_pct_asian': 0.00,
 'workers_wo_health_ins': 0.9}

def dv(input_data):
    input_df = pd.DataFrame(index=expl_variables, data=list(input_data.values()))
    input_df = input_df.T
    return input_df

# X = dv(input_data)
# xgb_loaded.predict(X)[0]


# In[135]:


# Save the final model
#with open('model.pkl', 'wb') as f:
#    pickle.dump((dv, xgb), f)


# In[136]:


# Loading the xgboost model with pickle
#with open('model.pkl', 'rb') as f:
#    (dv, xgb_loaded) = pickle.load(f)
#
# Verification that model saving & loading was correct
#y_verify = xgb_loaded.predict(data_test[expl_variables])
#
# Calculation of RMSE on the validation set
#rmse = mean_squared_error(y_verify, data_test['Trump']) ** 1/2
#print(rmse)
#plt.scatter(x=y_verify, y=data_test['Trump'])


# In[137]:


# Even better, let's transform dv and the xgb model into a pipeline

def pipeline(X):
    return xgb.predict(dv(X))[0]

pipeline


# In[138]:


# Save the final model
with open('model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

# Loading the xgboost model with pickle
with open('model.pkl', 'rb') as f:
    pipeline = pickle.load(f)

pipeline(input_data)


# In[ ]:





# In[ ]:




