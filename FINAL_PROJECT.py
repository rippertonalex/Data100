#!/usr/bin/env python
# coding: utf-8

# # **Final Project**
# **Team Members:** Kevin Coulson, Chandler Ooms, Alex Ripperton
# 
# 1. Perform exploratory data analysis (EDA) and include in your report at least two data visualizations.
# 2. Describe any data cleaning or transformations that you perform and why they are motivated by your
# EDA.
# 3. Apply relevant inference or prediction methods (e.g., linear regression, logistic regression, or classification and regression trees), including, if appropriate, feature engineering and regularization.
# 4. Use cross-validation or test data as appropriate for model selection and evaluation. Make sure to
# carefully describe the methods you are using and why they are appropriate for the question to be
# answered.
# 5. Summarize and interpret your results (including visualization).
# 6. Provide an evaluation of your approach and discuss any limitations of the methods you used.
# 7. Describe any surprising discoveries that you made and future work.
# The analysis must involve at least one of the inference or prediction methods presented in this course.

# **Important Links:**
# 
# [Report Google Doc](https://docs.google.com/document/d/1TYyvvzBG0y7Z2ndw2CZvpdlg-F1Nrfa9-cLIiIIOzPo/edit?usp=sharing)
# 
# [Project Description](http://www.ds100.org/sp20/resources/assets/final_proj/final_proj_spec.pdf)
# 
# [Project Rubric](https://d1b10bmlvqabco.cloudfront.net/attach/k4zyqkjkyt33a2/j4f6z772zscwl/k9ns0abjn95w/Undergrad_DS_100_Final_Project_Rubric_Release.pdf)

# # Visualizations - Kevin

# In[5]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime as dt
get_ipython().run_line_magic('matplotlib', 'inline')



from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import tree
from sklearn import ensemble

from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    county_json = json.load(response)


# ## Import data

# In[6]:


states = pd.read_csv('4.18states.csv')
counties = pd.read_csv('abridged_couties.csv')
confirmed = pd.read_csv('time_series_covid19_confirmed_US.csv')
deaths = pd.read_csv('time_series_covid19_deaths_US.csv')


# ## Clean data

# In[7]:


states_clean = (states.fillna(0)
                [(states['ISO3'] == 'USA')]
               .set_index('Province_State'))
states_clean


# In[4]:


counties.loc[counties['StateName'] == 'AK', 'State'] = 'Alaska'
counties_clean = (counties.fillna(0)
                  [(counties['State'].isin(states_clean.index))]
                  .sort_values(by = ['State', 'CountyName']))
counties_clean['countyFIPS'] = pd.to_numeric(counties_clean['countyFIPS'])
counties_clean = counties_clean.set_index('countyFIPS')
counties_clean


# In[5]:


confirmed_clean = (confirmed.fillna(0)
             [(confirmed['iso3'] == 'USA') & (confirmed['Lat'] != 0) & (confirmed['FIPS'] > 0)])
confirmed_clean['FIPS'] = confirmed_clean['FIPS'].astype(int)
confirmed_clean = confirmed_clean.set_index('FIPS')
confirmed_clean


# In[6]:


deaths_clean = (deaths.fillna(0)
          [(deaths['iso3'] == 'USA') & (deaths['Lat'] != 0) & (deaths['FIPS'] > 0)])
deaths_clean['FIPS'] = deaths_clean['FIPS'].astype(int)
deaths_clean = deaths_clean.set_index('FIPS')
deaths_clean


# ## Visualization 1: On a state level, how do resources and population type affect mortality rate?

# **Weighted average by state function:**

# In[7]:


def WEIGHTED_DF(df, value_col, weight_col):
    new_col_name = value_col +'_weighted'
    df['times_weight'] = df[value_col] * df[weight_col]
    weighted_df = df.groupby('State').sum()
    return pd.DataFrame(weighted_df['times_weight'] / weighted_df[weight_col], columns = [new_col_name])


# **Sum by state function:**

# In[8]:


def SUM_DF(df, value_col):
    return df.groupby('State').sum()[[value_col]]


# **Create new columns per state based on county data:**

# In[9]:


state_pop_total = SUM_DF(counties_clean, 'PopulationEstimate2018')
states_clean['Population'] = state_pop_total

state_pop_senior = SUM_DF(counties_clean, 'PopulationEstimate65+2017')
states_clean['Population65+Pct'] = state_pop_senior['PopulationEstimate65+2017'] / states_clean['Population']

state_pop_density = WEIGHTED_DF(counties_clean, 'PopulationDensityperSqMile2010', 'PopulationEstimate2018')
states_clean['PopulationDensity'] = state_pop_density

state_age = WEIGHTED_DF(counties_clean, 'MedianAge2010', 'PopulationEstimate2018')
states_clean['MedianAge'] = state_age

state_urban_rural = WEIGHTED_DF(counties_clean, 'Rural-UrbanContinuumCode2013', 'PopulationEstimate2018')
states_clean['RUCC'] = state_urban_rural

state_heart_mort = WEIGHTED_DF(counties_clean, 'HeartDiseaseMortality', 'PopulationEstimate2018')
states_clean['HeartMortRate'] = state_heart_mort

state_smokers = WEIGHTED_DF(counties_clean, 'Smokers_Percentage', 'PopulationEstimate2018')
states_clean['SmokerPct'] = state_smokers

state_resp_mort = WEIGHTED_DF(counties_clean, 'RespMortalityRate2014', 'PopulationEstimate2018')
states_clean['RespMortRate'] = state_resp_mort

state_medicare_eligible = SUM_DF(counties_clean, '#EligibleforMedicare2018')
states_clean['MedicareEligibilityPct'] = state_medicare_eligible['#EligibleforMedicare2018'] / states_clean['Population']

state_medicare_enroll = SUM_DF(counties_clean, 'MedicareEnrollment,AgedTot2017')
states_clean['MedicareEnrollmentPct'] = state_medicare_enroll['MedicareEnrollment,AgedTot2017'] / states_clean['Population']

state_hosp_hrs = SUM_DF(counties_clean, '#FTEHospitalTotal2017')
states_clean['HospitalHoursPerCapita'] = state_hosp_hrs['#FTEHospitalTotal2017'] / states_clean['Population']

state_mds = SUM_DF(counties_clean, "TotalM.D.'s,TotNon-FedandFed2017")
states_clean['M.D.sPerCapita'] = state_mds["TotalM.D.'s,TotNon-FedandFed2017"] / states_clean['Population']

state_hospitals = SUM_DF(counties_clean, '#Hospitals')
states_clean['HospitalsPerCapita'] = state_hospitals['#Hospitals'] / states_clean['Population']

state_icus = SUM_DF(counties_clean, '#ICU_beds')
states_clean['ICU_bedsPerCapita'] = state_icus['#ICU_beds'] / states_clean['Population']

state_dem_rep = WEIGHTED_DF(counties_clean, 'dem_to_rep_ratio', 'PopulationEstimate2018')
states_clean['Dem-Rep-Ratio'] = state_dem_rep

state_SVI = WEIGHTED_DF(counties_clean, 'SVIPercentile', 'PopulationEstimate2018')
states_clean['SVI'] = state_SVI

state_HPSA_underserved = SUM_DF(counties_clean, 'HPSAUnderservedPop')
states_clean['HPSAUnderservedPct'] = state_HPSA_underserved['HPSAUnderservedPop'] / states_clean['Population']

states_clean.columns


# In[10]:


fig1, axs1 = plt.subplots(2,2)

x = 'HospitalsPerCapita'
y = 'ICU_bedsPerCapita'
color = 'Mortality_Rate'
size = 'RUCC'

cMap = axs1[0, 0].scatter(states_clean[x], states_clean[y], s = 400/states_clean[size], c = states_clean[color], cmap = 'jet')
axs1[0, 0].set(xlabel = 'Hospitals per Capita', ylabel = 'ICU Beds per Capita')

x = 'MedicareEnrollmentPct'
y = 'M.D.sPerCapita'
axs1[0, 1].scatter(states_clean[x], states_clean[y], s = 400/states_clean[size], c = states_clean[color], cmap = 'jet')
axs1[0, 1].set(xlabel = 'Medicare Enrollment %', ylabel = 'M.D.s per Capita')

x = 'PopulationDensity'
y = 'SVI'
axs1[1, 1].scatter(states_clean[x], states_clean[y], s = 400/states_clean[size], c = states_clean[color], cmap = 'jet')
axs1[1, 1].set(xlabel = 'Population Density', ylabel = 'SVI Percentile')
plt.xscale('log')

x = 'SmokerPct'
y = 'RespMortRate'
axs1[1, 0].scatter(states_clean[x], states_clean[y], s = 400/states_clean[size], c = states_clean[color], cmap = 'jet')
axs1[1, 0].set(xlabel = 'Smoker %', ylabel = 'Respritory Mortality Rate')

cbar = fig1.colorbar(cMap, ax = axs1)
cbar.ax.set_ylabel('Mortality Rate')

fig1.suptitle('Population Type and Resource Type Correlations with Mortality Rate and Rural-Urban Continuum Code \n Color - Mortality Rate \n Size - Urban-Rural Continuum Code (Larger -> More Urban)')
fig1.set_size_inches(18, 10)
plt.show()


# ## Visualization 2: On a county basis, how do resources and population type affect confirmed cases?

# **Append total confirmed cases and deaths to county data**

# In[11]:


total_confirmed = confirmed_clean[['4/18/20']]
counties_clean['Total_Confirmed'] = total_confirmed['4/18/20']
counties_clean['Total_ConfirmedPct'] = total_confirmed['4/18/20'] / counties_clean['PopulationEstimate2018']

total_dead = deaths_clean[['4/18/20']]
counties_clean['Total_Deaths'] = total_dead['4/18/20']
counties_clean['Total_DeathsPct'] = total_dead['4/18/20'] / counties_clean['PopulationEstimate2018']

counties_clean.columns


# **Map Dem-Rep ratio to better represent data points**

# In[12]:


counties_clean.loc[counties_clean['dem_to_rep_ratio'] >= 4, 'DemRepScore'] = 0
counties_clean.loc[(counties_clean['dem_to_rep_ratio'] >= 2) & (counties_clean['dem_to_rep_ratio'] < 4), 'DemRepScore'] = 0.2
counties_clean.loc[(counties_clean['dem_to_rep_ratio'] >= 1.1) & (counties_clean['dem_to_rep_ratio'] < 2), 'DemRepScore'] = 0.4
counties_clean.loc[(counties_clean['dem_to_rep_ratio'] >= 1/1.1) & (counties_clean['dem_to_rep_ratio'] < 1.1), 'DemRepScore'] = 0.5
counties_clean.loc[(counties_clean['dem_to_rep_ratio'] >= 1/2) & (counties_clean['dem_to_rep_ratio'] < 1/1.1), 'DemRepScore'] = 0.8
counties_clean.loc[(counties_clean['dem_to_rep_ratio'] >= 1/4) & (counties_clean['dem_to_rep_ratio'] < 1/2), 'DemRepScore'] = 0.6
counties_clean.loc[counties_clean['dem_to_rep_ratio'] < 1/4, 'DemRepScore'] = 1
plt.hist(counties_clean['DemRepScore'])


# In[13]:


fig2, axs2 = plt.subplots(2,2)

x = 'MedianAge2010'
y = 'Total_ConfirmedPct'
color = 'DemRepScore'

cMap2 = axs2[0,0].scatter(counties_clean[x], counties_clean[y], c = counties_clean[color], cmap = 'bwr')
axs2[0, 0].set(ylabel = 'Total Confirmed Case % of Population')

x = '#FTEHospitalTotal2017'
axs2[0, 1].scatter(counties_clean[x], counties_clean[y], c = counties_clean[color], cmap = 'bwr')

x = 'MedianAge2010'
y = 'Total_DeathsPct'
axs2[1, 0].scatter(counties_clean[x], counties_clean[y], c = counties_clean[color], cmap = 'bwr')
axs2[1, 0].set(xlabel = 'Median Age', ylabel = 'Total Deaths % of Population')

x = '#FTEHospitalTotal2017'
axs2[1, 1].scatter(counties_clean[x], counties_clean[y], c = counties_clean[color], cmap = 'bwr')
axs2[1, 1].set(xlabel = 'Hospital Hours')

cbar = fig1.colorbar(cMap2, ax = axs2)
cbar.ax.set_ylabel('Political Affiliation')
cbar.ax.set_yticklabels(['Democrat', '', '', '', '', 'Republican'])

fig2.suptitle('Population Type and Resource Type Correlations \n with Mortality/Case Totals by Political Affiliation')
fig2.set_size_inches(18, 10)
plt.show()


# ## Model: Predicting cases and deaths by state based on population type and resource availability.

# **Split Data**

# In[14]:


RES_cols = ['Population', 'RUCC', 'HospitalsPerCapita', 'ICU_bedsPerCapita', 'MedicareEnrollmentPct', 'HPSAUnderservedPct']
POP_cols = ['Population', 'RUCC', 'PopulationDensity', 'MedianAge', 'Population65+Pct', 'Dem-Rep-Ratio']

predicting = 'Confirmed'

TOT_cols = RES_cols + POP_cols

covid_train, covid_test = train_test_split(states_clean[TOT_cols + [predicting]], test_size = 0.2, random_state = 100)
covid_train = covid_train.fillna(0)
covid_test = covid_test.fillna(0)

model = (LinearRegression(fit_intercept = False, normalize = False)
                 .fit(covid_train[TOT_cols], covid_train[predicting]))


# In[1]:


covid_train['PredictedCases'] = model.predict(covid_train[TOT_cols])
covid_test['PredictedCases'] = model.predict(covid_test[TOT_cols])
train_accuracy = model.score(covid_train[TOT_cols], covid_train[predicting])
test_accuracy = model.score(covid_test[TOT_cols], covid_test[predicting])

confirmedTOT_test = covid_test['PredictedCases']
confirmedTOT_train = covid_train['PredictedCases']
confirmedTOT_true_test = covid_test['Confirmed']
confirmedTOT_true_train = covid_train['Confirmed']

train_accuracy, test_accuracy


# In[16]:


TOT_cols = RES_cols

covid_train, covid_test = train_test_split(states_clean[TOT_cols + [predicting]], test_size = 0.2, random_state = 100)
covid_train = covid_train.fillna(0)
covid_test = covid_test.fillna(0)

model = (LinearRegression(fit_intercept = False, normalize = True)
                 .fit(covid_train[TOT_cols], covid_train[predicting]))


# In[17]:


covid_train['PredictedCasesRES'] = model.predict(covid_train[TOT_cols])
covid_test['PredictedCasesRES'] = model.predict(covid_test[TOT_cols])
train_accuracy = model.score(covid_train[TOT_cols], covid_train[predicting])
test_accuracy = model.score(covid_test[TOT_cols], covid_test[predicting])

confirmedRES_test = covid_test['PredictedCasesRES']
confirmedRES_train = covid_train['PredictedCasesRES']
confirmedRES_true_test = covid_test['Confirmed']
confirmedRES_true_train = covid_train['Confirmed']

train_accuracy, test_accuracy


# In[18]:


TOT_cols = POP_cols

covid_train, covid_test = train_test_split(states_clean[TOT_cols + [predicting]], test_size = 0.2, random_state = 100)
covid_train = covid_train.fillna(0)
covid_test = covid_test.fillna(0)

model = (LinearRegression(fit_intercept = False, normalize = True)
                 .fit(covid_train[TOT_cols], covid_train[predicting]))


# In[19]:


covid_train['PredictedCasesPOP'] = model.predict(covid_train[TOT_cols])
covid_test['PredictedCasesPOP'] = model.predict(covid_test[TOT_cols])
train_accuracy = model.score(covid_train[TOT_cols], covid_train[predicting])
test_accuracy = model.score(covid_test[TOT_cols], covid_test[predicting])

confirmedPOP_test = covid_test['PredictedCasesPOP']
confirmedPOP_train = covid_train['PredictedCasesPOP']
confirmedPOP_true_test = covid_test['Confirmed']
confirmedPOP_true_train = covid_train['Confirmed']

train_accuracy, test_accuracy


# In[20]:


predicting = 'Deaths'

TOT_cols = RES_cols + POP_cols

covid_train, covid_test = train_test_split(states_clean[TOT_cols + [predicting]], test_size = 0.2, random_state = 100)
covid_train = covid_train.fillna(0)
covid_test = covid_test.fillna(0)

model = (LinearRegression(fit_intercept = False, normalize = False)
                 .fit(covid_train[TOT_cols], covid_train[predicting]))


# In[21]:


covid_train['PredictedCases'] = model.predict(covid_train[TOT_cols])
covid_test['PredictedCases'] = model.predict(covid_test[TOT_cols])
train_accuracy = model.score(covid_train[TOT_cols], covid_train[predicting])
test_accuracy = model.score(covid_test[TOT_cols], covid_test[predicting])

deathsTOT_test = covid_test['PredictedCases']
deathsTOT_train = covid_train['PredictedCases']
deathsTOT_true_test = covid_test['Deaths']
deathsTOT_true_train = covid_train['Deaths']

train_accuracy, test_accuracy


# In[22]:


TOT_cols = RES_cols

covid_train, covid_test = train_test_split(states_clean[TOT_cols + [predicting]], test_size = 0.2, random_state = 100)
covid_train = covid_train.fillna(0)
covid_test = covid_test.fillna(0)

model = (LinearRegression(fit_intercept = False, normalize = True)
                 .fit(covid_train[TOT_cols], covid_train[predicting]))


# In[23]:


covid_train['PredictedCasesRES'] = model.predict(covid_train[TOT_cols])
covid_test['PredictedCasesRES'] = model.predict(covid_test[TOT_cols])
train_accuracy = model.score(covid_train[TOT_cols], covid_train[predicting])
test_accuracy = model.score(covid_test[TOT_cols], covid_test[predicting])

deathsRES_test = covid_test['PredictedCasesRES']
deathsRES_train = covid_train['PredictedCasesRES']
deathsRES_true_test = covid_test['Deaths']
deathsRES_true_train = covid_train['Deaths']

train_accuracy, test_accuracy


# In[24]:


TOT_cols = POP_cols

covid_train, covid_test = train_test_split(states_clean[TOT_cols + [predicting]], test_size = 0.2, random_state = 100)
covid_train = covid_train.fillna(0)
covid_test = covid_test.fillna(0)

model = (LinearRegression(fit_intercept = False, normalize = True)
                 .fit(covid_train[TOT_cols], covid_train[predicting]))


# In[25]:


covid_train['PredictedCasesPOP'] = model.predict(covid_train[TOT_cols])
covid_test['PredictedCasesPOP'] = model.predict(covid_test[TOT_cols])
train_accuracy = model.score(covid_train[TOT_cols], covid_train[predicting])
test_accuracy = model.score(covid_test[TOT_cols], covid_test[predicting])

deathsPOP_test = covid_test['PredictedCasesPOP']
deathsPOP_train = covid_train['PredictedCasesPOP']
deathsPOP_true_test = covid_test['Deaths']
deathsPOP_true_train = covid_train['Deaths']

train_accuracy, test_accuracy


# In[26]:


fig3, axs3 = plt.subplots(2, 2)

axs3[0, 0].scatter(confirmedTOT_true_test, confirmedTOT_test)
axs3[0, 0].scatter(confirmedRES_true_test, confirmedRES_test)
axs3[0, 0].scatter(confirmedPOP_true_test, confirmedPOP_test)
axs3[0, 0].scatter(confirmedPOP_true_test, confirmedPOP_true_test)
axs3[0, 0].set(xlabel = 'True Confirmed Case Count', ylabel = 'Predicted Confirmed Case Count')
axs3[0,0].title.set_text('Test Data')

axs3[1, 0].scatter(deathsTOT_true_test, deathsTOT_test)
axs3[1, 0].scatter(deathsRES_true_test, deathsRES_test)
axs3[1, 0].scatter(deathsPOP_true_test, deathsPOP_test)
axs3[1, 0].scatter(deathsPOP_true_test, deathsPOP_true_test)
axs3[1, 0].set(xlabel = 'True Death Count', ylabel = 'Predicted Death Count')

axs3[0, 1].scatter(confirmedTOT_true_train, confirmedTOT_train)
axs3[0, 1].scatter(confirmedRES_true_train, confirmedRES_train)
axs3[0, 1].scatter(confirmedPOP_true_train, confirmedPOP_train)
axs3[0, 1].scatter(confirmedPOP_true_train, confirmedPOP_true_train)
axs3[0, 1].set(xlabel = 'True Confirmed Case Count', ylabel = 'Predicted Death Count')
axs3[0,1].title.set_text('Training Data')

axs3[1, 1].scatter(deathsTOT_true_train, deathsTOT_train)
axs3[1, 1].scatter(deathsRES_true_train, deathsRES_train)
axs3[1, 1].scatter(deathsPOP_true_train, deathsPOP_train)
axs3[1, 1].scatter(deathsPOP_true_train, deathsPOP_true_train)
axs3[1, 1].set(xlabel = 'True Death Count', ylabel = 'Predicted Death Count;')

axs3[0, 1].legend(['Population and resource prediction', 'Resource prediction only', 'Population type prediction only', 'True results'])

fig3.suptitle('Linear Regression Model Accuracy for Test and Training Data')
fig3.set_size_inches(12, 10)
plt.show()


# ## Alex: Visualizations

# In[29]:


states = pd.read_csv('4.18states.csv')
counties = pd.read_csv('abridged_couties.csv')
confirmed = pd.read_csv('time_series_covid19_confirmed_US.csv')
deaths = pd.read_csv('time_series_covid19_deaths_US.csv')


# In[30]:


counties_grouped_by_state = counties.groupby('State').sum()
total_state_population = counties_grouped_by_state['PopulationEstimate2018']


# In[31]:


grouped_byState_confrimed = confirmed.groupby('Province_State').sum()
grouped_byState_confrimed_cleaned = grouped_byState_confrimed.drop(grouped_byState_confrimed.columns[[0,1,2,3,4,5,6,7,8,9,10]], axis = 1)
confirmed_cases_by_state = pd.DataFrame(grouped_byState_confrimed_cleaned['4/18/20'])
cleaned_confirmed_cases_by_state = confirmed_cases_by_state.drop(['American Samoa', 'Diamond Princess', 'Alaska', 'Grand Princess', 'Guam', 'Hawaii' , 'Northern Mariana Islands', 'Puerto Rico', 'Virgin Islands'])
per_capita_Percentage_49_state_cases = cleaned_confirmed_cases_by_state['4/18/20']/ np.array(total_state_population)


# In[32]:


county_count_tab = counties.groupby('State').count()
county_count = county_count_tab['lat']


# In[33]:


avg_pop_density_by_state = counties_grouped_by_state['PopulationDensityperSqMile2010'] / county_count


# In[34]:


vis_2 = {'Percentage of People in state with confirmed case': per_capita_Percentage_49_state_cases , 'State avg pop density' : avg_pop_density_by_state}
vis_2_df = pd.DataFrame(vis_2).drop(['District Of Columbia', 'District of Columbia'])


# In[35]:


plt.bar(vis_2_df['State avg pop density'], vis_2_df['Percentage of People in state with confirmed case'], width = 100)
plt.title('Cases and Population Density by state')
plt.ylabel('Percent of state population with case')
plt.xlabel('Avg Population Density')


# In[36]:


num_icu_beds = pd.DataFrame(counties_grouped_by_state['#ICU_beds']) 
num_icu_beds_per_capita = pd.DataFrame(num_icu_beds['#ICU_beds']/ np.array(total_state_population))


# In[37]:


percent_smokers = counties_grouped_by_state['Smokers_Percentage'] / county_count


# In[38]:


death_group_by_state = deaths.groupby('Province_State').sum()
death_group_by_state_cleaned = death_group_by_state.drop(['American Samoa', 'Diamond Princess', 'Alaska', 'Grand Princess', 'Guam', 'Hawaii' , 'Northern Mariana Islands', 'Puerto Rico', 'Virgin Islands' ])
deaths_by_state = death_group_by_state_cleaned['4/18/20']
Percentage_of_confirmed_cases_resulting_in_death = 100 *(deaths_by_state / cleaned_confirmed_cases_by_state['4/18/20'])
viz_3_dict = {'percent of confirmed cases resulting in death': Percentage_of_confirmed_cases_resulting_in_death, 'ICU Beds per capita' : num_icu_beds_per_capita['#ICU_beds'], 'Smoker Percentages': percent_smokers}
viz_3_df = pd.DataFrame(viz_3_dict).drop(['District Of Columbia', 'District of Columbia'])


# In[39]:


plt.scatter(viz_3_df['Smoker Percentages'], viz_3_df['percent of confirmed cases resulting in death'])
plt.title('Smoker Percentages and COVID Mortality')
plt.xlabel('Percent of People that are Smokers')
plt.ylabel('Percent of COVID cases that are fatal')


# The relationship between the amount of smokers in a state and the percentage of COVID patients that end up dying from COVID. Seems to not be a strong correlation which is suprising.

# # Chandler: Visualizations

# ## Step 1: Load and Clean Data + EDA
# 1. Perform exploratory data analysis (EDA) and include in your report at least two data visualizations.
# 2. Describe any data cleaning or transformations that you perform and why they are motivated by your EDA.

# In[3]:


# 4 Data Sets
states = pd.read_csv('4.18states.csv')
counties = pd.read_csv('abridged_couties.csv')
confirmed = pd.read_csv('time_series_covid19_confirmed_US.csv')
deaths = pd.read_csv('time_series_covid19_deaths_US.csv')


# In[4]:


print(states.shape)
print(counties.shape)
print(confirmed.shape)
print(deaths.shape)


# ## Cleaning Datasets into Useful Information
# **IMPORTANT ALL COUNTY INFORMATION ONLY INCLUDES THE 50 STATES AND SHOULD ONLY INCLUDE 3142 COUNTIES**
# 1. create a list of county names for use later
# 2. narrow down our core data into `case_stats` for easier access throughout the project
# 3. eliminate all data from outside the 50 states/unassigned counties
# 4. drop new york because it is an outlier

# In[5]:


county_names = list(deaths[['Admin2']].drop([0,1,2,3,4]).iloc[0:3142,:]['Admin2'])

cleaned_deaths = deaths[['UID','FIPS','Admin2','Province_State','Population','4/18/20']].rename({'Admin2':"County",'4/18/20':'deaths 4/18','Province_State':'State'}, axis=1).drop([0,1,2,3,4]).reset_index(drop=True)
cleaned_deaths = cleaned_deaths[cleaned_deaths['FIPS']<80000]
cleaned_confirmed = confirmed[['UID','FIPS','Admin2','Province_State','4/18/20']].rename({'Admin2':"County",'4/18/20':'confirmed 4/18','Province_State':'State'}, axis=1).drop([0,1,2,3,4]).reset_index(drop=True)
cleaned_confirmed['Population'] = cleaned_deaths['Population']
cleaned_confirmed = cleaned_confirmed[cleaned_confirmed['FIPS']<80000]

cleaned_counties = counties.copy()
cleaned_counties = cleaned_counties[cleaned_counties['STATEFP']<60]
cleaned_counties = (counties.fillna(0)
                  [(counties['State'].isin(states['Province_State']))]
                  .sort_values(by = ['State', 'CountyName']))

# includes deaths/num confirmed cases
cleaned_counties['deaths'] = cleaned_deaths['deaths 4/18']
cleaned_counties['confirmed'] = cleaned_confirmed['confirmed 4/18']

# case stats combines cleaned death/confirmed into one table
case_stats = cleaned_deaths.copy()
case_stats['confirmed 4/18'] = cleaned_confirmed['confirmed 4/18']


# In[6]:


def clean(string):
    if string < 10000:
        return '0{0}'.format(string)
    else:
        return string
    
map_data = case_stats.copy()
map_data['FIPS'] = map_data['FIPS'].astype('int').apply(lambda x: clean(x))
map_data.head()


# ### Figure 1 
# Figure 1 aims to show the amount of confirmed cases and the political affiliations of each county in the United States. 

# In[1]:


fig = px.choropleth(map_data, geojson=county_json, locations='FIPS', color='confirmed 4/18',
                           color_continuous_scale="plasma",
                           range_color=(0, 100),
                           scope="usa",
                           labels={'confirmed 4/18':'Total Confirmed Cases'}
                          )
fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0}, 
                  title='Confirmed Cases of COVID-19 by County (Fig 1.1)',
                  width=700,
                  height=350)
fig.show()
fig2 = px.choropleth(cleaned_counties, geojson=county_json, locations='countyFIPS', color='dem_to_rep_ratio',
                           color_continuous_scale="bluered_r",
                           range_color=(0, 2),
                           scope="usa",
                           labels={'dem_to_rep_ratio':'Political Affiliation by County'}
                          )
fig2.update_layout(margin={"r":0,"t":40,"l":0,"b":0},
                   title='Political Affiliation by County 2016 (Fig 1.2)',
                   width=700,
                   height=350)
fig2.show()
fig = px.choropleth(cleaned_counties, geojson=county_json, locations='countyFIPS', color='PopulationDensityperSqMile2010',
                           color_continuous_scale="inferno",
                           range_color=(0, 200),
                           scope="usa",
                           hover_data = ['CountyName'],
                           labels={'PopulationDensityperSqMile2010':'Population Density'}
                          )
fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0}, 
                  title='Population Density by County (Fig 1.3)',
                  width=700,
                  height=400)
fig.show()
fig = px.choropleth(cleaned_counties, geojson=county_json, locations='countyFIPS', color='SVIPercentile',
                           color_continuous_scale="OrRd",
                           range_color=(0, 1),
                           scope="usa",
                           hover_data = ['CountyName'],
                           labels={'SVIPercentile':'SVI'}
                          )
fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0}, 
                  title='Social Vulnerability Index by County (Fig 1.4)',
                  width=700,
                  height=400)
fig.show()


# ### Choropleth Maps of COVID-19
# 
# The exploratory data analysis for this plot required me to fix the `FIPS` (county identification number) so that it would match up with the the given county location data from plotly's `geojson` data. I further wanted to ignore anything outside the continental United States.
# 
# Through these maps of the United States, our team hopes to show the disparity of COVID-19 cases throughout the differing population and resource makeups of the United States. Fig 1.1 shows the amount of confirmed cases, with yellow meaning 100+ cases. It connects closely with Fig 1.3 where higher population density leads to more confirmed cases. This further connects with Fig 1.2 and political affiliation as typically more densley populated counties lean more liberal in affiliation. However, I was hoping Figure 1.4 would provide more insight into social vulnerability of a county (and in connection with poorer health care opportunities). 
# 

# ## Social Vulnerability Index and Mortality Rates
# In this visualization, we are going to calculate a county's `SVIPercentile` from `county` which tells us the overall percentile ranking indicating the CDC's Social Vulnerability Index (SVI); higher ranking indicates greater social vulnerability. We will then compare it against the county's `mortality rate` which is calculated from $\frac{Deaths}{Population} * 1000$
# 
# **Takeaways:** Clearly there is nothing of value in this plot, however it doesn't really account for anything such as rural or urban, etc. It does not seem to matter if the county is socially vulnerable or not. 

# In[8]:


svi_mortality = case_stats.copy()
svi_mortality['mortality rate'] = svi_mortality['deaths 4/18']/svi_mortality['Population']*1000
svi_mortality['SVI'] = cleaned_counties['SVIPercentile'].iloc[:3142]
svi_mortality= svi_mortality[svi_mortality['mortality rate']>.3]
svi_mortality


# In[9]:


fig = px.scatter(svi_mortality,
                x='SVI',
                y='mortality rate',
                text=list(svi_mortality['County']),
                hover_data=['State'])
fig.update_layout(height=450,
                  width =800,
                  title_text='Mortality Rate per 1000 individuals (Fig 2.1)')
fig.show();


# ### Figure 2.1 Mortality Rate per 1000 
# 
# By plotting out the mortality rates on a per capita basis, I wanted to understand if there was any sort of trend between my earlier observations regarding SVI, population and confirmed cases. Based on the data observed here, it didn't reveal any trends. Further explanation would require me to better account for population density, speed of transmission and 

# **Let's try that again and this time do deaths/confirmed cases**

# In[10]:


svi_deaths = case_stats.copy()
svi_deaths['death rate'] = svi_deaths['deaths 4/18']/svi_mortality['confirmed 4/18']
svi_deaths['SVI'] = cleaned_counties['SVIPercentile'].iloc[:3142]

svi_deaths= svi_deaths[svi_deaths['death rate']>.05]
svi_deaths.head()


# In[11]:


fig = px.scatter(svi_deaths,
                x='SVI',
                y='death rate',
                hover_data=['State'])
fig.update_layout(height=600,
                  width = 800,
                  title_text='Death Rate for Confirmed Coronavirus Cases (Fig 2.2)')
fig.show();


# ### Figure 2.2 Death Rate per 1000 given County SVI
# 
# EDA for this data meant I had to eliminate all the counties that had under .05 death rate or negligible confirmed cases. There is an increase in death rates when the SVI surpasses $.6$. However, There are not a lot of data points to go off of, but when considering the dispersal of resources - it is important to remember that historically disadvantaged communities are in need of support.
