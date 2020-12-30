#!/usr/bin/env python
# coding: utf-8

# # Data 100 Final Project Covid-19
# Anna Litskevitch and Teresa Bodart
#
# ## Importing Necessary Modules

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Also installing and importing geopandas modules we will use in our visualizations
get_ipython().system('pip install geopandas')
get_ipython().system('pip install descartes')
import geopandas
from shapely.geometry import Point, Polygon
from sklearn.model_selection import KFold

from sklearn.base import clone
from sklearn import linear_model as lm


# ## Importing Data



# This provided data provides county level data about cases of COVID-19 from 1/23/20 t0 4/18/20
confirmed = pd.read_csv('Data/time_series_covid19_confirmed_US.csv')
# This provided data provides county level data about deaths from COVID-19 from 1/23/20 t0 4/18/20
deaths = pd.read_csv('Data/time_series_covid19_deaths_US.csv')
# This data has more specific information about county data
abridged_counties = pd.read_csv('Data/abridged_couties.csv')



# This is a shapefile imported as a geopandas data frame that I will use for visualizations.
from geopandas import gpd
countyshapes = pd.read_csv('Data/tl_2017_us_county.csv')
countyshapes['geometry'] = countyshapes.apply(lambda row: Point(row.X, row.Y), axis=1)
countyshapes  = gpd.GeoDataFrame(countyshapes)
countyshapes.head()


# ## Cleaning Data Frames



# First I will focus on the confirmed data frame
# I filter out all the counties that do not have a 840 code3, so that all the counties are located
# in the United States, which is the region we want to analyze
confirmed = confirmed[confirmed['code3'] == 840]

# I wanted to add information about how many days since the first case of COVID-19 occured in a county to
# 4/18/20, as it can give insight about how much a county may currently be affected
confirmednumbers = confirmed.loc[:,'1/22/20':'4/18/20']
numberofzeros = confirmednumbers.apply( lambda s : s.value_counts().get(0,0), axis=1)
confirmed['dayssincefirstcase'] = confirmed.shape[1] - numberofzeros

# I also divided the number of cases confirmed on April 18 divided by the days since the first
# case of COVID-19 which is a very crude parameter to represent how fast the epidemic is spreading
# in the county once it appears. fillna(0) accounts for counties with no cases
confirmed['rate'] = (confirmed['4/18/20']/confirmed['dayssincefirstcase']).fillna(0)

# Standardizing the county codes, so that they can be used to match on with other dataframes
# fillna(0) used to filter out rows that weren't real counties
confirmed['GEOID'] = confirmed['FIPS'].fillna(0).astype(int).astype(str)

# Renaming the column as the data is aggregated
confirmed['confirmedcases'] = confirmed['4/18/20']

# Selecting only the columns we believe to be relevent to our question
confirmed = confirmed[['GEOID', 'confirmedcases', 'dayssincefirstcase', 'rate']]
confirmed.head()



# Now for the deaths data frame
# I filter out all the counties that do not have a 840 code3, so that all the counties are located
# in the United States, which is the area we want to analyze
deaths = deaths[deaths['code3'] == 840]

# Renaming the column as 4/18/20 is the total
deaths['confirmeddeaths'] = deaths['4/18/20']

# Standardizing the county codes, so that they can be used to match on with other dataframes
# fillna(0) used to filter out rows that weren't real counties
deaths['GEOID'] = deaths['FIPS'].fillna(0).astype(int).astype(str)

# Selecting only the columns we believe to be relevent to our question
deaths = deaths[['GEOID', 'confirmeddeaths']]
deaths.head()




# Cleaning Abridged Counties
cleaned_abridged_counties = abridged_counties

# Filtering out counties not in the continental United States
# according to State FIPS codes https://www.nrcs.usda.gov/wps/portal/nrcs/detail/?cid=nrcs143_013696
cleaned_abridged_counties = cleaned_abridged_counties[cleaned_abridged_counties['STATEFP'] <= 56]

# Standardizing the county codes, so that they can be used to match with the other dataframes
cleaned_abridged_counties['GEOID'] = cleaned_abridged_counties['countyFIPS'].fillna(0).astype(int).astype(str)

# Removing very sparse columns that will not useful as parameters due to lack of data
# as I could see simply from scanning the dataset
sparsecolumns = ['3-YrDiabetes2015-17', '3-YrMortalityAge<1Year2015-17', '3-YrMortalityAge1-4Years2015-17',
                '3-YrMortalityAge5-14Years2015-17', '3-YrMortalityAge15-24Years2015-17',
                '3-YrMortalityAge25-34Years2015-17', '3-YrMortalityAge35-44Years2015-17',
                'mortality2015-17Estimated', 'HPSAShortage', 'HPSAServedPop',
                'HPSAUnderservedPop']
cleaned_abridged_counties.drop(sparsecolumns, axis=1, inplace=True)

# Removing columns that give redundant information
redundant = ['State', 'lat', 'lon', 'POP_LATITUDE', 'POP_LONGITUDE', 'CensusRegionName', 'COUNTYFP', 'countyFIPS']
cleaned_abridged_counties.drop(redundant, axis=1, inplace=True)
cleaned_abridged_counties.head()




# Getting rid of the sparse columns handled almost all of the NaN data in the dataframe, but
# one county in this dataframe that has no data is Yellowstone County with FIPS 30113. This county in not
# included in the confirmed or deaths dataframes. After researching, I found that this county had been
# integrated into Gallatin County 30031 in 1970, meaning that we can drop this county from our dataframe.
cleaned_abridged_counties = cleaned_abridged_counties[cleaned_abridged_counties['GEOID'] != '30113']
# Another county with limited data is Shannon County 46113, which also does not appear in confirmed or deaths.
# After another round of research I found that this county was renamed to Oglala Lakota County 46102, which
# does appear in the confirmed and deaths datasets but is not in abridged_couties. I am replacing the
# Shannon County row with FIPS 46102 to match with confirmed and deaths I will then fill the NaNs
# of this county with the mean values of the counties located in South Dakota, specifically those with a Rural-Urban
# continuum code the same as Oglala Lakota County of 9 (taken from the United States Department of Agriculture),
# as they likely resemble this county the closest.
southdakota9 = cleaned_abridged_counties.copy()
southdakota9 = southdakota9[southdakota9['StateName'] == 'SD']
southdakota9 = southdakota9[southdakota9['Rural-UrbanContinuumCode2013'] == 9.0]
southdakota9 = southdakota9.loc[:, 'Rural-UrbanContinuumCode2013': 'SVIPercentile']
southdakota9 = southdakota9.mean()

# Dataframe of original Shannon county to be appended to cleaned_abridged_counties
onlyshannon = cleaned_abridged_counties.copy()
onlyshannon = onlyshannon[onlyshannon['GEOID'] == '46113'].copy()
onlyshannon = onlyshannon.fillna(southdakota9)

# Based on other nearby counties
onlyshannon['CensusDivisionName'] = 'West North Central'

# Changing the county code to the new one used in confirmed and deaths
onlyshannon['GEOID'] = '46102'

# Replacing current row with the new filled in one
cleaned_abridged_counties = cleaned_abridged_counties[cleaned_abridged_counties['GEOID'] != '46113']
cleaned_abridged_counties = cleaned_abridged_counties.append(onlyshannon)
cleaned_abridged_counties.tail()




# I wanted to check whether other columns had a large amount of NaN values that I did not catch visually
nans = cleaned_abridged_counties.copy()
nans = nans.isnull().sum().to_frame().reset_index()
nans = nans[nans[0] != 0]
plt.figure(figsize=(10,5))
plt.xticks(rotation=90)

plt.bar(nans['index'], nans[0])
plt.title('Fig. 1: Null Values per column in Cleaned abridged_counties')
plt.xlabel('Column')
plt.ylabel('Number of Null Values');


# When going back and looking back on the columns that describe the times certain restrictions were put into place in the county, the null values corresponded to states that have not issued such guideline. I then think it would be fine to replace those values as 0, which would represent that no guidlines were put into place. I will also get rid of the 3 year mortality for 45-54 year olds column, as almost a third of the values are missing. The rest of columns for me have an acceptable amount of missing values, so that replacing the values with the mean of the column should be an acceptle guess for the missing value.



# Dropping column with extreme number of null values and replacing others with mean of column
cleaned_abridged_counties.drop(['3-YrMortalityAge45-54Years2015-17'],
                                                           axis=1, inplace=True)
cleaned_abridged_counties['stay at home'] = cleaned_abridged_counties['stay at home'].fillna(0)
cleaned_abridged_counties['>50 gatherings'] = cleaned_abridged_counties['>50 gatherings'].fillna(0)
cleaned_abridged_counties['>500 gatherings'] = cleaned_abridged_counties['>500 gatherings'].fillna(0)
cleaned_abridged_counties['public schools'] = cleaned_abridged_counties['public schools'].fillna(0)
cleaned_abridged_counties['restaurant dine-in'] = cleaned_abridged_counties['restaurant dine-in'].fillna(0)
cleaned_abridged_counties['entertainment/gym'] = cleaned_abridged_counties['entertainment/gym'].fillna(0)
cleaned_abridged_counties['federal guidelines'] = cleaned_abridged_counties['federal guidelines'].fillna(0)
cleaned_abridged_counties['foreign travel ban'] = cleaned_abridged_counties['foreign travel ban'].fillna(0)
cleaned_abridged_counties = cleaned_abridged_counties.fillna(cleaned_abridged_counties.mean())




# Checking that there are no null values
nans = cleaned_abridged_counties.copy()
nans = nans.isnull().sum().to_frame().reset_index()
nans = nans[nans[0] != 0]
nans




# Cleaning the shapefile dataframe to only include counties in the continental US
# and remove unnecessary columns
countyshapes = countyshapes[(countyshapes["INTPTLAT"].astype(float) > 24.00)
                            & (countyshapes["INTPTLON"].astype(float) < 100.00)
                            & (countyshapes["INTPTLAT"].astype(float) < 50.00)]

# Standardizing the county codes, so that they can be used to match on with other dataframes
countyshapes['GEOID'] = countyshapes['GEOID'].astype(int).astype(str)
countyshapes = countyshapes[['GEOID','INTPTLAT', 'INTPTLON', 'geometry']]




# Merging the data we cleaned into one comprehensive dataframe! Whew
cleaned_abridged_counties = cleaned_abridged_counties.merge(confirmed, how = 'left', on = 'GEOID')
cleaned_abridged_counties = cleaned_abridged_counties.merge(deaths, how = 'left', on = 'GEOID')
fulldata = countyshapes.merge(cleaned_abridged_counties, how = "left", on = "GEOID")



# Creating a geovisualization of total deaths across the U.S. current to 04/18/2020
f, ax = plt.subplots(1, figsize=(20, 8))
plt.title('Fig. 2: Total Deaths Across U.S.')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
ax = fulldata.plot(column = 'confirmeddeaths', ax = ax, legend = True, cmap = 'YlGnBu')

# Circle to show where the data is
plt.scatter(-74, 40.8, s=2000, color="none", edgecolor="red")
plt.show();


# Huh! The only thing I can see is a dot near New York...
# This makes sense, as New York has been hit the hardest, but it does mean this visualization is not very helpful. At least we know that New York is going to be a pretty large outlier. To make this visualization more useful, I am going to take the log of the number of deaths, hopefully dealing with the magnitude of the death count.



# Taking log of number of deaths and plotting
fulldata['loggeddeaths'] = np.log(fulldata['confirmeddeaths']).replace(-np.inf, 0)
f, ax = plt.subplots(1, figsize=(20, 8))
plt.title('Fig. 3: Log of Total Deaths Across U.S.')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
ax = fulldata.plot(column = 'loggeddeaths', ax = ax, legend = True, cmap = 'YlGnBu')
plt.show();


# Much better! Now we can see more information about the thing



# Now I am adding in columns for features I think will be important for our model, created from
# data already in other columns in the dataset. Case fatality rate is the fatality calculated from
# deaths/confirmed cases.
fulldata['case_fatality_rate'] = fulldata['confirmeddeaths']/fulldata['confirmedcases']

# Cause specific mortality million is the proportion of the population that has died due COVID-19,
# and we multiplied the rate by a million to make the numbers easier to work with and to better understand
# our loss function
fulldata['cause_specific_mortality_million'] = fulldata['confirmeddeaths']*1000000/fulldata['PopulationEstimate2018']

# Incidence rate ten thousand is the incidence of COVID-19 per 10,000 people
fulldata['incidence_rate_tenthousand'] = fulldata['confirmedcases']*10000/fulldata['PopulationEstimate2018']

# ICU beds per person by county
fulldata['icuperperson'] = fulldata['#ICU_beds']/fulldata['PopulationEstimate2018']

# ICU beds per hospital by county
fulldata['icuperhospital'] = (fulldata['#ICU_beds']/fulldata['#Hospitals']).fillna(0)

# Proportion of the population 65+ per county
fulldata['proportion65+'] = (fulldata['PopulationEstimate65+2017']/fulldata['PopulationEstimate2018'])




f, ax = plt.subplots(1, figsize=(20, 8))
plt.title('Fig. 4: Cause Specific Mortality Per Million People')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
ax = fulldata.plot(column = 'cause_specific_mortality_million', ax = ax, legend = True, cmap = 'YlGnBu')
plt.show();


# Only New York and a few other counties are showing up on our map. Again, to make the visualization more useful, I am taking the log of cause specific mortality rate.



# Taking log of cause specific mortality per million
fulldata['loggedcausespecific'] = np.log(fulldata['cause_specific_mortality_million']).replace(-np.inf, 0)
f, ax = plt.subplots(1, figsize=(20, 8))
plt.title('Fig. 5: Log of Cause Specific Mortality Per Million People')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
ax = fulldata.plot(column = 'loggedcausespecific', ax = ax, legend = True, cmap = 'YlGnBu')
plt.show();




# Creating a population density geovisualization to compare with Death Total map and cause specific mortality map
f, ax = plt.subplots(1, figsize=(20, 8))
plt.title('Fig. 6: Population Density Per Square Mile')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
fulldata['loggeddensity'] = np.log(fulldata['PopulationDensityperSqMile2010'])
ax = fulldata.plot(column = 'loggeddensity', ax = ax, legend = True, cmap = 'YlGnBu')
plt.show();




# Creating a plot to show whether or not stroke, heart disease, and respiratory mortality rates are linearly independent
# We concluded that there was a  linear relationship and that we should refrain from using more than one in our model
import plotly.express as px
x = fulldata['StrokeMortality']
y = fulldata['RespMortalityRate2014']
z = fulldata['HeartDiseaseMortality']

print('Fig 7: Correlation Between Stroke Mortality, Respiratory Mortality and Heart Disease Mortality')
px.scatter_3d(fulldata, x='StrokeMortality', y='RespMortalityRate2014', z = 'HeartDiseaseMortality',
              color = np.linspace(0, 100, 3108),
              range_x = [x.min(), x.max()], range_y = [y.min(), y.max()], range_z = [z.min(), z.max()])


# ## Model



def process_couties_data(data, outcome_column, columns):
    data = data[[outcome_column] + columns]
    # Return predictors and response variables separately
    X = data.drop([outcome_column], axis = 1)
    y = data.loc[:, outcome_column]

    return X, y




def rmse(predicted, actual):
    # Function taken from Homework 6
    """
    Calculates RMSE from actual and predicted values
    Input:
      predicted (1D array): vector of predicted/fitted values
      actual (1D array): vector of actual values
    Output:
      a float, the root-mean square error
    """
    return np.sqrt(np.mean((actual - predicted)**2))

def cross_validate_rmse(model, X, y):
    # Function taken from Homework 6
    model = clone(model)
    five_fold = KFold(n_splits=5)
    rmse_values = []
    for tr_ind, va_ind in five_fold.split(X):
        model.fit(X.iloc[tr_ind,:], y.iloc[tr_ind])
        rmse_values.append(rmse(y.iloc[va_ind], model.predict(X.iloc[va_ind,:])))
    return np.mean(rmse_values)




# First attempt at a linear model, using all of the numerical, non-redundant columns, and our created columns
allcolumns = list(fulldata.loc[:,'Rural-UrbanContinuumCode2013':'rate'].columns)
allcolumns = allcolumns + ['loggeddensity',
       'icuperperson', 'icuperhospital', 'proportion65+', 'incidence_rate_tenthousand']
linear_model = lm.LinearRegression(fit_intercept=True)
X, y = process_couties_data(fulldata, 'cause_specific_mortality_million', allcolumns)
error = cross_validate_rmse(linear_model, X, y)
error




# Our error seems fairly reasonable, but we decided to look into the residuals to see how well the linear model
# fit the data
linear_model.fit(X,y)
y_pred = linear_model.predict(X)
from scipy.stats import gaussian_kde
xy = np.vstack([y,y - y_pred])
z = gaussian_kde(y*(y - y_pred))(y*(y - y_pred))

fig, ax = plt.subplots()
ax.scatter(y, y - y_pred, c=z, s=100, cmap ='winter')
plt.title('Fig. 8: Residuals vs Observed')
plt.xlabel('Observed Values')
plt.ylabel('Residuals')
plt.show();


# As you can see from the above plot, the residuals are not random, but rather fan out as the observed value increases. This suggests there is some heteroscedasticity, which is a limitation of our model.



# To evaluate which features were having the largest effect on our model, we created a bar graph showing the coefficients of each feature
coefficients = list(linear_model.coef_)
plt.figure(figsize=(20,5))
plt.xticks(rotation=90)
plt.title('Fig 9. First Model Coefficients')
plt.ylabel('Coefficient')
plt.xlabel('Column Names')
plt.bar(allcolumns, coefficients);


# As you can see, only a very few features have any significant coefficient, meaning that we can filter out ineffectual features. ICU beds per person clearly has a very large impact on our model. ICU Per person can reflect the county's preparedness an use of hospital facilities, and its capacity to deal with cases of increased healthcare needs.



# New list of most impactfull features
usefulcolumns = ['icuperperson', 'proportion65+', 'incidence_rate_tenthousand', 'FracMale2017',
                'foreign travel ban']

# Second linear model, using only the features of significance in Fig. 9 above, barely improves our error.
# Maybe we should think more deeply on our feature selection
linear_model_2 = lm.LinearRegression(fit_intercept=True)
X, y = process_couties_data(fulldata, 'cause_specific_mortality_million', usefulcolumns)
error = cross_validate_rmse(linear_model_2, X, y)
error




# Final linear model. Most features selected through trial and error and intuition from our EDA
# This was the best model we were able to create.
linear_model_final = lm.LinearRegression(fit_intercept=True)
columns = ['#EligibleforMedicare2018',
                      'Rural-UrbanContinuumCode2013',
                      'PopulationDensityperSqMile2010',
                      'confirmedcases',
                      'proportion65+',
                      'incidence_rate_tenthousand',
                      'StrokeMortality',
                      'Smokers_Percentage',
                      'dem_to_rep_ratio',
                      'Smokers_Percentage',
                      'dayssincefirstcase',
                      'icuperperson',
                      'icuperhospital',
                      '#Hospitals',
                       'SVIPercentile'
                      ]
X, y = process_couties_data(fulldata, 'cause_specific_mortality_million', columns)
error = cross_validate_rmse(linear_model_final, X, y)
error




# Visualization showing coefficients for our best model, linear_model_final
linear_model = lm.LinearRegression(fit_intercept=True)
linear_model.fit(X,y)
y_pred = linear_model.predict(X)
coefficients = list(linear_model.coef_)
plt.figure(figsize=(20,5))
plt.xticks(rotation=90)
plt.ylabel('Coefficients')
plt.xlabel('Column Names')
plt.title('Fig. 10: Coefficients of our Best Model')
plt.bar(columns, coefficients)
plt.show();




# Plotting coefficients without ICU per person to zoom in on smaller values
columnswithouticuperperson = ['#EligibleforMedicare2018',
                      'Rural-UrbanContinuumCode2013',
                      'PopulationDensityperSqMile2010',
                      'confirmedcases',
                      'proportion65+',
                      'incidence_rate_tenthousand',
                      'StrokeMortality',
                      'Smokers_Percentage',
                      'dem_to_rep_ratio',
                      'Smokers_Percentage',
                      'dayssincefirstcase',
                      'icuperhospital',
                      '#Hospitals',
                       'SVIPercentile'
                      ]
X, y = process_couties_data(fulldata, 'cause_specific_mortality_million', columnswithouticuperperson)
error = cross_validate_rmse(linear_model, X, y)
linear_model = lm.LinearRegression(fit_intercept=True)
linear_model.fit(X,y)
y_pred = linear_model.predict(X)
coefficients = list(linear_model.coef_)
plt.figure(figsize=(20,5))
plt.xticks(rotation=90)
plt.ylabel('Coefficients')
plt.xlabel('Column Names')
plt.title('Fig. 11: Coefficients of our Best Model without ICU per person')
plt.bar(columnswithouticuperperson, coefficients);
