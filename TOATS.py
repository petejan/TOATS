#!/usr/bin/env python
# coding: utf-8

# # Best practices for assessing <u>T</u>rends of <u>O</u>cean <u>A</u>cidification <u>T</u>ime <u>S</u>eries (TOATS)
# 
# Roman Battisti<sup>1,2</sup> and Adrienne J. Sutton<sup>2</sup>
# 
# <sup>1</sup>Cooperative Institute for Climate, Ocean, and Ecosystem Studies, University of Washington, Seattle, WA, 98115
# 
# <sup>2</sup>Pacific Marine Environmental Laboratory, NOAA, Seattle, Washington, USA
# 
# 
# 
# A supplement to Sutton et al., Advancing best practices for assessing trends of ocean acidification time series, submitted to _Frontiers in Marine Science_ Research Topic “Best Practices in Ocean Observing”.

# In[ ]:


# import libraries

# standard libraries
from datetime import datetime
from copy import deepcopy
from collections import OrderedDict
import math
import re
#import requests
import warnings

# 3rd party libraries
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import statsmodels.api as sm
from tabulate import tabulate

import sys

warnings.filterwarnings('ignore')

# set size of figures
#get_ipython().run_line_magic('matplotlib', 'notebook')
#plt.rcParams['figure.figsize'] = (8, 3)


# ### ASSUMPTIONS
# ***
# - Data collation and quality control are completed prior to loading the time series data. 
# - Data are normally distributed.
# - All data are subject to the same biases and have equal precision.
# - If data gaps exist, they will not impact the calculations of climatological monthly or annual means.
# - Autocorrelation is present in ocean biogeochemical time series but accounted for in the following approaches for linear regression and trend detection time.
# - The user of this code has read the associated README.md file.

# ### LOAD TIME SERIES DATA
# ***
# Running the next cell will open dialogue boxes asking the user to enter the name of the .csv or .txt time series data file, the site name, and additional information about each carbon variable.  For more information on the required structure and formating of the data file, see associated README.md.

# In[ ]:


# function for reading tab- and comma-delimited files 
def separator_check(text):
    for sep in [',', '\t']:
        if re.search(sep, text):
            return sep
    return None

# functions to load data file to dataframe 
def load_file_to_pandas(file_name, comment_indicator=None): 
    with open(file_name, 'r') as f:
        sep = None
        for data in f:
            sep = separator_check(data)
            break
    return pd.read_csv(file_name, sep=sep, engine='python', dtype={0: str})

def trend_options(df_columns):
    print('')
    for i, col in enumerate(df_columns[1:]):
        print('{}:\t{}'.format(i + 1, col))
    selection = int(input('Select the number associated with the parameter to process for trends: '))
    print('{} selected...'.format(df_columns[selection]))
    print('')
    return df_columns[selection]

class ExtensionException(Exception):
    pass

# UI for entry of source file and site name
comment_indicator = None
file_name = input("Input the data file name with extension: ")
site_name = input("Input the site name: ")
ts_df = load_file_to_pandas(file_name)

pp = PdfPages(site_name + '-' + file_name + '-TOATS.pdf')

print('\nBeginning of the dataframe with the imported data:')
datetime_col = ts_df.columns[0]
variable_names = list(ts_df.columns[1:])
var_sigfig_dict = {}
var_unc_dict = {}
var_unit_dict = OrderedDict()
for var in variable_names:
    var_sigfig_value = int(input(f"Please provide the number of decimal places for {var}: "))
    var_uncertainty_value = float(input(f"Please provide the uncertainty for {var}: "))
    var_units = input(f"Please provide the units of {var}: ")
    print("\n")
    var_sigfig_dict[var] = var_sigfig_value
    var_unc_dict[var] = var_uncertainty_value
    var_unit_dict[var] = var_units

ts_df.head()


# In[ ]:


# Date format test
def build_formatted_string(string: str, char_requirements_dict: dict):
    from collections import Counter
    string = np.array(list(string.lower()))
    _, idx = np.unique(string, return_index=True)
    char_counter = Counter(string)
    string = string[np.sort(idx)]
    output = []
    
    # check for required characters
    if 'y' in char_requirements_dict:
        for c in char_requirements_dict.keys():
            if c not in char_counter:
                raise FormatError(f"The character {c} is missing in your date format and is required.")
    elif 'h' in char_requirements_dict:
        if 'h' not in char_counter:
            raise FormatError("The character h is missing in your time format and is required")
    
    for c in string:
        # check that there are no meaningless additional characters
        if c not in char_requirements_dict:
            raise FormatError(f"{c} is not a valid character for a datetime format or your datetime is ordered incorrectly.")
        # make sure number of characters is consistent with formatting constraints
        if char_counter[c] not in char_requirements_dict[c]:
            raise FormatError(f"{c} must have {char_requirements_dict[c]} number of characters.")
        elif char_counter[c] == char_requirements_dict[c][0]:
            output.append(c)
        elif char_counter[c] == char_requirements_dict[c][1]:
            output.append(c.upper())
    return '%' + '%'.join(output)

def convert_datetime_format_string(string: str):
    date_requirements_dict = {'m': [2], 'd': [2], 'y': [2, 4]}
    time_requirements_dict = {'h': [None, 2], 'm': [None, 2], 's': [None, 2]}
    separated_string = string.split(' ')
    if len(separated_string) > 2:
        raise FormatError("There are too many inputs for a datetime string format. Make sure your datetime only has one space between the date and time.")
    output = []
    for i in range(len(separated_string)):
        # assume first part is date, second part is time
        if i == 0:
            output.append(build_formatted_string(separated_string[i], date_requirements_dict))
        elif i == 1:
            output.append(build_formatted_string(separated_string[i], time_requirements_dict))
        else:
            raise FormatError("There are too many spaces in your date/time format. There should be at most two, one for the date and one for the time.")
    return ' '.join(output)


class FormatError(Exception):
    pass


dayfirst = 'n'
dayformat = None

try:  # if date is entirely numeric with no separators
    date_test = np.array([t.split() for t in ts_df.iloc[:, 0]]).astype(int)
    dayformat = input("What is the date format of your data (ex yyyymmdd, ddmmyyyy HHMMSS, etc)? ").upper()
    dayformat = convert_datetime_format_string(dayformat)
    
except ValueError:
    day_test_df = ts_df.iloc[:, 0].str.split('[,\s/-]+|T', expand=True)
    if len(day_test_df.columns) > 2:  # if datetime can be separated. Otherwise, try to let pandas handle it.
        int_columns = []
        for col in day_test_df.columns[:3]:  # test whether each column in split date can be converted to int
            try:
                day_test_df = day_test_df.astype({col: int})
                int_columns.append(col)
            except ValueError:  # assume an error results from the month in string form
                pass
        day_test_max = day_test_df.iloc[:, int_columns].aggregate(max)
        daypos = [col for col in day_test_max.index if day_test_max[col] > 12 and day_test_max[col] < 32]
        if daypos and int_columns[daypos[0]] == 0:
            dayfirst = 'y'
        elif not dayfirst:
            dayfirst = input("Are days first in the datetimes being imported (y or n)? ")
except FormatError as e:
    print("The following exception was raised: ", e)
    print("Please rerun the cell with a properly formatted date format or reformat your datetimes in your file to an acceptable format.")

dp_str_dict = {k: '%.' + str(var_sigfig_dict[k]) + 'f' for k in var_sigfig_dict.keys()}


# In[ ]:


def decimal_month(years, months, days):
    """Compute the decimal month (ex. March 15 would be 3.48 or 3 + 15/31). Account for number of days in February using year."""
    days_in_month = {1: 31, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
    feb_days = {y: 29 if y%400 == 0 or (y%4 == 0 and not y%100==0) else 28 for y in np.unique(years)}
    output = []
    for y, m, d in zip(years, months, days):
        if m == 2:
            output.append(m + (d - 1) / feb_days[y])
        else:
            output.append(m + (d - 1) / days_in_month[m])
    return output

# make datetime the index and calculate decimal year
if dayformat:
    ts_df[datetime_col] = pd.to_datetime(ts_df[datetime_col], format=dayformat)
elif dayfirst == 'y':
    ts_df[datetime_col] = pd.to_datetime(ts_df[datetime_col],dayfirst=True)
else:
    ts_df[datetime_col] = pd.to_datetime(ts_df[datetime_col])

ts_df.index = ts_df[datetime_col]
ts_df['year'] = pd.DatetimeIndex(ts_df[datetime_col]).year
ts_df['month'] = pd.DatetimeIndex(ts_df[datetime_col]).month
ts_df['day'] = pd.DatetimeIndex(ts_df[datetime_col]).day
ts_df['decimal_month'] = decimal_month(ts_df['year'], ts_df['month'], ts_df['day'])
ts_df['decimal_year'] = ts_df['year'] + (ts_df['decimal_month'] - 1) / 12

# create dictionary of dataframes only containing one variable each
additional_columns = ['year', 'month', 'day', 'decimal_month', 'decimal_year']
ts_df_dict = {var: deepcopy(ts_df[[var] + additional_columns]) for var in variable_names}

# remove NaNs
for k in ts_df_dict.keys():
    ts_df_dict[k].dropna(how='any', inplace=True)


# ### 1) ASSESS DATA GAPS
# ***
# Assess plot and monthly statistics to determine whether obervations are sufficient to constrain climatological monthly means and the annual climatological mean.

# In[ ]:


# calculate and display monthly and annual statistics
ts_stats = {}

for k, v in ts_df_dict.items():
    ts_monthly_stats = round(v.groupby('month').agg({k: ['mean', 'std', 'count']}), var_sigfig_dict[k])
    ts_annual_stats = round(v.groupby('year').agg({k: ['mean', 'std', 'count']}), var_sigfig_dict[k])
    ts_stats[k] = {'monthly': ts_monthly_stats, 'annual': ts_annual_stats}

    # plot monthly means and histogram of monthly measurements
    plt.figure()
    plt.errorbar(ts_monthly_stats.index, ts_monthly_stats[k]['mean'], 
                    ts_monthly_stats[k]['std'], marker= 'o', elinewidth=1, 
                    linewidth=2)
    plt.title(f"{k} monthly means and std")
    plt.xlabel('Month')
    plt.xticks(np.arange(1, 13, step=1))
    plt.ylabel(f"seawater {k} ({var_unit_dict[k]})")
    pp.savefig()
    #plt.show()
    plt.close()

    plt.figure()
    plt.hist(ts_df_dict[k]['month'], bins=np.arange(14)-0.5, edgecolor='black', rwidth=0.8)
    plt.title(f"{k} monthly measurement distribution")
    plt.xlabel("month") 
    plt.ylabel("# of measurements") 
    plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12])
    plt.xlim([0.5,12.5])
    pp.savefig()
    #plt.show()
    plt.close()
    
print('\nBefore continuing, the user should confirm that monthly measurement distributions are sufficient to constrain climatological monthly means and the annual climatological mean.','\n')


# ### 2) REMOVE PERIODIC (SEASONAL) SIGNAL
# ***
# The following steps characterize the seasonal signal and remove that noise from the time series of monthly means, creating a time series of de-seasoned monthly means used in the trend analysis:
# - Temporarily remove the overall linear trend by applying a simple linear regression and removing the slope from the original data set. 
# - Using this de-trended time series, calculate climatological monthly means and determine the climatological annual mean (mean of the climatological monthly means).
# - Determine the monthly adjustments by subtracting the climatological annual mean from the climatological monthly means.  
# - Subtract the adjustment value for each month from the time series of monthly means, which was not de-trended. 

# In[ ]:


var_trends = {}

for k, v in ts_df_dict.items():
    # fit linear model
    decimal_year = np.reshape(v['decimal_year'].to_numpy(), (-1,1))
    decimal_year = sm.add_constant(decimal_year)
    ts_variable = v[k].to_numpy()
    ts_variable = np.reshape(ts_variable, (-1,1))
    model = sm.OLS(ts_variable, decimal_year).fit(cov_type='HAC', cov_kwds={'maxlags':1})
    
    # calculate trend
    trend_to_remove = model.predict(decimal_year)

    # temporarily remove trend
    detrended = [ts_variable[i]-trend_to_remove[i] for i in range(0, len(decimal_year))]

    # extract slope and error values from OLS results and print results
    trend_to_remove_slope = dp_str_dict[k] % model.params[1]
    trend_to_remove_slope_error = dp_str_dict[k] % model.bse[1]
    var_trends[k] = {'model': model, 'detrended': detrended, 'slope_str': trend_to_remove_slope, 'slope_error_str': trend_to_remove_slope_error}


# In[ ]:


#create new dataframe with de-trended values
for k, v in var_trends.items():
    detrended = np.round(v['detrended'], var_sigfig_dict[k])
    detrended_df = pd.DataFrame(data=detrended, index=ts_df_dict[k].index, columns=['detrended_variable'])
    detrended_df['month'] = detrended_df.index.month
    detrended_df['year'] = detrended_df.index.year
    v['detrended'] = detrended_df


# In[ ]:


# climatological monthly mean of de-trended values
for k, v in var_trends.items():
    ts_month = v['detrended'].groupby('month')
    climatological_df = round(ts_month.agg({'detrended_variable': ['mean']}), var_sigfig_dict[k])

    # climatological annual mean of de-trended values
    annual_mean = np.mean(climatological_df['detrended_variable']['mean'])

    # monthly seasonal adjustment 
    climatological_df['monthly_adj'] = np.round(climatological_df['detrended_variable']['mean'].values - annual_mean, var_sigfig_dict[k])

    # seasonal amplitude and interannual variability for display in summary report
    season_max = climatological_df.detrended_variable['mean'].max()
    season_min = climatological_df.detrended_variable['mean'].min()
    seasonal_amplitude = round(season_max - season_min, var_sigfig_dict[k])

    annual_means = v['detrended'].groupby('year') 
    annual_means_df = annual_means.agg({'detrended_variable':['mean']})
    IAV = round(np.mean(abs(annual_means_df.detrended_variable['mean'])), var_sigfig_dict[k])
    v['climatological'] = climatological_df
    v['seasonal_amplitude'] = seasonal_amplitude
    v['annual_means'] = annual_means


# In[ ]:


# apply monthly adjustment to monthly means of the original time series, which was not de-trended
for k, v in var_trends.items():
    ts_year_month = ts_df_dict[k].groupby(['year','month'])

    ts_mean = round(ts_year_month.agg({k: ['mean']}), var_sigfig_dict[k])
    ts_mean[k,'datetime_mean'] = [datetime(i[0],i[1],15,0,0,0) for i in ts_mean.index]

    adj_mean = []
    for i in ts_mean.index:
        temp = ts_mean[k]['mean'][i] - v['climatological']['monthly_adj'][i[1]]
        adj_mean.append(temp)

    # create time series of de-seasoned monthly means (variable name = adj_mean)
    ts_mean[k, 'adj_mean'] = adj_mean
    v['ts_mean'] = ts_mean


# ### 3) ASSESS LINEAR FIT TO THE DATA WITH THE PERIODIC SIGNAL REMOVED
# ***
# Determines the linear trend of de-seasoned monthly means and regression statistics. The Weighted Least Squares linear regression model from the statsmodels Python module (www.statsmodels.org/dev/regression.html) is used with Newey-West standard errors to account for heteroskedasticity and autocorrelation. For a description of all elements of the output, refer to the open-source statsmodels documentation (www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.OLSResults.html)
# 
# Newey, Whitney K; West, Kenneth D (1987). "A Simple, Positive Semi-definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix". Econometrica 55 (3): 703–708. doi:10.2307/1913610

# In[ ]:


# create decimal year
for k, v in var_trends.items():
    ts_mean = v['ts_mean']
    ts_mean[k,'year'] = pd.DatetimeIndex(ts_mean[k]['datetime_mean']).year
    ts_mean[k,'month'] = pd.DatetimeIndex(ts_mean[k]['datetime_mean']).month
    ts_mean[k,'day'] = pd.DatetimeIndex(ts_mean[k]['datetime_mean']).day
    ts_mean[k,'decimal_month'] = decimal_month(ts_mean[k]['year'], ts_mean[k]['month'], ts_mean[k]['day'])
    ts_mean[k, 'decimal_year'] = ts_mean[k]['year'] + (ts_mean[k]['decimal_month'] - 1) / 12
    v['ts_mean'] = ts_mean


# In[ ]:


# create series of dates adjusted to start at 0 for use in the regression model 
# this prevents the y-intercept value from being highly sensitive to input dates
wls_model_dict = {}
for k, v in var_trends.items():
    ts_mean = v['ts_mean']
    decimal_year_deseasoned = np.reshape(ts_mean[k]['decimal_year'].values, (-1, 1))
    min_year = np.amin(decimal_year_deseasoned)
    decimal_year_zero = decimal_year_deseasoned-min_year
    decimal_year_zero = sm.add_constant(decimal_year_zero)
    ts_variable_deseasoned = ts_mean[k]['adj_mean'].values
    ts_variable_deseasoned = np.reshape(ts_variable_deseasoned,(-1, 1))

    # Weights are based on user input uncertainty
    weights = var_unc_dict[k] * np.ones(len(ts_variable_deseasoned))

    # fit linear model
    weights = var_unc_dict[k]  # uncertainties
    model = sm.WLS(ts_variable_deseasoned,decimal_year_zero, weights).fit(cov_type='HAC',cov_kwds={'maxlags':1})
    wls_model_dict[k] = {'model': model, 'decimal_year_zero': decimal_year_zero, 'ts_variable_deseasoned': ts_variable_deseasoned}


# In[ ]:


# calculate and plot trend 
for k, v in wls_model_dict.items():
    model = v['model']
    trend = model.predict(v['decimal_year_zero'])

    # extract slope and error values from WLS results and print results
    slope_str = dp_str_dict[k]
    slope = slope_str % model.params[1]
    slope_error = slope_str % model.bse[1]
    wls_model_dict[k]['trend'] = trend
    wls_model_dict[k]['slope_str'] = slope
    wls_model_dict[k]['slope_err_str'] = slope_error


# ### 4) ESTIMATE TREND DETECTION TIME
# ***
# This section uses the trend detection methods of Weatherhead et al. 1998 to estimate trend detection time.
# 
# Weatherhead, E. C., Reinsel, G. C., Tiao, G. C., Meng, X.-L., Choi, D., Cheang, W.-K., et al. (1998). Factors affecting the detection of trends: Statistical considerations and applications to environmental data. Journal of Geophysical Research: Atmospheres, 103(D14), 17149-17161. http://dx.doi.org/10.1029/98JD00995

# In[ ]:


TDTi_dict = {}  # time of detection
# autocorrelation at lag 1 of the time series noise
for k, v in wls_model_dict.items():
    ts_variable_deseasoned = v['ts_variable_deseasoned']
    decimal_year_zero = v['decimal_year_zero']
    autocorr = sm.tsa.stattools.acf(ts_variable_deseasoned,fft=False,nlags=1)[1:]
    ts_mean = var_trends[k]['ts_mean']

    # standard deviation of detrended monthly anomalies
    model = v['model']
    trend_to_remove_TDT = model.predict(decimal_year_zero)
    detrended_TDT = [ts_variable_deseasoned[i]-trend_to_remove_TDT[i] for i in range(0, len(ts_mean[k]['datetime_mean']))]
    std_dev = np.std(detrended_TDT)

    # time of detection 
    TDTi = np.round((((3.3*std_dev)/(abs(model.params[1:])))*(np.sqrt(((1+autocorr)/(1-autocorr)))))**(2/3), 1)
    ts_length = round(np.max(decimal_year_zero[:, 1]), 1)

    # uncertainties of time of detection due to unknown variance and autocorrelation
    uncert_factor = (4/(3*np.sqrt(len(ts_variable_deseasoned))))*(np.sqrt(((1+autocorr)/(1-autocorr))))
    upper_conf_intervali = TDTi * math.exp(uncert_factor)
    lower_conf_intervali = TDTi * math.exp(-uncert_factor)
    uncert_TDTi = np.round(((upper_conf_intervali-TDTi)+(TDTi-lower_conf_intervali))/2,1)

    dp_str = dp_str_dict[k]
    TDT = dp_str % TDTi[0]
    uncert_TDT = dp_str % uncert_TDTi[0]
    upper_conf_interval = dp_str % upper_conf_intervali[0]
    lower_conf_interval = dp_str % lower_conf_intervali[0]
    
    TDTi_dict[k] = {'TDTi': TDTi, 'ts_length': ts_length, 'TDT': TDT, 'uncert_TDT': uncert_TDT, 'upper_conf_interval': upper_conf_interval, 'lower_conf_interval': lower_conf_interval}


# ### Summary report:
# ### 5) CONSIDER UNCERTAINTY IN THE MEASUREMENTS AND REPORTED TRENDS
# ### 6) PRESENT RESULTS IN THE CONTEXT OF NATURAL VARIABILITY AND UNCERTAINTY
# 
# After running the following cell, the user will be prompted to enter one of the variable names. After the variable is entered, running the next cell will generate summary statistics for that variable. The time series of de-seasoned monthly means for that variable will be exported to a .csv file saved to the same location as this jupyter notebook. These two cells can be rerun, selecting different variables to summarize without needing to rerun the entire notebook.
# ***

# In[ ]:


for k in var_unit_dict.keys():
    print(k)
print("\n")

param_to_summerize = input('Please select from the available parameters to see a summary: ')

while True:
    test = wls_model_dict.get(param_to_summerize, False)
    if not test:
        param_to_summerize = input("The parameter you've selected doesn't exist, please select from the available parameters: ")
    else:
        break

# organize parameters
ts_df = ts_df_dict[param_to_summerize]
dp = var_sigfig_dict[param_to_summerize]
units = var_unit_dict[param_to_summerize]
model = wls_model_dict[param_to_summerize]['model']

ts_annual_stats = ts_stats[param_to_summerize]['annual']
ts_monthly_stats = ts_stats[param_to_summerize]['monthly']

slope = wls_model_dict[param_to_summerize]['slope_str']
slope_error = wls_model_dict[param_to_summerize]['slope_err_str']

ts_mean = var_trends[param_to_summerize]['ts_mean']
ts_variable_deseasoned = wls_model_dict[param_to_summerize]['ts_variable_deseasoned']
trend = wls_model_dict[param_to_summerize]['trend']

td = TDTi_dict[param_to_summerize]
TDTi = td['TDTi']
ts_length = td['ts_length']
TDT = td['TDT']
uncert_TDT = td['uncert_TDT']
upper_conf_interval = td['upper_conf_interval']
lower_conf_interval = td['lower_conf_interval']


# In[ ]:

print()

# re-direct output to a text file
original = sys.stdout
sys.stdout = open(site_name + '-' + file_name + '-TOATS.txt', 'w')

# summary of OLS Regression Results
print('---Summary of OLS Regression Results---','\n')

# create table header
head = ["Statistic", "Value", "Result"]

# create CI and variance values
CI = model.conf_int()
CI_values = np.round(CI[1],dp)
CI_low = CI_values[0] 
CI_high = CI_values[1] 
adj_r2 = round(model.rsquared_adj,2)
adj_r2_percent = int(adj_r2 * 100)


if model.pvalues[1:] >= 0.05: # model.pvalues[:1] >= 0.05 and
    statsdata = [["P > |z|", "> 0.05", "The slope coefficient is not significant"], 
      ["Adjusted r2", adj_r2, "The model describes {}% of the variation in {} over time".format(adj_r2_percent, param_to_summerize)], 
      ["Standard error", slope_error, "The estimated {} trend is {} ± {} {} per year".format(param_to_summerize, slope, slope_error, units)], 
      ["[0.025 and 0.975]", "{} to {}".format(CI_low, CI_high), "The 95% CI for the trend is {} to {} {} per year".format(CI_low, CI_high, units)]]

    # display table
    print(tabulate(statsdata, headers=head, tablefmt="rst"))

    print('The p-value for how likely the slope coefficient is measured through the linear model by chance is > 0.05, suggesting the coefficient is not statistically significant.', '\n')
    print("High slope error and low adjusted r-squared should also suggest lack of significance in the trend. In this case slope error is {} uatm per year compared to a slope of {} uatm per year and adjusted r-squared is {}".format(slope_error, slope, round(model.rsquared_adj,dp)),'\n')

else:
    statsdata = [["P > |z|", "< 0.05", "The slope coefficient is significant"], 
      ["Adjusted r2", adj_r2, "The model describes {}% of the variation in {} over time".format(adj_r2_percent, param_to_summerize)], 
      ["Standard error", slope_error, "The estimated {} trend is {} ± {} {} per year".format(param_to_summerize, slope, slope_error, units)], 
      ["[0.025 and 0.975]", "{} to {}".format(CI_low, CI_high), "The 95% CI for the trend is {} to {} {} per year".format(CI_low, CI_high, units)]]

    # display table
    print(tabulate(statsdata, headers=head, tablefmt="rst"))

    print('The p-value for how likely the slope coefficient is measured through the linear model by chance is < 0.05, suggesting the coefficient is statistically significant.', '\n')
    print("Low slope error and high adjusted r-squared should also accompany a low p-value if the trend is statistically significant. In this case slope error is {} {} per year compared to a slope of {} {} per year and adjusted r-squared is {}.".format(slope_error, units, slope, units, round(model.rsquared_adj,dp)),'\n')


# summary of trend detection results
print('---Summary of Trend Detection Results---','\n')
print("The number of years to detect a trend of {} {} per year at {} is approximately {} ± {} with a confidence interval of {} to {} years.".format(slope,units,site_name,TDT,uncert_TDT,lower_conf_interval,upper_conf_interval),'\n')
if ts_length < TDTi:
    print("The {} time series length of {} years may not be long enough to detect a statistically-significant trend.".format(site_name,ts_length),'\n')
else:
    print("The {} time series length of {} years may be long enough to detect a statistically-significant trend.".format(site_name,ts_length),'\n')

plt.figure()
plt.scatter(ts_df.index, ts_df[param_to_summerize], marker='.', label='observations')
plt.plot(ts_mean[param_to_summerize]['datetime_mean'], ts_variable_deseasoned, c='r', marker= 's', markersize=4, linewidth=0, label='monthly anomalies')
plt.plot(ts_mean[param_to_summerize]['datetime_mean'], ts_mean[param_to_summerize]['mean'], c='k', marker= 'o', markersize=4, linewidth=0, label='monthly means')
plt.plot(ts_mean[param_to_summerize]['datetime_mean'], trend, c='r', linewidth=2, label='trend')
plt.title("{} time series".format(site_name)) 
plt.ylabel("seawater {} ({})".format(param_to_summerize, units)) 
plt.legend()
pp.savefig()
#plt.show()
plt.close()

# assumptions
print('---Assumptions and considerations---','\n')
print('These trend results are only valid if the data are normally distributed and all data are subject to the same biases and have equal precision.','\n')
print("Characterizing and removing periodic signal(s) in time series prior to estimating trends reduces noise in the data set, thereby reducing uncertainty in the resulting trend. The method used here removes the seasonal signal, which is a seasonal amplitude of {} {} for this time series.  Daily, tidal, interannual, or decadal signals could also be characterized and removed to further reduce trend uncertainty.".format(seasonal_amplitude,units),'\n')
print("The impact of interannual or decadal variability on trends could also be assessed by calculating trends over different time periods within the data set. For example, to interrogate how different phases of the El Niño Southern Oscillation (ENSO) impact long-term change, the data set could be separated into El Niño, La Niña, and neutral time periods and trends assessed separately for the different ENSO phases.  Interannual variability of {} for this time series is {} {}.".format(param_to_summerize, IAV, units),'\n')
print("The seasonal amplitude and interannual variability estimates are based on the data below.  If data are nonuniform across months and years, these estimates may not be valid.")
      
# plot monthly means
fig, ax = plt.subplots()
ax.errorbar(ts_monthly_stats.index, ts_monthly_stats[param_to_summerize]['mean'], 
                ts_monthly_stats[param_to_summerize]['std'], marker= 'o', elinewidth=1, 
                linewidth=2)
ax.set_title("{} monthly means and std".format(site_name)) 
ax.set_xlabel('Month')
ax.set_xticks(np.arange(1, 13, step=1))
ax.set_ylabel("seawater {} ({})".format(param_to_summerize, units)) 
pp.savefig()
#plt.show()
plt.close()

plt.figure()
plt.hist(ts_df['month'], bins=np.arange(14)-0.5, edgecolor='black', rwidth=0.8)
plt.title("{} monthly measurement distribution".format(site_name)) 
plt.xlabel("month") 
plt.ylabel("# of measurements") 
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12])
plt.xlim([0.5,12.5])
pp.savefig()
#plt.show()
plt.close()

#display(ts_monthly_stats)

# plot annual means
fig, ax = plt.subplots()
ax.errorbar(ts_annual_stats.index, ts_annual_stats[param_to_summerize]['mean'], 
                ts_annual_stats[param_to_summerize]['std'], marker= 'o', elinewidth=1, 
                linewidth=2)
ax.set_title("{} annual means and std".format(site_name)) 
ax.set_xlabel('Year')
ax.set_ylabel("seawater {} ({})".format(param_to_summerize, units)) 
pp.savefig()
#plt.show()
plt.close()

maxyear = max(ts_df['year'])
minyear = min(ts_df['year'])
nyears = maxyear-minyear
allyears = np.append([np.unique(ts_df['year'])],[maxyear+1])

plt.figure()
plt.hist(ts_df['year'], bins=allyears-0.5, edgecolor='black', rwidth=0.8)
plt.title("{} annual measurement distribution".format(site_name)) 
plt.xlabel("year") 
plt.ylabel("# of measurements") 
pp.savefig()
#plt.show()
plt.close()

pp.close()

#display(ts_annual_stats)

# export time series of de-seasoned monthly means 
month_year = np.reshape(ts_mean[param_to_summerize]['datetime_mean'].values,
               (len(ts_mean[param_to_summerize]['datetime_mean']),1))
deseasoned_df = pd.DataFrame(data = [month_year[:,0],np.round(ts_variable_deseasoned[:,0], dp)]).T
deseasoned_df.columns = ["date", "deseasoned_monthly_mean_{}".format(param_to_summerize)]
deseasoned_df.to_csv(f'{site_name}_{param_to_summerize}_deseasoned_monthly_means.csv', index = False, header = True)
print('\n---De-seasoned data export---','\n')
print("The time series of de-seasoned {} monthly means has be exported to a file called deseasoned_monthly_means.csv saved to the same location as this jupyter notebook.".format(param_to_summerize),'\n')


# In[ ]:

sys.stdout = original

