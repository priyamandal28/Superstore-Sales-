#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


import warnings
warnings.filterwarnings('ignore')


# # DATA CLEANING

# LOADING THE DATSETS

# In[2]:


train_data = pd.read_csv(r"C:\Users\Priya\Downloads\store_forecasting_data (1)\store_forecasting_data\train.csv")
train_data.head()


# In[3]:


stores = pd.read_csv(r"C:\Users\Priya\Downloads\store_forecasting_data (1)\store_forecasting_data\stores.csv")
stores.head()


# In[4]:


oil = pd.read_csv(r"C:\Users\Priya\Downloads\store_forecasting_data (1)\store_forecasting_data\oil.csv")
oil.head()


# In[5]:


holiday_events = pd.read_csv(r"C:\Users\Priya\Downloads\store_forecasting_data (1)\store_forecasting_data\holidays_events.csv")
holiday_events.head()


# Handling Missing Values

# In[6]:


oil.isnull().sum()


# #we have missing values in dcoilwtico column 
# #Filling missing values by interpolation method

# In[7]:


oil['dcoilwtico']=oil['dcoilwtico'].interpolate()


# In[8]:


oil.isnull().sum()


# In[9]:


# linear interpolation fills the missing oil prices based on previous and next values


# In[10]:


#Converting date columns to proper date-time formats


# In[11]:


train_data.info()


# In[12]:


train_data['date']=pd.to_datetime(train_data['date'],format ="%Y-%m-%d")


# In[13]:


train_data.info()


# In[14]:


holiday_events.info()


# In[15]:


holiday_events['date']=pd.to_datetime(holiday_events['date'],format = "%Y-%m-%d")


# In[16]:


holiday_events.info()


# In[17]:


oil.info()


# In[18]:


oil['date']= pd.to_datetime(oil['date'],format = "%Y-%m-%d")


# In[19]:


oil.info()


# #merging

# In[20]:


train_data = train_data.merge(stores,on = "store_nbr",how = "left")
train_data = train_data.merge(oil,on = "date",how = "left")
train_data = train_data.merge(holiday_events,on = "date",how = "left")


# In[21]:


train_data.head(20)


# # Feature Engineering

# In[22]:


#extracting time based features


# In[23]:


train_data['Year']= train_data['date'].dt.year
train_data['Month']= train_data['date'].dt.month
train_data['Day']= train_data['date'].dt.day
train_data['weekday']= train_data['date'].dt.weekday
train_data['week']=train_data['date'].dt.isocalendar().week


# In[24]:


train_data.head(20)


# In[25]:


train_data.tail(20)


# identifying the seasonal trends

# In[26]:


#grouping by month and calculated avg sales
monthly_sales = train_data.groupby("Month")['sales'].mean()


# In[27]:


#plot sales trends
plt.figure(figsize=(10,5))
monthly_sales.plot(kind = 'bar',color = 'Green')
plt.title('Average sales trends per month')
plt.xlabel('Month')
plt.ylabel('Average Sales')
plt.show()


# From the above visualization we can conclude that the average sales are high in December month

# #extracting event based features

# In[28]:


#created binary flag for holiday
train_data["is_holiday"] = train_data["type_y"].apply(lambda x:1 if x in ["Holiday","Event"] else 0)


# In[29]:


train_data.head()


# In[30]:


#created binary flag for promotions
train_data["has_promotion"]= train_data["onpromotion"].apply(lambda x:1 if x>0 else 0)


# In[31]:


train_data.head()


# In[32]:



train_data["Last_Day"] = train_data["date"].dt.days_in_month 


# In[33]:


train_data.head()


# In[34]:


#identify government payday (15th and last day of month)
train_data["is_payday"] = train_data.apply(lambda row: 1 if row["Day"] in [15, row["Last_Day"]] else 0, axis=1)


# In[35]:


train_data.head(20)


# In[36]:


#consider earthquake impact on 16th April 2016
train_data["earthquake_impact"]=train_data["date"].apply(lambda x:1 if x== pd.Timestamp("2016-04-16")else 0)


# In[37]:


train_data.head(5)


# In[38]:


train_data.drop(['Last_Day'],axis = 1,inplace = True)


# In[39]:


train_data.head()


# Rolling Statistics

# In[40]:


#rolling mean for 7-days and 30-days(moving average)
train_data["rolling_mean_7"] = train_data["sales"].rolling(window = 7).mean()
train_data["rolling_mean_30"]=train_data["sales"].rolling(window = 30).mean()


# In[41]:


#rolling standard deviation for 7-days and 30-days(rolling std)
train_data["rolling_std_7"]=train_data["sales"].rolling(window = 7).std()
train_data["rolling_std_30"]=train_data["sales"].rolling(window = 30).std()


# In[42]:


train_data.tail(20)


# creating Lag features

# In[43]:


#sales for previous week and previous month

train_data["lag_7"]=train_data["sales"].shift(7)
train_data["lag_30"]= train_data["sales"].shift(30)


# In[44]:


#dropping rows having NAN values
train_data.dropna(inplace = True)


# In[45]:


train_data.head()


# STORE SPECIFIC AGGREGATIONS

# In[46]:


#Average sales per store type
store_avg_sales = train_data.groupby("store_nbr")["sales"].mean().reset_index()
store_avg_sales.rename(columns={"sales": "avg_sales_per_store"}, inplace=True)


# In[47]:


#merging with main train dataset

train_data = train_data.merge(store_avg_sales,on = "store_nbr",how = "left")


# In[48]:


train_data.head(10)


# In[49]:


#identifying top selling product families per cluster
top_products = train_data.groupby(["store_nbr", "family"])["sales"].sum().reset_index()
top_products = top_products.sort_values(["store_nbr", "sales"], ascending=[True, False])


# In[50]:


print(top_products.head())


# # Exploratory Data Analysis

# VISUALIZING SALE TRENDS OVER TIME

# In[51]:


plt.figure(figsize=(14, 6))
plt.plot(train_data["date"], train_data["sales"], color="blue", alpha=0.7)
plt.title("Sales Trends Over Time", fontsize=14)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Sales", fontsize=12)
plt.grid()
plt.show()


# ANALYZE SALES BEFORE AND AFTER HOLIDAYS AND PROMOTIONS

# In[52]:


plt.figure(figsize=(10, 5))
sns.boxplot(x=train_data["is_holiday"], y=train_data["sales"], palette="coolwarm")
plt.title("Sales Distribution: Holidays vs. Non-Holidays")
plt.xlabel("Is Holiday (0=No, 1=Yes)")
plt.ylabel("Sales")
plt.show()


# From above we can say there are more sales during Holiday period

# In[53]:


#correlations between oil price and sales

plt.figure(figsize=(10, 5))
sns.scatterplot(x=train_data["dcoilwtico"], y=train_data["sales"], alpha=0.5)
plt.title("Correlation Between Oil Prices and Sales")
plt.xlabel("Oil Price (USD)")
plt.ylabel("Sales")
plt.grid()
plt.show()


# In[54]:


correlation = train_data["sales"].corr(train_data["dcoilwtico"])
print(f"Correlation between Sales and Oil Prices: {correlation:.4f}")


# In[55]:


#identifying Anomalies 


# Using IQR method to detect anomalies
Q1 = train_data["sales"].quantile(0.25)
Q3 = train_data["sales"].quantile(0.75)
IQR = Q3 - Q1

# Define anomaly thresholds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify anomalies
train_data["is_anomaly"] = ((train_data["sales"] < lower_bound) | (train_data["sales"] > upper_bound)).astype(int)

# Plot anomalies
plt.figure(figsize=(14, 6))
plt.plot(train_data["date"], train_data["sales"], color="blue", alpha=0.7, label="Sales")
plt.scatter(train_data["date"][train_data["is_anomaly"] == 1], 
            train_data["sales"][train_data["is_anomaly"] == 1], 
            color="red", label="Anomalies", zorder=3)
plt.title("Sales Trends with Anomalies Highlighted", fontsize=14)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Sales", fontsize=12)
plt.legend()
plt.grid()
plt.show()


# In[56]:


test_data = pd.read_csv(r"C:\Users\Priya\Downloads\store_forecasting_data (1)\store_forecasting_data\test.csv")
test_data.head()


# In[57]:


test_data.info()


# In[58]:


test_data['date']=pd.to_datetime(test_data['date'],format ="%Y-%m-%d")


# In[59]:


test_data.info()


# In[60]:


#set date as a index value for time series model


# In[61]:


train_data.set_index("date",inplace = True)
test_data.set_index("date",inplace = True)


# In[62]:


train_data.head()


# In[63]:


test_data.head()


# In[64]:


#Decomposition to check dataset component


# In[65]:


from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(train_data['sales'], period = 7)
decomposition.plot()
plt.show()


# In[66]:


# To check whether my data is stationary or non-stationary
# check name - "Augumented Dickey Fuller Test"
from statsmodels.tsa.stattools import adfuller


# In[67]:


adfuller(train_data['sales'])


# In[68]:


def adf_check(timeseries):
    result = adfuller(timeseries)
    print("Augmented Dickey Fuller Test - Stationary or Non-Staionary")
    labels = ['ADF Test Statistics','p-value', '#Lags', 'No of Obs']
    
    for a, b in zip(result, labels):
        print(b + " : " + str(a))
        
    if result[1] <=0.05:
        print("Strong evidence against null hypothesis and my timeseries is Stationary")
    else:
        print("Weak evidence against null hypothesis and my timeseries is Non-Stationary")    


# In[69]:


adf_check(train_data['sales'])


# In[71]:


#Now my time series data is stationary

# AIC = -2LL + 2K
# K = Parameter
# Parameter = trend (p d q) / seasonality (P D Q)

# D / d = difference - Integrated 

# Trend
# d = 1
# p = ?
# q = ?

# Seasonality 
# D = ?
# P = ?
# Q = ?

# ARIMA = AutoRegressive Integrated Moving Avg
# AR - P/p
# I - D/d
# MA - Q/q


# In[72]:


# create a seasonality
train_data['Seasonality'] = train_data['sales'] - train_data['sales'].shift(7)


# In[73]:


train_data.head(20)


# In[77]:


adf_check(train_data['Seasonality'].dropna())


# In[ ]:


# Trend
# d = 1
# p = ?
# q = ?

# Seasonality 
# D = 1
# P = ?
# Q = ?

# To calculate p,q and P, Q 
# how to find it  - by the help of acf and pacf method
# acf - autocorrelation
# pacf - partial autocorrelation

# ARIMA = AutoRegressive Integrated Moving Avg
# AR - P/p - pacf - partial autocorrelation
# I - D/d - got it (stationary)
# MA - Q/q - acf - autocorrelation


# In[78]:


# To find P/p and Q/q value 
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# In[79]:


# Trend : d = 1, p = ? and q = ?
# to find p value
plot_pacf(train_data['sales'].dropna(), lags = 30)
plt.show()


# In[80]:


# to find  q value
plot_acf(train_data['sales'].dropna(), lags = 30)
plt.show()


# In[ ]:


#Trend: p = 1,q=1,d=1


# In[81]:


#Seasonality
# to find p value
plot_pacf(train_data['Seasonality'].dropna(), lags = 30)
plt.show()


# In[82]:


# to find  q value
plot_acf(train_data['Seasonality'].dropna(), lags = 30)
plt.show()


# In[ ]:


#seasonality: p=1,d=1,q=5


# In[83]:


# Building Time Series Forecasting

from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm


# In[84]:


model = sm.tsa.statespace.SARIMAX(train_data['sales'],
                                 order = (1,1,1), seasonal_order=(1,1,5,12))


# In[85]:


# Auto ARIMA approach
import itertools

p = d = q = range(0,2)

pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]

print("Few Parameter combinations are :")
print('{} x {}'.format(pdq[1], seasonal_pdq[1]))
print('{} x {}'.format(pdq[2], seasonal_pdq[2]))


# In[ ]:


# We are implemention the above parameter by using permutation and combination approach to get best AIC value

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            model = sm.tsa.statespace.SARIMAX(train_data['sales'],
                                              order = param, 
                                              seasonal_order=param_seasonal,
                                              enforce_stationarity=False,
                                              enforce_invertibility=False)
            results = model.fit()
            print('ARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic))
        except :
            continue        


# In[ ]:


#ARIMA(0, 0, 0)x(0, 0, 1, 12) - AIC:6002998.575023256  is the best one because of less AIC


# In[ ]:


#Prediction


# In[ ]:


len(train_data)


# In[ ]:


dataset.head()


# In[ ]:


# validate whether my model is right or wrong

train_data['forecast'] = results.predict(start=3000874, end=3000888, dynamic=True)
train_data[['sales', 'forecast']].plot()


# In[ ]:


dataset.tail()


# In[ ]:


# my time series model is absolutely fine and we are ready to forecaset the sales value

from pandas.tseries.offsets import DateOffset


# In[ ]:


future_dates = [dataset.index[-1] + DateOffset(days=x) for x in range(0,20) ]


# In[ ]:


future_dates_df = pd.DataFrame(index =future_dates[1:], columns = train_data.columns )


# In[ ]:


future_dates_df.head()


# In[ ]:


future_dates_df.tail()


# In[ ]:


future_df = pd.concat([train_data, future_dates_df])


# In[ ]:


future_df.head()


# In[ ]:


future_df.tail()


# In[ ]:


future_df['forecast'] = results.predict(start=3000888, end = 3000903, dynamic = True)
future_df[['sales', 'forecast']].plot()


# In[ ]:


future_df.tail(50)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




