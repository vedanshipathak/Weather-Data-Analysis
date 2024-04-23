#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[14]:


df_weather=pd.read_csv(r"C:\Users\VEDANSHI\Downloads\weatherHistory.csv.zip")


# In[15]:


df_weather_row_count, df_weather_column_count=df_weather.shape
print('Total number of rows:', df_weather_row_count)
print('Total number of columns:', df_weather_column_count)


# In[16]:


df_weather.info()


# In[17]:


Summary_Weather=df_weather["Summary"].value_counts().reset_index()
Summary_Weather.columns=["Weather Type","Count"]
Summary_Weather


# In[18]:


wt_missing =df_weather.isna().sum()
wt_missing 


# In[19]:


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming df_weather is your DataFrame

# Calculate missing values
wt_missing = df_weather.isna().sum()

# Set up Seaborn for better aesthetics
sns.set(style="whitegrid")

# Create a colorful box plot
plt.figure(figsize=(10, 6))
plt.boxplot(wt_missing, vert=False, widths=0.7, patch_artist=True, boxprops=dict(facecolor='lightblue'))
plt.yticks([1], ['Missing Values'])
plt.xlabel('Count of Missing Values', fontsize=12)
plt.title('Box Plot of Missing Values in df_weather DataFrame', fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Display the plot
plt.show()


# In[22]:


weatherdata=['formatted date','temperature','apparent temperature','windspeed','visibility','pressure','daily summary']
data=[96429,7574,8984,2484,949,4979,214]
plt.pie(data,labels=weatherdata)
plt.show()


# In[8]:


t_cells = np.product(df_weather.shape)
t_missing = wt_missing.sum()
percent_missing = (t_missing/t_cells) * 100
print(percent_missing)


# In[9]:


df_weather['Precip Type'].fillna(df_weather['Precip Type'].value_counts().index[0],inplace=True)
df_weather.isna().sum()


# In[10]:


df_weather.head().iloc[:5]


# In[11]:


#number of times the weather was Clear
#filtering
df_weather.head(2)


# In[12]:


#value count 
df_weather.Summary.value_counts()


# In[13]:


summary_data=['Partly Cloudy','Mostly Cloudy','Overcast','Clear','Foggy']
data=[31733,28094,16597,10890,7148]
plt.pie(data,labels=summary_data)
plt.show()


# In[18]:


#data when weather was clear using filtering method

df_weather[df_weather.Summary=='Clear']


# In[14]:


#TREND IN DATA FOR TEMP,HUMIDITY ,VISIBILITY ,WIND SPEED,APPARENT TEMP
import pandas as pd
df_w1=pd.DataFrame(df_weather)


# In[17]:


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming df_w1 is your DataFrame

# Increase the size of the heatmap and change color coordination
plt.figure(figsize=(10, 8))
sns.heatmap(df_w1[df_w1.columns[:8]].corr(), cmap='coolwarm', annot=True, fmt=".2f", linewidths=.5)

# Set the title
plt.title('Correlation Heatmap of the First 8 Columns in df_w1', fontsize=14)

# Display the plot
plt.show()


# In[42]:


import datetime as dt
from datetime import timedelta


# In[43]:


df_weather["Formatted Date"]=pd.to_datetime(df_weather["Formatted Date"])


# In[44]:


plt.figure(figsize=(12,7))
plt.xticks(rotation=90)
sns.barplot(data=df_weather, x="Summary", y="Temperature (C)",hue="Precip Type")


# In[29]:


#bar plot for humidty vs summary 
plt.figure(figsize=(12,6))
plt.xticks(rotation=90)
sns.barplot(data=df_weather, x="Summary", y="Humidity")


# In[30]:


#rename the column name Daily Summary to Weather Condition
df_weather.head(3)


# In[46]:


df_weather.rename(columns={'Daily Summary':'Weather Condition'})


# In[47]:


#calculating the mean humidity
df_weather.head(2)


# In[48]:


import pandas as pd


# In[49]:


df_w=pd.DataFrame(df_weather)


# In[50]:


df_w.Humidity.mean()


# In[51]:


#standard Deviation of pressure
df_w.Pressure(millibars).std()


# In[31]:


#finding all the instances when the wind speed was more than 20 and visbility >15
df_weather.head(3)


# In[32]:


df_weather[(df_weather['Wind Speed (km/h)']>20)& (df_weather['Visibility (km)']>15)]


# In[9]:


#showing all records where the data is is clear and the humidity is greater than 20 or visibility greater than 15
data=pd.DataFrame(df_weather)


# In[10]:


data.head(2)


# In[11]:


data[(data['Summary']=='Clear' ) & (data['Humidity'] >20) | (data['Visibility (km)']> 15)]


# In[54]:


import pandas as pd
import matplotlib.pyplot as plt


filtered_data = data[(data['Summary'] == 'Clear') & ((data['Humidity'] > 20) | (data['Visibility (km)'] > 15))]

 #you can create a scatter plot of 'Humidity' vs. 'Visibility (km)where the data points are those which meets the condn of filetered data'
plt.scatter(filtered_data['Humidity'], filtered_data['Visibility (km)'])
plt.xlabel('Humidity')
plt.ylabel('Visibility (km)')
plt.title('Scatter Plot of Humidity vs. Visibility')
plt.show()


# In[63]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the weather data from the CSV file
df_weather = pd.read_csv(r"C:\Users\VEDANSHI\Downloads\weatherHistory.csv.zip")

# Convert the 'Formatted Date' column to a datetime object
df_weather['Formatted Date'] = pd.to_datetime(df_weather['Formatted Date'])

# Sort the data by date
df_weather = df_weather.sort_values(by='Formatted Date')

# Smooth wind speed data (e.g., using a rolling mean) to reduce noise
df_weather['Smoothed Wind Speed'] = df_weather['Wind Speed (km/h)'].rolling(window=7, min_periods=1).mean()

# Detect anomalies in wind speed
threshold = 5.0  # Adjust this threshold as needed
df_weather['Wind Speed Anomaly'] = np.abs(df_weather['Wind Speed (km/h)'] - df_weather['Smoothed Wind Speed']) > threshold

# Plot wind speed and anomalies
plt.figure(figsize=(10, 6))
plt.plot(df_weather['Formatted Date'], df_weather['Wind Speed (km/h)'], label='Wind Speed (km/h)', color='blue')

plt.scatter(df_weather[df_weather['Wind Speed Anomaly']]['Formatted Date'], 
            df_weather[df_weather['Wind Speed Anomaly']]['Wind Speed (km/h)'],
            color='red', marker='o', label='Anomalies')
plt.xlabel('Date')
plt.ylabel('Wind Speed (km/h)')
plt.title('Wind Speed and Anomalies')
plt.legend()
plt.grid(True)
plt.show()


# In[59]:


#To visualize the distribution of daily temperatures using histograms or kernel density plots with the provided dataset
#kernel density checks for the estimated probability density
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# on x we have value of temp as bins and on y we have freq count of temp within each bin
df_weather = pd.read_csv(r"C:\Users\VEDANSHI\Downloads\weatherHistory.csv.zip")
plt.figure(figsize=(8, 6))
sns.histplot(data=df_weather, x='Temperature (C)', bins=30, kde=True, color='red')
plt.xlabel('Temperature (C)')
plt.ylabel('Frequency')
plt.title('Distribution of Daily Temperatures')
plt.grid(True)
plt.show()


# In[21]:


import pandas as pd
from windrose import WindroseAxes
import matplotlib.pyplot as plt

wind_speed = df['Wind Speed']
wind_direction = df['Wind Direction']

# Create a wind rose chart
fig, ax = plt.subplots(subplot_kw=dict(projection="windrose"))
ax.bar(wind_direction, wind_speed, normed=True, opening=0.8, edgecolor='white')

# Customize the appearance of the wind rose chart
ax.set_legend(title='Wind Speed (m/s)')
ax.set_title('Wind Rose Chart')

# Display the plot
plt.show()


# In[22]:


pip install windrose


# In[35]:


import pandas as pd
from windrose import WindroseAxes
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming df_weather is your DataFrame
wind_speed = df_weather['Wind Speed (km/h)']
wind_direction = df_weather['Wind Bearing (degrees)']

# Set a different color palette (e.g., "viridis")
sns.set_palette("plasma")

# Create a wind rose chart
fig, ax = plt.subplots(subplot_kw=dict(projection="windrose"))
ax.bar(wind_direction, wind_speed, normed=True, opening=0.8, edgecolor='white')

# Customize the appearance of the wind rose chart
ax.set_legend(title='Wind Speed (km/h)', loc='upper left', bbox_to_anchor=(1, 1))
ax.set_title('Wind Rose Chart', fontsize=16, fontweight='bold')

# Add a grid for better readability
ax.grid(linestyle='--', alpha=0.7)

# Display the plot
plt.show()


# In[7]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set the color palette for the plot
sns.set_palette("viridis")

# Create a bar plot with improved aesthetics
plt.figure(figsize=(12, 8))
sns.barplot(data=df_weather, y="Summary", x="Humidity", ci=None)

# Add labels and title
plt.xlabel("Humidity")
plt.ylabel("Weather Summary")
plt.title("Humidity Distribution by Weather Summary")

# Rotate y-axis labels for better readability
plt.xticks(rotation=45)

# Display the plot
plt.show()


# In[12]:


import pandas as pd
import matplotlib.pyplot as plt

# Filter the data
filtered_data = data[(data['Summary'] == 'Clear') & ((data['Humidity'] > 20) | (data['Visibility (km)'] > 15))]

# Create a scatter plot with improved aesthetics
plt.figure(figsize=(10, 6))
plt.scatter(filtered_data['Humidity'], filtered_data['Visibility (km)'], c='blue', alpha=0.6, s=50, edgecolors='w')

# Add labels and title
plt.xlabel('Humidity')
plt.ylabel('Visibility (km)')
plt.title('Scatter Plot of Humidity vs. Visibility for Clear Weather')

# Adjust plot appearance
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(['Clear Weather'], loc='upper right')
plt.tight_layout()

# Display the plot
plt.show()


# In[1]:


pip install pandas wordcloud matplotlib


# In[3]:


#Create a word cloud or bar chart to display the most frequent terms or phrases in the "Daily Summary" column.
#This can provide a quick overview of common weather patterns described in the dataset.
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load the CSV file into a pandas DataFrame
  # Replace with the actual file path
df = pd.read_csv(r"C:\Users\VEDANSHI\Downloads\weatherHistory.csv.zip")

# Specify the column for which you want to create the word cloud
column_name = 'Summary'  # Replace with the actual column name

# Combine all the text in the specified column
text = ' '.join(df[column_name].astype(str))

# Create the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

# Display the word cloud using matplotlib
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[7]:


import pandas as pd
import matplotlib.pyplot as plt

 # Replace with the actual file path
df = pd.read_csv(r"C:\Users\VEDANSHI\Downloads\weatherHistory.csv.zip")

# Specify the columns for the hexbin plot
temperature_column = 'Temperature (C)'  # Replace with the actual column name
apparent_temperature_column = 'Apparent Temperature (C)'  # Replace with the actual column name

# Create a hexbin plot
plt.figure(figsize=(10, 8))
plt.hexbin(df[temperature_column], df[apparent_temperature_column], gridsize=50, cmap='viridis')
plt.xlabel('Temperature (C)')
plt.ylabel('Apparent Temperature (C)')
plt.title('Hexbin Plot: Temperature vs. Apparent Temperature')
plt.colorbar(label='Count')
plt.tight_layout()

# Show the plot
plt.show()


# In[ ]:


#. This type of plot provides a smoothed representation of the data density, allowing you to visualize areas of high and low correlation
#highelights the high correlation for the above  code


# In[ ]:


#Visibility is an essential weather parameter. Analyzing its distribution helps meteorologists understand typical visibility patterns, detect anomalies, and predict potential weather-related events


# In[3]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file into a pandas DataFrame
  # Replace with the actual file path
df = pd.read_csv(r"C:\Users\VEDANSHI\Downloads\weatherHistory.csv.zip")

# Specify the column for visibility
visibility_column = 'Visibility (km)'  # Replace with the actual column name

# Create a histogram for visibility distribution
plt.figure(figsize=(10, 6))
sns.histplot(df[visibility_column], bins=20, kde=True, color='skyblue')
plt.xlabel('Visibility (km)')
plt.ylabel('Frequency')
plt.title('Visibility Distribution')
plt.tight_layout()

# Find the bin with the maximum frequency
bin_edges, bin_heights, _ = plt.hist(df[visibility_column], bins=20, alpha=0)  # Use alpha=0 to hide the plotted histogram
max_bin_index = bin_heights.argmax()
max_bin_value = bin_edges[max_bin_index], bin_edges[max_bin_index + 1]

print(f"The bin with the maximum frequency is between {max_bin_value[0]} and {max_bin_value[1]} km.")

# Show the plot
plt.show()


# In[ ]:




