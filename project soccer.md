#Prediction question:

Which teams based on the current stats will be the top 4 teams in 2020-21 season of England Premier League?


```python
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
```


```python
url_1="https://fbref.com/en/comps/9/3232/2019-2020-Premier-League-Stats"
url_2="https://fbref.com/en/comps/9/3232/shooting/2019-2020-Premier-League-Stats"
url_3="https://fbref.com/en/comps/9/3232/misc/2019-2020-Premier-League-Stats"
url_4="https://fbref.com/en/comps/9/3232/possession/2019-2020-Premier-League-Stats"
url_5="https://fbref.com/en/comps/9/3232/keepers/2019-2020-Premier-League-Stats"
url_6="https://fbref.com/en/comps/9/3232/stats/2019-2020-Premier-League-Stats"
url1="https://fbref.com/en/comps/9/1889/2018-2019-Premier-League-Stats"
url2="https://fbref.com/en/comps/9/1889/shooting/2018-2019-Premier-League-Stats"
url3="https://fbref.com/en/comps/9/1889/misc/2018-2019-Premier-League-Stats"
url4="https://fbref.com/en/comps/9/1889/possession/2018-2019-Premier-League-Stats"
url5="https://fbref.com/en/comps/9/1889/keepers/2018-2019-Premier-League-Stats"
url6="https://fbref.com/en/comps/9/1889/stats/2018-2019-Premier-League-Stats"
url7="https://fbref.com/en/comps/9/1631/2017-2018-Premier-League-Stats"
url8="https://fbref.com/en/comps/9/1631/shooting/2017-2018-Premier-League-Stats"
url9="https://fbref.com/en/comps/9/1631/misc/2017-2018-Premier-League-Stats"
url10="https://fbref.com/en/comps/9/1631/possession/2017-2018-Premier-League-Stats"
url10a="https://fbref.com/en/comps/9/1631/keepers/2017-2018-Premier-League-Stats"
url10b="https://fbref.com/en/comps/9/1631/stats/2017-2018-Premier-League-Stats"
url11="https://fbref.com/en/comps/9/1526/2016-2017-Premier-League-Stats"
url12="https://fbref.com/en/comps/9/1526/shooting/2016-2017-Premier-League-Stats"
url13="https://fbref.com/en/comps/9/1526/misc/2016-2017-Premier-League-Stats"
url14="https://fbref.com/en/comps/9/1526/possession/2016-2017-Premier-League-Stats"
url15="https://fbref.com/en/comps/9/1526/keepers/2016-2017-Premier-League-Stats"
url16="https://fbref.com/en/comps/9/1526/stats/2016-2017-Premier-League-Stats"
url17="https://fbref.com/en/comps/9/1467/2015-2016-Premier-League-Stats"
url18="https://fbref.com/en/comps/9/1467/shooting/2015-2016-Premier-League-Stats"
url19="https://fbref.com/en/comps/9/1467/misc/2015-2016-Premier-League-Stats"
url20="https://fbref.com/en/comps/9/1467/possession/2015-2016-Premier-League-Stats"
url21="https://fbref.com/en/comps/9/1467/keepers/2015-2016-Premier-League-Stats"
url22="https://fbref.com/en/comps/9/1467/stats/2015-2016-Premier-League-Stats"
url23="https://fbref.com/en/comps/9/733/2014-2015-Premier-League-Stats"
url24="https://fbref.com/en/comps/9/733/shooting/2018-2019-Premier-League-Stats"
url25="https://fbref.com/en/comps/9/733/misc/2014-2015-Premier-League-Stats"
url26="https://fbref.com/en/comps/9/733/possession/2014-2015-Premier-League-Stats"
url27="https://fbref.com/en/comps/9/733/keepers/2014-2015-Premier-League-Stats"
url28="https://fbref.com/en/comps/9/733/stats/2014-2015-Premier-League-Stats"
url29="https://fbref.com/en/comps/9/Premier-League-Stats"
url30="https://fbref.com/en/comps/9/shooting/Premier-League-Stats"
url31="https://fbref.com/en/comps/9/misc/Premier-League-Stats"
url32="https://fbref.com/en/comps/9/possession/Premier-League-Stats"
url33="https://fbref.com/en/comps/9/keepers/Premier-League-Stats"
url34="https://fbref.com/en/comps/9/stats/Premier-League-Stats"
```

#Data clean

I collected the data from 2014-15 season up to 2020-21 season. To get a table I wanted, I extracted necessary columns from mutiple tables and merged them into the table I wanted. After getting a table for each season from 2014-15 to 2019-20, I combine all of them into one big dataframe ("soccer"). For the current season, I made a seperate dataframe ("soccer_current").


*It takes a bit to run


```python
#19-20 season
general_stats=pd.read_html(url_1)
target=pd.read_html(url_2)
Fouls=pd.read_html(url_3)
possession=pd.read_html(url_4)
saves=pd.read_html(url_5)
age=pd.read_html(url_6)

df1=general_stats[0]
df3=target[0]
df4=Fouls[0]
df5=possession[0]
df6=saves[0]
df7=age[0]

df3=df3.iloc[20:40,[0,5]]
df3.columns = df3.columns.droplevel()
df3.rename(columns={'Squad':'Team',"SoT":"Shots_on_Target"},inplace=True)
df4=df4.iloc[20:40,[0,6]]
df4.columns = df4.columns.droplevel()
df4.rename(columns={'Squad':'Team',"Fls":"Fouls"},inplace=True)
df5=df5.iloc[20:40,[0,2]]
df5.columns = df5.columns.droplevel()
df5.rename(columns={'Squad':'Team',"Poss":"Possession"},inplace=True)
df6=df6.iloc[20:40,[0,8]]
df6.columns = df6.columns.droplevel()
df6.rename(columns={'Squad':'Team',"Saves":"Saves_by_Goalkeeper"},inplace=True)
df7=df7.iloc[20:40,[0,2]]
df7.columns = df7.columns.droplevel()
df7.rename(columns={'Squad':'Team',"Age":"Age"},inplace=True)

df=pd.DataFrame()
df['Team']=df1['Squad']
df['Points']=df1['Pts']
df['Goals']=df1['GF']
df['Goals_Allowed']=df1['GA']
df=pd.merge(df,df3,on='Team')
df=pd.merge(df,df4,on='Team')
df=pd.merge(df,df5,on='Team')
df=pd.merge(df,df6,on='Team')
df=pd.merge(df,df7,on='Team')

df=df[['Team', 'Points', "Goals_Allowed", 'Goals' ,'Shots_on_Target', 'Possession', 'Fouls', 'Saves_by_Goalkeeper', 'Age']]
df['Team'] = df['Team'] + "_19"
df_19=df
```


```python
#18-19 season
general_stats=pd.read_html(url1)
target=pd.read_html(url2)
Fouls=pd.read_html(url3)
possession=pd.read_html(url4)
saves=pd.read_html(url5)
age=pd.read_html(url6)

df1=general_stats[0]
df3=target[0]
df4=Fouls[0]
df5=possession[0]
df6=saves[0]
df7=age[0]

df3=df3.iloc[20:40,[0,5]]
df3.columns = df3.columns.droplevel()
df3.rename(columns={'Squad':'Team',"SoT":"Shots_on_Target"},inplace=True)
df4=df4.iloc[20:40,[0,6]]
df4.columns = df4.columns.droplevel()
df4.rename(columns={'Squad':'Team',"Fls":"Fouls"},inplace=True)
df5=df5.iloc[20:40,[0,2]]
df5.columns = df5.columns.droplevel()
df5.rename(columns={'Squad':'Team',"Poss":"Possession"},inplace=True)
df6=df6.iloc[20:40,[0,8]]
df6.columns = df6.columns.droplevel()
df6.rename(columns={'Squad':'Team',"Saves":"Saves_by_Goalkeeper"},inplace=True)
df7=df7.iloc[20:40,[0,2]]
df7.columns = df7.columns.droplevel()
df7.rename(columns={'Squad':'Team',"Age":"Age"},inplace=True)

df=pd.DataFrame()
df['Team']=df1['Squad']
df['Points']=df1['Pts']
df['Goals']=df1['GF']
df['Goals_Allowed']=df1['GA']
df=pd.merge(df,df3,on='Team')
df=pd.merge(df,df4,on='Team')
df=pd.merge(df,df5,on='Team')
df=pd.merge(df,df6,on='Team')
df=pd.merge(df,df7,on='Team')

df=df[['Team', 'Points', 'Goals','Goals_Allowed' ,'Shots_on_Target', 'Possession', 'Fouls', 'Saves_by_Goalkeeper', 'Age']]
df['Team'] = df['Team'] + "_18"
df_18=df
```


```python
#17-18 season
general_stats=pd.read_html(url7)
target=pd.read_html(url8)
Fouls=pd.read_html(url9)
possession=pd.read_html(url10)
saves=pd.read_html(url10a)
age=pd.read_html(url10b)

df1=general_stats[0]
df3=target[0]
df4=Fouls[0]
df5=possession[0]
df6=saves[0]
df7=age[0]

df3=df3.iloc[20:40,[0,5]]
df3.columns = df3.columns.droplevel()
df3.rename(columns={'Squad':'Team',"SoT":"Shots_on_Target"},inplace=True)
df4=df4.iloc[20:40,[0,6]]
df4.columns = df4.columns.droplevel()
df4.rename(columns={'Squad':'Team',"Fls":"Fouls"},inplace=True)
df5=df5.iloc[20:40,[0,2]]
df5.columns = df5.columns.droplevel()
df5.rename(columns={'Squad':'Team',"Poss":"Possession"},inplace=True)
df6=df6.iloc[20:40,[0,8]]
df6.columns = df6.columns.droplevel()
df6.rename(columns={'Squad':'Team',"Saves":"Saves_by_Goalkeeper"},inplace=True)
df7=df7.iloc[20:40,[0,2]]
df7.columns = df7.columns.droplevel()
df7.rename(columns={'Squad':'Team',"Age":"Age"},inplace=True)

df=pd.DataFrame()
df['Team']=df1['Squad']
df['Points']=df1['Pts']
df['Goals']=df1['GF']
df['Goals_Allowed']=df1['GA']
df=pd.merge(df,df3,on='Team')
df=pd.merge(df,df4,on='Team')
df=pd.merge(df,df5,on='Team')
df=pd.merge(df,df6,on='Team')
df=pd.merge(df,df7,on='Team')

df=df[['Team', 'Points', 'Goals','Goals_Allowed' ,'Shots_on_Target', 'Possession', 'Fouls', 'Saves_by_Goalkeeper', 'Age']]
df['Team'] = df['Team'] + "_17"
df_17=df
```


```python
#16-17 season
general_stats=pd.read_html(url11)
target=pd.read_html(url12)
Fouls=pd.read_html(url13)
possession=pd.read_html(url14)
saves=pd.read_html(url15)
age=pd.read_html(url16)

df1=general_stats[0]
df3=target[0]
df4=Fouls[0]
df5=possession[0]
df6=saves[0]
df7=age[0]

df3=df3.iloc[20:40,[0,5]]
df3.columns = df3.columns.droplevel()
df3.rename(columns={'Squad':'Team',"SoT":"Shots_on_Target"},inplace=True)
df4=df4.iloc[20:40,[0,6]]
df4.columns = df4.columns.droplevel()
df4.rename(columns={'Squad':'Team',"Fls":"Fouls"},inplace=True)
df5=df5.iloc[20:40,[0,2]]
df5.columns = df5.columns.droplevel()
df5.rename(columns={'Squad':'Team',"Poss":"Possession"},inplace=True)
df6=df6.iloc[20:40,[0,8]]
df6.columns = df6.columns.droplevel()
df6.rename(columns={'Squad':'Team',"Saves":"Saves_by_Goalkeeper"},inplace=True)
df7=df7.iloc[20:40,[0,2]]
df7.columns = df7.columns.droplevel()
df7.rename(columns={'Squad':'Team',"Age":"Age"},inplace=True)

df=pd.DataFrame()
df['Team']=df1['Squad']
df['Points']=df1['Pts']
df['Goals']=df1['GF']
df['Goals_Allowed']=df1['GA']
df=pd.merge(df,df3,on='Team')
df=pd.merge(df,df4,on='Team')
df=pd.merge(df,df5,on='Team')
df=pd.merge(df,df6,on='Team')
df=pd.merge(df,df7,on='Team')


df=df[['Team', 'Points', 'Goals','Goals_Allowed' ,'Shots_on_Target', 'Possession', 'Fouls', 'Saves_by_Goalkeeper', 'Age']]
df['Team'] = df['Team'] + "_16"
df_16=df
```


```python
#15-16 season
general_stats=pd.read_html(url17)
target=pd.read_html(url18)
Fouls=pd.read_html(url19)
possession=pd.read_html(url20)
saves=pd.read_html(url21)
age=pd.read_html(url22)

df1=general_stats[0]
df3=target[0]
df4=Fouls[0]
df5=possession[0]
df6=saves[0]
df7=age[0]

df3=df3.iloc[20:40,[0,5]]
df3.columns = df3.columns.droplevel()
df3.rename(columns={'Squad':'Team',"SoT":"Shots_on_Target"},inplace=True)
df4=df4.iloc[20:40,[0,6]]
df4.columns = df4.columns.droplevel()
df4.rename(columns={'Squad':'Team',"Fls":"Fouls"},inplace=True)
df5=df5.iloc[20:40,[0,2]]
df5.columns = df5.columns.droplevel()
df5.rename(columns={'Squad':'Team',"Poss":"Possession"},inplace=True)
df6=df6.iloc[20:40,[0,8]]
df6.columns = df6.columns.droplevel()
df6.rename(columns={'Squad':'Team',"Saves":"Saves_by_Goalkeeper"},inplace=True)
df7=df7.iloc[20:40,[0,2]]
df7.columns = df7.columns.droplevel()
df7.rename(columns={'Squad':'Team',"Age":"Age"},inplace=True)

df=pd.DataFrame()
df['Team']=df1['Squad']
df['Points']=df1['Pts']
df['Goals']=df1['GF']
df['Goals_Allowed']=df1['GA']
df=pd.merge(df,df3,on='Team')
df=pd.merge(df,df4,on='Team')
df=pd.merge(df,df5,on='Team')
df=pd.merge(df,df6,on='Team')
df=pd.merge(df,df7,on='Team')


df=df[['Team', 'Points', 'Goals' ,'Goals_Allowed','Shots_on_Target', 'Possession', 'Fouls', 'Saves_by_Goalkeeper', 'Age']]
df['Team'] = df['Team'] + "_15"
df_15=df
```


```python
#14-15 season
general_stats=pd.read_html(url23)
target=pd.read_html(url24)
Fouls=pd.read_html(url25)
possession=pd.read_html(url26)
saves=pd.read_html(url27)
age=pd.read_html(url28)

df1=general_stats[0]
df3=target[0]
df4=Fouls[0]
df5=possession[0]
df6=saves[0]
df7=age[0]

df3=df3.iloc[20:40,[0,5]]
df3.columns = df3.columns.droplevel()
df3.rename(columns={'Squad':'Team',"SoT":"Shots_on_Target"},inplace=True)
df4=df4.iloc[20:40,[0,6]]
df4.columns = df4.columns.droplevel()
df4.rename(columns={'Squad':'Team',"Fls":"Fouls"},inplace=True)
df5=df5.iloc[20:40,[0,2]]
df5.columns = df5.columns.droplevel()
df5.rename(columns={'Squad':'Team',"Poss":"Possession"},inplace=True)
df6=df6.iloc[20:40,[0,8]]
df6.columns = df6.columns.droplevel()
df6.rename(columns={'Squad':'Team',"Saves":"Saves_by_Goalkeeper"},inplace=True)
df7=df7.iloc[20:40,[0,2]]
df7.columns = df7.columns.droplevel()
df7.rename(columns={'Squad':'Team',"Age":"Age"},inplace=True)

df=pd.DataFrame()
df['Team']=df1['Squad']
df['Points']=df1['Pts']
df['Goals']=df1['GF']
df['Goals_Allowed']=df1['GA']
df=pd.merge(df,df3,on='Team')
df=pd.merge(df,df4,on='Team')
df=pd.merge(df,df5,on='Team')
df=pd.merge(df,df6,on='Team')
df=pd.merge(df,df7,on='Team')


df=df[['Team', 'Points', 'Goals','Goals_Allowed' ,'Shots_on_Target', 'Possession', 'Fouls', 'Saves_by_Goalkeeper', 'Age']]
df['Team'] = df['Team'] + "_14"
df_14=df
```


```python
soccer=pd.concat([df_19,df_18,df_17,df_16,df_15,df_14])
soccer=soccer[['Team', 'Points', 'Goals','Goals_Allowed' ,'Shots_on_Target', 'Possession', 'Fouls', 'Saves_by_Goalkeeper', 'Age']]
```

    C:\Users\jmlim\Anaconda3\lib\site-packages\ipykernel_launcher.py:1: FutureWarning: Sorting because non-concatenation axis is not aligned. A future version
    of pandas will change to not sort by default.
    
    To accept the future behavior, pass 'sort=False'.
    
    To retain the current behavior and silence the warning, pass 'sort=True'.
    
      """Entry point for launching an IPython kernel.
    


```python
#current season (20-21 season)
general_stats=pd.read_html(url29)
target=pd.read_html(url30)
Fouls=pd.read_html(url31)
possession=pd.read_html(url32)
saves=pd.read_html(url33)
age=pd.read_html(url34)

df1=general_stats[0]
df3=target[0]
df4=Fouls[0]
df5=possession[0]
df6=saves[0]
df7=age[0]

df3=df3.iloc[20:40,[0,5]]
df3.columns = df3.columns.droplevel()
df3.rename(columns={'Squad':'Team',"SoT":"Shots_on_Target"},inplace=True)
df4=df4.iloc[20:40,[0,6]]
df4.columns = df4.columns.droplevel()
df4.rename(columns={'Squad':'Team',"Fls":"Fouls"},inplace=True)
df5=df5.iloc[20:40,[0,2]]
df5.columns = df5.columns.droplevel()
df5.rename(columns={'Squad':'Team',"Poss":"Possession"},inplace=True)
df6=df6.iloc[20:40,[0,8]]
df6.columns = df6.columns.droplevel()
df6.rename(columns={'Squad':'Team',"Saves":"Saves_by_Goalkeeper"},inplace=True)
df7=df7.iloc[20:40,[0,2]]
df7.columns = df7.columns.droplevel()
df7.rename(columns={'Squad':'Team',"Age":"Age"},inplace=True)

df=pd.DataFrame()
df['Team']=df1['Squad']
df['Points']=df1['Pts']
df['Goals']=df1['GF']
df['Goals_Allowed']=df1['GA']
df=pd.merge(df,df3,on='Team')
df=pd.merge(df,df4,on='Team')
df=pd.merge(df,df5,on='Team')
df=pd.merge(df,df6,on='Team')
df=pd.merge(df,df7,on='Team')


df=df[['Team', 'Points', 'Goals', "Goals_Allowed",'Shots_on_Target', 'Possession', 'Fouls', 'Saves_by_Goalkeeper', 'Age']]
df['Team'] = df['Team'] + "_20"
soccer_current=df
```

#EDA


```python
#check first 10 rows of the soccer data
soccer.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Team</th>
      <th>Points</th>
      <th>Goals</th>
      <th>Goals_Allowed</th>
      <th>Shots_on_Target</th>
      <th>Possession</th>
      <th>Fouls</th>
      <th>Saves_by_Goalkeeper</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Liverpool_19</td>
      <td>99</td>
      <td>85</td>
      <td>33</td>
      <td>222</td>
      <td>63.4</td>
      <td>331</td>
      <td>75</td>
      <td>26.6</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Manchester City_19</td>
      <td>81</td>
      <td>102</td>
      <td>35</td>
      <td>246</td>
      <td>66.9</td>
      <td>361</td>
      <td>73</td>
      <td>26.9</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Manchester Utd_19</td>
      <td>66</td>
      <td>66</td>
      <td>36</td>
      <td>200</td>
      <td>56.2</td>
      <td>423</td>
      <td>92</td>
      <td>24.8</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Chelsea_19</td>
      <td>66</td>
      <td>69</td>
      <td>54</td>
      <td>210</td>
      <td>60.7</td>
      <td>386</td>
      <td>65</td>
      <td>25.5</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Leicester City_19</td>
      <td>62</td>
      <td>67</td>
      <td>41</td>
      <td>181</td>
      <td>57.6</td>
      <td>418</td>
      <td>95</td>
      <td>26.2</td>
    </tr>
    <tr>
      <td>5</td>
      <td>Tottenham_19</td>
      <td>59</td>
      <td>61</td>
      <td>47</td>
      <td>155</td>
      <td>52.2</td>
      <td>423</td>
      <td>130</td>
      <td>26.7</td>
    </tr>
    <tr>
      <td>6</td>
      <td>Wolves_19</td>
      <td>59</td>
      <td>51</td>
      <td>40</td>
      <td>142</td>
      <td>48.3</td>
      <td>406</td>
      <td>80</td>
      <td>26.6</td>
    </tr>
    <tr>
      <td>7</td>
      <td>Arsenal_19</td>
      <td>56</td>
      <td>56</td>
      <td>48</td>
      <td>144</td>
      <td>54.0</td>
      <td>420</td>
      <td>143</td>
      <td>25.8</td>
    </tr>
    <tr>
      <td>8</td>
      <td>Sheffield Utd_19</td>
      <td>54</td>
      <td>39</td>
      <td>39</td>
      <td>109</td>
      <td>43.1</td>
      <td>410</td>
      <td>103</td>
      <td>26.9</td>
    </tr>
    <tr>
      <td>9</td>
      <td>Burnley_19</td>
      <td>54</td>
      <td>43</td>
      <td>50</td>
      <td>124</td>
      <td>41.4</td>
      <td>412</td>
      <td>112</td>
      <td>28.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#check last 10 rows of the soccer data
soccer.tail(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Team</th>
      <th>Points</th>
      <th>Goals</th>
      <th>Goals_Allowed</th>
      <th>Shots_on_Target</th>
      <th>Possession</th>
      <th>Fouls</th>
      <th>Saves_by_Goalkeeper</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>10</td>
      <td>Everton_14</td>
      <td>47</td>
      <td>48</td>
      <td>50</td>
      <td>167</td>
      <td>54.7</td>
      <td>387</td>
      <td>78</td>
      <td>27.9</td>
    </tr>
    <tr>
      <td>11</td>
      <td>West Ham_14</td>
      <td>47</td>
      <td>44</td>
      <td>47</td>
      <td>153</td>
      <td>45.6</td>
      <td>416</td>
      <td>127</td>
      <td>26.8</td>
    </tr>
    <tr>
      <td>12</td>
      <td>West Brom_14</td>
      <td>44</td>
      <td>38</td>
      <td>51</td>
      <td>132</td>
      <td>43.7</td>
      <td>420</td>
      <td>131</td>
      <td>28.0</td>
    </tr>
    <tr>
      <td>13</td>
      <td>Leicester City_14</td>
      <td>41</td>
      <td>46</td>
      <td>55</td>
      <td>137</td>
      <td>43.5</td>
      <td>485</td>
      <td>117</td>
      <td>27.9</td>
    </tr>
    <tr>
      <td>14</td>
      <td>Newcastle Utd_14</td>
      <td>39</td>
      <td>40</td>
      <td>63</td>
      <td>140</td>
      <td>47.7</td>
      <td>433</td>
      <td>98</td>
      <td>25.9</td>
    </tr>
    <tr>
      <td>15</td>
      <td>Sunderland_14</td>
      <td>38</td>
      <td>31</td>
      <td>53</td>
      <td>129</td>
      <td>45.2</td>
      <td>440</td>
      <td>138</td>
      <td>27.8</td>
    </tr>
    <tr>
      <td>16</td>
      <td>Aston Villa_14</td>
      <td>38</td>
      <td>31</td>
      <td>57</td>
      <td>127</td>
      <td>49.0</td>
      <td>398</td>
      <td>111</td>
      <td>26.3</td>
    </tr>
    <tr>
      <td>17</td>
      <td>Hull City_14</td>
      <td>35</td>
      <td>33</td>
      <td>51</td>
      <td>131</td>
      <td>44.2</td>
      <td>451</td>
      <td>102</td>
      <td>27.4</td>
    </tr>
    <tr>
      <td>18</td>
      <td>Burnley_14</td>
      <td>33</td>
      <td>28</td>
      <td>53</td>
      <td>124</td>
      <td>42.5</td>
      <td>406</td>
      <td>125</td>
      <td>27.2</td>
    </tr>
    <tr>
      <td>19</td>
      <td>QPR_14</td>
      <td>30</td>
      <td>42</td>
      <td>73</td>
      <td>149</td>
      <td>45.2</td>
      <td>450</td>
      <td>136</td>
      <td>28.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
#check soccer_current data
soccer_current
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Team</th>
      <th>Points</th>
      <th>Goals</th>
      <th>Goals_Allowed</th>
      <th>Shots_on_Target</th>
      <th>Possession</th>
      <th>Fouls</th>
      <th>Saves_by_Goalkeeper</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Tottenham_20</td>
      <td>24</td>
      <td>23</td>
      <td>9</td>
      <td>51</td>
      <td>49.8</td>
      <td>143</td>
      <td>27</td>
      <td>27.8</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Liverpool_20</td>
      <td>24</td>
      <td>26</td>
      <td>17</td>
      <td>63</td>
      <td>59.9</td>
      <td>115</td>
      <td>24</td>
      <td>27.3</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Chelsea_20</td>
      <td>22</td>
      <td>25</td>
      <td>11</td>
      <td>61</td>
      <td>59.0</td>
      <td>141</td>
      <td>18</td>
      <td>26.2</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Leicester City_20</td>
      <td>21</td>
      <td>21</td>
      <td>15</td>
      <td>43</td>
      <td>53.3</td>
      <td>103</td>
      <td>35</td>
      <td>27.4</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Southampton_20</td>
      <td>20</td>
      <td>21</td>
      <td>17</td>
      <td>50</td>
      <td>54.1</td>
      <td>142</td>
      <td>34</td>
      <td>27.4</td>
    </tr>
    <tr>
      <td>5</td>
      <td>Manchester Utd_20</td>
      <td>19</td>
      <td>19</td>
      <td>17</td>
      <td>47</td>
      <td>54.5</td>
      <td>129</td>
      <td>22</td>
      <td>26.2</td>
    </tr>
    <tr>
      <td>6</td>
      <td>Manchester City_20</td>
      <td>18</td>
      <td>17</td>
      <td>11</td>
      <td>53</td>
      <td>63.8</td>
      <td>118</td>
      <td>16</td>
      <td>26.5</td>
    </tr>
    <tr>
      <td>7</td>
      <td>West Ham_20</td>
      <td>17</td>
      <td>18</td>
      <td>14</td>
      <td>46</td>
      <td>40.5</td>
      <td>129</td>
      <td>29</td>
      <td>28.1</td>
    </tr>
    <tr>
      <td>8</td>
      <td>Everton_20</td>
      <td>17</td>
      <td>20</td>
      <td>18</td>
      <td>56</td>
      <td>52.4</td>
      <td>120</td>
      <td>33</td>
      <td>27.1</td>
    </tr>
    <tr>
      <td>9</td>
      <td>Wolves_20</td>
      <td>17</td>
      <td>11</td>
      <td>15</td>
      <td>43</td>
      <td>46.2</td>
      <td>125</td>
      <td>24</td>
      <td>27.1</td>
    </tr>
    <tr>
      <td>10</td>
      <td>Crystal Palace_20</td>
      <td>16</td>
      <td>17</td>
      <td>16</td>
      <td>41</td>
      <td>40.5</td>
      <td>124</td>
      <td>33</td>
      <td>29.5</td>
    </tr>
    <tr>
      <td>11</td>
      <td>Aston Villa_20</td>
      <td>15</td>
      <td>20</td>
      <td>13</td>
      <td>49</td>
      <td>49.6</td>
      <td>105</td>
      <td>21</td>
      <td>25.5</td>
    </tr>
    <tr>
      <td>12</td>
      <td>Newcastle Utd_20</td>
      <td>14</td>
      <td>12</td>
      <td>15</td>
      <td>26</td>
      <td>39.1</td>
      <td>105</td>
      <td>40</td>
      <td>27.4</td>
    </tr>
    <tr>
      <td>13</td>
      <td>Leeds United_20</td>
      <td>14</td>
      <td>16</td>
      <td>20</td>
      <td>57</td>
      <td>59.6</td>
      <td>122</td>
      <td>37</td>
      <td>26.6</td>
    </tr>
    <tr>
      <td>14</td>
      <td>Arsenal_20</td>
      <td>13</td>
      <td>10</td>
      <td>14</td>
      <td>32</td>
      <td>52.7</td>
      <td>113</td>
      <td>30</td>
      <td>26.6</td>
    </tr>
    <tr>
      <td>15</td>
      <td>Brighton_20</td>
      <td>10</td>
      <td>15</td>
      <td>18</td>
      <td>33</td>
      <td>52.7</td>
      <td>145</td>
      <td>14</td>
      <td>26.0</td>
    </tr>
    <tr>
      <td>16</td>
      <td>Fulham_20</td>
      <td>7</td>
      <td>11</td>
      <td>21</td>
      <td>37</td>
      <td>49.7</td>
      <td>138</td>
      <td>42</td>
      <td>26.4</td>
    </tr>
    <tr>
      <td>17</td>
      <td>Burnley_20</td>
      <td>6</td>
      <td>5</td>
      <td>18</td>
      <td>27</td>
      <td>41.4</td>
      <td>104</td>
      <td>34</td>
      <td>28.4</td>
    </tr>
    <tr>
      <td>18</td>
      <td>West Brom_20</td>
      <td>6</td>
      <td>8</td>
      <td>23</td>
      <td>35</td>
      <td>42.3</td>
      <td>130</td>
      <td>46</td>
      <td>26.7</td>
    </tr>
    <tr>
      <td>19</td>
      <td>Sheffield Utd_20</td>
      <td>1</td>
      <td>5</td>
      <td>18</td>
      <td>27</td>
      <td>39.2</td>
      <td>130</td>
      <td>42</td>
      <td>26.7</td>
    </tr>
  </tbody>
</table>
</div>




```python
#check the data's info
print(soccer.info())
print(soccer_current.info())
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 120 entries, 0 to 19
    Data columns (total 9 columns):
    Team                   120 non-null object
    Points                 120 non-null int64
    Goals                  120 non-null int64
    Goals_Allowed          120 non-null int64
    Shots_on_Target        120 non-null int64
    Possession             120 non-null float64
    Fouls                  120 non-null int64
    Saves_by_Goalkeeper    120 non-null int64
    Age                    120 non-null float64
    dtypes: float64(2), int64(6), object(1)
    memory usage: 9.4+ KB
    None
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 20 entries, 0 to 19
    Data columns (total 9 columns):
    Team                   20 non-null object
    Points                 20 non-null int64
    Goals                  20 non-null int64
    Goals_Allowed          20 non-null int64
    Shots_on_Target        20 non-null int64
    Possession             20 non-null float64
    Fouls                  20 non-null int64
    Saves_by_Goalkeeper    20 non-null int64
    Age                    20 non-null float64
    dtypes: float64(2), int64(6), object(1)
    memory usage: 1.6+ KB
    None
    


```python
#check to see if there is any missing values
print(soccer.isnull().sum())
print(soccer_current.isnull().sum())
```

    Team                   0
    Points                 0
    Goals                  0
    Goals_Allowed          0
    Shots_on_Target        0
    Possession             0
    Fouls                  0
    Saves_by_Goalkeeper    0
    Age                    0
    dtype: int64
    Team                   0
    Points                 0
    Goals                  0
    Goals_Allowed          0
    Shots_on_Target        0
    Possession             0
    Fouls                  0
    Saves_by_Goalkeeper    0
    Age                    0
    dtype: int64
    


```python
#check the pairplot
soccer.keys()
sns.pairplot(data=soccer,vars= ['Points', 'Goals','Goals_Allowed', 'Shots_on_Target', 'Possession', 'Fouls',
'Saves_by_Goalkeeper', 'Age'])
```




    <seaborn.axisgrid.PairGrid at 0x1d1fe61dd08>




![png](output_18_1.png)



```python
#set up X and y variables with data for 2014-15 through 2019-20.
X = soccer[['Goals','Goals_Allowed', 'Shots_on_Target', 'Possession', 'Fouls', 'Saves_by_Goalkeeper', 'Age']]
y = soccer['Points']
#set up X and y variables with data for current season (2020-21).
X1 = soccer_current[['Goals', 'Goals_Allowed', 'Shots_on_Target', 'Possession', 'Fouls', 'Saves_by_Goalkeeper', 'Age']]
y1 = soccer_current['Points']
```


```python
# Linear Regression cross validation
lm = LinearRegression()
scores_lm = cross_val_score(lm, X, y, cv=10,
                              scoring='neg_mean_squared_error')
mse_lm_cv = -1*scores_lm.mean()
print("CV", mse_lm_cv)
```

    CV 19.778848377974683
    


```python
# Lasso Regression cross validation
lo = Lasso(alpha=0.5)
scores_lasso = cross_val_score(lo, X, y, cv=10,
                               scoring='neg_mean_squared_error')
mse_lasso_cv = -1*scores_lasso.mean()
print("CV",mse_lasso_cv)
```

    CV 19.428978817032583
    


```python
# KNN model cross validation
knn = KNeighborsRegressor(n_neighbors=10)
scores_knn = cross_val_score(knn, X, y, cv=10, scoring='neg_mean_squared_error')
mse_knn_cv = -1*scores_knn.mean()
print("CV",mse_knn_cv)
```

    CV 76.56416666666665
    


```python
# Decision tree model cross validation
dtr = DecisionTreeRegressor(min_samples_leaf=3)
scores_dtr = cross_val_score(dtr, X, y, cv=10, scoring='neg_mean_squared_error')
mse_dtr_cv = -1*scores_dtr.mean()
print("CV",mse_dtr_cv)
```

    CV 39.07383333333333
    

It looks like Lasso is the best model with the lowest cross validation score, so I chose Lasso to fit a model.


```python
#fit a model and make a prediction
lo.fit(X,y)
prediction=lo.predict(X1)
```


```python
#rank the teams from top to bottom based on the predicted points
soccer_current['Predicted_Points']=np.round(prediction,decimals=2)
rank_prediction=soccer_current[['Team','Predicted_Points']]
rank_prediction=rank_prediction.sort_values(by='Predicted_Points',ascending=False).reset_index()
rank_prediction=rank_prediction[['Team','Predicted_Points']]
rank_prediction
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Team</th>
      <th>Predicted_Points</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Chelsea_20</td>
      <td>64.73</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Tottenham_20</td>
      <td>64.62</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Liverpool_20</td>
      <td>62.04</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Aston Villa_20</td>
      <td>60.62</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Leicester City_20</td>
      <td>60.38</td>
    </tr>
    <tr>
      <td>5</td>
      <td>Manchester City_20</td>
      <td>59.97</td>
    </tr>
    <tr>
      <td>6</td>
      <td>Southampton_20</td>
      <td>58.54</td>
    </tr>
    <tr>
      <td>7</td>
      <td>West Ham_20</td>
      <td>58.24</td>
    </tr>
    <tr>
      <td>8</td>
      <td>Everton_20</td>
      <td>57.37</td>
    </tr>
    <tr>
      <td>9</td>
      <td>Manchester Utd_20</td>
      <td>57.36</td>
    </tr>
    <tr>
      <td>10</td>
      <td>Crystal Palace_20</td>
      <td>56.53</td>
    </tr>
    <tr>
      <td>11</td>
      <td>Newcastle Utd_20</td>
      <td>54.36</td>
    </tr>
    <tr>
      <td>12</td>
      <td>Brighton_20</td>
      <td>54.05</td>
    </tr>
    <tr>
      <td>13</td>
      <td>Arsenal_20</td>
      <td>53.73</td>
    </tr>
    <tr>
      <td>14</td>
      <td>Leeds United_20</td>
      <td>53.65</td>
    </tr>
    <tr>
      <td>15</td>
      <td>Wolves_20</td>
      <td>53.20</td>
    </tr>
    <tr>
      <td>16</td>
      <td>Fulham_20</td>
      <td>49.66</td>
    </tr>
    <tr>
      <td>17</td>
      <td>Burnley_20</td>
      <td>47.87</td>
    </tr>
    <tr>
      <td>18</td>
      <td>Sheffield Utd_20</td>
      <td>47.54</td>
    </tr>
    <tr>
      <td>19</td>
      <td>West Brom_20</td>
      <td>46.40</td>
    </tr>
  </tbody>
</table>
</div>




```python
#find and plot important features
from matplotlib import pyplot
importance = lo.coef_
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
for i in range(0,len(X.columns)):
    if abs(importance[i]) > 0.3:
        print('Feature:', [i], X.columns[i], 'Score:', round(importance[i],2))

#Two most impactful features were discovered to be goals and goals allowed.
```


![png](output_27_0.png)


    Feature: [0] Goals Score: 0.66
    Feature: [1] Goals_Allowed Score: -0.62
    

#Answer to the prediction question

Based on the predicted points, the top 4 teams are Chelsea, Tottenham, Liverpool and Aston Villa, and two most features are goals and goals allowed.
