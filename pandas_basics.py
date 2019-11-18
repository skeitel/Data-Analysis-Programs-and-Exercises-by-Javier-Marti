#NOTES ON PANDAS LANGUAGE####################################
# Complete cheatsheet at https://elitedatascience.com/python-cheat-sheet
#############################################################

import pandas as pd
import numpy as np


#########################################################################
#BASIC OPENING I/O FILES. OPENING AND SAVING FILES ######################
#########################################################################
#opening table from url
import pandas as pd
tables = pd.read_html("https://coinmunity.co/")
print(tables[0])


#opening csv file
df = pd.read_csv('filename.csv') #full path to file 'D://directory/folder/file.csv'

#Opening CSV file that shows garbled data
fixed_df = pd.read_csv('../data/bikes.csv', sep=';', encoding='latin1', parse_dates=['Date'], dayfirst=True, index_col='Date')

#opening EXCEL file
df = pd.read_excel('city_employee_salaries.xlsx','sheet_name',header=[0,1])

#opening dictionary
df = pd.DataFrame(name_of_dictionary)

#LOADING DICTIONARY INTO DATAFRAME
df = pd.DataFrame(list(name_of_dict.items()), columns= ['col1','col2'])


#opening tuple
df = pd.DataFrame(name_of_tuple, columns= ['column1','column2','column3'])

#open from online file
url = 'www.locationoffile.com/file.csv'
df = pd.read_csv(url, sep = '\t') #you may also be able to use "read_table"

#loading data from SQL ::: CHECK OUT STONERIVERLEARNING COURSE Python Web Programming IN MY PURCHASED COURSES
con = sqlite3.connect('filename.sqlite')
df = pd.read_sql('SELECT * FROM weather_2012 LIMIT 3', con add index_col = 'id') #id is the column that will be used as index
#more than once column can be used as index with index = ['column1','column2']

#opening tsv file
df = pd.read_table('filename.tsv', sep= ',')

pd.read_csv(filename) # From a CSV file
pd.read_table(filename) # From a delimited text file (like TSV)
pd.read_json(json_string) # Reads from a JSON formatted string, URL or file.
pd.read_html(url) # Parses an html URL, string or file and extracts tables to a list of dataframes
pd.read_clipboard() # Takes the contents of your clipboard and passes it to read_table()
pd.DataFrame(dict) # From a dict, keys for columns names, values for data as lists


#convert html table imported as list to dataframe
import pandas as pd
import requests
from bs4 import BeautifulSoup
res = requests.get("https://www.coingecko.com/en")
soup = BeautifulSoup(res.content, 'lxml')
table = soup.find_all('table')[0]
df = pd.read_html(str(table))
df = df[0].dropna(axis=0, thresh=4)
df.describe()

#other options to add in the when opening the file()
skiprows = 1 #will skip that number of rows when opening the file
names = ['Column1', 'Column2'] #will open the file assigning those names to the columns
nrows = 3 #only reads 3 rows of the file (useful not to load complete large files)
startrow = 1 #start reading file from those rows
startcolumn = 2 #start reading file from those columns
parse_dates = [name_of_column] #will CONVERT date column to DATE TYPE, parsing dates as dates and NOT as strings
index_col = 'Column_name' #use that column as index
sep = '\t' #kind of separator used in the file. It can be tab or comma (',') or any other value like space (' ')

#SETTING UP DISPLAY OPTIONS
pd.set_option('display.max_rows', None) #Show all rows, or a value like '200' can also be used instead of 'None'
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_colwidth', 200)
pd.set_option('display.precision', 2) #To show only 2 decimal places
pd.set_option('display.float_format', {:,}.format) #Will add commas to mark more clearly thousands and millions
pd.reset_option(all) #resets all
#To get a description of currently engaged options:
pd.describe_option
pd.describe_option('rows')


#########################################################################
#WRITING TO FILES #######################################################
#########################################################################
df.to_csv(filename) # Writes to a CSV file
df.to_excel(filename) # Writes to an Excel file
df.to_sql(table_name, connection_object) # Writes to a SQL table
df.to_json(filename) # Writes to a file in JSON format
df.to_html(filename) # Saves as an HTML table
df.to_clipboard() # Writes to the clipboard

#Writing to excel as csv
df.to_csv('new.csv') #index = False will not write index. header = False will not write header


#Writing 2 dfs to one excel file as sheets:
with pd.ExcelWriter('Filename.xlsx') as writer:
    df_name_of_df1.to_excel(writer, sheet_name = 'Sheet1')
    df_name_of_df2.to_excel(writer, sheet_name = 'Sheet2')

#writing to sql
con.execute('DROP TABLE IF EXISTS weather_2012')
df.to_sql('weather_2012', con)


#########################################################################
#BASIC INFORMATION ABOUT THE DATAFRAME ##################################
#########################################################################
df.columns = ['a', 'b', 'c']  # Renames columns
pd.isnull()  # Checks for null Values, Returns Boolean Array
pd.notnull()  # Opposite of s.isnull()
pd.set_option('display.max_rows', 999) #will show only 999 rows of the dataframe
df.isnull().sum() #shows missing values in each column
df[np.isnan(df.column)]
df[df.isnull().any(axis=1)] #see rows containing null values

#Dropping and filling NAs
df.dropna()  # Drops all rows that contain null values
df.dropna(axis=1)  # Drops all columns that contain null values
df.dropna(axis=1, thresh=n)  # Drops all rows have have less than n non null values
df.fillna(x)  # Replaces all null values with x
df.fillna(df.mean())  # Replaces all null values with the mean (mean can be replaced with almost any function from the statistics section)
df.astype(float)  # Converts the datatype of the series to float
df.replace(1, 'one')  # Replaces all values equal to 1 with 'one'
df.replace([1, 3], ['one', 'three'])  # Replaces all 1 with 'one' and 3 with 'three'

#Renaming and setting index
df.rename(columns = lambda x: x + 1)  # Mass renaming of columns
df.rename(columns = {'old_name': 'new_ name'})  # Selective renaming
df.set_index('column_one')  # Changes the index
df.rename(index=lambda x: x + 1)  # Mass renaming of INDEX



#Removing duplicates
#DUPLICATE REMOVAL IS FALSE BY DEFAULT. USE "INPLACE = TRUE" FOR PERMANENT CHANGES
df.duplicated() #identify duplicates
df.duplicated(subset= ['age','zipcode']).sum
df.column.duplicated()
df.duplicated().value_counts()
df.loc(df.duplicated(), : ) #options: keep = 'first' or keep = 'last' or keep = False
df.drop_duplicates(keep = 'first').shape
df.drop_duplicates(subset='A', keep="last")

#delete column
del df['Real price']


#EXPLORING THE DATA ###############################
df.head() # or df.head(number)
df.tail(),# or df.tail(number)
df.describe()
df.column.describe()
df.describe().loc['25%', 'beer_servings'] #access one particular value from the describe dataframe
df.describe(include= 'all') #to see all columns
df.info()
df.sample(n = 10) #randomly select 10 rows
df.sample(frac = 0.2) #randomly select 20% of rows
df.column.size() #number of observations in that column
df.shape
df.columns
df.column.value_counts() #how many of that type
df.column.value_counts(normalize = True) #% of each value
df.column.unique #unique values present
df.column.nunique() #how many unique values present?
df.dtypes
df.loc['index_number'] #prints row at that index number
type(df['column_name']) #what type of data a column is
df.values #to see values
df.index #to see what's the index
df.value_counts(dropna=False) # Views unique values and counts
df.apply(pd.Series.value_counts) # Unique values and counts for all columns
df.nlargest(n, 'value') #Select and order top entries

#Basic exploratory operations on data
df.mean() # Returns the mean of all columns
df.corr() # Returns the correlation between columns in a DataFrame
df.count() # Returns the number of non-null values in each DataFrame column
df.max() # Returns the highest value in each column
df.min() # Returns the lowest value in each column
df.median() # Returns the median of each column
df.std() # Returns the standard deviation of each column
df.mode() #Most frequently appearing value
df.col.value_counts() #count number of each occurrence in column



########################################################################
#FILLING UP AND REPLACING NA VALUES #####################################
########################################################################

df[np.isnan(df['column name'])] #identifying NaN values
df.isnull().any() # check missing values of any kind

#dropping NaNs
df.dropna() #drops all without values
df.dropna(how = 'all') #drop it if ALL values are NA or 'any' if any NA is present
#options
tresh = 1 #drops the row if there's at least 1 cell that's empty


#replacing na_values
df = df.fillna(0, inplace = True) #replace all missing values with '0'

na_values = ['not available', 'n.a'] #replaces such occurrences with NaN. Also accepts a dictionary to replace selectively from each column
na_values = {'eps': ['not available', 'n.a'],
             'revenue': ['not available', 'n.a', -1],
             'people': ['not available', 'n.a']
             }


#simple consecutive replace
df.replace(['poor','average','good','exceptional'],['1,2,3,4']
           )

#simple replacement everywhere the value is found
df.replace({-99999: np.NaN, #if you find value -99999 anywhere replace it with NAN
             'No event': 'Sunny',   #replace 'no event' with "Sunny"
             })

#multiple replacement based on current column value
df.replace  ({'temperature': -99999, #if you find value -99999 in temperature cell
             'windspeed': -99999,   #if you find value -99999 in windspeed cell
             'event': '0'           #if you find value 0 in event cell
             }, np.NaN)            #then replace that value with NaN


#using a LOOP FUNCTION to replace values whilst opening the doc
def convert_people_cell(cell):
    if cell == 'n.a':
        return 'whatever_value_replacing_it'
    return cell
df = pd.read_csv('filename.csv', converters = convert_people_cell() )

#ESTIMATING VALUES from surrounding cells
import numpy as np
new_df = df.fillna (method = ffill) #forward fill
new_df = df.fillna (method = bfill) #backward fill
new_df = df.fillna (method = time) #fill based on date column
#options
limit = 1 #will limit any fill to only one replacement instance, be it forward or backwards


#FILLING MISSING DATES
df.interpolate(method = 'time') #approximation created by the computer, closer to the values in the surrounding cells
df = pd.date_range('01-01-2017','01-11-2017')
idx = pd.DatetimeIndex(df)
df = df.reindex(idx)

#creating time series
df.index.day
df.index.weekday
df.loc[:,'weekday'] = df.index.weekday
wk = df.groupby('weekday').aggregate(sum)
wk.index = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
wk.plot(kind = 'bar')

#Create DAY of the week column
df['day_of_week'] = df['Date'].dt.day_name()
df.head()

#Converting to Pandas' series
list_to_convert = [1,4,3,6,3,7,8,9]
var_name = pd.Series(list_to_convert)


#CLEANING DATA WITH REGEX - replacing text in the df
df.replace ({'temperature': '[A-Za-z]', #if you find any text in the cell
             'windspeed': '[A-Za-z]',   #if you find any text in the cell
             },'', regex = True)        #then replace that value with what is between the quotes (in this case, nothing)

#
new_df.max_salary = new_df.max_salary.apply(lambda x: x.replace(',', ''))


#converting STRINGS date columns to DATE realtime columns
df['date'] = pd.to_datetime(df['date'])

#converting UNIX TIME to regular time
df['date'] = pd.to_datetime(df['date'], unit = 's')

#Open file and convert date column to datetime in one go
df = pd.read_csv('file.csv', parse_dates = ['column that contains dates goes here'])

#Examples of datetime methods and functions(click tab after "df.column.dt." to see all methods
df.column.dt.dayofyear.head()
df.column.dt.dayofweek
ts = pd.to_datetime('1/1/1999')
df.loc[df.column >= ts, :]
df.column.max() - df.column.min() #once in datetime type, we can do operations with date information
df['2017-01'].column.mean() #will get mean of all values for month of January
df.column.resample('M').mean() #use df.column.resample + tab to see options available. 'W' is for weekly resample

#generating dates for data
rng = pd.date_range(start = '1-1-2017',end = '1-11-2017', freq = 'B') #B means only business working days will be included
df.set_index(rng, inplace = True)

#generating imaginary data for weekends when it's not present
df.asfreq('D', method = 'pad') #options are D(daily), M(monthly), H(hourly)

#generating data range based on periods
rng = pd.date_range(start = '1-1-2017', periods = 72, freq = 'M') #options are D(daily), M(monthly), H(hourly)

#generating custom calendars (including bank hols)
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
usb = CustomBusinessDay(calendar = USFederalHolidayCalendar())
rng = pd.date_range(start='1-1-2017', end='1-11-2017', freq= usb)
df.set_index(rng, inplace= True)
##NOTE: to create custom calendar open the custom holidays class from Pandas, and replace data with local holidays


#CREATING PERIODS to do arithmetic operations with them
y = pd.Period('2016')
y.start_time
y.end_time
m = pd.Period('2016-12', freq = 'M')
d = pd.Period('2016-12', freq = 'D')
d + 1 # will show us the following day
h = pd.Period('2016-02-28 23:59:99999999', freq = 'H')
q.pd.Period('2017Q1')
idx = pd.period_range('2011','2017',freq = 'Q-JAN') # or Q-FEB or Q-MARCH...
idx = pd.period_range('2011', periods='10', freq= 'Q') #will generate 10 periods


#creating random numbers
import numpy as np
fs = pd.series(np.random.randint(1,10,len(rng)), index = rng)



#exercise: cleaning up data with NA. Replace 'no clue' and 'N/A' with 0
na_values = ['no clue','N/A','0']
requests = pd.read_csv('filename.csv', na_values = na_values, dtype = {'Incident_zip': str})
#this tells the program that 'incident_zip' column will now be forced to be read as a string
#Now, clean rows with dashes:
rows_with_dashes = requests['incident_zip'].str.contains('-').fillna(False)
#Now research and find long zipcodes
long_zipcodes = requests['incident_zip'].str.len()>5
requests['incident_zip'][long_zipcodes].unique()
#next reduce any zipcode with more than 5 numbers:
requests['incident_zip']=requests['incident_zip'].str.slice(0,5)
#now delete all '00000' zip codes:
requests[requests['incident_zip']=='00000']
zero_zips = requests['incident_zip'] == '00000'
requests.loc[zero_zips, 'incident_zip'] = np.nan
#now let's see what zip codes are far or near:
zips = requests['incident_zip']
is_close = zips.str.startswith('0')|zips.str.startswith('1') #THE VERTICAL LINE MEANS 'OR'
is_far = ~(is_close) & zips.notnull() #NOTICE 'NOTNULL' MEANING 'EXISTS AND DOES NOT HAVE NULL VALUE'

#########################################################################
#SELECTING COLUMNS AND BASIC COLUMN OPERATIONS ##########################
#########################################################################

df.drop(['quality'], axis = 1) #we use all columns except the one we are dropping, 'quality' column

df[df.min_salary.str.contains('Salary negotiable')]#rows that contains certain words
df[df.min_salary.str.contains('Salary negotiable') == False]#rows that DON'T contains certain words
df.query('min_salary in ["Salary negotiable", "Competitive salary"]') #(WARNING comillas y dobles comillas imprescindibles) rows that contains certain words
grades_df.query("Test_1 < Test_2 and Test_2 < Test_3") #multiple conditions
grades_df.query("Test_1 < Test_2 < Test_3")
grades_df.query("Test_3 in [98, 99, 100]") #checks if students got 98, 99 or 100 in the last test

df[col]
df[2:5]
df['column_name']
df.values[3][4] #show contents of row 3 column 4
df[['event','day']] # show those two columns
df[['event','day']][2:6] # show only those columns and rows
df.set_index('day') # ('day', inplace = True) to replace data
df.reset_index(inplace = True) #caution: "inplace" replaces data
df[df.temperature==df['temperature'].max()] #shows full row with max temp
df[df['price']>229]
df['date'][df['Humidity'] > 50]
df['date'][df['Events']=='Rain']
movies.loc[movies.duration >= 200, 'Genre']
df.loc[0:1] #loc is for selecting things by labels, including on both sides
df.loc[[0,1,2],:]
df.loc[:,'city' :'state'] #show all rows but only columns from 'city' to 'state'
df.loc[df.city == 'Oakland', 'State']
army.loc[['Florida','California'],['battles', 'armored', 'readiness']]
s.loc[0]  # Selection by index (selects element at index 0)
#SEE MORE ABOUT LOC https://stackoverflow.com/questions/17071871/select-rows-from-a-dataframe-based-on-values-in-a-column-in-pandas
df.iloc[3:7, 3:6] #select rows 3 to 7 and columns 3 to 6. Iloc is to filter by integer portion. Inclusive start, exclusive end
df.iloc[:, 3:6] #select all rows and columns 3 to 6
df.iloc[0]  # Selection by position (selects first element)
df.iloc[0, :]  # First row
df.iloc[0, 0]  # First element of first column

army[(army['deaths'] > 500)]
df['column'] = df['column'].pct_change()
df.column.value_counts().sort_index() # will arrange text index in alphabetical order



#EXTRACTING VALUES FROM COLUMNS
df.mean()
df.min()
df.max()
df.std()
df.sort #arrange alphabetically
df.sort_values(ascending=False)
movies.sort_values('Title')
movies.sort_values(['rating','duration'])
movies.shape


###########################################################################################################
###########################################################################################################
#AND and OR and MULTIPLE CONDITIONS

#Fast method
filter_list = ['US', 'USA']
df[df.Region.isin(filter_list)]

#Other methods
df[(df['ClosePrice'] > 10) & (df['Category'] == 'Photography')]] #means "AND"
df[(df['Annual_Rt']> 50000) & (df[df.Unit == 'Fire'])]
df[(df['price'] > 335) & (df['bid'] > 10)]
df[(df['ClosePrice'] > 10) | (df['Category'] == 'Automotive')] #means "OR"
movies[movies.genre.isin(['crime','drama','action'])]
drinks[drinks.continent.isin(['Asia','Africa'])] #Show only drinks for Asia and Africa
drinks[~drinks.continent.isin(['Asia','Africa'])] #the '~' means 'EXCLUDE' or 'NOT'...so exclude Asia and Africa from results
drinks.groupby('continent').beer_servings.agg(['count','mean','max','min'])
df.groupby('country')['unemployment'].mean().sort_values().plot(kind='bar', figsize=(10,2))

#Create new column based on multi-variables testing
df['Test price increase']= ( (df['Price % change'] > 0.04) & (df['Volume % change'] > 5)).map({True: 'YES', False: 'NO'})





###########################################################################################################
###########################################################################################################

#FINDING CORRELATION
df.corr()
df.corr(method = 'pearson')
df.corr(method = 'kendall')
df.corr(method = 'spearman')

#FINDING PCT CHANGE
df.pct_change()
df.pct_change(3) is the same as s/s.shift(3)-1


#FINDING AND DELETING OUTLIERS
df=pd.DataFrame({'Data':np.random.normal(size=200)})  #example dataset of normally distributed data.
df[np.abs(df.Data-df.Data.mean())<=(3*df.Data.std())] #keep only the ones that are within +3 to -3 standard deviations in the column 'Data'.
df[~(np.abs(df.Data-df.Data.mean())>(3*df.Data.std()))] #or if you prefer the other way around

#changing single or multiple column types to numeric
pd.to_numeric(column_name, errors='ignore') #error can also be 'coerce' to force it to change type
df[['length_of_encounter_seconds','latitude']] = df[['length_of_encounter_seconds','latitude']].apply(pd.to_numeric, errors='coerce')

#changing several cols to numeric at the same time
col_list = ['Land Value', 'Improvement Value', 'Total Value', 'Sale Price']
for item in col_list:
    props[item] = pd.to_numeric(props[item].str.replace('$', ''), errors='coerce')


#APPLYING APPLY, MAP AND APPLYMAP TO PERFORM OPERATIONS ON THE DATAFRAME
#'Apply' applies a function to EACH element of the df
df['name_length'] = df.name.apply(len)
df['train_fare_rounded'] = df.train_fare.apply(np.ceil) # Rounds fares
df.loc[:,'beer_servings':'wine_servings'].apply(max, axis = 1) # Find biggest value in column or row (depending axis 1 or 0)
df.loc[:,'beer_servings':'wine_servings'].apply(np.argmax, axis = 1) #Returns what column has the largest value

#Another simple apply function
def majority_of_age(x):
    if x >= 18:
        return 'Can drink. Ok!'
    if x <= 18:
        return 'Too young to drink'
df['legal_drinker'] = df.age.apply(majority_of_age) #creates a new column and applies the function
df.head()

#Insert value in cell depending on values from other cells
def impute_age(cols):
    age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age



#'Map' is to assign values to cells based on predefined values
df['young_male'] = ((df.sex == 'male')& (df.age <30)).map({True: 'yes', False: 'no'})
(df.continent == 'Asia').astype(int) # This will create a column type series that has '1' for Asia and '0' for any other value
#'Applymap' applies the operation to each and ALL values in the df



#Split a string separated by commas
df.column.str.split(',')


#Examples extracting specific data from columns and time range
#once our index is date in a stock table that gives us date, open, high, low, close and volume:
df['2017-1-1','2017-11-1'].close.min()
df['2017-1-1','2017-11-1'].volume.max()
df['2017-1-1','2017-11-1'].low.std()
#to consider monthly data only when date is index:
df.close.resample('M').mean() #M means monthly



#examples and exercises #######################

#Resampling daily and getting the dates with the largest price increases
(df.resample('D')['Price % change'].mean()).nlargest(n=10, keep='first')


#discover male ratio per occupation and sort it
def gender_to_numeric(x):
    if x == 'M'
        return 1
    if x == 'F':
        return 0
users['gender_n' = users['gender'].apply(gender_to_numeric)] #this is how to apply a function to the data
a = users.groupby('occupation').gender_n.sum() / users.occupation.value_counts() * 100
a = sort.values (ascending = False)


#transform date column to pandas' datetime64
df.Date = pd.to_datetime(apple.Date)

#are there any duplicate values?
df.index.is_unique

#arrange in ascending order
df.sort_index(ascending = True)

#Product name in cell 19?
df.values[18][7]

#most ordered item?
df.column.value_counts().head(1)

#how many items in total
df.column.sum()

#how many orders were made?
df.order_id.value_counts().count()

#average amount per order?
df.groupby(by=['order_id']).sum()

    #quantity ordered of the most expensive item?
df.sort_values(by='item_price', ascending = False).head(1)

#how many times people ordered more than one can of soda?
df[(df.item_name == 'canned_soda')&(df.quantity > 1)]

#sort values by yellow card and red card
df = df.sort_values(['Yellow cards', 'Red cards'], ascending = False)

#what's the value that appears the least?
df.column.value_counts().tail(1)


#########################################################################
#GROUPING OBJECTS #######################################################
#########################################################################
df[df[col] > 0.5] # Rows where the col column is greater than 0.5
df[(df[col] > 0.5) & (df[col] < 0.7)] # Rows where 0.5 < col < 0.7
df.sort_values(col1) # Sorts values by col1 in ascending order
df.sort_values(col2,ascending=False) # Sorts values by col2 in descending order
df.sort_values([col1,col2], ascending=[True,False]) # Sorts values by col1 in ascending order then col2 in descending order
df.groupby(col) # Returns a groupby object for values from one column
df.groupby([col1,col2]) # Returns a groupby object values from multiple columns
df.groupby(col1)[col2].mean() # Returns the mean of the values in col2, grouped by the values in col1 (mean can be replaced with almost any function from the statistics section)
df.pivot_table(index=col1, values= [col2,col3], aggfunc=mean) # Creates a pivot table that groups by col1 and calculates the mean of col2 and col3
df.pivot_table(index = ['Reviewer_Nationality'], values = ['Reviewer_Score', 'Total_Number_of_Reviews'], aggfunc = min)
df.groupby(col1).agg(np.mean) # Finds the average across all columns for every unique column 1 group
df.apply(np.mean) # Applies a function across each column
df.apply(np.max, axis=1) # Applies a function across each row


#simple grouping
g = df.groupby('city')
for city,city_df in g:
    print(city)
    print(city_df)


#Finding the BEST BALANCE between lowest Compete rank and highest Quantcast rank
df_balance = df.groupby('Company').agg({'Compete rank': 'min', 'Quantcast rank':'max'})[['Compete rank', 'Quantcast rank']].reset_index()
df_balance

#creating group and plotting from text
weather_description = weather_2012['Weather_data']
is_snowing = weather_description.str.contains('Snow')
is_snowing.plot()

#examples
df[df['regiment']=='Nighthawks'].groupby('regiment').mean() #mean of group
df.groupby('company').column.mean()
regiment.groupby(['company','regiment']).size()

#iterate and show name and data for each regiment
for name,group in regiment.groupby('regiment'):
    print(name)
    print(group)

#plotting group
g = df.groupby('city')
g.get_group('mumbai')
%matplotlib inline
plt.rcParams['figure.figsize'] = (15,5)
g.plot()

#grouping several elements
df.pivot_table(index = pd.Grouper(freq = 'M', key = 'date', columns = 'city'))

#MELT method to reshape dfs
pd.melt(df, id_vars = ['day'])
df1[df1['variable']== 'Chicago']
import matplotlib as plt
    %matplotlib inline
pd.set_option('display.mpl_style', 'default')
plt.reParams['figure.figsize'] = (15,5)
#to graph a particular column: df['column'].plot()

#########################################################################
#MAPPING FUNCTIONS TO EXECUTE FUNCTIONS IN CELLS ########################
#########################################################################
import random
def function(data):
    x = random.randrange(0,5)
    return data * x

df['new_column'] = map(function, df['some_column'])

#########################################################################
#CONCATENATING AND MERGING DATAFRAMES ###################################
#########################################################################
df1.append(df2) # Adds the rows in df1 to the end of df2 (columns should be identical)
pd.concat([df1, df2],axis=1) # Adds the columns in df1 to the end of df2 (rows should be identical)
df1.join(df2,on=col1,how='inner') # SQL-style joins the columns in df1 with the columns on df2 where the rows for col have identical values. how can be one of 'left', 'right', 'outer', 'inner'

df = pd.concat([india_weather, us_weather],     axis = 1, index = [1,0], keys = 'India', 'US'])
#axis will concatenate data in a new column and index will align the dataframes

df.loc['US'] #access US subset of data

#MERGING dataframes
df3 = pd.merge(df1, df2, on = 'city', how = 'inner', indicator = True)
#'how' can be 'inner', 'outter', 'left' or 'right'. Indicators shows us where the data is missing

#changing data orientation vertical to horizontal
df.transpose() or df.T
df.stack()
df.unstack(level = 1)


#changing data orientation (pivoting)
df.pivot(index = 'date', columns = 'city', values = 'humidity')
df.pivot_table(index = 'city', columns = 'date', aggfunc = 'sum', margin = True)
#'sum' is a Numpy function. Margin = True will add a column for totals


#crosstab/Contingency table to view frequency distribution
import numpy as np
pd.crosstab(df.Sex, df.handedness, margins = True, normalize = 'index', values = df.Age, aggfunc = np.average)

pd.crosstab(movies.genre, movies.content_rating)



######################################################
#LINEAR REGRESSION WITH SKLEARN
######################################################

from sklearn.linear_model import LinearRegression as LinReg
x = df.index.values.reshape(-1, 1)
y = df.price.values

reg = LinReg()
reg.fit(x,y)
y_preds = reg.predict(x)
print(reg.score(x,y))

plt.figure(figsize = (15,5))
plt.title('Linear Regression')
plt.scatter(x=x, y=y_preds)
plt.scatter(x=x, y=y, c='r')


######################################################
######################################################
#MAKING GRAPHS with Matplotlib
######################################################
######################################################
#Simple plot of two columns values
import matplotlib.pyplot as plt
%matplotlib inline
x = df.Open[1:50]
y = df.Close[1:50]
z = df.Volume[1:50]/100000
plt.plot(x, label = 'Open')
plt.plot(y, label = 'Close')
plt.plot(z, label = 'Volume')
plt.xlabel('Date')
plt.ylabel('Open, Close and Volume')
plt.title('Title\nSecond line of title')
plt.show()


#Simple multi-bar chart
import matplotlib.pyplot as plt
%matplotlib inline
w = [2,4,6,8,10]
x = [3,6,7,3,7]
y = [1,3,5,7,9]
z = [4,5,7,2,2]
plt.bar(w,x, label = 'w, x', color = 'Red')
plt.bar(y,z, label = 'y,z', color = 'c')

plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Title\nSecond line of title')
plt.legend()
plt.show()


#Simple histogram with bins
import matplotlib.pyplot as plt
%matplotlib inline
ages = [4,7,22,34,34,23,54,23,12,34,56,76,54,34,23,45,67]
bins = [10,20,30,40,50,60,70,80]
plt.hist(ages, bins, histtype = 'bar', rwidth = 0.8)

plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Title\nSecond line of title')
plt.legend()
plt.show()


#Simple scatterplot
import matplotlib.pyplot as plt
%matplotlib inline
x = [3,4,3,5,3,7,8,4,7,3]
y = [3,6,7,3,7,4,7,8,5,3]
plt.scatter(x,y, label='Scatterplot', color = 'b', marker = '*', s = 80)

plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Title\nSecond line of title')
plt.legend()
plt.show()

#Simple stackplot
import matplotlib.pyplot as plt
%matplotlib inline
days = [1,2,3,4,5]
sleeping = [9,3,4,5,2]
eating = [3,5,3,2,6]
working = [3,3,4,6,2]

plt.plot([],[], color = 'm', label = 'sleeping', linewidth = 5)
plt.plot([],[], color = 'c', label = 'eating', linewidth = 5)
plt.plot([],[], color = 'r', label = 'working', linewidth = 5)

plt.stackplot(days, sleeping, eating, working, colors = ['m','c','r'])

plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Title\nSecond line of title')
plt.legend()
plt.show()


#Simple pie chart
import matplotlib.pyplot as plt
%matplotlib inline

sleeping = [9,3,4,5,2]
eating = [3,5,3,2,6]
working = [3,3,4,6,2]

slices = [7, 9, 8]
activities = ['sleeping', 'eating', 'working']
cols = ['c', 'm', 'r']

plt.pie(slices,
    labels = activities,
    colors = cols,
    startangle = 90,
    shadow = True,
    explode = (0,0.1,0),
    autopct = '%1.1f%%')

plt.title('Title\nSecond line of title')
plt.show()


#Quick view of correlations among all the data
%matplotlib inline
import matplotlib.pyplot as plt
pd.plotting.scatter_matrix(column, figsize = (15,15), diagonal = 'hist')
plt.show()


#Selecting what column to plot
df.plot(x= df.column, style = '.-')


#Simple plot
import matplotlib.pyplot as plt
%matplotlib inline
appl_open = apple['Adj Close'].plot(title = "Apple Stock")
fig = appl_open.get_figure()
fig.set_size_inches(13.5, 9)

df.close.resample('M').mean().plot()


#Simple bigger graph with labels
plt.figure(figsize = (15,10))
plt.plot(df['Price % change'])
plt.title('Price $ change over time')
plt.xlabel('date')
plt.ylabel('% price variation')


#Basic scatterplot bidtime/bid ratio
plt.figure(figsize = (15,5))
plt.scatter(x = df['bidtime'].index, y = df['bid'])
plt.title = ('Bidtime/Bid ratio')
plt.xlabel = ('Bidtime')
plt.ylabel = ('Bid')
plt.show()

#Simple violin distribution graph
var_name = "sales"
col_order = np.sort(data[var_name].unique()).tolist()
plt.figure(figsize=(12,6))
sns.violinplot(x=var_name, y='average_montly_hours', data=data, order=col_order)
plt.xlabel(var_name, fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title("Distribution of average_monthly_hours variable with "+ var_name, fontsize=15)
plt.show()


#Simple pie chart
by_race = data.groupby('race').size().sort_values(ascending=False).rename('counts').head(9)
labels = ['White non-Hispanic',
    'Asian',
    'Hispanic (of any race)',
    'Black',
    'Mixed',
    'Indian',
    'Middle Eastern',
    'Multi',
    'European']
bomb = (0,0,0,0,0.1,0.1,0.1,0.1,0.1)
fig, ax = plt.subplots(figsize=(8,8))
ax.pie(by_race, explode=bomb, labels=labels, shadow=True, startangle=90, autopct='%1.1f%%')
ax.axis('equal')
plt.show()


#Example of overlapped graphs.#####################################
# without ax = ax we shall get several graphs, not one ############
#example from http://jonathansoma.com/lede/algorithms-2017/classes/fuzziness-matplotlib/how-pandas-uses-matplotlib-plus-figures-axes-and-subplots/
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
df.groupby('country').plot(x='year', y='unemployment', ax=ax, legend=False)


#Two graphs in one plot #############################
fig = plt.figure()
ax1 = fig.add_subplot(211) # Divide the figure into a 2x1 grid, and give me the first section
ax2 = fig.add_subplot(212) # Divide the figure into a 2x1 grid, and give me the second section
df.groupby('country').plot(x='year', y='unemployment', ax=ax1, legend=False)
df.groupby('country')['unemployment'].mean().sort_values().plot(kind='barh', ax=ax2)


######################################################
######################################################
#MAKING GRAPHS with SEABORN
######################################################
######################################################
import pandas as pd
import numpy as np
import matplotlib.plyplot as plt
import seaborn as sns
%matplotlib inline
sns.set_style('white')

#setting up the matplotlib figure to show 3 graphs like "simple distribution plot" in one line
f, axes = plt.subplots(ncols=3, figsize=(15, 6))

#simple scatter plot
plt.scatter(df['TV'],df['Radio'], color='red')

# simple distribution plot
sns.distplot(df.evaluation, kde=False, color="r", ax=axes[1]).set_title('Employee Evaluation Distribution')

#Scaterred plot WITH LINEAR REGRESSION for multiple columns
#We can see that the TV platform is the one that historically has been more in line with spending. But has it been profitable?
import seaborn as sns
sns.pairplot(df, x_vars = ['TV','Radio','Newspaper'], y_vars = 'Sales', size = 7, aspect = 0.7, kind = 'reg')

#Histogram
url = ('https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/07_Visualization/Tips/tips.csv')
tips = pd.read_csv(url)
sns.set_style('white')
ttbill = sns.distplot(tips.total_bill)
ttbill.set(xlabel = 'Value', ylabel = 'Frequency', title = 'Total Bill')
sns.despine #Plot the total_bill column histogram
sns.jointplot(x ="total_bill", y ="tip", data = tips)#Create a scatter plot presenting the relationship between total_bill and tip
sns.pairplot(tips) #Create one image with the relationship of total_bill, tip and size.
sns.stripplot(x = "day", y = "total_bill", data = tips, jitter = True); #Present the relationship between days and total_bill value
sns.stripplot(x = "tip", y = "day", hue = "sex", data = tips, jitter = True); #Create a scatter plot with the day as the y-axis and tip as the x-axis, differ the dots by sex
sns.boxplot(x = "day", y = "total_bill", hue = "time", data = tips); #box plot presenting the total_bill per day differetiation the time (Dinner or Lunch)¶


#Correlation Matrix
corr = df.corr()
corr = (corr)
sns.heatmap(corr,
           xticklabels = corr.columns.values,
           yticklabels = corr.columns.values)
sns.plt.title('Heatmap of correlation Matrix')


#Now create two histograms of the tip value based for Dinner and Lunch. They must be side by side.
# better seaborn style
sns.set(style = "ticks")
# creates FacetGrid
g = sns.FacetGrid(tips, col = "time")
g.map(plt.hist, "tip")


#Now create two scatterplots graphs, one for Male and another for Female, presenting the total_bill value and tip relationship, differing by smoker or no smoker
g = sns.FacetGrid(tips, col = "sex", hue = "smoker")
g.map(plt.scatter, "total_bill", "tip", alpha =.7)
g.add_legend()


#example 1
f,axarr = fig, ax = plt.subplots(ncols=1, nrows=1,figsize=(12,10))
data = votes.groupby('companyAlias').mean()
sns.barplot(x = data.index, y = data['vote'])

#example 2
week_happ = votes.groupby('wday').mean()['vote']
sns.barplot(x = week_happ.index, y = week_happ.values)

#example 3
sns.regplot(data['vote'].values, data['churn_perc'].values)

#example 4 - Correlation map
f,ax = plt.subplots(1,1, figsize = (15,10))
red_emp = employee.drop(['companyAlias','lastParticipationDate','wday'],axis=1)
sns.heatmap(red_emp.corr())
plt.title('Features Correlation Heatmap',fontsize=24)
plt.show()

##############################################################
#CONDITIONAL PANDAS STYLES AND COLOURS FOR CELLS #############
##############################################################

import pandas as pd
import numpy as np
#generating random dataframe
np.random.seed(24)
df = pd.DataFrame({'A': np.linspace(1, 10, 10)})
df = pd.concat([df, pd.DataFrame(np.random.randn(10, 4), columns=list('BCDE'))],
               axis=1)
df.iloc[0, 2] = np.nan


#creating function to colour negative values in red
#WARNING reset index may be needed to apply, as in df.reset_index(drop = True).style.applymap(_color_red_or_green)
def color_negative_red(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    color = 'red' if val < 0 else 'black'
    return 'color: %s' % color



#now we highlight in yellow max value in a column
#WARNING reset index may be needed to apply, as in df.reset_index(drop = True).style.applymap(_color_red_or_green)
def highlight_max(s):
    '''
    highlight the maximum in a Series yellow.
    '''
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]

df.style.apply(highlight_max)


#now we highlight in yellow max value in a column
#WARNING reset index may be needed to apply, as in df.reset_index(drop = True).style.applymap(_color_red_or_green)
def _color_red_or_green(val):
    color = 'red' if val < 0 else 'green'
    return 'color: %s' % color


#now we higlight in blue any value with a correlation superior than X
hr.corr().style.applymap(color_negative_blue)

def color_corr_blue(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    color = 'blue' if val > 0.5 else 'black'
    return 'color: %s' % color

df.corr().style.applymap(color_corr_blue)

##############################################################
#OPTIMIZING MEMORY USE  ######################################
##############################################################
'''Objects take much more space to be stored than integers
Reducing memory use: use "category" when you have only few items that are used repeatedly

'''
df.memory_usage(deep = True) #querying memory usage
df['column'] = df.continent.astype('category') #To create a category converting objects to integers
#Assigning categories to use operations on them:
df['quality'] = df.quality.astype('category', categories['good','very good','excellent'], ordered = True)
#Once this assignment is done you can do things like:
df.sort_values('quality')
df.loc[df.quality > good, :]



OTHER NOTES ##################
##############################
Jupyter ctrl enter to run command
Jupyter shift tab to get help where cursor is
Jupyter shift enter to run command starting a new line at the same time
TO CONVERT JUPYTER NOTEBOOK TO CODE, USE "NBCONVERT"

''' What's a significant correlation value?
As a rule of thumb, for absolute value of r:
0.00-0.19: very weak
0.20-0.39: weak
0.40-0.59: moderate 
0.60-0.79: strong
0.80-1.00: very strong.
You should keep in mind that these classes are arbitrary and they may differ for different context.
'''

##############################################################
#sample program ##############
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

web_stats = {'Day': [1,2,3,4,5,6],
             'Visitors': [43, 53, 34, 45, 64, 34],
             'Bounce_Rate': [65, 72, 62, 64, 54, 66]}

df = pd.DataFrame(web_stats)