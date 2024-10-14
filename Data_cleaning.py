import numpy as np
import pandas as pd

uploaded_file = 'C:/Users/Admin/Python_Projects/Startup_Fail_Success/Startup_dataset.csv'

def load_csv():
    csv = pd.read_csv(uploaded_file,encoding_errors='ignore')
    return csv
df = load_csv()

df.columns = df.columns.str.replace(' ', '')
def str_to_float(row):
    if '-' in row or ',' in row:
        row = row.replace(',','')
        row = np.nan
    else:
        row = float(row)
    return row

df['funding_total_usd'] = df['funding_total_usd'].apply(str_to_float)
funding_rounds_group = df.groupby('funding_rounds')['funding_total_usd'].aggregate(['count',np.nanmean,np.nanmedian,np.nanstd])

funding_filled = []

for h,i in enumerate(df['funding_total_usd']):
    if np.isnan(i):
        i = int(funding_rounds_group.loc[df['funding_rounds'].iloc[h],'nanmean'])
        funding_filled.append(i)
    else:
        funding_filled.append(int(i))
        
df['funding_filled'] = funding_filled

# Creating a new column country to fill missing values in country_code
country = []

for i in df.country_code:
    if not isinstance(i,str):
        country.append(np.random.choice(df.country_code[df.country_code.notnull()]))
    else:
        country.append(i)

df['country'] = country

years = []
for fou, first in zip(df.founded_at, df.first_funding_at):
    if isinstance(fou,str):
        years.append(int(fou.split('-')[0]))
    elif not isinstance(fou,str) and isinstance(first,str):
        years.append(int(first.split('-')[0]))
    else:
        date = int(np.random.choice(df.founded_at[df.founded_at.notnull()]).split('-')[0])
        years.append(date)

df['year'] = years

main_category = []
for i in df['category_list']:
    if not isinstance(i,str):
        main_category.append('Other')
    else:
        main_category.append(i.split('|')[0])

df['main_category'] = main_category

status = []
# Creating a new column status_class to classify the status of the startup
for i in df['status']:
    if i in ['acquired','ipo']:
        status.append('success')
    elif i == 'closed':
        status.append('fail')
    else:
        status.append('operating')
        
df['status_class'] = status

# Creating dummies for the status_class column
status_class = pd.get_dummies(df['status_class'])
df = pd.concat([df,status_class],axis=1)

# Drop the columns that we are not going to use in the analysis
df_cleaned = df[['name','country','year','city','main_category','funding_rounds','funding_filled',
            'first_funding_at','last_funding_at','status','status_class','fail','operating','success']]

# df_cleaned.to_csv('C:/Users/Admin/Python_Projects/Startup_Fail_Success/df_cleaned.csv', index=False)

df_cleaned.head()










































































