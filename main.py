import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import squarify
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

from Data_cleaning import *

st.title("Startup_Fail/Success :chart_with_upwards_trend:")
st.sidebar.title("Startup_Fail_Success :chart_with_upwards_trend:")
st.markdown("The objective of the project is to predict whether a startup that is currently operating will become a success or a failure. " 
            "The success of a company is defined as the event that provides the founders of the company with a large sum of money through the M&A (Merger and Acquisition) process or an IPO (Initial Public Offering). " 
            "A company would be considered a failure if it had to close. ")

st.image('C:/Users/Admin/Python_Projects/Startup_Fail_Success/Final_GUI/Startup_Fail_Success.png')

raw_data = 'C:/Users/Admin/Python_Projects/Startup_Fail_Success/Final_GUI/Startup_dataset.csv'

clean_data = 'C:/Users/Admin/Python_Projects/Startup_Fail_Success/Final_GUI/df_cleaned.csv'

@st.cache_data
def load_csv(uploaded_file):
    csv = pd.read_csv(uploaded_file,encoding_errors='ignore')
    return csv

df_raw = load_csv(raw_data)
df = load_csv(clean_data)

st.sidebar.subheader('**Data Overview**')

st.sidebar.markdown(f"There are {df_raw.shape[0]} rows and {df_raw.shape[1]} columns in our dataset.")
cols = df_raw.columns.tolist()
st.sidebar.write(cols)

st.sidebar.markdown("Click below button to see the sample of the data.")
if st.sidebar.button("Show Raw Data"):
    st.subheader('**Raw Data**')
    st.write(df_raw.sample(5))

# Pandas Profiling Report
st.sidebar.markdown("**Basic EDA using Pandas Profiling**")
st.sidebar.markdown("Click the hide button to see the report.")
if not st.sidebar.checkbox("Hide", True):
    pr = ProfileReport(df_raw, explorative=True)
    st.write('---')
    st.header('**Pandas Profiling Report**')
    st_profile_report(pr)


#######################################################  Data Wrangling  ####################################################### 

st.sidebar.subheader("**Data Wrangling**")
with st.expander("Description of Data Wrangling done on dataset:"):
    st.subheader("**Data Wrangling**")
    st.markdown("Data wrangling is the process of cleaning and unifying messy and complex data sets for easy access and analysis.")

    st.markdown("At first we create a function to convert the columns from string to float."
                "After appyling this function to funding_total_usd we get : ")

    st.markdown("Let's see the statistics of the column funding_total_usd grouped by the number of funding rounds.")
    st.image('C:/Users/Admin/Python_Projects/Startup_Fail_Success/funding_rounds_group.png')

    st.markdown("Let's fill missing values in the column funding_total_usd with the mean calculated above. "  
                "Depending on the number of funding rounds would be the mean assigned.")

    st.markdown("Let's see the number of missing values in the column funding_filled.")
    st.write(df_cleaned['funding_filled'].isnull().sum())

    st.markdown("Filling country code & founded_at column with Random Sample Imputation. "  
                "We create new column year where for missing values in the foundation date we will replace those values by the date of the first funding")
    st.markdown("The more startups are created in that country, "
                "the more likely it is that the missing value corresponds to that country.")

    st.dataframe(df_cleaned[['country','year','first_funding_at']].sample(5))

    st.markdown("""We simplify the category_list column by creating a new column named main category, 
                where we're going to assume that the first description in the category_list refers to the main category.   
                Also we apply One Hot Encoding to the status_class where   
                1. success(1) = 'acquired' or 'ipo' and   
                2. fail(0) = 'closed'
                """)

    st.subheader("**Data Wrangling Over**, Now let's see our data which we will use for further analysis.")
    st.dataframe(df_cleaned.sample(5))
        
############################################################################################################################################################################
    
# st.markdown(""" 
#             Random Sample Imputation with founded_at column.
#             For missing values in the foundation date we will replace those values by the date of the first funding, 
#             and for those cases where we do not have either of the two values (24 inputs) we will fill them by doing a random sample imputation.
#             """)


# df['year'] = df['year'].apply(str_to_float)

############################################################################################################################################################################

# st.markdown("""
#             **Simplify the category_list column**
#             Let's create a column named main category, we're going to assume that the first description in
#             the category_list refers to the main category
#             """)


    
############################################################################################################################################################################


        
#######################################################  Data Wrangling Over ####################################################### 

######################################  Define Functions & Variables  ####################################################### 

success_ratio = df.success.mean()*100
fail_ratio = df.fail.mean()*100

def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i-0.1,y[i],y[i])
        
def group_by_status(column_name):
    # Group by a feature and calculate the fail/success rate
    group = df.groupby(column_name)[['success','fail','operating']].sum()
    group['total'] = group.sum(axis=1)
    group['success_ratio'] = group.success/group.total
    group['fail_ratio'] = group.fail/group.total
    return group

#######################################  Functions & Variables Over ####################################################### 

st.sidebar.subheader('**Multivariate Analysis**')    
st.sidebar.markdown("Understanding the behaviour of Startup by different variables.")
variable = st.sidebar.selectbox('Choose a Variable:', ['Foundation Year', 'Country', 'Categories'])

###################################################################################################################### 

if not st.sidebar.checkbox("Hide", True,key='1'):
    if variable == 'Foundation Year':
        st.subheader("**Let's visualize how is the behavior of startups depending on the Foundation Year :chart_with_upwards_trend:**")
        st.markdown("Let's see only the startups founded after 1990 i.e. 1990 - 2023, "
                    "cause we want to see if there's a pattern between the most recent startups")
        df = df[(df.year > 1990) & (df.year < 2023)]
        
        year_status = group_by_status('year')
        st.dataframe(year_status.sort_values(by='year', ascending=False).head())
        st.markdown("We can see a very clear outlier, and it corresponds to the year 2016, " 
                    "this is due to the small amount of the sample (only 2 entries), so the percentage is not representative.")
        
        df = df[df.year < 2016]
        year_status = year_status[year_status.index < 2016]
        plt.figure(figsize=(8,6))
        plt.title('Number of startups by status_class after filtering by year', size=16)
        sns.barplot(x= df.status_class.value_counts().index, y=df.status_class.value_counts())
        addlabels(df.status_class.value_counts().index,df.status_class.value_counts())
        st.pyplot(plt.gcf())
        
        st.write(f'Number of startups: {df.count().max()}')
        st.write(f'{df.success[df.success == 1].count()} startups reach success, what means {df.success.mean()*100:.2f}% of total')
        st.write(f'{df.fail[df.fail == 1].count()} startups fail, what means {df.fail.mean()*100:.2f}% of total')
        
        plt.figure(figsize=(12,6))
        plt.title('Startups created by year', size=18)
        plt.ylabel('Number of startups',size=12)
        plt.grid()
        sns.lineplot(data=year_status, x=year_status.index, y=year_status.total, linewidth=5)
        plt.axvline(x=2008, color="red", label="2008 Recesion",linestyle=':')
        plt.axvline(x=2000, color="red", label="2000 Dot-com bubble",linestyle=':')
        plt.legend()
        st.pyplot(plt.gcf())
        
        st.markdown("""
                    We can see in the trend line that there was a clear upward trend in terms of startup creation, with only two setbacks, located in 2000 and 2008, 
                    which makes sense, due to the crisis experienced during those years.
                    However, we can observe a slight drop in 2014 and a huge drop in 2015.   
                    This could be explained by the nature of the dataset and the possible lack of entries in the most recent years, 
                    as happened in a more extreme way with 2016
                    """)
        
        plt.figure(figsize=(12,6))
        plt.title('Startups created by year', size=18)
        plt.ylabel('Status Ratio',size=12)
        plt.plot(year_status[year_status.index < 2016].index,year_status[year_status.index < 2016].success_ratio,
                label='Success Ratio',color='#62CB3A',linewidth=3)
        plt.plot(year_status[year_status.index < 2016].index,year_status[year_status.index < 2016].fail_ratio,
                label='Fail Ratio',color='#EA664A',linewidth=3)
        plt.axvline(x=2008, color="blue", label="2008 Recesion",linestyle=':')
        plt.axvline(x=2000, color="blue", label="2000 Dot-com bubble",linestyle=':')
        plt.axhline(df.success.mean(), label='Global Success Ratio', linestyle='-.')
        plt.grid()
        plt.legend()
        st.pyplot(plt.gcf())
        
        st.markdown("""
                    We can see a disturbing trend graph of both the percentage of startups that have succeeded and those that have failed over the years, 
                    we can see how in both crises, the trend went through an inflection point, 
                    in fact, it is just during the 2008 crisis that both trends intersect.  
                    The most distressing thing of all is The clear downward trend in the success rate of startups since its peak in 1997.
                    """)
        
        df_year = pd.DataFrame()
        df_year['original'] = df.year
        df_year['success'] = df.year[df.success == 1]
        df_year['fail'] = df.year[df.fail == 1]

        plt.figure(figsize=(8,5))
        plt.title('Years distribution by status',size=18)
        df_year.boxplot(color='#69D3DF', patch_artist=True, vert=0,showmeans=True, medianprops={'color':'white'})
        st.pyplot(plt.gcf())
        
        st.markdown("""
                    Successful startups tend to be founded in more recent years. 
                    This aligns with the lower success rate observed earlier, suggesting a more competitive landscape.
                    Despite the apparent decline in startups per year (potentially due to data collection issues), an upward trend might be present. 
                    Increased accessibility of previously expensive tools could be driving this rise. 
                    However, it also creates more competition, leading to fewer successes compared to the total number of startups.
                    """)
        
    ######################################################################################################################

    elif variable == 'Country':
        st.markdown("Let's see only the top countries with the startups > 50.")
        country_status = group_by_status('country')

        # Filter by countries with at least 50 startups
        countries_most = country_status.total[country_status.total >= 50]
        st.write(f'There are {countries_most.count()} countries with at least 50 startups')  
        
        # Let's the proportion of the total startupos by countries
        countries_analysis = country_status[country_status.total >= 50]
        top_ten = countries_analysis.total.sort_values(ascending=False).head(10)
        rest_world = pd.Series([countries_analysis.total.sort_values(ascending=False)[10:].sum()], index=['REST OF WORLD'])
        world = pd.concat([top_ten, rest_world])
        
        plt.figure(figsize=(12,8))
        plt.title('Top 10 countries with more startups vs. Rest of world', size=18)
        squarify.plot(sizes=world, label=world.index, color=sns.color_palette("magma",len(world)), alpha = 0.7)
        plt.axis('off')
        plt.show()
        st.pyplot(plt.gcf())
        
        st.markdown(f"""
                    We can see that there is quite a wide difference between the US and the rest of the countries, 
                    in fact only the US represents more than 60% of the dataset, and has 10 times more startups than the second (UK) in the sample.
                    """)
        df_nousa = df[df.country != 'USA']

        n = len(['success_ratio','fail_ratio'])
        x = np.arange(n)
        width = .25

        plt.figure(figsize=(8,6))
        plt.bar(x, df[['success','fail']].mean(), width=width, label='With USA')
        plt.bar(x - width, df_nousa[['success','fail']].mean(), width=width, label='Without USA')
        plt.title('Status Ratio with USA vs. without', size=16)
        plt.xticks(x,['success_ratio','fail_ratio'])
        plt.legend()
        plt.show()
        st.pyplot(plt.gcf())
        
        st.markdown("A large part of the world success rate is influenced by the success rate in the USA," 
                    "which is not the case for the failure rate.")
        
        st.markdown("""
                    **US cities**
                    Since the USA is by far the largest producer of startups worldwide,
                    let's take a closer look at which cities have the largest number of startups, as well as their success/failure ratios.
                    """)
        # We're gonna ignore those null values in city

        usa_startups = df[df.country == 'USA']
        usa_startups = usa_startups[usa_startups.city.notnull()]

        usa_cities = usa_startups.groupby('city')[['fail','success','operating']].sum()
        usa_cities['total'] = usa_cities.sum(axis=1)
        usa_cities['success_ratio'] = usa_cities.success/usa_cities.total
        usa_cities['fail_ratio'] = usa_cities.fail/usa_cities.total
        usa_cities_most = usa_cities[usa_cities.total > 50]

        successful_cities = usa_cities[usa_cities.total > 50].sort_values(by='success_ratio', ascending=False)

        plt.figure(figsize=(8,6))
        plt.scatter(usa_cities_most.success_ratio,usa_cities_most.fail_ratio, color='deeppink', edgecolor='k')
        plt.title('Success/Fail Rate by US cities (> 50 startups)', size=18)
        plt.ylabel('Fail Ratio', size=14)
        plt.xlabel('Success Ratio', size=14)
        plt.axhline(0.094, label='Global Fail Ratio', linestyle='-', color='#EA664A', alpha=.7)
        plt.axvline(0.1069, label='Global Success Ratio', linestyle='-',color='#62CB3A', alpha=.7)
        st.pyplot(plt.gcf())
        
        uscities_bestratio = usa_cities_most[(usa_cities_most.success_ratio > success_ratio/100) & (usa_cities_most.fail_ratio < fail_ratio/100)]
        uscities_bestratio = uscities_bestratio.success_ratio/uscities_bestratio.fail_ratio
        uscities_bestratio = uscities_bestratio.sort_values(ascending=False)
        
        plt.figure(figsize=(40,10))
        sns.barplot(x=uscities_bestratio.index,y=uscities_bestratio)
        plt.title('US cities in the lower right quadrant (order by Success/Fail rate)', size=70)
        plt.ylabel('Number of successful startups per failed startups', size = 25)
        plt.xticks(rotation=90, fontsize=35)
        plt.yticks(fontsize=35)
        addlabels(uscities_bestratio.index,round(uscities_bestratio,2))
        plt.show()
        st.pyplot(plt.gcf())
        
        st.markdown("""
                    Tucson has zero closed startups registered so far, so, we cannot calculate a ratio of successful startups versus failures. 
                    We see that there are many cities with very high ratios, which means that there are many geographic areas in the USA where it is much more likely to succeed than to fail.

                    We can see the big difference in spreads between the global success/failure rates (which are very close to each other), 
                    and on the other hand, there is a very big spread when we compare that of the cities with the most startups in the USA.

                    1. From the 25 US cities with the most startups, only 6 do not exceed the level of the worldwide success ratio.
                    And 5 of them more than double the global success rate.
                    2. Only Los Angeles has a failure rate higher than the success rate.
                    """)
        
    ######################################################################################################################

    elif variable == 'Categories':
        category_ratio = group_by_status('main_category')
        category_ratio = category_ratio[category_ratio.index != 'Other']
        category_ratio = category_ratio[category_ratio.total > 50]

        plt.figure(figsize=(8,6))
        plt.title('Success/Fail Rate by categories (> 50 startups)', size=20)
        plt.ylabel('Fail Ratio', size=14)
        plt.xlabel('Success Ratio', size=14)
        plt.scatter(category_ratio.success_ratio, category_ratio.fail_ratio, edgecolor='k')
        plt.axhline(0.094, label='Global Fail Ratio', linestyle='-', color='#EA664A', alpha=.7)
        plt.axvline(0.1069, label='Global Success Ratio', linestyle='-',color='#62CB3A', alpha=.7)
        plt.legend(loc='upper right')
        st.pyplot(plt.gcf())
        
        categories_bestratio = category_ratio[(category_ratio.success_ratio > success_ratio/100) & (category_ratio.fail_ratio < fail_ratio/100)]
        categories_bestratio = categories_bestratio.success_ratio/categories_bestratio.fail_ratio
        categories_bestratio = categories_bestratio.sort_values(ascending=False)

        plt.figure(figsize=(16,6))  
        plt.title('Categories in the lower right quadrant (order by Success/Fail rate)', size=20)
        plt.ylabel('Number of successful startups per failed startups')
        plt.xticks(rotation=70)
        sns.barplot(x=categories_bestratio.index,y=categories_bestratio)

        addlabels(categories_bestratio.index,round(categories_bestratio,2))
        plt.show()
        st.pyplot(plt.gcf())
        
        st.markdown("""
                    As expected, the vast majority of startups that appear in the top with the highest success rate are companies dedicated to the technology sector.

                    We see that there are categories where startups have ratios of up to 5:1, 
                    i.e. for every startup that is closed 5 were acquired or went public.
                    """)
    
# #####################################################################################################################




















