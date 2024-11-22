#!/usr/bin/env python
# coding: utf-8

# # What are the character strengths differences among homeless veterans and non-homeless veterans

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy.stats as stats


import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.size'] = 12

pd.set_option('display.max_rows', 300)
pd.set_option('display.max_columns', 300)


# # Data Reading, Merging and Cleaning

# In[2]:


veteran_1 = pd.read_excel("C:\\Users\\kgduodu\OneDrive - The University of Texas at El Paso\\Desktop\\veterans.xlsx", sheet_name='Demographics')
veteran_2 = pd.read_excel("C:\\Users\kgduodu\\OneDrive - The University of Texas at El Paso\\Desktop\\veterans.xlsx", sheet_name='VIA Results')


# In[3]:


print('veteran 1: ', veteran_1.shape)
print('Unique Veteran 1 ID: ', len(veteran_1['Session ID'].unique()))
print('veteran 2: ', veteran_2.shape)
print('Unique Veteran 2 ID: ', len(veteran_2['Session ID'].unique()))


# In[4]:


#merge veteran 1 and 2 datasets by session ID
veterans_dat = pd.merge(veteran_1, veteran_2, on='Session ID', how='outer')
print('veteran: ', veterans_dat.shape)



# In[5]:


us_identifiers = [
    'United States', 'USA', 'Usa', 'usa', 'U.S.', 'US', 'us',
    'New York', 'Los Angeles', 'Chicago', 'California', 'Texas', 'Virginia', 'Florida', 'Georgia',
    'Puerto Rico', 'Guam', 'U.S. Virgin Islands', 'American Samoa', 'Ohio', 'Colorado', 'Iowa',
    'Wyoming', 'Washington', 'Philadelphia',
    'California', 'New York', 'North Carolina', 'United States', 'United States Minor Outlying Islands',
    'Washington State', 'New jersey', 'New york city', 'America', 'Decataur', 'Michigan', 'United states',
    'Fort Hood', 'Jersey', 'chicago, il', 'Washington', 'Baltimore md', 'Northern Mariana Islands', 
    'California', 'Vernal utah', 'Houston TX', 'Maryland', 'West Texas', 'Nevada', 
    'Big Bend Sector', 'Perth', 'Yuma az', 'Illinois', 'Seattle', 'Pennsylvania', 'Bethlehem, GA',
    'Cannon AFB', 
    'DC', 'Philadelphia', 'Arkham', 'Texas', 'Tx','jblm', 'Alpine Texas', 'Grand Rapids, MI',
    'New york', 'Kempner TX', 'Becker', 'Vermont', 'Washington DC', 'Fairchild Air Force Base',
    'Georgia', 'San Diego', 'Utah', 'New Mexico', 'Schofield barracks', 
    'Burlington', 'Arizona', 'San Antonio, Tx', 'Aland Islands', 'Memphis, Tn', 'Fairfax va',
    'North Dakota', 'Cincinnati Ohio', 'sing', 'new york', 'San Diego, CA', 'Fort polk', 'mn',
    'Chesterfield, MI', 'Seattle, WA', 'Joint Base Andrews', 'Brussels', 'Denver', 'Colorado Springs', 
    'San Diego', 'Adelaide', 'Montana', 'Fayetteville WV', 'Grover Beach, California', 
    'United Stateso', 'new zealand', 'Washington', 'Los Angeles CA', 'Fort Huachuca', 'VA', 'Ellsworth AFB',
    'Camp Humphreys', 'America', 'New Jersey', 'United sttes', 'Danvers',
    'North Carolina', 'Whiteman', 'Az', 'in a small village of 80 people', 'Yardley, PA',
    'Charleston, SC', 'Aurora, CO',

    #for Non_veterans
    #       
    'United States','USA', 'Oklahoma ','MI','Connecticut', 'San diego',
       'Oregon','new jersey','Philadelphia, PA','texas', 'usa', 'New York ',
        'ohio', 'Jersey', 'Boston, MA ','San Francisco', 'North Dakota',
        'Utah', 'California', 'Los Angeles, CA','Boston', 'Chicago', 
        'Indiana','Arizona ', 'Thousand Oaks, California', 'Cleveland',
        'Dayton, Ohio','new york','San Francisco ','Las vegas nevada','Cincinnati, ohio',
        'Missouri','Illinois', 'Kentucky', 'Texas', 'New York',  'Denver, CO', 'Usa', 
        'South Carolina',  'Ohio', 'Greenville',
       'Oklahoma', 'United States Minor Outlying Islands', 'United States of America',
       'Portland, oregon ', 'Rhode Island', 'United ',
       'VIrginia', 'Guernsey', 'Los Angeles ', 'New York, NY', 'Alexandria Virginia', 'US', 'Montana',
       'North Carolina','Alabama','Provo, ut','New Hampshire', 'West palm beach florida ',
        'Salt lake city utah ', 'Pittsburgh, PA','Washington dc', 'Columbus, Ohio','37 The Crescent',
         'New york', 'Nebraska ', 'Long Island',
       'Pittsburgh pa ', 'New Jersey', 'Houston, Tx', 'Minnesota',
       'Maryland', 'ok ','Cincinnati ohio', 'Anchorage alaska', 'Arkansas','San fran','Superior WI',
       'Tennessee', 'Chicago, IL', 'Brooklyn, Ny', 'florida', 'Orlando',
     'Portland','Un', 'United','Virginia, USA', 'miami','Colorado, USA','Scott AFB',
      'United states', 'Buckeye az', 'Pomona, CA',
       'Central Queensland', 'Austin, Texas', 'Massachusetts',
       'Washington, DC','Pennsylvania', 'Michigan', 'Florida','Virginia','Idaho', 'Holloman AFB', 'Arizona', 'Princeton, nj',
        'New york ', 'utah', 'United Statesw', 'Los Angeles','cincinnati',
         'Cincinnati, Ohio', 'San Diego', 'Macon, GA','Nashville, TN', 'Northern New York', 'Colorado ',
         'Brighton co', 'Livingston, NJ', 'jersey', 'Minnesota ','philadelphia', 'washington',
          'Louisville,  KY ','United States ', 'Coastal South Carolina','Truckee, CA',
          'Maine','Atlanta, GA', 'Sacramento ','St. Catharines, Ontario','woodridge', 'NJ', 'Honolulu, Hawaii', 
           'Grand forks ', 'Marion, IN', 'california ', 'phoenix, AZ','Charleston', 'tampa', 'United state', 'Ohio USA', 'Hawaii',
       'Brooklyn NY', 'Wisconsin', 'united states', '8 Fawn circle Malvern Pa ', 'Columbus OH',
       'Washington state', 'Albuquerque', 'Colorado', 'California ','montana','Austin', 'Atlanta', 'Arcata, CA',
        'Miami',  'Cincinnati ','Aspendale', 'PA', 'American Samoa', 'CA','Berkeley', 'Reading PA', 'Alberta', 
         'United states ','indiana', 'NYC','Oakland, CA', 'Charlottesville ','Denver','St Catharines Ontario', 'United States America ', 
         'Atlanta GA','american ','St. Louis', 'NY', 'San diego, california','Longmeadow MA','Bakersfield', 'Fort Worth Texas', 'Texas ', 'SC',
       'Richmond, VA', 'chantilly, va', 'Georgia, United stats','Tampa, Fl','Eugene, OR', 
       'Bend, Oregon', 'San antonio ', 'Cincinnati', 'Upstate New York','exeter', 'Stretton', 'Unites','Orlando, FL, USA', 'Provo, utah',
       'Cardington', 'Washington ','America', 'NEW Jersey ', 'Nevada ', 'Newton, NC','New jersey', 'colorado', 'Las Vegas',
       'Eagle Pass, Texas', 'Charlotte, NC', 'Forest hill','louisiana', 'Nyc', 'USA ','Miami ', 'NORTH CAROLINA', 'U', 
       'Kansas City', 'Dallas','Livingston nj','San Antonio TX','Nevada', 'Maryland ', 'FL ','Friedheim, MO. USA','Friedheim, MO. USA', 'New Jersey USA',
       'St Albans', 'ma','Plymouth ma ','Compton','South Carolina ', 'Alaska',
       'boulder, Colorado', 'Columbus Ohio', 'Columbus ', 'Gahanna, Ohio', 'Youngstown ','Bennington VT','Fort hood tx ','Oregon ','Washington','Florida Broward County ',
       'Pittston ','Byron, MI','Walla walla, washington', 'Canton','Las Vegas, Nevada','Ann arbor, mi','Goodfellow AFB','arkansas',  'GLENDORA , CA','New York City','Chapel Hill',
       'Kansas Oklahoma', 'NEW YORK CITY', 'Joliet, il', 'Dallas TX', 'Midwest usa','Minneapolis ', 'Seattle, WA',
        'Units states of America ', 'Marietta, GA','United States12','Portland, Oregon','napa california','Atlanta ', 'Louisiana', 'Menlo Park, California', 'Atlanta, Georgia', 'Cypress,Texas',
        'Portland, OR','Virgin Islands, U.S.', 'Montana ', 'Seguin, TX','Wright Patterson AFB', '6817 41st Street Ct NW', 'boston, ma','Memphis, TN', 'west virginia','NV', 'Delaware ', 'Santa Barbara',
        'las vegas', 'Waterloo, Ontario, CA','Estados Unidos','Mind Springs Health','lexington ky','us','Santa Barbara California',
        'Arizona, United States', 'Unites states of america', 'United States of America ', 'Arlington, VA','Springfield, MO','NEw YOrk', 'CHICAGO', 
        'Knoxville', 'united states of america','Burbank','LOS ANGELES', 'washington, dc','Frankston',
       'Newport Beach, Ca','Madison', 'Houston','America ','Washoe Valley, Nevada','Philadelphia','cape may, nj', 'Beverly Hills','Newkensingtion ', 'Walnut Creek',
       'usa ', 'Ca','Franklin, MA', 'nyc','in', 'New Mexico', 'Berwick', 'Pasadena', 'Pomona','San Diego, CA ', 'Las vegas ', 'Provo, Utah','columbus',  'California, USA', 'MA', 'Massachusets', 'Sharpsville ',
       'fort huachuca, arizona', 'Mi', 'Santa Monica', 'Birmingham, AL','6427 Freeport Rd Fayetteville NC', 'Georgia ','pharr texas', 'Wichita State University', 'Kansas','University of Cincinnati',
       'Frederick, MD',  'chicago', 'San Diego, CA', 'Wilkes-Barre','bowling green','Tempe', 'Philadelphia ', 'Greensboro','california', 'United State', 'Salt lake city ut','nebraska','Columbus, OH','San francisco',
       'Ysis, NM', 'Northampton, Massachusetts','Fort myers','San Antonio','fort collins','Cartersville', 'Columbus, Ga', 'Phoenix',
        'Buffalo,NY', 'allison park', 'oakland california', 'Columbus ohio',
       'Cleveland, Ohio','Raleigh','Charlotte, nc', 'Ohio ', 'United States19146',
       'Brooklyn, NY', 'Ambler, PA', 'CO','america', 'Memphis', 'Springfield','UT', 'oklahoma','Rochester NY','Virginia Beach','Las vegas','New jersey ', 'Rochester ny'
]   

def classify_location(location):
    if location in us_identifiers:
        return 'usa'
    else:
        return 'non_usa'

# Apply the function to the column
veterans_dat['location'] = veterans_dat['Where are you located?'].apply(classify_location)

# Check the updated value counts
print(veterans_dat['location'].value_counts(dropna=False))


data =veterans_dat
print('data: ', data.shape)

#create age columns
year_of_birth = pd.to_datetime(data['What is your date of birth?'], format='%Y-%M-%d', errors='coerce').dt.year
# how many value for year beyond 2010 and erase them (people who are not above 18yo)
year_of_birth[year_of_birth > 2010] = np.nan
year_of_birth[year_of_birth < 1904] = np.nan
data['age'] = 2023 - year_of_birth

#restrain data to where location is usa
data = data[data['location'] == 'usa']
#restrain data to 18yo and above
data = data[data['age'] >= 18]



# In[6]:


data[data['What is your current employment status?']=='Retired']['Are you currently experiencing homelessness?'].value_counts(dropna=False)


# In[7]:


#Employment binary variable creation
data['employment_status_binary'] = np.nan
for i,emp in enumerate(data['What is your current employment status?']):
    if emp in ['Employed full time (40 or more hours per week)'
                , 'Active Military'
                ,'Full-time Student '
                ,'Employed part time (up to 39 hours per week)'
                ,'Homemaker'] :
        data['employment_status_binary'].iloc[i] = 'employed'
    elif emp in ['Disabled or Unable to Work','Unemployed']:
        data['employment_status_binary'].iloc[i] = 'unemployed'
    else: #here are included retired, other  and NA
        data['employment_status_binary'].iloc[i] = np.nan


# In[8]:


#create new columns based on the character strength category
categories = {'Wisdom': ['Curiosity', 'Creativity', 'Love of Learning', 'Judgment', 'Perspective'],
              'Courage': ['Bravery', 'Perseverance ', 'Honesty', 'Zest'],
              'Humanity': ['Love', 'Kindness', 'Social Intelligence'],
              'Justice': ['Teamwork', 'Fairness', 'Leadership'],
              'Temperance': ['Forgiveness', 'Humility', 'Prudence', 'Self-Regulation'],
              'Transcendence': ['Appreciation of Beauty & Excellence', 'Gratitude', 'Hope', 'Humor', 'Spirituality']}

for category in categories.keys():
    data[category] = data[categories[category]].median(axis=1) 


# In[9]:


#Those are the codes for disabilities but right now we are not using the distinction between them
code_to_disability = {
    "2364": "Autism/Aspergers",
    "2365": "Hearing Impairment",
    "2366": "Intellectual Disability",
    "2367": "Mental Disorder",
    "2368": "Physical Impairment",
    "2369": "Specific Learning Disability",
    "2370": "Speech Impairment",
    "2371": "Traumatic Brain Injury",
    "2372": "Visual Impairment",
    "2373": "War Related Disability",
    "2374": "Yes i'd rather not say"}

#count the number of codes separated by commas in each row of 'Are you experiencing any of the following? (Please check all that apply)' column 
data['disability_count'] = data['Are you experiencing any of the following? (Please check all that apply)'].str.count(',') + 1
# where nan, replace with 0 (means no disability)
data['disability_count'] = data['disability_count'].fillna(0)
#bar plot for disabilities count
#data['disability_count'].loc[data['Are you a veteran of the military?'] == 'YES'].value_counts(dropna=True).plot(kind='bar')


# In[10]:


# Update rows based on the condition
data['disability_count'] = data['disability_count'].apply(lambda x: 'No disability' if x == 0 else 'Disability')


# In[11]:


data.columns


# In[12]:


# Define the mapping of old values to new shortened names
education_mapping = {
    "Completed Master's, Doctorate, or Professional degree (post-Bachelor's)": "Post-Bachelor's",
    "Some college but no degree": "Some College",
    "Some graduate or professional school": "Grad/Prof School",
    "Bachelor's degree": "Bachelor's",
    "Certificate or technical degree": "Cert/Tech",
    "High school degree or GED": "High School/GED",
    "Associate's degree": "Associate's",
    "Less than a high school degree": "Less than HS"
}

# Apply the mapping to the column
data["What is your highest level of education?"] = data["What is your highest level of education?"].replace(education_mapping)

# Verify the changes
print(data["What is your highest level of education?"].unique())


# In[13]:


#calculate age based on birth year
legends_level_of_ed = ['Post-graduate degree'
                       , "Bachelor's degree"
                       , 'Some college (no degree)'
                       , 'High school degree or GED'
                       , 'Some graduate or\nprofessional school'
                       , "Associate's degree"
                       , 'Certificate or technical degree'
                       , 'less that high school degree']

#legends for ages by category:
legends_age = ['18-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75+']
ages = pd.cut(data['age'], bins=[18, 25, 35, 45, 55, 65, 75, 100], labels=legends_age)


# In[14]:


# Filter rows where the column matches 'NO' or 'YES'
data= data[data['Are you currently experiencing homelessness?'].isin(['NO', 'YES'])]


# In[15]:


# Define the mapping dictionary
gender_mapping = {
    'Female': 'Female',
    'Male': 'Male',
    'Prefer not to say': 'Other',
    'Non-binary / third gender': 'Other',
    np.nan: 'Other'  # Handle NaN values as 'Other'
}

# Apply the mapping to the gender column
data['How do you identify your gender?'] = data['How do you identify your gender?'].replace(gender_mapping)


# In[16]:


# Filter rows to exclude 'Other', 'Retired', and 'N/A'
data = data[~data['What is your current employment status?'].isin(["Other", "Retired", "N/A"])]

# Define the categories for employed and unemployed
Employed = [
    "Employed part time (up to 39 hours per week)",
    "Employed full time (40 or more hours per week)",
    "Homemaker",
    "Full-time Student",
    "Active Military"
]

Unemployed = [
    "Unemployed",
    "Disabled or Unable to Work"
]

# Update the column values based on the conditions
data['What is your current employment status?'] = data['What is your current employment status?'].apply(
    lambda x: "Employed" if x in Employed else "Unemployed"
)


# In[17]:


# Given data
job_data_dict = {
    "Armed Forces": 8365,
    "Other": 7201,
    # ... (rest of your data)
    "Chief Executives, Senior Officials, Legislators, Business and Administration Professionals": 3
}

job_data_series = pd.Series(job_data_dict)

# Define mapping of jobs to broader domains
job_to_domain_mapping = {
    "Armed Forces": "Military & Public Service",
    "Legislators and Senior Officials": "Military & Public Service",
    "Legislators": "Military & Public Service",
    "Religious Professionals": "Military & Public Service",
    "Information and Communications Technology Professionals": "Information & Technology",

    "Science and Engineering Professionals": "Science & Engineering",

    "Production and Specialized Services Managers": "Management & Executive Roles",
    "Hospitality, Retail and Other Services Managers": "Management & Executive Roles",
    "Managing Directors and Chief Executives": "Management & Executive Roles",
    "Chief Executives, Senior Officials, Legislators, Business and Administration Professionals": "Management & Executive Roles",

    "Health Professionals Not Elsewhere Classified": "Health & Medical",
    "Nursing and Midwifery Professionals": "Health & Medical",
    "Medical Doctors": "Health & Medical",
    "Paramedical Practitioners": "Health & Medical",
    "Traditional and Complementary Medicine Professionals": "Health & Medical",
    "Dentists": "Health & Medical",
    "Health Professionals": "Health & Medical",
    "Health Coach": "Health & Medical",

    "Secondary Education Teachers": "Education & Academia",
    "Other Teaching Professionals": "Education & Academia",
    "Primary School and Early Childhood Teachers": "Education & Academia",
    "University and Higher Education Teachers": "Education & Academia",
    "Vocational Education Teachers": "Education & Academia",
    "Teaching Professionals": "Education & Academia",

    "Finance Professionals": "Finance, Sales & Marketing",
    "Sales, Marketing and Public Relations Professionals": "Finance, Sales & Marketing",

    "Legal Professionals": "Legal & Counseling",
    "Social Work and Counselling Professionals": "Legal & Counseling",
    "Psychologists": "Legal & Counseling",
    "Legal, Social/Mental Health and Cultural Professionals": "Legal & Counseling",

    "Administration Professionals": "Human Resources & Administration",
    "Human resources Professional": "Human Resources & Administration",

    "Other": "Others",
    "Retired": "Others",
    "NaN": "Others",
    "Wellness, Executive, and/or Life Coach": "Others"
}


# Apply the mapping to reclassify job categories into broader domains
data['job_domain'] = data['What is your occupation?'].map(job_to_domain_mapping)
data['job_domain'].value_counts(dropna = False)


# # EXPLORATORY ANALYSIS

# In[18]:


#make pie chart of data column dtypes in a 3x2 grid
fig,axs = plt.subplots(2,3, figsize=(12,8))
fig.suptitle('Data Types and Demographics (Veterans)', fontsize=16)
axs[0,0].pie(data.dtypes.value_counts(), autopct='%1.1f%%', shadow=False
             , startangle=90,pctdistance=0.75)

axs[0,0].legend(data.dtypes.value_counts().index, loc='lower right')
axs[0,0].set_title('Data Types')
axs[0,0].set_ylabel('')
#on 0,1 make a pie chart for the civilian vs veteran ammount in complete data
axs[0,1].pie(data['Are you a veteran of the military?'].value_counts(dropna=False)
             , autopct='%1.2f%%', shadow=False, startangle=90,pctdistance=0.6)
axs[0,1].legend(data['Are you a veteran of the military?'].value_counts(dropna=False).index, loc='upper right')
axs[0,1].set_title('Are you a veteran\n of the military?')

#on 0,2 make a pie chart for "How do you identify your gender?" column
axs[0,2].pie(data['How do you identify your gender?'].value_counts()
             , autopct='%1.2f%%', shadow=False, startangle=180
             , labeldistance=1.8,pctdistance=1.1)
axs[0,2].legend(data['How do you identify your gender?'].value_counts().index
                , bbox_to_anchor=[0.9,0.62])
axs[0,2].set_title('How do you identify\n your gender?')

#on 1,0 make a pie chart for What is your highest level of education? column
axs[1,0].pie(data['What is your highest level of education?'].value_counts()
                , autopct='%1.2f%%', shadow=False, startangle=180
                , labeldistance=1.8,pctdistance=1.1)
axs[1,0].legend(legends_level_of_ed, bbox_to_anchor=[1.4,0.05])
axs[1,0].set_title('What is your highest\n level of education?')

#on 1,1 make a pie chart for What is your total annual household income column
axs[1,1].pie(data['What is your total annual household income'].value_counts()
                , autopct='%1.2f%%', shadow=False, startangle=180
                , labeldistance=1.8,pctdistance=1.1)
axs[1,1].legend(data['What is your total annual household income'].value_counts().index
                , bbox_to_anchor=[1,0])
axs[1,1].set_title('What is your total \nannual household income')

axs[1,2].pie(ages.value_counts()
                , autopct='%1.2f%%', shadow=False, startangle=180   
                , labeldistance=1.8,pctdistance=1.1)
axs[1,2].legend(legends_age, bbox_to_anchor=[0.7,0.05])
axs[1,2].set_title('What is your Age?')

fig.tight_layout()
fig.patch.set_facecolor('white')
plt.show()


# In[19]:


data_temp = data[data['Are you a veteran of the military?'] == 'YES']
fig,axs = plt.subplots(2,3, figsize=(12,8))
fig.suptitle('Demographics of Veterans', fontsize=18)
axs[0,0].pie(data_temp['Are you currently experiencing homelessness?'].value_counts(), autopct='%1.1f%%', shadow=False
             , startangle=90,pctdistance=0.75)
axs[0,0].legend(data_temp['Are you currently experiencing homelessness?'].value_counts().index
                , bbox_to_anchor=[0.9,0.62])
axs[0,0].set_title('Are you currently \nexperiencing homelessness?')
axs[0,0].set_ylabel('')
#on 0,1 make a pie chart for the civilian vs veteran ammount in complete data
axs[0,1].pie(data_temp['employment_status_binary'].value_counts(dropna=False)
             , autopct='%1.2f%%', shadow=False, startangle=90,pctdistance=1.2)
axs[0,1].legend(data_temp['employment_status_binary'].value_counts(dropna=False).index
                , bbox_to_anchor=[0.9,0.62])
axs[0,1].set_title('Employment Status')

#on 0,2 make a pie chart for "How do you identify your gender?" column
axs[0,2].pie(data_temp['How do you identify your gender?'].value_counts()
             , autopct='%1.2f%%', shadow=False, startangle=180
             , labeldistance=1.8,pctdistance=1.1)
axs[0,2].legend(data_temp['How do you identify your gender?'].value_counts().index
                , bbox_to_anchor=[0.9,0.82])
axs[0,2].set_title('How do you identify\n your gender?')

#on 1,0 make a pie chart for What is your highest level of education? column
axs[1,0].pie(data_temp['What is your highest level of education?'].value_counts()
                , autopct='%1.2f%%', shadow=False, startangle=180
                , labeldistance=1.8,pctdistance=1.1)
axs[1,0].legend(legends_level_of_ed, bbox_to_anchor=[1.4,0.05])
axs[1,0].set_title('What is your highest\n level of education?')

#on 1,1 make a pie chart for What is your total annual household income column
axs[1,1].pie(data_temp['What is your total annual household income'].value_counts()
                , autopct='%1.2f%%', shadow=False, startangle=180
                , labeldistance=1.8,pctdistance=1.1)
axs[1,1].legend(data_temp['What is your total annual household income'].value_counts().index
                , bbox_to_anchor=[1,0])
axs[1,1].set_title('What is your total \nannual household income')

axs[1,2].pie(ages.value_counts()
                , autopct='%1.2f%%', shadow=False, startangle=180   
                , labeldistance=1.8,pctdistance=1.1)
axs[1,2].legend(legends_age, bbox_to_anchor=[0.7,0.05])
axs[1,2].set_title('What is your Age?')

fig.tight_layout()
fig.patch.set_facecolor('white')
plt.show()


# In[20]:


# Identifying the columns that contain 'Rank' and extract the first word from each such column
rank_columns = [col for col in data.columns if 'Rank' in col]
#strength_columns = [col.split()[0] for col in rank_columns]  # Split by space and take the first word
strength_columns = [col.replace(" Rank", "") for col in rank_columns if " Rank" in col]

# Ensuring we only include the strengths that have corresponding columns
strength_columns = [col for col in strength_columns if col in data.columns]

# List of character strength variables
strengths = strength_columns
# pairwise correlation matrix
correlation_matrix = veterans_dat[strengths].corr()

# Converting the correlation matrix into a DataFrame
correlation_df = pd.DataFrame(correlation_matrix)
#correlation_df


# In[21]:


# Assuming 'data' contains your strength columns (replace 'strength_columns' with your actual column names)
# Adjust the figure size to accommodate all strength names
plt.figure(figsize=(10,10))  # You can increase the size as needed

# Create the heatmap with correlation
sns.heatmap(data[strengths].corr(), cmap='Blues', vmin=0, vmax=1, fmt=".2f")



# Show the plot
plt.show()


# In[22]:


from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

# Convert correlation matrix to distance matrix
distance_matrix = np.sqrt(2 * (1 - correlation_df))

# Perform hierarchical clustering
linked = linkage(distance_matrix, 'single')

# Plotting the dendrogram
plt.figure(figsize=(14, 7))
dendrogram(linked, labels=strengths, orientation='top', distance_sort='descending')
plt.title("Dendrogram for Character Strengths")
plt.xlabel("Character Strengths")
plt.xticks(rotation=90)
plt.ylabel("Distance")
plt.grid()
plt.show()


# # Testing for association with descriptive approach

# In[23]:


data[data['Are you a veteran of the military?'] == 'YES']['Are you currently experiencing homelessness?'].value_counts(dropna=False)


# In[24]:


df = data[data['Are you a veteran of the military?'] == 'YES'].reset_index(drop=True)
#drop df data on nans in Are you currently experiencing homelessness? column
df = df.dropna(subset=['Are you currently experiencing homelessness?'])


# In[25]:


#bar plot for imbalanced target variable are you experiencing homelesness column
df['Are you currently experiencing homelessness?'].value_counts(dropna=False)


# In[26]:


#Bar plot of average wisdom, courage, humanity, justice, temperance, and transcendence by homelessness status
fig,axs = plt.subplots(1,1, figsize=(5,5))
fig.suptitle('Character Strengths by \nHomelessness Status (Veterans)', fontsize=16)
pos_no = df[df['Are you currently experiencing homelessness?'] == 'NO']
pos_yes = df[df['Are you currently experiencing homelessness?'] == 'YES']
pos_no[['Wisdom', 'Courage', 'Humanity', 'Justice', 'Temperance'
        , 'Transcendence']].mean(axis=0).plot(kind='bar', ax=axs, color='blue', label='No homeless', alpha=0.2)
pos_yes[['Wisdom', 'Courage', 'Humanity', 'Justice', 'Temperance'
         , 'Transcendence']].mean(axis=0).plot(kind='bar', ax=axs, color='red', label='Homeless', alpha=0.2)
fig.legend(loc='lower right')
plt.show()


# In[27]:


# data['employment_status_binary'].value_counts(dropna=False)
data = data.dropna(subset=['employment_status_binary'])


# In[28]:


data['How do you identify your gender?'].value_counts(dropna=False)


# In[29]:


variables_to_analyze = ['age',  'How do you identify your gender?']
character_strengths = [ 'Appreciation of Beauty & Excellence', 'Bravery', 'Love', 'Prudence', 'Teamwork', 
                                           'Creativity', 'Curiosity','Fairness','Forgiveness', 'Gratitude', 'Honesty','Hope',
                                           'Humor','Perseverance ', 'Judgment','Kindness', 'Leadership','Love of Learning',
                                           'Humility', 'Perspective', 'Self-Regulation','Social Intelligence', 
                                           'Spirituality', 'Zest']
categories_var1 = ['young adults', 'middle aged', 'elderly']
categories_var2 = ['Male', 'Female', 'other']
categories_var3 = ['No disability', 'Disability']
categories_var4 = ['employed', 'unemployed']
categories_var5 =["Post-Bachelor's", "Bachelor's", 'Some College' ,'Grad/Prof School',
 'Cert/Tech', 'High School/GED', "Associate's" , 'Less than HS']



# create columns with the categorycal division of our selected vars
data['age_category'] = pd.cut(data['age'], bins=[18, 35, 55, 120], labels=categories_var1) #this is a cut if your variable numeric
data['gender_category'] = np.nan
for i, x in enumerate(data['How do you identify your gender?'].values): #this is how you cut if your variable is categorical
    if x == 'Male':
        data['gender_category'].iloc[i] = x
    elif x == 'Female':
        data['gender_category'].iloc[i] = x
    else:
        
        data['gender_category'].iloc[i] = 'other'


# In[30]:


#p_values for AGE
#first iterate for the categories of the first variable
#pvalues_age = np.zeros((len(categories_var1), len(character_strengths))) * np.nan #create empty array to fill with pvalues
numbers_rows = np.zeros((len(categories_var1), 2)) * np.nan #create empty array to fill with pvalues
for i, cat in enumerate(categories_var1):
    data_subset = data[data['age_category'] == cat] #here i selected each age group to study
    for j, cs in enumerate(character_strengths):
        #get the two populations to compare (i.e. homeless vs non homeless)
        homeless_pop = data_subset[data_subset['Are you currently experiencing homelessness?'] == 'YES']
        non_homeless_pop = data_subset[data_subset['Are you currently experiencing homelessness?'] == 'NO']
        #to tell the difference between the two populations we use a t-test
        ttest = stats.ttest_ind(homeless_pop[cs], non_homeless_pop[cs], equal_var=False, nan_policy='omit')
        numbers_rows[i, 0] = len(homeless_pop)
        numbers_rows[i, 1] = len(non_homeless_pop)
        #pvalues_age[i, j] = round(ttest.pvalue, 3)


# In[31]:


#heatmap for number of people in each category
fig,axs = plt.subplots(1,1, figsize=(12,5))
fig.suptitle('Number of people in each category', fontsize=16)
sns.heatmap(numbers_rows, annot=True, fmt=".0f", cmap='Blues', ax=axs,yticklabels=categories_var2)
axs.set_xticklabels(['homeless', 'non_homeless_pop'])
# axs.set_xlabel('Character Strengths')
axs.set_ylabel('Gender Category')
plt.show()


# In[32]:


pvalues_age = np.zeros((len(categories_var1), len(character_strengths))) * np.nan #create empty array to fill with pvalues
numbers_rows = np.zeros((len(categories_var1), 2)) * np.nan #create empty array to fill with pvalues
formatted_pvalues = np.empty((len(categories_var1), len(character_strengths)), dtype=object)

for i, cat in enumerate(categories_var1):
    data_subset = data[data["age_category"] == cat]  # Here I selected each employment group group
    for j, cs in enumerate(character_strengths):
        # Get the two populations to compare (i.e., homeless vs non-homeless)
        homeless_pop = data_subset[data_subset['Are you currently experiencing homelessness?'] == 'YES']
        non_homeless_pop = data_subset[data_subset['Are you currently experiencing homelessness?'] == 'NO']
        # Perform t-test to compare populations
        ttest = stats.ttest_ind(homeless_pop[cs], non_homeless_pop[cs], equal_var=False, nan_policy='omit')
        numbers_rows[i, 0] = len(homeless_pop)
        numbers_rows[i, 1] = len(non_homeless_pop)
        p_value = round(ttest.pvalue, 3)
        pvalues_age[i, j] = p_value

        # Add asterisk for significant p-values
        if p_value < 0.05:
            formatted_pvalues[i, j] = f"{p_value}"
        else:
            formatted_pvalues[i, j] = f"{p_value}"

# Convert the formatted array for displaying in heatmap
formatted_pvalues_array = np.array(formatted_pvalues)

# Plot the heatmap
fig, ax = plt.subplots(figsize=(35, 10))
sns.heatmap(
    pvalues_age,
    annot=formatted_pvalues_array,
    cmap='Blues_r',
    ax=ax,
    xticklabels=character_strengths,
    yticklabels=categories_var1,
    annot_kws={"size": 18},
    cbar_kws={'label': 'p-value'},
    fmt=''
)

# Overlay red asterisks for significant values
for i in range(pvalues_age.shape[0]):
    for j in range(pvalues_age.shape[1]):
        if pvalues_age[i, j] < 0.05:  # Significant threshold
            ax.text(
                j + 0.5,  # x-coordinate
                i + 0.5,  # y-coordinate
                "***",  # Three asterisks
                color='red',  # Red color for asterisks
                ha='center',  # Horizontal alignment
                va='center',  # Vertical alignment
                fontsize=18  # Font size
            )

# Customize the plot
ax.set_title(
    'Are the character strengths different between \nhomeless and non-homeless Age status?',
    fontsize=25
)
ax.set_xlabel('Character Strengths', fontsize=25)
ax.set_ylabel('Age Status', fontsize=25)

# Adjust tick labels
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)

# Show the plot
plt.show()


# In[33]:


pvalues_disable = np.zeros((len(categories_var3), len(character_strengths))) * np.nan #create empty array to fill with pvalues
numbers_rows = np.zeros((len(categories_var3), 2)) * np.nan #create empty array to fill with pvalues
formatted_pvalues = np.empty((len(categories_var3), len(character_strengths)), dtype=object)

for i, cat in enumerate(categories_var3):
    data_subset = data[data["disability_count"] == cat]  # Here I selected each employment group group
    for j, cs in enumerate(character_strengths):
        # Get the two populations to compare (i.e., homeless vs non-homeless)
        homeless_pop = data_subset[data_subset['Are you currently experiencing homelessness?'] == 'YES']
        non_homeless_pop = data_subset[data_subset['Are you currently experiencing homelessness?'] == 'NO']
        # Perform t-test to compare populations
        ttest = stats.ttest_ind(homeless_pop[cs], non_homeless_pop[cs], equal_var=False, nan_policy='omit')
        numbers_rows[i, 0] = len(homeless_pop)
        numbers_rows[i, 1] = len(non_homeless_pop)
        p_value = round(ttest.pvalue, 3)
        pvalues_disable[i, j] = p_value

        # Add asterisk for significant p-values
        if p_value < 0.05:
            formatted_pvalues[i, j] = f"{p_value}"
        else:
            formatted_pvalues[i, j] = f"{p_value}"

# Convert the formatted array for displaying in heatmap
formatted_pvalues_array = np.array(formatted_pvalues)

# Plot the heatmap
fig, ax = plt.subplots(figsize=(35, 7))
sns.heatmap(
    pvalues_disable,
    annot=formatted_pvalues_array,
    cmap='Blues_r',
    ax=ax,
    xticklabels=character_strengths,
    yticklabels=categories_var3,
    annot_kws={"size": 18},
    cbar_kws={'label': 'p-value'},
    fmt=''
)

# Overlay red asterisks for significant values
for i in range(pvalues_disable.shape[0]):
    for j in range(pvalues_disable.shape[1]):
        if pvalues_disable[i, j] < 0.05:  # Significant threshold
            ax.text(
                j + 0.5,  # x-coordinate
                i + 0.5,  # y-coordinate
                "***",  # Three asterisks
                color='red',  # Red color for asterisks
                ha='center',  # Horizontal alignment
                va='center',  # Vertical alignment
                fontsize=18  # Font size
            )

# Customize the plot
ax.set_title(
    'Are the character strengths different between \nhomeless and non-homeless Disability status?',
    fontsize=25
)
ax.set_xlabel('Character Strengths', fontsize=25)
ax.set_ylabel('Disability Status', fontsize=25)

# Adjust tick labels
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)

# Show the plot
plt.show()


# In[34]:


pvalues_employment = np.zeros((len(categories_var4), len(character_strengths))) * np.nan #create empty array to fill with pvalues
numbers_rows = np.zeros((len(categories_var4), 2)) * np.nan #create empty array to fill with pvalues
formatted_pvalues = np.empty((len(categories_var4), len(character_strengths)), dtype=object)

for i, cat in enumerate(categories_var4):
    data_subset = data[data["employment_status_binary"] == cat]  # Here I selected each employment group group
    for j, cs in enumerate(character_strengths):
        # Get the two populations to compare (i.e., homeless vs non-homeless)
        homeless_pop = data_subset[data_subset['Are you currently experiencing homelessness?'] == 'YES']
        non_homeless_pop = data_subset[data_subset['Are you currently experiencing homelessness?'] == 'NO']
        # Perform t-test to compare populations
        ttest = stats.ttest_ind(homeless_pop[cs], non_homeless_pop[cs], equal_var=False, nan_policy='omit')
        numbers_rows[i, 0] = len(homeless_pop)
        numbers_rows[i, 1] = len(non_homeless_pop)
        p_value = round(ttest.pvalue, 3)
        pvalues_employment[i, j] = p_value

        # Add asterisk for significant p-values
        if p_value < 0.05:
            formatted_pvalues[i, j] = f"{p_value}"
        else:
            formatted_pvalues[i, j] = f"{p_value}"

# Convert the formatted array for displaying in heatmap
formatted_pvalues_array = np.array(formatted_pvalues)

# Plot the heatmap
fig, ax = plt.subplots(figsize=(35, 7))
sns.heatmap(
    pvalues_employment,
    annot=formatted_pvalues_array,
    cmap='Blues_r',
    ax=ax,
    xticklabels=character_strengths,
    yticklabels=categories_var4,
    annot_kws={"size": 18},
    cbar_kws={'label': 'p-value'},
    fmt=''
)

# Overlay red asterisks for significant values
for i in range(pvalues_employment.shape[0]):
    for j in range(pvalues_employment.shape[1]):
        if pvalues_employment[i, j] < 0.05:  # Significant threshold
            ax.text(
                j + 0.5,  # x-coordinate
                i + 0.5,  # y-coordinate
                "***",  # Three asterisks
                color='red',  # Red color for asterisks
                ha='center',  # Horizontal alignment
                va='center',  # Vertical alignment
                fontsize=18  # Font size
            )

# Customize the plot
ax.set_title(
    'Are the character strengths different between \nhomeless and non-homeless employment status?',
    fontsize=25
)
ax.set_xlabel('Character Strengths', fontsize=25)
ax.set_ylabel('Employment Status', fontsize=25)

# Adjust tick labels
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)

# Show the plot
plt.show()


# In[35]:


pvalues_education = np.zeros((len(categories_var5), len(character_strengths))) * np.nan #create empty array to fill with pvalues
numbers_rows = np.zeros((len(categories_var5), 2)) * np.nan #create empty array to fill with pvalues
formatted_pvalues = np.empty((len(categories_var5), len(character_strengths)), dtype=object)


for i, cat in enumerate(categories_var5):
    data_subset = data[data["What is your highest level of education?"] == cat]  # Here I selected each education group
    for j, cs in enumerate(character_strengths):
        # Get the two populations to compare (i.e., homeless vs non-homeless)
        homeless_pop = data_subset[data_subset['Are you currently experiencing homelessness?'] == 'YES']
        non_homeless_pop = data_subset[data_subset['Are you currently experiencing homelessness?'] == 'NO']
        # Perform t-test to compare populations
        ttest = stats.ttest_ind(homeless_pop[cs], non_homeless_pop[cs], equal_var=False, nan_policy='omit')
        numbers_rows[i, 0] = len(homeless_pop)
        numbers_rows[i, 1] = len(non_homeless_pop)
        p_value = round(ttest.pvalue, 3)
        pvalues_education[i, j] = p_value

        # Add asterisk for significant p-values
        if p_value < 0.05:
            formatted_pvalues[i, j] = f"{p_value}"
        else:
            formatted_pvalues[i, j] = f"{p_value}"

# Convert the formatted array for displaying in heatmap
formatted_pvalues_array = np.array(formatted_pvalues)

# Plot the heatmap
fig, ax = plt.subplots(figsize=(35, 10))
sns.heatmap(
    pvalues_education,
    annot=formatted_pvalues_array,
    cmap='Blues_r',
    ax=ax,
    xticklabels=character_strengths,
    yticklabels=categories_var5,
    annot_kws={"size": 18},
    cbar_kws={'label': 'p-value'},
    fmt=''
)

# Overlay red asterisks for significant values
for i in range(pvalues_education.shape[0]):
    for j in range(pvalues_education.shape[1]):
        if pvalues_education[i, j] < 0.05:  # Significant threshold
            ax.text(
                j + 0.5,  # x-coordinate
                i + 0.5,  # y-coordinate
                "***",  # Three asterisks
                color='red',  # Red color for asterisks
                ha='center',  # Horizontal alignment
                va='center',  # Vertical alignment
                fontsize=18  # Font size
            )

# Customize the plot
ax.set_title(
    'Are the character strengths different between \nhomeless and non-homeless Education status?',
    fontsize=25
)
ax.set_xlabel('Character Strengths', fontsize=25)
ax.set_ylabel('Education Status', fontsize=25)

# Adjust tick labels
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)

# Show the plot
plt.show()


# In[36]:


selected_columns = ['Are you currently experiencing homelessness?', 'Appreciation of Beauty & Excellence', 'Bravery', 'Love', 'Prudence', 'Teamwork', 'Creativity', 'Curiosity','Fairness','Forgiveness', 'Gratitude', 'Honesty','Hope','Humor','Perseverance ', 'Judgment','Kindness', 'Leadership','Love of Learning','Humility', 'Perspective', 'Self-Regulation','Social Intelligence', 'Spirituality', 'Zest']


# In[37]:


vet=data[selected_columns]


# In[38]:


from math import pi  # Import pi from the math module


# Group by homelessness status and calculate mean for each strength
mean_strengths = vet.groupby('Are you currently experiencing homelessness?').mean()
#mean_strengths_nv=df_selected_non_vet.groupby('Are you currently experiencing homelessness?').mean()

# Transpose the DataFrame for easier plotting
mean_strengths = mean_strengths.transpose()
#mean_strengths_nv=mean_strengths_nv.transpose()

# Number of variables
num_vars = len(mean_strengths.index)

# Calculate angle for each axis in the plot
angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
angles += angles[:1]

# Create a spider plot
plt.figure(figsize=(7, 7))
ax = plt.subplot(111, polar=True)

# Plot each category
for i, status in enumerate(mean_strengths.columns):
    values = mean_strengths[status].values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, label=status, linewidth=2, linestyle='solid')

# Add character strength names to each axis
ax.set_xticks(angles[:-1])

# Mark specific indices with an asterisk
#marked_indices = [1, 4, 7, 8, 9, 10, 11, 13, 14, 15, 19]

xticklabels = ['Appr. of Beauty & Excll', '*Bravery', 'Love', 'Prudence', '*Teamwork', 'Creativity', 'Curiosity','*Fairness','*Forgiveness', '*Gratitude', '*Honesty','*Hope','Humor','*Perseverance ', '*Judgment','*Kindness', 'Leadership','Love of Learning','Humility', '*Perspective', 'Self-Regulation','Social Intelligence', 'Spirituality', 'Zest']



#ax.set_xticks([i for i in range(len(xticklabels2))])  # Set the number of ticks based on xticklabels2
ax.set_xticklabels(xticklabels, ha='center')



# Adjust the size of xtick labels
#ax.tick_params(axis='x', labelsize=12)

# Add legend and title
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.annotate("* significant", xy=(1.15, 1), xycoords="axes fraction", ha="left", fontsize=10)
plt.title('Veterans Means of Character Strengths by Homelessness Status')

# Show the plot
plt.show()

