# Character Strengths Analysis of Homeless and Non-Homeless U.S. Veterans

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Overview](#dataset-overview)
- [Project Workflow](#project-overview)
- [Requirements](#requirements)
  

## Project Overview  
This project analyzes character strengths among U.S. veterans, comparing differences between homeless and non-homeless veterans. Using data from the VIA Institute on Character, the goal is to uncover correlations between character traits and homelessness.
This study explores the character strengths of U.S. war veterans and examines how these attributes correlate with the risk of homelessness within this demographic. Utilizing a comprehensive dataset, we conducted an extensive exploratory data analysis (EDA) to investigate the character strengths of both homeless and non-homeless veterans. The analysis encompassed a range of variables, including age, gender, employment status, race, education, marital status, income, and disability status. We applied statistical methods to calculate p-values, aiming to identify significant differences in character strengths between the homeless and non-homeless groups across these variables. Our findings reveal insightful patterns and disparities, providing a nuanced understanding of the character strengths prevalent among U.S. war veterans. The study's implications extend beyond academic interest, offering valuable insights for policymakers and organizations dedicated to veteran welfare and homelessness prevention. By identifying the key character strengths associated with reduced homelessness risk, this research contributes to the development of targeted support strategies for this vulnerable population.

## Dataset Overview  
Source: VIA Institute on Character  
The dataset includes [The dataset contains 29361 records of U.S. veterans, with the following key details:
- `homeless_status`: Categorical variable indicating homelessness (Yes/No).
- `age`: Age of the veteran.
- `gender`: Male or female.
- `character_strengths`: Scores for 24 VIA character strengths, such as gratitude, hope, perseverance, brave etc.
- `education status`: The highest level education of veterans
- `employment status`: The current occupation of veterans
The target variable is `homeless_status`, which categorizes veterans as homeless or non-homeless. Missing values in character strength scores were imputed using the mean of the respective group.].

## Project Workflow  
### 1. Data Preprocessing
   * Data Loading and inspection
   * Handling missing values
   * Data cleaning and Formatting
   
### 2. Exploratory Data Analysis (EDA)
This study's EDA aims to uncover patterns and insights within the character strengths of U.S. war veterans, and to evaluate these findings in the context of homelessness risk. Our analysis delves into 29361 survey sessions, examining a diverse range of variables, including employment, education, income, disability, and homelessness status.


 - Calculated descriptive statistics (mean, median, standard deviation)  and check the number of people in each category across homeless and non-homeless veterans
  ```python
   #heatmap for number of people in each category
fig,axs = plt.subplots(1,1, figsize=(12,5))
fig.suptitle('Number of people in each category', fontsize=16)
sns.heatmap(numbers_rows, annot=True, fmt=".0f", cmap='Blues', ax=axs,yticklabels=categories_var2)
axs.set_xticklabels(['homeless', 'non_homeless_pop'])
# axs.set_xlabel('Character Strengths')
axs.set_ylabel('Gender Category')
plt.show()
```
#### Distribution of veteran status
![Number of People in each category](/plots/poluationnumber.png)
   
- Plotted correlation heat map and dendograms of character strenghts across homeless and non-homless veterans.We used a heatmap to visualize the correlation matrix of character strengths among veterans, revealing low correlation coefficients, suggesting the distinctness of these traits. Hierarchical clustering further delineated the relationships among various character strengths, indicating that while some traits may co-occur, they largely represent distinct constructs
```python
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
# Create the heatmap with correlation
sns.heatmap(data[strengths].corr(), cmap='Blues', vmin=0, vmax=1, fmt=".2f")
# Show the plot
plt.show()
```
#### correlation heatmap
![Dendogram](/plots/corrheatmap.png)
Heatmap depicting the correlations between various character strengths, with darker shades indicating stronger associations. This analysis informs on the research on U.S. war veterans by highlighting patterns in character strengths that may relate to the risk of homelessness within this group.

```python
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
```
#### Dendograms of characrter strengths
![Dendogram](/plots/dendogramcs.png)
The dendrogram groups character strengths based on their similarity, with closer clusters indicating more closely related traits. This clustering can provide insights into the interrelationships among character strengths of U.S. war veterans, which may influence their risk of homelessness.

   - Examined the distribution of  `age`, `gender`, `homeless_status`, `education status`,`employment status` and `household income` of veterans through a pie plot
```python
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
```
#### Distribution veterans demographics
![Dendogram](/plots/pieplots.png)


### 3.  Statistical Analysis
* In this phase, we performed statistical tests to identify significant differences in character strengths between homeless and non-homeless veterans across different categories (`age`,`disabilty`,`employemnt`,`education`). A t-test was used to compare the means of the two groups (homeless vs. non-homeless) for each character strength. Below is the code that calculates p-values for age category across each comparison:
  
```python
import numpy as np
from scipy import stats

# Initialize arrays to store p-values, sample sizes, and formatted p-values
pvalues_age = np.zeros((len(categories_var1), len(character_strengths))) * np.nan
numbers_rows = np.zeros((len(categories_var1), 2)) * np.nan
formatted_pvalues = np.empty((len(categories_var1), len(character_strengths)), dtype=object)

# Loop through age categories and character strengths
for i, cat in enumerate(categories_var1):
    data_subset = data[data["age_category"] == cat]  # Select each age category
    for j, cs in enumerate(character_strengths):
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
            formatted_pvalues[i, j] = f"{p_value}*"
        else:
            formatted_pvalues[i, j] = f"{p_value}"

# Convert the formatted array for display
formatted_pvalues_array = np.array(formatted_pvalues)
```
#### HEATMAPS OF ASSOCIATION BETWEEN HOMELESS AND NON-HOMELESS VETERANS ACROSS VARIOUS CATEGORIES WITH CHARACTER STRENGTH

Each shaded cell in the chart indicates the level of statistical significance for a given character strength, with darker shades representing greater significance. By utilizing an alpha threshold of 0.05. The shaded cell with red asterisk indicates a p-values less than 0.05. It means there is enough evidence to suggest that the scores for the given character strength differ significantly between the two groups (homeless vs. non-homeless).

![Association](/plots/agestatus.png)

 Character strengths like bravery, Teamwork, Honesty all indicates a significant difference among young adults who are veterans (homeless vs. non-homeless). Among elderly veterans too appreciation of beauty and excellence differ significantly. This tells us to look for these character strength among young adults and the elderly, it might be a sign to tell us if a veterans can be homeless or not. For the ages we categorized young adults as adults from the age of 18 to 35, middle-aged from 35 to 55 and elderly 55+.   

![Association](/plots/employstatus.png)

Employment status and race have had interesting findings, although intuition tells us that we should consider obvious factors such as education level or simply age. When considering education, notable variations are visible, such as those with a bachelor's degree showing significant differences in "Appreciation of Beauty & Excellence," indicating that education level may influence the manifestation of certain character strengths in relation to homelessness. In each cell the darker the color the bigger the chances of character strength showing significant difference.

![Association](/plots/disabilitystatus.png)

Using the same test of association, we can see that strengths like bravery, teamwork, honesty, etc shows a significant difference among disable veterans. There are no sign of differences in character strengths among able veterans. This gives a glimpse of strenghts we should take into consideration when we are looking veterans with disability. They might be some of the causes of homelessness in veterans shows a sign of these strength. In each cell the darker the color the bigger the chances of character strength showing significant difference.

![Association](/plots/educatiostatus.png)

Whiles bravery showed a significant difference in our initial studies when we considered the age category and emplymet status. There is no sign of difference in education status. Character strengths like Forgiveness showed a difference between homeless and non-homeless veterans who are post-bachelor's. When we look at Hope and Humilty, they both showed significant difference among homeless and non-homeless veterans that has highest level of education being in Profesional school or a level less than High school. In each cell the darker the color the bigger the chances of character strength showing significant difference. The intersting thing is veterans with post bachelor's degree showed no bravery and love in their character strengths.

![Association](/plots/spiderplot.png)

The radar chart titled "Veterans Means of Character Strengths by Homelessness Status" and displays the mean scores of various character strengths for homeless (YES) and non-homeless (NO) veterans. The chart is marked with asterisks to denote significant differences in character strengths, such as "Honesty" and "Perseverance." The blue line represents non-homeless veterans, and the orange line represents homeless veterans. Character strengths like "Honesty," "Hope," and "Kindness" show higher mean values for non-homeless veterans compared to their homeless counterparts, suggesting these traits may be more pronounced in veterans who are not experiencing homelessness.

### 4. Findings & Results  
-  In the age category, most of the character strengths showed difference between homeless veterans and non-homeless veterans that are young adults
-  In the disability category, most of the character strengths showed difference in homeless and non-homeless veterans that are disable
-  In the employment status, most of the character strength showed difference in homeless and non-homeless veterans that are employed.
-  In the employment status at least five of the character strength showed significant differences between homeless and non-homeless veterans that level of higher education being Professional school, Associate's and less than high school.
-  With the help of the radar chart, the most prominent character strengths among non-homeless veterans are Honesty, Judgement, and Kindness, whereas homeless veterans exhibit higher means in Curiosity, Honesty, and Creativity.

   

## Requirements  
- Python 3.9+  
- Required libraries: pandas, numpy, matplotlib, seaborn, scikit-learn  

## Setup and Installation  
1. Clone the repository:  
   `git clone https://github.com/kobbyg184/project1.git`  
2. Install dependencies:  
   `pip install -r requirements.txt`  
3. Run the main script:  
   `python main.py`  




