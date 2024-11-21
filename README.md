# Character Strengths Analysis of Homeless and Non-Homeless U.S. Veterans

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Overview](#dataset-overview)
- [Project Workflow](#project-overview)
  

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
   
   - Plotted correlation heat map and dendograms of character strenghts across homeless and non-homless veterans
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


3. Statistical Analysis
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
   
4. Visualization  
5. Conclusion  

## Requirements  
- Python 3.9+  
- Required libraries: pandas, numpy, matplotlib, seaborn, scikit-learn  

## Setup and Installation  
1. Clone the repository:  
   `git clone https://github.com/your_username/character-strengths-analysis.git`  
2. Install dependencies:  
   `pip install -r requirements.txt`  
3. Run the main script:  
   `python main.py`  

## Results  
Homeless veterans exhibit significantly lower scores in strengths such as gratitude and hope, compared to non-homeless veterans.

## Visualization  
![Bar chart of character strengths](images/bar_chart.png)

## License  
This project is licensed under the MIT License.

## Acknowledgments  
Thanks to the VIA Institute on Character for providing the dataset.
