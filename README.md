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
The target variable is `homeless_status`, which categorizes veterans as homeless or non-homeless. Missing values in character strength scores were imputed using the mean of the respective group.

].

## Project Workflow  
1. Data Preprocessing
   * Data Loading and inspection
   * Handling missing values
   * Data cleaning and Formatting
   
2. Exploratory Data Analysis (EDA)
   - Calculated descriptive statistics (mean, median, standard deviation)  and check the number of people in each category across homeless and non-homeless veterans
   - 
   - Plotted correlation heat map and dendograms of character strenghts across homeless and non-homless veterans
   - Examined the distribution of  `age`, `gender`, `homeless_status`, `education status`,`employment status` and `household income` of veterans through a pie plot


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
