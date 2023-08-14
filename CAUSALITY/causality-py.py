# ---------------------------------------------------------------------------- #
#                                   Causality                                  #
# ---------------------------------------------------------------------------- #

import pandas as pd
import numpy as np

# ----------- Section 2.1 Racial Discrimination in the Labor Market ---------- #

resume = pd.read_csv('resume.csv')

resume.shape

resume.head()

resume.dtypes # firstname, sex, and race are currently strings

resume.describe() # by default, only summarizes numeric variables

'''
In 2.2.5, when we discuss categorical variables, we will also explore overriding 
the `describe()` default behavior and alternatives for summarizing non-numeric 
data.
'''

# contingency table (crosstab)
race_call_tab = pd.crosstab(resume.race, resume['call'])
# note the two ways to access a column in a data frame

race_call_tab

type(race_call_tab) # a data frame

# the data frame's index and columns both have names
print(race_call_tab.columns)
print(race_call_tab.index)

# crosstab with margins
pd.crosstab(resume.race, resume.call, margins=True)

# overall callback rate: total callbacks divided by sample size 

## using positional selection and number of rows
race_call_tab.iloc[:,1].sum() / resume.shape[0] 

# callback rate for each race race
race_call_tab.loc['black', 1] / race_call_tab.loc['black'].sum() # black
race_call_tab.loc['white', 1] / race_call_tab.loc['white'].sum() # white

race_call_tab.iloc[0] # the first row, using positions
race_call_tab.loc['black'] # the first row, using names

race_call_tab.iloc[:,1] # the second column, using positions
race_call_tab.loc[:,1] # the second column, using names
'''
By coincidence, the name of the second column is also the number 1. In pandas, 
column names can be numeric.
'''

resume['call'].mean() # overall callback rate

# --------------------- Section 2.2 Subsetting in pandas --------------------- #

# 2.2.1 Boolean values and Logical Operators

type(True)

int(True)
int(False)
 
x = pd.Series([True, False, True]) # a vector with boolean values 

x.mean() # proportion of True values
x.sum() # number of True values

False & True

True & True

True | False 

False | False

True & False & True

# Parentheses evaluate to False (NB: QSS source comment is flipped)
(True | False) & False

# Parentheses evaluate to True (NB: QSS source comment is flipped)
True | (False & False)

# Vector-wise logical operations 
TF1 = pd.Series([True, False, False])
TF2 = pd.Series([True, False, True])
TF1 | TF2
TF1 & TF2

# 2.2.2 Relational Operators 

4 > 3

"Hello" == "hello" # Python is case-sensitive

"Hello" != "hello"

x = pd.Series([3, 2, 1, -2, -1])
x >= 2
x != 1

# logical conjunction of two vectors with boolean values
(x > 0) & (x <= 2)
# logical disjunction of two vectors with boolean values
(x > 2) | (x <= -1)

x_int = (x > 0) & (x <= 2) # logical vector 
x_int

x_int.mean() # proportion of True values
x_int.sum() # number of True values

# 2.2.3 Subsetting

# callback rate for black-sounding names
resume['call'][resume['race'] == 'black'].mean()

# race of the first 5 observations
resume['race'][0:5]

# comparison of first 5 observations
resume['race'][0:5] == 'black'

resume.shape # dimensions of the original data frame

# subset blacks only
resumeB = resume.loc[resume['race'] == 'black'].copy()

resumeB.shape # this data frame has fewer rows than the original 

resumeB['call'].mean() # callback rate for blacks

# subset observations with black, female-sounding names
# keep only the "call" and "firstname" variables 
resumeBf = (resume.loc[(resume.race == 'black') & 
                       (resume.sex == 'female'), ['call', 'firstname']])
'''
Notice, we can use parentheses to break up a line and circumvent the python 
white space rules. 
'''

resumeBf.head(n=6)

# black male
resumeBm = resume.loc[(resume.race == 'black') & (resume.sex == 'male')]

# white female
resumeWf = resume.loc[(resume.race == 'white') & (resume.sex == 'female')]

# white male
resumeWm = resume.loc[(resume.race == 'white') & (resume.sex == 'male')]

# racial gaps
resumeWf['call'].mean() - resumeBf['call'].mean() # among females
resumeWm['call'].mean() - resumeBm['call'].mean() # among males

# 2.2.4 Simple conditional statements 

# where() from numpy implements vectorized if-else
resume['BlackFemale'] = (np.where((resume.race == 'black') & 
                                  (resume.sex == 'female'), 1, 0))

# three-way crosstab
pd.crosstab([resume.race, resume.sex], resume.BlackFemale)

# drop the BlackFemale column in place
resume.drop('BlackFemale', axis=1, inplace=True)


# 2.2.5 Categorical variables

'''
Recall, firstname, sex, and race are currently strings, but for analytical
purposes, they are categorical variables because values in these columns belong 
to one of a limited number of groups. Let's convert firstname, sex, and race to 
the pandas categorical data type. 
'''

# first, store the variable names in a list for more compact code
cat_vars = ['firstname', 'sex', 'race']

resume[cat_vars] = resume[cat_vars].astype('category')

resume.dtypes # now the variables are categorical

resume['race'][0:5]

resume['race'].cat.categories

resume['race'].cat.codes

resume['race'].value_counts()

resume['race'].value_counts(normalize=True)

resume[cat_vars].describe()

resume.describe(include='all') # output is not visually appealing

# create a factor variable 

resume['type'] = np.nan
(resume.loc[(resume.race == "black") & 
            (resume.sex == "female"), 'type']) = 'BlackFemale'
(resume.loc[(resume.race == "black") & 
            (resume.sex == "male"), 'type']) = 'BlackMale'
(resume.loc[(resume.race == "white") & 
            (resume.sex == "female"), 'type']) = 'WhiteFemale'
(resume.loc[(resume.race == "white") & 
            (resume.sex == "male"), 'type']) = 'WhiteMale'

# A faster alternative: 

# create a list of n-1 conditions 
conditions = [
      (resume.race == "black") & (resume.sex == "female")
    , (resume.race == "black") & (resume.sex == "male")
    , (resume.race == "white") & (resume.sex == "female")
]

# create a list of choices corresponding to the conditions
choices  = ['BlackFemale', 'BlackMale', 'WhiteFemale']

# create a new column in the DF based on the conditions
# the third argument is the default value if none of the conditions are met
resume["type_alt"] = np.select(conditions, choices, 'WhiteMale')

# check that the results are the same
resume['type'].equals(resume['type_alt'])

# drop the alternative column
resume.drop('type_alt', axis=1, inplace=True)

resume.dtypes # type is still a string

# coerce the new variable into a categorical variable
resume['type'] = resume['type'].astype('category')

# list the categories
resume['type'].cat.categories

# obtain the number of observations in each category
resume['type'].value_counts(sort=False)

# compute callback rate for each category
resume.groupby('type')['call'].mean()

# compute callback rate for each first name using groupby
callback_name = resume.groupby('firstname')['call'].mean()

# look at the names with the lowest callback rates
callback_name.sort_values().head(n=10)

# look at the names with the highest callback rates
callback_name.sort_values(ascending=False).head(n=10)

# ------------ Section 2.3: Causal Effects and the Counterfactual ------------ #

resume.iloc[0]

# ----------------- Section 2.4: Randomized Controlled Trials ---------------- #

# ----------------- Section 2.4.1: The Role of Randomization ----------------- #

# ------------- Section 2.4.2: Social Pressure and Voter Turnout ------------- #

social = pd.read_csv('social.csv')

social.describe().round(2)

social.info()

# convert sex and messages to categorical variables
social[['sex', 'messages']] = social[['sex', 'messages']].astype('category')

social['messages'].cat.categories

# re-order the categories, so the control group is first
social['messages'] = social['messages'].cat.reorder_categories(
    ['Control', 'Civic Duty', 'Hawthorne', 'Neighbors'])

social['messages'].cat.categories

'''
Even though we re-ordered the levels, we have not converted messages to an 
ordered categorical variable.
'''
social['messages'].cat.ordered

# turnout for each group
social.groupby('messages')['primary2006'].mean()

# turnout for control group
social['primary2006'][social.messages == 'Control'].mean()

# subtract control group turnout from each group
(social.groupby('messages')['primary2006'].mean() - 
 social['primary2006'][social.messages == 'Control'].mean())

social['age'] = 2006 - social['yearofbirth'] # create age variable

# calculate mean of age for each message type
social.groupby('messages')['age'].mean()

# calculate the mean of primary2004 for each message type
social.groupby('messages')['primary2004'].mean()

# calculate the mean of hhsize for each message type
social.groupby('messages')['hhsize'].mean()

# -------------------- Section 2.5: Observational Studies -------------------- #

# Section 2.5.1: Minimum Wage and Unemployment

'''
If we know that certain variables should be categorical ahead of time, we can
specify that in pd.read_csv() using the dtype argument and a dictionary. 
'''
minwage = pd.read_csv('minwage.csv', 
                      dtype={'chain': 'category', 'location': 'category'})

minwage.info()

minwage.shape

minwage.describe().round(2)

minwage['chain'].value_counts()

minwage['location'].value_counts()

# subsetting the data into two states
minwageNJ = minwage.loc[minwage.location != 'PA'].copy()
minwagePA = minwage.loc[minwage.location == 'PA'].copy()

# proportion of restaurants whose wage is less than $5.05
(minwageNJ['wageBefore'] < 5.05).mean() # NJ before

(minwageNJ['wageAfter'] < 5.05).mean() # NJ after

(minwagePA['wageBefore'] < 5.05).mean() # PA before

(minwagePA['wageAfter'] < 5.05).mean() # PA after

# create a variable for proportion of full-time employees in NJ and PA
minwageNJ['fullPropAfter'] = (
    minwageNJ['fullAfter'] / (minwageNJ['fullAfter'] + minwageNJ['partAfter'])
    )

minwagePA['fullPropAfter'] = (
    minwagePA['fullAfter'] / (minwagePA['fullAfter'] + minwagePA['partAfter'])
    )

# compute the difference in means
minwageNJ['fullPropAfter'].mean() - minwagePA['fullPropAfter'].mean()

# ---------------------- Section 2.5.2: Confounding Bias --------------------- #

minwageNJ['chain'].value_counts(sort=False, normalize=True)

minwagePA['chain'].value_counts(sort=False, normalize=True)

# subset Burger King only
minwageNJ_bk = minwageNJ.loc[minwageNJ.chain == 'burgerking'].copy()
minwagePA_bk = minwagePA.loc[minwagePA.chain == 'burgerking'].copy()

# comparison of full-time employment rates
minwageNJ_bk['fullPropAfter'].mean() - minwagePA_bk['fullPropAfter'].mean()

minwageNJ_bk_subset = (
    minwageNJ_bk.loc[(minwageNJ_bk.location != 'shoreNJ') & 
                     (minwageNJ_bk.location != 'centralNJ')].copy()
)

(minwageNJ_bk_subset['fullPropAfter'].mean() - 
 minwagePA_bk['fullPropAfter'].mean())

# --- Section 2.5.3: Before-and-After and Difference-in-Differences Design --- #

# full-time employment proportion in the previous period for NJ
minwageNJ['fullPropBefore'] = (
    minwageNJ['fullBefore'] / 
    (minwageNJ['fullBefore'] + minwageNJ['partBefore'])
)

# mean difference before and after the minimum wage increase for NJ
NJdiff = minwageNJ['fullPropAfter'].mean() - minwageNJ['fullPropBefore'].mean()
NJdiff

# full-time employment proportion in the previous period for PA
minwagePA['fullPropBefore'] = (
    minwagePA['fullBefore'] / 
    (minwagePA['fullBefore'] + minwagePA['partBefore'])
)

# mean difference before and after the minimum wage increase for PA
PAdiff = minwagePA['fullPropAfter'].mean() - minwagePA['fullPropBefore'].mean()

# difference-in-differences
NJdiff - PAdiff

# --------- Section 2.6: Descriptive Statistics for a Single Variable -------- #

# Section 2.6.1: Quantiles

# cross-section comparison between NJ and PA
minwageNJ['fullPropAfter'].median() - minwagePA['fullPropAfter'].median()

# before and after comparison
NJdiff_med = (minwageNJ['fullPropAfter'].median() - 
              minwageNJ['fullPropBefore'].median())

NJdiff_med.round(3)

# median difference-in-differences
PAdiff_med = (minwagePA['fullPropAfter'].median() - 
              minwagePA['fullPropBefore'].median())

NJdiff_med - PAdiff_med

# describe() shows quartiles as well as minimum, maximum, and mean
minwageNJ['wageBefore'].describe().round(2)

minwageNJ['wageAfter'].describe().round(2)

# find the interquartile range (IQR)
(minwageNJ['wageBefore'].quantile(0.75) - 
 minwageNJ['wageBefore'].quantile(0.25)).round(2)

minwageNJ['wageAfter'].quantile(0.75) - minwageNJ['wageAfter'].quantile(0.25)

# deciles (10 groups)
# use np.arange(start, stop, step) to generate sequence; stop is not included 
minwageNJ['wageBefore'].quantile(np.arange(0, 1.1, 0.1))

minwageNJ['wageAfter'].quantile(np.arange(0, 1.1, 0.1))

# Section 2.6.2: Standard Deviation
(np.sqrt((minwageNJ['fullPropAfter'] - 
          minwageNJ['fullPropBefore']).pow(2).mean()))

(minwageNJ['fullPropAfter'] - minwageNJ['fullPropBefore']).mean()

# standard deviation
minwageNJ['fullPropBefore'].std()
minwageNJ['fullPropAfter'].std()

# variance
minwageNJ['fullPropBefore'].var()
minwageNJ['fullPropAfter'].var()
