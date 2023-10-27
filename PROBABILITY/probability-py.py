# ---------------------------------------------------------------------------- #
#                                  Probability                                 #
# ---------------------------------------------------------------------------- #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import comb, exp, factorial, log

# ------------------------- Section 6.1: Probability ------------------------- #

# Section 6.1.1: Frequentist versus Bayesian

# Section 6.1.2: Definition and Axioms

# Section 6.1.3: Permutations

def birthday(k):
    logdenom = k * log(365) + log(factorial(365 - k)) # log denominator
    lognumer = log(factorial(365)) # log numerator
    # P(at least two have the same bday) = 1 - P(nobody has the same bday)
    pr = 1 - exp(lognumer - logdenom) # transform back
    return pr

k = pd.Series(np.arange(1, 51))

bday = k.apply(birthday) # apply the function to each element of k

bday.index = k # add labels 

sns.set_style('ticks')

sns.relplot(
    x=k, y=bday, color='white', edgecolor='black', height=4, aspect=1.5
).set(ylabel='Probability that at least\n two people have the same birthday',
      xlabel='Number of people').despine(right=False, top=False)

# horizontal line at 0.5
plt.axhline(0.5, color='black', linewidth=0.75)

bday.loc[20:25] 

# Section 6.1.4: Sampling With and Without Replacement

k = 23 # number of people
sims = 10000 # number of simulations
event = 0 # initialize counter

for i in range(sims):
    days = np.random.choice(np.arange(1,366), size=k, replace=True)
    days_unique = np.unique(days) # number of unique days
    '''
    if there are duplicates, the number of unique birthdays will be less than
    the number of birthdays, which is 'k
    '''
    if len(days_unique) < len(days):
        event += 1

answer = event / sims
answer

# Section 6.1.5: Combinations

comb(84, 6)

# ------------------- Section 6.2: Conditional Probability ------------------- #

# Section 6.2.1: Conditional, Marginal, and Joint Probabilities

FLVoters = pd.read_csv('FLVoters.csv')

FLVoters.shape # before removal of missing data

FLVoters.info() # there is one missing surname

# print the record with the missing surname
FLVoters[FLVoters['surname'].isnull()]

'''
Looking at the raw data, it turns out that one voter's surname is Null. 
pandas treated the name as missing. We need to override this behavior and treat 
Ms. Null's name as a string.
'''

FLVoters.head() # the surnames are in all caps

FLVoters['surname'] = np.where(
    FLVoters['surname'].isnull(), 'NULL', FLVoters['surname'])

FLVoters = FLVoters.dropna()

FLVoters.shape # after removal of missing data

margin_race = FLVoters['race'].value_counts(normalize=True).sort_index()
margin_race

margin_gender = FLVoters['gender'].value_counts(normalize=True)
margin_gender

FLVoters['race'][FLVoters.gender == 'f'].value_counts(
    normalize=True).sort_index()

joint_p = pd.crosstab(FLVoters.race, FLVoters.gender, normalize=True)
joint_p

'''
To obtain the row sums in pandas, we specify axis='columns' in the .sum()
method. This may seem counterintuitive, but the logic is that we need to
collapse the columns to calculate the sum of each row.
'''

# row sums
joint_p.sum(axis='columns')

# column sums
joint_p.sum(axis='rows')

# Develop age group categories; start with a list of n-1 conditions
conditions = [
      (FLVoters.age <= 20)
    , (FLVoters.age > 20) & (FLVoters.age <= 40)
    , (FLVoters.age > 40) & (FLVoters.age <= 60)
]

choices  = [1, 2, 3]

# Assign 4 to voters older than 60
FLVoters["age_group"] = np.select(conditions, choices, 4)

joint3 = pd.crosstab([FLVoters.race, FLVoters.age_group], FLVoters.gender,
                     normalize=True)

# print the first 8 rows
joint3.head(8)

# marginal probabilities for age groups
margin_age = FLVoters['age_group'].value_counts(normalize=True).sort_index()
margin_age

# take a look at the joint3 index for a few observations
joint3.index[:3]

# select elements from a multi-index using .loc and tuples
joint3.loc[('asian', 3), 'f']

# P(black and female | above 60)
joint3.loc[('black', 4), 'f'] / margin_age[4]

# two-way joint probability table for age group and gender
joint2 = pd.crosstab(FLVoters['age_group'], FLVoters['gender'], 
                     normalize=True)
joint2

# P(above 60 and female)
joint2.loc[4, 'f']

# P(black | female and above 60)
joint3.loc[('black', 4), 'f'] / joint2.loc[4, 'f']

# Section 6.2.2: Independence 

# store plotting parameters
lims = (-0.02, 0.42)
ticks = [0, .1, .2, .3, .4]

sns.relplot(
    x=margin_race * margin_gender['f'], y=joint_p['f'],
    color='white', edgecolor='black', height=4, aspect=1.5
).set(xlabel='P(race) * P(female)', ylabel='P(race and female)',
      xlim=lims, ylim=lims, xticks=ticks, yticks=ticks).despine(
          right=False, top=False)

plt.gca().axline((0, 0), slope=1, color='black', linewidth=0.5)

# subplots for joint and conditional independence
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

lims = (-0.02, 0.32)

# joint independence
sns.scatterplot(
    x=joint3.loc[(slice(None), 4), 'f'].droplevel('age_group'), 
    y=margin_race * margin_age[4] * margin_gender['f'],
    color='white', edgecolor='black', ax=axs[0]
).set(xlabel='P(race and above 60 and female)', 
      ylabel='P(race) * P(above 60) * P(female)',
      title='Joint Independence', xlim=lims, ylim=lims)

axs[0].axline((0, 0), slope=1, color='black', linewidth=0.5)

# conditional independence given female
sns.scatterplot(
    x=(joint3.loc[(slice(None), 4), 'f'] / 
       margin_gender['f']).droplevel('age_group'), 
    y=((joint_p['f'] / margin_gender['f']) * 
       (joint2.loc[4, 'f'] / margin_gender['f'])),
    color='white', edgecolor='black', ax=axs[1]
).set(xlabel='P(race and above 60 | female)', 
      ylabel='P(race | female) * P(above 60 | female)',
      title='Conditional Independence', xlim=lims, ylim=lims)

axs[1].axline((0, 0), slope=1, color='black', linewidth=0.5)

# Monty Hall problem
sims = 1000
doors = np.array(['goat', 'goat', 'car'])
# Store empty vector of strings with same dtype as doors
result_switch = np.empty(sims, dtype=doors.dtype)
result_noswitch = np.empty(sims, dtype=doors.dtype)

for i in range(sims):
    # randomly choose the initial door
    first = np.random.choice(np.arange(0,3))
    result_noswitch[i] = doors[first]
    remain = np.delete(doors, first) # remaining two doors
    if doors[first] == 'car': # two goats left
        monty = np.random.choice(np.arange(0,2))
    else: # one goat and one car left
        monty = np.arange(0,2)[remain=='goat']
    result_switch[i] = np.delete(remain, monty)[0]

(result_noswitch == 'car').mean()
(result_switch == 'car').mean()

# Section 6.2.3: Bayes' Rule

# Section 6.2.4: Predicting Race Using Surname and Residence Location

cnames = pd.read_csv('names.csv')

cnames.info() # one surname is missing

# As with FLVoters, ensure the surname "NULL" is treated as a string
cnames['surname'] = np.where(
    cnames['surname'].isnull(), 'NULL', cnames['surname'])

cnames.shape

# merge the two data frames (inner join)
FLVoters = pd.merge(FLVoters, cnames, on='surname')

FLVoters.shape

# store relevant variables
vars = ["pctwhite", "pctblack", "pctapi", "pcthispanic", "pctothers"]

# Whites 
whites = FLVoters.loc[FLVoters.race == 'white'].copy()
(whites[vars].max(axis='columns') == whites['pctwhite']).mean()

# Blacks
blacks = FLVoters.loc[FLVoters.race == 'black'].copy()
(blacks[vars].max(axis='columns') == blacks['pctblack']).mean()

# Hispanics
hispanics = FLVoters.loc[FLVoters.race == 'hispanic'].copy()
(hispanics[vars].max(axis='columns') == hispanics['pcthispanic']).mean()

# Asian
asians = FLVoters.loc[FLVoters.race == 'asian'].copy()
(asians[vars].max(axis='columns') == asians['pctapi']).mean()

# White false discovery rate 
1 - (FLVoters['race'][FLVoters[vars].max(axis='columns') == 
                      FLVoters['pctwhite']] == "white").mean()

# Black false discovery rate
1 - (FLVoters['race'][FLVoters[vars].max(axis='columns') == 
                      FLVoters['pctblack']] == "black").mean()

# Hispanic false discovery rate
1 - (FLVoters['race'][FLVoters[vars].max(axis='columns') == 
                      FLVoters['pcthispanic']] == "hispanic").mean()

# Asian false discovery rate
1 - (FLVoters['race'][FLVoters[vars].max(axis='columns') == 
                      FLVoters['pctapi']] == "asian").mean()

FLCensus = pd.read_csv('FLCensusVTD.csv')

# compute proportions by applying np.average to each column with pop weight
census_race = ['white', 'black', 'api', 'hispanic', 'others']

race_prop = FLCensus[census_race].apply(
    lambda x: np.average(x, weights=FLCensus['total.pop']))

race_prop # race proportions in Florida

# store total count from original cnames data
total_count = cnames['count'].sum()

# P(surname | race) = P(race | surname) * P(surname) / P(race) in Florida
FLVoters['name_white'] = (
       (FLVoters['pctwhite'] / 100) * (FLVoters['count'] / total_count) /
       race_prop['white'])

FLVoters['name_black'] = (
       (FLVoters['pctblack'] / 100) * (FLVoters['count'] / total_count) /
       race_prop['black'])

FLVoters['name_hispanic'] = (
       (FLVoters['pcthispanic'] / 100) * (FLVoters['count'] / total_count) /
       race_prop['hispanic'])

FLVoters['name_asian'] = (
       (FLVoters['pctapi'] / 100) * (FLVoters['count'] / total_count) /
       race_prop['api'])

FLVoters['name_others'] = (
       (FLVoters['pctothers'] / 100) * (FLVoters['count'] / total_count) /
       race_prop['others'])

# merge FLVoters with FLCensus by county and VTD using left join
FLVoters = pd.merge(FLVoters, FLCensus, on=['county', 'VTD'], how='left')

# P(surname | residence) = sum_race P(surname | race) P(race | residence)
FLVoters['name_residence'] = (
    FLVoters['name_white'] * FLVoters['white'] + 
    FLVoters['name_black'] * FLVoters['black'] + 
    FLVoters['name_hispanic'] * FLVoters['hispanic'] + 
    FLVoters['name_asian'] * FLVoters['api'] + 
    FLVoters['name_others'] * FLVoters['others'])

'''
P(race | surname, residence) = P(surname | race) * P(race | residence) /
P(surname | residence)
'''

FLVoters['pre_white'] = (FLVoters.name_white * FLVoters.white / 
                         FLVoters.name_residence)

FLVoters['pre_black'] = (FLVoters.name_black * FLVoters.black /
                         FLVoters.name_residence)

FLVoters['pre_hispanic'] = (FLVoters.name_hispanic * FLVoters.hispanic /
                            FLVoters.name_residence)

FLVoters['pre_asian'] = (FLVoters.name_asian * FLVoters.api /
                         FLVoters.name_residence)

FLVoters['pre_others'] = (1 - FLVoters.pre_white - FLVoters.pre_black -
                          FLVoters.pre_hispanic - FLVoters.pre_asian)

# relevant variables
vars1 = ['pre_white', 'pre_black', 'pre_hispanic', 'pre_asian', 'pre_others']

# Whites 
whites = FLVoters.loc[FLVoters.race == 'white'].copy()
(whites[vars1].max(axis='columns') == whites['pre_white']).mean()

# Blacks
blacks = FLVoters.loc[FLVoters.race == 'black'].copy()
(blacks[vars1].max(axis='columns') == blacks['pre_black']).mean()

# Hispanics
hispanics = FLVoters.loc[FLVoters.race == 'hispanic'].copy()
(hispanics[vars1].max(axis='columns') == hispanics['pre_hispanic']).mean()

# Asians
asians = FLVoters.loc[FLVoters.race == 'asian'].copy()
(asians[vars1].max(axis='columns') == asians['pre_asian']).mean()

# proportion of blacks among those with surname "White"
cnames['pctblack'][cnames.surname == "WHITE"]

# predicted probability of being black given residence location 
FLVoters['pre_black'][FLVoters.surname == "WHITE"].describe()

# Whites
1 - (FLVoters['race'][FLVoters[vars1].max(axis='columns') == 
                      FLVoters['pre_white']] == "white").mean()

# Blacks
1 - (FLVoters['race'][FLVoters[vars1].max(axis='columns') == 
                      FLVoters['pre_black']] == "black").mean()

# Hispanics
1 - (FLVoters['race'][FLVoters[vars1].max(axis='columns') == 
                      FLVoters['pre_hispanic']] == "hispanic").mean()

# Asians
1 - (FLVoters['race'][FLVoters[vars1].max(axis='columns') == 
                      FLVoters['pre_asian']] == "asian").mean()

# -------- Section 6.3: Random Variables and Probability Distributions ------- #

# Section 6.3.1: Random Variables

# Section 6.3.2: Bernoulli and Uniform Distributions

from scipy import stats

# uniform PDF: x = 0.5, interval = [0,1]
stats.uniform.pdf(x=0.5, loc=0, scale=1) # loc = a, scale = b-a

# uniform CDF: x = 1, interval = [-2, 2]
a = -2
b = 2
stats.uniform.cdf(x=1, loc=a, scale=b-a)

sims = 1000
p = 0.5 # success probabilities
x = stats.uniform.rvs(size=sims, loc=0, scale=1)
type(x) # a numpy array
x[:6]

y = (x <= p).astype(int)
y[:6]

y.mean() # close to success probability p, proportion of 1's vs. 0's

# Section 6.3.3: Binomial Distribution

# PMF: k = 2, n = 3, p = 0.5
stats.binom.pmf(k=2, n=3, p=0.5)

# CDF: k = 1, n = 3, p = 0.5
stats.binom.cdf(k=1, n=3, p=0.5)

# number of voters who turn out 
voters = np.array([1000, 10000, 100000])
stats.binom.pmf(voters/2, n=voters, p=0.5)

# Section 6.3.4: Normal Distribution

# plus minus 1 standard deviation from the mean
stats.norm.cdf(1) - stats.norm.cdf(-1)

# plus minus 2 standard deviations from the mean
stats.norm.cdf(2) - stats.norm.cdf(-2)

mu = 5
sigma = 2

# plus minus 1 standard deviation from the mean
(stats.norm.cdf(mu + sigma, loc=mu, scale=sigma) - 
 stats.norm.cdf(mu - sigma, loc=mu, scale=sigma))

# plus minus 2 standard deviations from the mean
(stats.norm.cdf(mu + 2*sigma, loc=mu, scale=sigma) - 
 stats.norm.cdf(mu - 2*sigma, loc=mu, scale=sigma))

# Replicate model from 4.2.5
pres08 = pd.read_csv('pres08.csv')

# import pres12 from the PREDICTION folder
pres12 = pd.read_csv('../PREDICTION/pres12.csv')

# merge the two elections by state
pres = pd.merge(pres08, pres12, on='state')

# Use the scipy zscore function to standardize Obama's vote share
# Set ddof=1 to ensure the standard deviation denominator is n-1
pres['Obama2008_z'] = stats.zscore(pres['Obama_x'], ddof=1)
pres['Obama2012_z'] = stats.zscore(pres['Obama_y'], ddof=1)

'''
Note: In chapter 4, we built a function to calculate the z-score, which used
the pandas .std() method. The default ddof=1 for the pandas method. By 
contrast, the default ddof=0 for the numpy std function and the scipy zscore 
function.
'''

import statsmodels.formula.api as smf

fit1 = smf.ols('Obama2012_z ~ -1 + Obama2008_z', data=pres).fit()

e = fit1.resid

# z-score of residuals
e_zscore = stats.zscore(e, ddof=1) 

# alternatively, we can divide the residuals by the standard deviation
e_zscore = e / np.std(e, ddof=1)

# Plot a histogram and Q-Q plot of the standardized residuals

## First, calculate some inputs for the plots
x = np.arange(-3, 3, 0.01)
x_pdf = stats.norm.pdf(x) # PDF of x

## Find quantiles for Q-Q plot using scipy.stats.probplot
quantiles = stats.probplot(e_zscore)
osm = quantiles[0][0] # ordered statistic medians (theoretical quantiles)
osr = quantiles[0][1] # ordered statistic ranks (sample quantiles)

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Histogram of residuals
sns.histplot(e_zscore, stat='density', color='gray', ax=axs[0]).set(
    xlabel='Standardized residuals',
    title='Distribution of Standardized Residuals')

# Overlay the normal density 
sns.lineplot(x=x, y=x_pdf, color='black', ax=axs[0])

# Q-Q plot
sns.scatterplot(x=osm, y=osr, color='gray', ax=axs[1]).set(
    xlabel='Theoretical quantiles', ylabel='Sample quantiles',
    title='Normal Q-Q Plot', xlim=(-3.2, 3.2), ylim=(-3.2, 3.2))

# 45-degree line
axs[1].axline((0, 0), slope=1, color='black', linestyle='--')

'''
Note that we could have used `probplot` to create a Q-Q plot directly by 
passing a plot or an axis to the `plot` argument. However, obtaining the 
quantiles enables us to customize the plot a bit more.
'''

# e is a pandas series; we can use the pandas .std() method
e_sd = e.std()
e_sd

CA_2008 = pres['Obama2008_z'][pres['state'] == 'CA']
CA_2008

# CA_2008 is a series with index 4; extract the value using .iloc
CA_mean2012 = fit1.params * CA_2008.iloc[0]
CA_mean2012

# area to the right; greater than CA_2008
1 - stats.norm.cdf(CA_2008, loc=CA_mean2012, scale=e_sd)

TX_2008 = pres['Obama2008_z'][pres['state'] == 'TX']
TX_mean2012 = fit1.params * TX_2008.iloc[0]
TX_mean2012

1 - stats.norm.cdf(TX_2008, loc=TX_mean2012, scale=e_sd)

# Section 6.3.5: Expectation and Variance

# theoretical variance: p was set to 0.5 earlier
p * (1-p)

# sample variance using 'y' generated earlier through simulation
y.var(ddof=1)

# Section 6.3.6: Predicting Election Outcomes with Uncertainty

# In Progress