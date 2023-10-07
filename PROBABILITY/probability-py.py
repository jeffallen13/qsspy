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

# In Progress
