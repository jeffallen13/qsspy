# ---------------------------------------------------------------------------- #
#                                  Measurement                                 #
# ---------------------------------------------------------------------------- #

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------- Section 3.1: Measuring Civilian Victimization during Wartime ------- #

afghan = pd.read_csv('afghan.csv')

afghan['age'].describe().round(2)

afghan['educ.years'].describe().round(2)

afghan['employed'].describe().round(2)

afghan['income'].describe()

afghan['income'].value_counts(sort=False, dropna=False)

# Convert income to a categorical variable and specify levels
afghan['income'] = afghan['income'].astype('category').cat.reorder_categories(
    ['less than 2,000', '2,001-10,000', '10,001-20,000', '20,001-30,000', 
     'over 30,000']
)

afghan['income'].value_counts(sort=False, dropna=False)

pd.crosstab(afghan['violent.exp.ISAF'], afghan['violent.exp.taliban'],
            rownames=['ISAF'], colnames=['Taliban'], normalize=True)

# --------------- Section 3.2: Handling Missing Data in Pandas --------------- #

# print income data for first 10 respondents
afghan['income'].head(10)

# indicate whether respondents' income is missing
afghan['income'].isnull().head(10)

# count of missing values
afghan['income'].isnull().sum()

# proportion of missing values
afghan['income'].isnull().mean()

x = pd.Series([1, 2, 3, np.nan])

# pandas ignores missing values by default
x.mean()

# we can override the default behavior
x.mean(skipna=False)

'''
The pd.crosstab method does not have an argument for including missing values
in a contingency table. Instead, we can use the fillna method to supply a name 
for the missing values. 
'''
pd.crosstab(afghan['violent.exp.ISAF'].fillna('Nonresponse'),
            afghan['violent.exp.taliban'].fillna('Nonresponse'), 
            rownames=['ISAF'], colnames=['Taliban'], normalize=True)

# listwise deletion
afghan_sub = afghan.dropna()

afghan_sub.shape[0]

afghan['income'].dropna().shape[0]

# ----------- Section 3.3: Visualizing the Univariate Distribution ----------- #

# Section 3.3.1: Bar Plot

# a vector of proportions to plot
ISAF_ptable = (afghan['violent.exp.ISAF'].
               value_counts(normalize=True, dropna=False).reset_index())

ISAF_ptable

# add a response column for plotting convenience
ISAF_ptable['response'] = ['No harm', 'Harm', 'Nonresponse']

# plot using the catplot family and kind='bar'
sns.catplot(
    data=ISAF_ptable, x='response', y='proportion', color='gray',
    kind='bar', estimator=sum, height=4, aspect=1.5
).set(title='Civilian victimization by the ISAF', 
      xlabel='Response category', ylabel='Proportion of the respondents',
      ylim=(0, 0.7))

'''
Notice, we use estimator=sum because seaborn bar plots aggregate the data by
a given function. The default aggregation function is mean. Since we have
already calculated proportions, we can use sum to ensure there is no further
aggregation. Another strategy for creating the bar plot is to use the mean 
aggregation directly on the original data frame categories. 

Additionally, we set the height and aspect ratios directly. The default height
is 5 inches and the default aspect ratio is 1. The aspect ratio is the ratio of
the width to the height. Therefore, the default width is 5 inches.
'''

# repeat the same for the Taliban
Taliban_ptable = (afghan['violent.exp.taliban'].
                   value_counts(normalize=True, dropna=False).reset_index())

Taliban_ptable['response'] = ['No harm', 'Harm', 'Nonresponse']

sns.catplot(
    data=Taliban_ptable, x='response', y='proportion', color='gray',
    kind='bar', estimator=sum, height=4, aspect=1.5
).set(title='Civilian victimization by the Taliban', 
      xlabel='Response category', ylabel='Proportion of the respondents',
      ylim=(0, 0.7))

# Section 3.3.2: Histogram

sns.displot(
    data=afghan, x='age', stat='density', color='gray', 
    height=4, aspect=1.5
).set(title="Distribution of respondents' age", xlabel='Age')

# histogram of education
# use binrange and binwidth to control the bins
sns.displot(
    data=afghan, x='educ.years', stat='density', color='gray', 
    binrange=(-0.5, 18.5), binwidth=1, height=4, aspect=1.5
).set(title="Distribution of respondents' education", 
      xlabel='Years of education')

# add a vertical line representing the median
plt.axvline(x=afghan['educ.years'].median(), color='black', linestyle='--')

# add a text label for the median
plt.text(x=1.5, y=0.5, s='median')


# Section 3.3.3: Box Plot

# convert province to a categorical variable
# not necessary for plotting, but useful for other analyses
afghan['province'] = afghan['province'].astype('category')

sns.catplot(
    data=afghan, x='province', y='educ.years', kind='box', color='gray',
    height=4, aspect=1.5
).set(title='Education by province', xlabel='', ylabel='Years of education')

afghan.groupby('province')['violent.exp.taliban'].mean()

afghan.groupby('province')['violent.exp.ISAF'].mean()

# Section 3.3.4: Saving Plots

# Option 1: Save via point-and-click in IDE

# Option 2: Run plot code plus plt.savefig()

sns.catplot(
    data=afghan, x='province', y='educ.years', kind='box', color='gray',
    height=4, aspect=1.5
).set(title='Education by province', xlabel='', ylabel='Years of education')

plt.savefig('education-by-province.png', bbox_inches='tight')

plt.close() # preventing plot from re-displaying

# ----------------------- Section 3.4: Survey Sampling ----------------------- #

# Section 3.4.1: The Role of Randomization 

# load village data 
afghan_village = pd.read_csv('afghan-village.csv')

# add a more descriptive variable for survey status to aid plotting 
afghan_village['village_surveyed_desc'] = (
    np.where(afghan_village['village.surveyed']==1, 'Sampled', 'Nonsampled')
    )

# boxplots for altitude 
sns.catplot(
    data=afghan_village, x='village_surveyed_desc', y='altitude', kind='box',
    color='gray', height=4, aspect=1.5
).set(ylabel='Altitude (meters)', xlabel='')


# add the natural log of population to the data frame
afghan_village['log_pop'] = np.log(afghan_village['population'])

# boxplots for log population 
sns.catplot(
    data=afghan_village, x='village_surveyed_desc', y='log_pop', kind='box',
    color='gray', height=4, aspect=1.5
).set(ylabel='Log population', xlabel='')

# Section 3.4.2: Nonresponse and Other Sources of Bias 

afghan.groupby('province')['violent.exp.taliban'].apply(
    lambda x: x.isnull().mean()
)

afghan.groupby('province')['violent.exp.ISAF'].apply(
    lambda x: x.isnull().mean()
)

(afghan['list.response'][afghan['list.group'] == 'ISAF'].mean() - 
 afghan['list.response'][afghan['list.group'] == 'control'].mean())

afghan['list.group'] = (
    afghan['list.group'].astype('category').cat.reorder_categories(
        ['control', 'ISAF', 'taliban'])
)

pd.crosstab(afghan['list.response'], afghan['list.group'],
            colnames=['group'], rownames=['response'])


# --------------- Section 3.5: Measuring Political Polarization -------------- #

# ------------- Section 3.6: Summarizing Bivariate Relationships ------------- #

# Section 3.6.1: Scatter Plot

congress = pd.read_csv('congress.csv')

congress.head()

congress.dtypes

# store some plotting parameters for re-use
xlab='Economic liberalism/conservatism'
ylab='Racial liberalism/conservatism'
lim=(-1.5, 1.5)

# scatterplot for 80th congress
sns.relplot(
    data=congress.loc[(congress['congress'] == 80) & 
                      (congress['party'] != 'Other')],
    x='dwnom1', y='dwnom2', hue='party', style='party', palette=['b', 'r'],
    height=4, aspect=1.5
).set(title='80th Congress', xlabel=xlab, ylabel=ylab, xlim=lim, ylim=lim)

# scatterplot for 112th congress
sns.relplot(
    data=congress.loc[(congress['congress'] == 112) & 
                      (congress['party'] != 'Other')],
    x='dwnom1', y='dwnom2', hue='party', style='party', palette=['b', 'r'],
    height=4, aspect=1.5
).set(title='112th Congress', xlabel=xlab, ylabel=ylab, xlim=lim, ylim=lim)

# Find the median for combinations of party and congress
dwn1_med = (congress.loc[congress.party != 'Other'].
            groupby(['party', 'congress'])['dwnom1'].median().reset_index())

sns.relplot(
    data=dwn1_med, x='congress', y='dwnom1', hue='party', kind='line',
    palette=['b', 'r'], height=4, aspect=1.5
).set(ylim=(-1, 1), xlabel='Congress', 
      ylabel='DW-NOMINATE score (1st dimension)')

# Section 3.6.2: Correlation

gini = pd.read_csv('USGini.csv')

'''
Calculate the difference between the Republican and Democratic medians.

pandas will try to align indexes in conducting vector arithmetic. Therefore, 
it is best to reset the index and drop the old one so that the indexes are the
same. An alternative is to use numpy arrays. 
'''
med_diff = (
    dwn1_med['dwnom1'][dwn1_med.party=='Republican'].reset_index(drop=True) - 
    dwn1_med['dwnom1'][dwn1_med.party=='Democrat'].reset_index(drop=True)
)

# time series plot for partisan differences 
# notice, we can feed x and y directly
sns.relplot(
    x=np.arange(1947.5, 2012.5, step=2), y=med_diff, kind='line', 
    color='black', height=4, aspect=1.5
).set(title='Political Polarization', xlabel='Year',
      ylabel='Republican median - Democratic median')

# time-series plot for Gini coefficient
sns.relplot(
    data=gini, x='year', y='gini', kind='line', color='black',
    height=4, aspect=1.5
).set(title='Income Inequality', ylabel='Gini coefficient', xlabel='Year')

'''
Correlate the partisan difference with the Gini coefficient. 
We need to select every other observation for the Gini starting with the second
observation. 
'''

(gini['gini'].iloc[np.arange(1, gini.shape[0], step=2)].
 reset_index(drop=True).corr(med_diff))

# Section 3.6.3: Quantile-Quantile Plot 

dem112 = congress.loc[(congress['congress'] == 112) & 
                      (congress['party'] == 'Democrat')]

rep112 = congress.loc[(congress['congress'] == 112) & 
                      (congress['party'] == 'Republican')]

sns.displot(
    data=dem112, x='dwnom2', stat='density', color='gray',
    height=4, aspect=1.5
).set(title='Democrats', xlabel='Racial liberalism/conservatism dimension',
      xlim=(-1.5, 1.5), ylim=(0, 1.75))

sns.displot(
    data=rep112, x='dwnom2', stat='density', color='gray',
    height=4, aspect=1.5
).set(title='Republicans', xlabel='Racial liberalism/conservatism dimension',
      xlim=(-1.5, 1.5), ylim=(0, 1.75))

'''
Quantile-Quantile

Seaborn does not have a built-in function for Q-Q plots. However, we can 
create a scatterplot of the quantiles of two variables. The quantiles we plot
need to be the same length. Below, we plot percentiles.
'''
quantiles = np.linspace(0, 1, 101)

demq = dem112['dwnom2'].quantile(quantiles)
repq = rep112['dwnom2'].quantile(quantiles)

sns.relplot(
    x = demq, y = repq, height=4, aspect=1.5
).set(xlabel='Democrats', ylabel='Republicans',
      title='Racial liberalism/conservatism dimension',
      ylim=(-1.5, 1.5), xlim=(-1.5, 1.5))

plt.gca().axline((0, 0), slope=1, color='red', linestyle='--')


# -------------------------- Section 3.7: Clustering ------------------------- #

# Section 3.7.1: Numpy Arrays

# One-dimensional arrays as vectors

# create a one-dimensional numpy array

## from a list
x = np.array([10, 20, 30, 40, 50])

x

## from a sequence
y = np.arange(10, 60, 10)

y

## from random draws from a uniform distribution between 50 and 100
z = np.random.uniform(low=50, high=100, size=10)

z

# select the first observation from z
## recall, Python uses zero-based indexing
z[0]

# select the first five observations from z
# recall, Python uses "up to but not including" slicing semantics
z[0:5]

# select the fifth observation onward
z[4:]

# conduct vectorized arithmetic: multiply each element by .25
z * .25

# conduct conditional vectorized arithmetic
## if an element is above 75, multiply by .25; otherwise, multiply by .75
np.where(z > 75, z * .25, z * .75)

# obtain the sum of the elements
z.sum()

# obtain the mean of the elements
z.mean()

# Two-dimensional arrays as matrices

# create a two-dimensional numpy array from a range
mat = np.arange(0, 10).reshape(5, 2)

mat

# select the first row
mat[0]

# select the second column
mat[:,1]

# select the first two rows and the second column
mat[0:2, 1]

# calculate the sum of the columns
mat.sum(axis=0)

# calculate the mean of the rows
mat.mean(axis=1)

# calculate the standard deviation of the columns
mat.std(axis=0)


'''
A matrix generally must have the same data type for all elements. A data frame
can have different data types for each column.
'''

df = pd.DataFrame({'x': ['a', 'b', 'c'], 'y': [1, 2, 3]})

df.dtypes # contains a string and an integer

np.array(df).dtype # produces a dtype 'O' for object; in other words, a string 

# Section 3.7.2: Objects in Python

# check the object class 
type(congress)

# review an object's methods and attributes; print the first 15
dir(congress)[0:15]

# use a list comprehension to get the non-private attributes and methods
[item for item in dir(congress) if not item.startswith('_')][0:15]

# use the data frame's value_counts "method"
congress['party'].value_counts()

# review the data frame's shape "attribute"
congress.shape

# Section 3.7.3: The k-Means Algorithm

from sklearn.cluster import KMeans

dwnom80 = congress.loc[congress['congress']==80, ['dwnom1', 'dwnom2']].copy()

dwnom112 = congress.loc[congress['congress']==112, ['dwnom1', 'dwnom2']].copy()

# kmeans with two clusters

## instantiate the model with parameters
k80two = KMeans(n_clusters=2, n_init=5)
k112two = KMeans(n_clusters=2, n_init=5)

'''
Note: If you are working on Windows, you may get a warning about about memory 
leakage associated with using KMeans on Windows. The warning will likely
recommend setting the environmental variable OPM_NUM_THREADS to a certain value.
To do so, follow these steps: 
(1) Click on the Windows Search button
(2) Type "Edit the system environment variables"
(3) Select "Environment Variables"
(4) Click "New" under "User variables for <your username>"
(5) Enter "OMP_NUM_THREADS" for the variable name and '1' or the number 
recommended in the warning for the variable value
(6) Click "OK" and close the windows

'''

## fit the model to the data
k80two.fit(dwnom80)
k112two.fit(dwnom112)

## predict the clusters
k80two_labels = k80two.predict(dwnom80)
k112two_labels = k112two.predict(dwnom112)

type(k80two_labels) # numpy.ndarray

# Use a list comprehension to view the non-private methods
[item for item in dir(k80two) if not item.startswith('_')]

# final centroids
k80two.cluster_centers_
k112two.cluster_centers_

type(k112two.cluster_centers_) # numpy.ndarray

# number of observations for each cluster by party
pd.crosstab(congress['party'][congress.congress == 80], 
            k80two_labels, colnames=['cluster'])

pd.crosstab(congress['party'][congress.congress == 112],
            k112two_labels, colnames=['cluster'])

# k means with four clusters
k80four = KMeans(n_clusters=4, n_init=5)
k112four = KMeans(n_clusters=4, n_init=5)

k80four.fit(dwnom80)
k112four.fit(dwnom112)

k80four_labels = k80four.predict(dwnom80)
k112four_labels = k112four.predict(dwnom112)

# plot the centroids over the clusters using subplots
fix, ax = plt.subplots(1,1)

sns.scatterplot(
    data=dwnom80, x='dwnom1', y='dwnom2', hue=k80four_labels, legend=False,
    palette='pastel', ax=ax,
    ).set(title='80th Congress', xlabel=xlab, ylabel=ylab, xlim=lim, ylim=lim)

sns.scatterplot(
    x=k80four.cluster_centers_[:,0], y=k80four.cluster_centers_[:,1], 
    legend=False, color='black', s=100, marker='X', ax=ax,
    )

# repeat for 112th congress
fix, ax = plt.subplots(1,1)

sns.scatterplot(
    data=dwnom112, x='dwnom1', y='dwnom2', hue=k112four_labels, legend=False,
    palette='pastel', ax=ax,
    ).set(title='112th Congress', xlabel=xlab, ylabel=ylab, xlim=lim, ylim=lim)

sns.scatterplot(
    x=k112four.cluster_centers_[:,0], y=k112four.cluster_centers_[:,1], 
    legend=False, color='black', s=100, marker='X', ax=ax,
    )
