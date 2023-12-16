# ---------------------------------------------------------------------------- #
#                                   Discovery                                  #
# ---------------------------------------------------------------------------- #

# ------------------------- Section 5.1: Textual Data ------------------------ #

# Section 5.1.1: The Disputed Authorship of ‘The Federalist Papers’

#-- Importing textual data into a DataFrame --#

import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt

# Get a list of all txt files in the federalist directory
file_paths = glob.glob('federalist/*.txt')

# Create an empty list
file_contents = []

# Read txt files into the empty list
for file in file_paths:
    # with: open and close file automatically
    # open(file, 'r'): open file in read mode
    # assign opened file to f
    with open(file, 'r') as f:
        file_contents.append(f.read())

# Take a look at the first 100 characters of essay number 10
file_contents[9][:100]

# Create a data frame with essay number, a placeholder for author, and the text
federalist = pd.DataFrame({'fed_num': np.arange(1,86), 'author': None,
                           'text': file_contents})

# store authorship information
hamilton = ([1] + list(range(6,10)) + list(range(11, 14)) + 
            list(range(15, 18)) + list(range(21, 37)) + list(range(59, 62)) + 
            list(range(65, 86)))

madison = [10] + [14] + list(range(37, 49)) + [58]

jay = list(range(2,6)) + [64]

joint = [18, 19, 20] # Madison and Hamilton

# store conditions for authorship
conditions = [
      federalist['fed_num'].isin(hamilton),
      federalist['fed_num'].isin(madison),
      federalist['fed_num'].isin(jay),
      federalist['fed_num'].isin(joint)
]

choices  = ['Hamilton', 'Madison', 'Jay', 'Joint']

# populate the author column; assign 'Disputed' to unassigned essays
federalist['author'] = np.select(conditions, choices, 'Disputed')

federalist

federalist['author'].value_counts()

#-- Pre-processing textual data --#

import re # regular expressions
import string # string manipulation
import nltk # natural language toolkit

# Pre-process the text using regular expressions, list comprehensions, apply() 

# make lower case and remove punctuation
federalist['text_processed'] = (
    federalist['text'].apply(lambda x: "".join(
        [word.lower() for word in x if word not in string.punctuation])
    )
)

federalist[['text', 'text_processed']].head()

# download stopwords: only need to run once
# nltk.download('stopwords')

# save and inspect stopwords
stopwords = nltk.corpus.stopwords.words('english')
stopwords[:10]

stopwords[-10:] # interestingly, includes wouldn't but not would

type(stopwords)

'''
We can add to the list as appropriate. For example, 'would' is included in 
many stopword dictionaries. 
'''
stopwords.append('would')

# instantiate the Porter stemmer to stem the words
ps = nltk.PorterStemmer()

'''
It is more efficient to define a function to apply to the text column than to 
use a lambda function for every step. 
'''
def preprocess_text(text):
    # make lower case
    text = text.lower()
    # remove punctuation
    text = "".join([word for word in text if word not in string.punctuation])
    # remove numbers 
    text = re.sub('[0-9]+', '', text)
    # create a list of individual tokens, removing whitespace
    tokens = re.split('\W+', text)
    # remove stopwords
    tokens = [word for word in tokens if word not in stopwords]
    # remove any empty strings associated with trailing spaces
    tokens = [word for word in tokens if word !='']
    # finally, stem each word
    tokens = [ps.stem(word) for word in tokens]
    return tokens

# apply function to the text column; no need for lambda with a named function
federalist['text_processed'] = federalist['text'].apply(preprocess_text)

federalist[['text', 'text_processed']].head()

# each element of the text_processed column is a list of tokens
type(federalist['text_processed'][0])

# compare the pre-processed text to the original text for essay number 10
federalist['text_processed'][9][:15]

federalist['text'][9][:100]

# Section 5.1.2: Document-Term Matrix

from sklearn.feature_extraction.text import CountVectorizer

'''
Instantiate the CountVectorizer and pass the preprocess_text function to the
analyzer argument.
'''
count_vect = CountVectorizer(analyzer=preprocess_text)

# transform the text_processed column into a document-term matrix
dtm = count_vect.fit_transform(federalist['text'])

# the dtm is a sparse matrix
type(dtm)

# convert the sparse matrix to a dense matrix and store in a DataFrame
dtm_mat = pd.DataFrame(dtm.toarray(), 
                       columns=count_vect.get_feature_names_out())

dtm_mat.iloc[:,:10].head()

# Section 5.1.3: Topic Discovery

from wordcloud import WordCloud

essay_12 = dtm_mat.iloc[11,:]
essay_24 = dtm_mat.iloc[23,:]

# Essay 12 word cloud
wordcloud_12 = WordCloud(
    width=800, height=400, background_color ='white'
).generate_from_frequencies(essay_12)

# Essay 24 word cloud
wordcloud_24 = WordCloud(
    width=800, height=400, background_color ='white'
).generate_from_frequencies(essay_24)

# plot word clouds vertically
fig, axs = plt.subplots(2, 1, figsize=(8,8))

axs[0].imshow(wordcloud_12)
axs[0].axis('off')
axs[0].set_title('Essay 12')

axs[1].imshow(wordcloud_24)
axs[1].axis('off')
axs[1].set_title('Essay 24')

# Import the tf-idf vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Create a tf-idf dtm following the same steps as before 
tfidf_vect = TfidfVectorizer(analyzer=preprocess_text)

dtm_tfidf = tfidf_vect.fit_transform(federalist['text'])

dtm_tfidf_mat = pd.DataFrame(dtm_tfidf.toarray(), 
                             columns=tfidf_vect.get_feature_names_out())

# 10 most important words for Paper No. 12
dtm_tfidf_mat.iloc[11,:].sort_values(ascending=False).head(10)

# 10 most important words for Paper No. 24
dtm_tfidf_mat.iloc[23,:].sort_values(ascending=False).head(10)

from sklearn.cluster import KMeans

'''
subset The Federalist papers written by Hamilton using the author column of 
the federalist DataFrame
'''
dtm_tfidf_hamilton = dtm_tfidf_mat[federalist['author']=='Hamilton']

k = 4 # number of clusters
# instantiate the KMeans object; set random_state for reproducibility
km_out = KMeans(n_clusters=k, n_init=1, random_state=1234) 
# fit the model
km_out.fit(dtm_tfidf_hamilton) 

# check convergence; number of iterations may vary
km_out.n_iter_

# create data frame from the cluster centers
centers = pd.DataFrame(km_out.cluster_centers_, 
                       columns=dtm_tfidf_hamilton.columns)

# extract Hamilton's papers from the federalist DataFrame
hamilton_df = (federalist.loc[federalist['author']=='Hamilton']
               .copy().reset_index(drop=True))

km_out.labels_ # cluster labels

# add the cluster labels + 1 to the Hamilton DataFrame
hamilton_df['cluster'] = km_out.labels_ + 1

hamilton_df.head()

# store cluster numbers
clusters = np.arange(1, k+1)

# loop through the clusters and print the 10 most important words
for i in range(len(clusters)):
    print(f'CLUSTER {clusters[i]}')
    print('Top 10 words:')
    print(centers.iloc[i].sort_values(ascending=False).head(10))
    # store the essay numbers associated with each cluster
    essays = hamilton_df.loc[hamilton_df['cluster']==clusters[i], 'fed_num']
    print(f'Federalist Papers: {list(essays)}')
    print('\n')

'''
A few themes that emerge:
Cluster 1: courts, law, jurisprudence
Cluster 2: state power, tax, revenue
Cluster 3: institutional design, executive, legislature
Cluster 4: state power, national government
'''

# Section 5.1.4: Authorship Prediction

import statsmodels.formula.api as smf

'''
Customize the preprocessing function to make stemming and stopword removal
optional and to optionally return strings instead of lists of tokens.
'''
def preprocess_text(text, remove_stopwords=True, stem=True,
                    return_string=False):
    # make lower case
    text = text.lower()
    # remove punctuation
    text = "".join([word for word in text if word not in string.punctuation])
    # remove numbers 
    text = re.sub('[0-9]+', '', text)
    # create a list of individual tokens, removing whitespace
    tokens = re.split('\W+', text)
    # remove stopwords if remove_stopwords=True
    if remove_stopwords:
        tokens = [word for word in tokens if word not in stopwords]
    # remove any empty strings associated with trailing spaces
    tokens = [word for word in tokens if word !='']
    # stem each word if stem=True
    if stem:
        tokens = [ps.stem(word) for word in tokens]
    if return_string:
        return ' '.join(tokens)
    else:
        return tokens

# If we preprocess before using the CountVectorizer, it expects strings
federalist['text_processed_v2'] = (
    federalist['text'].apply(lambda x: preprocess_text(
        x, stem=False, remove_stopwords=False, return_string=True))
)

federalist['text_processed_v2'].head()

# this time, do not pass the preprocess_text function to the analyzer argument
count_vect1 = CountVectorizer()

dtm1 = count_vect1.fit_transform(federalist['text_processed_v2'])

dtm1_mat = pd.DataFrame(dtm1.toarray(), 
                        columns=count_vect1.get_feature_names_out())

# term frequency per 1000 words
row_sums = dtm1_mat.sum(axis='columns')
tfm = dtm1_mat.div(row_sums, axis='rows')*1000

# words of interest
words = ['although', 'always', 'commonly', 'consequently', 'considerable',
         'enough', 'there', 'upon', 'while', 'whilst']

# select only these words
tfm = tfm.loc[:, words]

# average among Hamilton/Madison essays
tfm_ave = (pd.concat(
    [tfm.loc[federalist['author']=='Hamilton'].sum(axis='rows') / len(hamilton),
     tfm.loc[federalist['author']=='Madison'].sum(axis='rows') / len(madison)],
     axis=1
)).T # transpose 

tfm_ave

# add tfm to the federalist data frame
federalist = pd.concat([federalist, tfm], axis=1)

model_words = ['upon', 'there', 'consequently', 'whilst']

select_vars = ['fed_num', 'author'] + model_words

hm_data = (
    federalist.loc[federalist['author'].isin(['Hamilton', 'Madison']),
                   select_vars]
).copy().reset_index(drop=True)

hm_data['author_y'] = np.where(hm_data['author'] == "Hamilton", 1, -1)

hm_data.head()

hm_model = 'author_y ~ upon + there + consequently + whilst'

hm_fit = smf.ols(hm_model, data=hm_data).fit()

hm_fit.params

hm_fitted = hm_fit.fittedvalues

np.std(hm_fitted)

# Section 5.1.5: Cross-Validation

# proportion of correctly classified essays for Hamilton
(hm_fitted[hm_data['author_y']==1] > 0).mean()

# proportion of correctly classified essays for Madison
(hm_fitted[hm_data['author_y']==-1] < 0).mean()

n = len(hm_data)

# a container vector
hm_classify = np.zeros(n)

for i in range(n):
    # fit the model to the data after removing the ith observation
    sub_fit = smf.ols(hm_model, data=hm_data.drop(i)).fit()
    # predict the authorship for the ith observation
    # [[]] ensures the row remains a data frame
    # finally, extract value from prediction Series without index
    hm_classify[i] = sub_fit.predict(hm_data.iloc[[i]]).iloc[0]

# proportion of correctly classified essays for Hamilton
(hm_classify[hm_data['author_y']==1] > 0).mean()

# proportion of correctly classified essays for Madison
(hm_classify[hm_data['author_y']==-1] < 0).mean()

# subset essays with disputed authorship
disputed = federalist.loc[federalist['author']=='Disputed', select_vars]

# predict the authorship of the disputed essays
pred = hm_fit.predict(disputed)
pred

# prepare the data for plotting
hm_data['pred'] = hm_fitted
disputed['pred'] = pred

plot_vars = ['fed_num', 'author', 'pred']

plot_data = pd.concat([hm_data[plot_vars], disputed[plot_vars]], 
                      axis=0, ignore_index=True)

import seaborn as sns

sns.set_style('ticks')

(sns.relplot(
    data=plot_data, x='fed_num', y='pred', hue='author', style='author', 
    palette=['red', 'blue', 'black'], markers = ['s', 'o', '^'],
    height=4, aspect=1.5
).set(xlabel='Federalist Papers', ylabel='Predicted values')
.despine(right=False, top=False)._legend.set_title('Author'))

plt.axhline(y=0, color='black', linestyle='--', linewidth=1)

# ------------------------- Section 5.2: Network Data ------------------------ #

# Section 5.2.1: Marriage Network in Renaissance Florence

florence = pd.read_csv('florentine.csv', index_col='FAMILY')

florence.iloc[:5,:5]

florence.sum(axis='columns')

# Section 5.2.2: Undirected Graph and Centrality Measures

# Note: if installing from conda forge, install 'python-igraph'
import igraph as ig

florence_g = ig.Graph.Adjacency(florence, mode='undirected')

# plot the graph
fig, ax = plt.subplots(figsize=(6,6))

ig.plot(
    florence_g,
    target=ax,
    vertex_size=0.6,
    vertex_label=florence_g.vs["name"],
    vertex_label_size=6.0,
    vertex_color='gray'
)

florence_g.degree() # a list

florence_g.vs['name']

pd.Series(florence_g.degree(), index=florence_g.vs['name'])

pd.Series(florence_g.closeness(normalized=False), index=florence_g.vs['name'])

1 / (pd.Series(florence_g.closeness(normalized=False), 
               index=florence_g.vs['name']) * 15)

pd.Series(florence_g.betweenness(directed=False), index=florence_g.vs['name'])

close = pd.Series(florence_g.closeness(normalized=False), 
                  index=florence_g.vs['name'])

close['PUCCI'] = 0

fig, ax = plt.subplots(figsize=(6,6))

ig.plot(
    florence_g,
    target=ax,
    vertex_size=close * 25,
    vertex_label=florence_g.vs["name"],
    vertex_label_size=6.0,
    vertex_color='gray',
    bbox=(0, 0, 300, 300),
    margin=20
).set(title='Closeness')


fig, ax = plt.subplots(figsize=(6,6))

ig.plot(
    florence_g,
    target=ax,
    vertex_size=pd.Series(florence_g.betweenness(directed=False)) / 50,
    vertex_label=florence_g.vs["name"],
    vertex_label_size=6.0,
    vertex_color='gray',
    bbox=(0, 0, 300, 300),
    margin=20
).set(title='Betweenness')

# Section 5.2.3: Twitter-Following Network

# In Progress