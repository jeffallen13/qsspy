# ---------------------------------------------------------------------------- #
#                                   Discovery                                  #
# ---------------------------------------------------------------------------- #

# ------------------------- Section 5.1: Textual Data ------------------------ #

# Section 5.1.1: The Disputed Authorship of ‘The Federalist Papers’

#-- Importing textual data into a DataFrame --#

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob

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
km_out = KMeans(n_clusters=k, n_init=1, random_state=42) 
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
Cluster 1: armed forces
Cluster 2: institutional design, executive, legislature
Cluster 3: state power, national government
Cluster 4: courts, law, jurisprudence
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
    vertex_size=40,
    vertex_label=florence_g.vs["name"],
    vertex_label_size=6.0,
    vertex_color='lightgray'
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
    vertex_size=close * 1500,
    vertex_label=florence_g.vs["name"],
    vertex_label_size=6.0,
    vertex_color='lightgray'
)

ax.set(title='Closeness')


fig, ax = plt.subplots(figsize=(6,6))

ig.plot(
    florence_g,
    target=ax,
    vertex_size=pd.Series(florence_g.betweenness(directed=False)) * 1.5,
    vertex_label=florence_g.vs["name"],
    vertex_label_size=6.0,
    vertex_color='lightgray'
)

ax.set(title='Betweenness')

# Section 5.2.3: Twitter-Following Network

twitter = pd.read_csv('twitter-following.csv')
senator = pd.read_csv('twitter-senator.csv')

n = senator.shape[0] # number of senators

# initialize adjacency matrix
twitter_adj = pd.DataFrame(np.zeros((n, n)), 
                           columns=senator['screen_name'], 
                           index=senator['screen_name'])

# change 0 to 1 when edge goes from node i to node j
for i in range(len(twitter)):
    twitter_adj.loc[twitter.loc[i,'following'], twitter.loc[i,'followed']] = 1

twitter_g = ig.Graph.Adjacency(twitter_adj, mode='directed')

# Section 5.2.4: Directed Graph and Centrality

senator['indegree'] = twitter_g.indegree()
senator['outdegree'] = twitter_g.outdegree()

# 5 greatest indegree
senator.sort_values(by='indegree', ascending=False).head(5)

# 5 greatest outdegree
senator.sort_values(by='outdegree', ascending=False).head(5)

# closeness for incoming and outgoing paths
senator['close_in'] = twitter_g.closeness(mode='in', normalized=False)
senator['close_out'] = twitter_g.closeness(mode='out', normalized=False)

# directed and undirected betweenness
senator['betweenness_d'] = twitter_g.betweenness(directed=True)
senator['betweenness_u'] = twitter_g.betweenness(directed=False)

fig, axs = plt.subplots(1, 2, figsize=(12,5))

sns.scatterplot(
    data=senator, x='close_in', y='close_out', ax=axs[0],
    hue='party', palette=['r', 'b', 'k'], legend=False,
    style='party', markers=['o', '^', 'X']
).set(title='Closeness', xlabel='Incoming path', ylabel='Outgoing path')

sns.scatterplot(
    data=senator, x='betweenness_d', y='betweenness_u', ax=axs[1],
    hue='party', palette=['r', 'b', 'k'], legend=False,
    style='party', markers=['o', '^', 'X']
).set(title='Betweenness', xlabel='Directed', ylabel='Undirected')

# senator PageRank
senator['pagerank'] = twitter_g.pagerank()

# save colors for plotting
v_color = np.where(senator.party=='R', 'red', 
                   np.where(senator.party=='D', 'blue', 'black'))

fig, ax = plt.subplots(figsize=(6,6))

ig.plot(
    twitter_g,
    target=ax,
    vertex_size=senator['pagerank'] * 2000,
    vertex_color=v_color,
    edge_color='lightgray',
    edge_width=0.5,
    edge_arrow_size=0.75,
)

ax.set(title='Page Rank')


def PageRank(n, A, d, pr):
    g = ig.Graph.Adjacency(A)
    deg = g.outdegree()
    for j in range(n):
        pr[j] = (1 - d) / n + d * sum(adj[:,j] * pr / deg)
    return pr

nodes = 4

# adjacency matrix with arbitrary values
adj = (np.array([0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0]).
       reshape(nodes, nodes))

# typical choice of constant
d = 0.85 

# starting values
pr = np.array([1/nodes] * nodes)

# maximum absolute difference; use value greater than threshold
diff = 100

# while loop with 0.001 as the threshold
while diff > 0.001:
    # save the previous iteration
    pr_pre = pr.copy()
    pr = PageRank(n=nodes, A=adj, d=d, pr=pr)
    diff = max(abs(pr - pr_pre))

pr

# ------------------------- Section 5.3: Spatial Data ------------------------ #

# Section 5.3.1: The 1854 Cholera Outbreak in Action

# Section 5.3.2: Spatial Data with GeoPandas

import geopandas as gpd

# read in the shapefile (.shp) of the U.S. states
# Source: U.S. Census Bureau's Cartographic Boundary Files
usa = gpd.read_file('cb_2022_us_state_500k/cb_2022_us_state_500k.shp')

type(usa) # a GeoDataFrame

# a GeoDataFrame is a pandas DataFrame with 'GeoSeries.'
usa.head() 

usa.shape

'''
The Census Bureau uses the North American Datum 1983 (NAD83) Coordinate 
Reference System (CRS)
'''
usa.crs

# focus on the continental U.S.
non_cont = ['Alaska', 'Hawaii', 'Puerto Rico', 'United States Virgin Islands',
            'Commonwealth of the Northern Mariana Islands', 'Guam', 
            'American Samoa']

usa_cont = usa.loc[~usa['NAME'].isin(non_cont)].copy().reset_index(drop=True)

usa_cont.boundary.plot(edgecolor='black', linewidth=0.5).axis('off')

# import cities data; source: Becker and others (2021)
us_cities = pd.read_csv('us_cities.csv')

# convert to GeoDataFrame
us_cities = gpd.GeoDataFrame(
    us_cities, 
    geometry=gpd.points_from_xy(us_cities['long'], us_cities['lat']),
    # specify the CRS associated with lat and long measurements
    crs='EPSG:4326'
)

us_cities.crs

# subset capitals of continental U.S. states
usa_cont_capitals = (
    us_cities.loc[(us_cities['capital']==2) & 
                  ~us_cities['country_etc'].isin(['AK', 'HI'])]
                  .copy().reset_index(drop=True)
)
    
# Re-project the usa_cont GeoDataFrame to match the CRS of the us_cities
usa_cont = usa_cont.to_crs(us_cities.crs)

usa_cont.crs

# plot capitals on top of state map
base_map = usa_cont.plot(color='white', edgecolor='black', linewidth=0.5)

usa_cont_capitals.plot(ax=base_map, markersize=usa_cont_capitals['pop']/10000)

base_map.set_axis_off()

base_map.set_title('US state capitals')

california = usa_cont.loc[usa_cont['NAME']=='California']

cal_cities = us_cities.loc[us_cities['country_etc']=='CA']

top7 = cal_cities.sort_values(by='pop', ascending=False).head(7)

# Extract the city name from the name column (i.e., remove 'CA')
top7['city_name'] = top7['name'].str[:-3]

# plot top 7 cities on top of California
cal_map = california.boundary.plot(edgecolor='black', linewidth=0.75)

top7.plot(ax=cal_map, color='black')
  
for i in range(len(top7)):
    plt.annotate(top7.iloc[i]['city_name'], 
                 (top7.iloc[i]['long'] + 0.25, top7.iloc[i]['lat']),
                 fontsize=8)

cal_map.set_axis_off()

cal_map.set_title('Largest cities in California')

# review geometric attributes of states

# geometry type
usa_cont.geom_type.head(5)

# geometries
usa_cont.geometry.head(5)

# bounds of each state
usa_cont.bounds.head(5)

# Section 5.3.3: Colors in Matplotlib

import matplotlib.colors as mcolors

# base colors with intensities on rgb scale
mcolors.BASE_COLORS

# Number of supported colors from different color palettes
print(len(mcolors.TABLEAU_COLORS))
print(len(mcolors.CSS4_COLORS))
print(len(mcolors.XKCD_COLORS))

# Colors in the CSS4 palette with Hex codes
pd.Series(mcolors.CSS4_COLORS)

red = (1, 0, 0)
green = (0, 1, 0)
blue = (0, 0, 1)

# case-insensitive hex codes
print(f'''
Red: {mcolors.to_hex(red)}
Green: {mcolors.to_hex(green)}
Blue: {mcolors.to_hex(blue)}''')

black = (0, 0, 0)
white = (1, 1, 1)

print(f'''
Black: {mcolors.to_hex(black)}
White: {mcolors.to_hex(white)}''')

purple = (0.5, 0, 0.5)
yellow = (1, 1, 0)

print(f'''
Purple: {mcolors.to_hex(purple)}
Yellow: {mcolors.to_hex(yellow)}''')

# semi-transparent blue; specify alpha (r, g, b, alpha)
blue_trans = (0, 0, 1, 0.5)
# semi-transparent black
black_trans = (0, 0, 0, 0.5)

x = [1, 1, 2, 2, 3, 3, 4, 4]
y = [1, 1.2, 2, 2.2, 3, 3.2, 4, 4.2]

colors = [black]*2 + [black_trans]*2 + [blue]*2 + [blue_trans]*2
 
# completely colored dots difficult to distinguish
# semi-transparent dots easier to distinguish
plt.figure() # open a new figure
plt.scatter(x, y, s=500, color=colors)
plt.xlim(0.5, 4.5)
plt.ylim(0.5, 4.5)

# Section 5.3.4: US Presidential Elections

pres08 = pd.read_csv('pres08.csv')

# two-party vote share
pres08['Dem'] = pres08['Obama'] / (pres08['Obama'] + pres08['McCain'])
pres08['Rep'] = pres08['McCain'] / (pres08['Obama'] + pres08['McCain'])

# assign red and blue colors based on two-party vote share
pres08['color'] = np.where(pres08['Rep'] > pres08['Dem'], 'r', 'b')

# add tuples of rgb values based on two-party vote share
pres08['color_p'] = pres08.apply(lambda x: (x['Rep'], 0, x['Dem']), axis=1)

pres08['color_p'].head(5)

fig, axs = plt.subplots(1, 2, figsize=(8,4))

# California as a blue state
california.plot(ax=axs[0], 
                color=pres08['color'].loc[pres08.state=='CA'].iloc[0])

axs[0].axis('off')

# California as a purple state
california.plot(ax=axs[1], 
                color=pres08['color_p'].loc[pres08.state=='CA'].iloc[0])

axs[1].axis('off')

# merge the GeoDataFrame and the colors from pres08 on state abbreviations
usa_cont = pd.merge(
    usa_cont, pres08[['state', 'color', 'color_p']],
    left_on='STUSPS', right_on='state', how='left'
).drop('state', axis='columns')

usa_cont.columns

fig, axs = plt.subplots(1, 2, figsize=(12,6))

usa_cont.plot(ax=axs[0], color=usa_cont['color'], edgecolor='black', 
              linewidth=0.5).axis('off')

usa_cont.plot(ax=axs[1], color=usa_cont['color_p'], edgecolor='black', 
              linewidth=0.5).axis('off')

# Section 5.3.5: Expansion of Walmart

walmart = pd.read_csv('walmart.csv')

walmart.head()

walmart['type'].value_counts()

# create store_type column for easier plotting
walmart['store_type'] = np.where(
    walmart['type']=='Wal-MartStore', 'Store',
    np.where(walmart['type']=='SuperCenter', 'Supercenter', 'Distribution')
)

# convert to categorical and reorder categories
walmart['store_type'] = (
    walmart['store_type'].astype('category').cat.reorder_categories(
        ['Store', 'Supercenter', 'Distribution'])
) 

# add marker size column
walmart['msize'] = np.where(walmart['store_type']=='Distribution', 30, 10)

# convert to GeoDataFrame
walmart = gpd.GeoDataFrame(
    walmart, 
    geometry=gpd.points_from_xy(walmart['long'], walmart['lat']),
    crs='EPSG:4326'
)

# define colors and transparency
store = (1, 0, 0, 1/3)
supercenter = (0, 1, 0, 1/3)
distribution = (0, 0, 1, 1/3)

# plot Walmart locations on top of state map
usa_map = usa_cont.plot(color='white', edgecolor='black', linewidth=0.5)

walmart.plot(ax=usa_map, column='store_type', categorical=True, legend=True,
             markersize=walmart['msize'],
             # define custom colormap 
             cmap=mcolors.ListedColormap([store, supercenter, distribution]))

usa_map.set_axis_off()

# Section 5.3.6: Animation in Matplotlib

# convert 'opendate' to datetime
walmart['opendate'] = pd.to_datetime(walmart['opendate'])

# extract year
walmart['year'] = walmart['opendate'].dt.year

# define a function to plot Walmart locations as of year-end for a given year
def walmart_map(base_map, data, year, ax=None):
    
    # if ax is not specified, use the current axis or create a new one
    if ax is None:
        ax = plt.gca()

    # define colors and transparency
    store = (1, 0, 0, 1/3)
    supercenter = (0, 1, 0, 1/3)
    distribution = (0, 0, 1, 1/3)
    
    walmart_sub = data.loc[data['year'] <= year]

    base_map.plot(ax=ax, color='white', edgecolor='black', linewidth=0.5)

    walmart_sub.plot(
        ax=ax, column='store_type', categorical=True, legend=False,
        markersize=walmart['msize'],
        cmap=mcolors.ListedColormap([store, supercenter, distribution]))

    ax.set_axis_off()

    ax.set_title(f'{year}')


fig, axs = plt.subplots(2, 2, figsize=(12,6))

walmart_map(usa_cont, walmart, 1975, ax=axs[0,0])

walmart_map(usa_cont, walmart, 1985, ax=axs[0,1])

walmart_map(usa_cont, walmart, 1995, ax=axs[1,0])

walmart_map(usa_cont, walmart, 2005, ax=axs[1,1])

## Animation using FuncAnimation
# from matplotlib.animation import FuncAnimation

# years = range(walmart['year'].min(), walmart['year'].max() + 1)

# fig, ax = plt.subplots()

# ani = FuncAnimation(
#     fig, lambda year: walmart_map(usa_cont, walmart, year, ax), 
#     frames=years, repeat=False
# )

# ani.save('walmart.html', writer='html', fps=2)


# -------------------------------- References -------------------------------- #

'''
Becker, Richard A., Allan R. Wilks, Ray Brownrigg, Thomas P. Minka, and Alex
Deckmyn. 2021. maps: Draw Geographical Maps. R package version 3.4.0. Original 
S code by Richard A. Becker and Allan R. Wilks. R version by Ray Brownrigg. 
Enhancements by Thomas P Minka and Alex Deckmyn. 
https://CRAN.R-project.org/package=maps.
'''
