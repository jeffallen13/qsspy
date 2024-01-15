# ---------------------------------------------------------------------------- #
#                                 Introduction                                 #
# ---------------------------------------------------------------------------- #

# --------------------- Section 1.1: Overview of the Book -------------------- #

# --------------------- Section 1.2: How to Use this Book -------------------- #

# -------------- Section 1.3: Introduction to Python and Pandas -------------- #

# Section 1.3.1: Arithmetic Operations: Python as a Calculator 

5 + 3

5 - 3

5 / 3

5 ** 3

5 * (10 - 3)

from math import sqrt

sqrt(4)

# Section 1.3.2: Modules and Packages 

# earlier, we imported the sqrt function from the math module

# we can import the whole module 
import math

# use the dot notation to access functions 
math.sqrt(4)
math.log(1)

'''
External packages are typically installed from public repositories, such as the 
Python Package Index (PyPI) and conda-forge, as follows: 

- From PyPI: pip install <package_name>
- From conda-forge: conda install <package_name>

A good practice is to install required packages into a virtual environment.
'''

# import the external package Pandas and give it the conventional alias `pd`
import pandas as pd

# Section 1.3.3: Python Scripts

'''
This is the start of a Python script
The heading provides some information about the file

File name: testing_arithmetic.py
Author: <Your Name>
Date created: <Date>
Purpose: Practicing basic math commands and commenting in Python
'''

import math

5-3 # What is 5 minus 3?
5/3
5**3
5 * (10 - 3) # A bit more complex

# This function returns the square root of a number
math.sqrt(4)

# Section 1.3.4: Variables and Objects

# assign the sum of 5 and 3 to the variable result
result = 5 + 3
result

print(result)

result = 5 - 3
result

kosuke = 'instructor'
kosuke

kosuke = 'instructor and author'
kosuke

Result = '5'
Result

result

# add 5 to the variable result
result+=5
result

'''
Python is an object-oriented programming (OOP) language. Everything in Python 
is an object of a certain class. Section 3.7.2 discusses classes and objects in 
more detail. While we do not cover OOP techniques in this book, we interact
with classes and objects throughout.
'''

type(result)

type(Result)

type(math.sqrt)

# Section 1.3.4: Python Data Structures: Lists, Dictionaries, Tuples, Sets

'''
Lists, dictionaries, tuples, and sets are built-in Python data structures. In
this book, we primarily use them to manipulate inputs to and outputs from 
data structures and models in external packages. However, understanding them 
is critical for effective Python use. External packages come and go. Knowing 
how to use the built-in data structures enables you to pick up new data 
analysis packages and infrastructure paradigms more easily.
''' 

#-- List --#

world_pop = [2525779, 3026003, 3691173, 4449049, 5320817, 6127700, 6916183]
world_pop

pop_first = [2525779, 3026003, 3691173]
pop_second = [4449049, 5320817, 6127700, 6916183]
pop_all = pop_first + pop_second
pop_all

# Python uses zero-based indexing
world_pop[0] # the first observation
world_pop[1] # the second observation
world_pop[-1] # the last observation

# Python uses "up to but not including" slicing semantics
world_pop[0:4] # the first four observations
world_pop[:4] # an alternative was to slice the first four observations

# How many observations are in the list?
len(world_pop)

# Select the last three observations
world_pop[4:]

# return a sequence of decades as a list using range(start, stop, step)
list(range(1950, 2011, 10)) # specify 2011 to include 2010

# A list can contain different data types
mixed_list = [10, 10.5, True, 'USA', 'Canada']
mixed_list

# Lists are mutable
mixed_list[4] = 'Mexico'
mixed_list

mixed_list.append('Canada')
mixed_list

# Assignment with mutable objects can be tricky because variables are pointers 
alt_list = mixed_list
alt_list

mixed_list.append('USA')
mixed_list

alt_list # alt_list is also updated!

# the .copy() method overrides this behavior
alt_list = mixed_list.copy() 
alt_list

mixed_list.append(10)
mixed_list

alt_list # alt_list is not updated

'''
Built-in Python data structures are generally not vectorized. For example, 
multiplying world_pop by 2 will concatenate the list twice, rather than 
multiply each element by 2. 
'''
world_pop*2

# We can use a 'list comprehension' to perform element-wise arithmetic
pop_million = [pop / 1000 for pop in world_pop]
pop_million

'''
In this book, we use pandas and numpy to conduct vectorized operations. 
Nevertheless, list comprehensions are very useful for returning observations 
that meet certain conditions. 
'''

# return strings from mixed_list
[item for item in mixed_list if isinstance(item, str)]

#-- Dictionary --#

# dictionaries are key-value pairs; they need not be ordered
world_pop_dict = {'1950': 2525779, '1960': 3026003, '1970': 3691173, 
                  '1980': 4449049, '1990': 5320817, '2000': 6127700}
world_pop_dict

world_pop_dict['1990']

# add a new key-value pair
world_pop_dict['2010'] = 6916183
world_pop_dict

# return the keys
world_pop_dict.keys()

# return the values
world_pop_dict.values()

# return the values as a list
list(world_pop_dict.values())

#-- Tuple --#

# tuples are like lists, but they are immutable
world_pop_t = (2525779, 3026003, 3691173, 4449049, 5320817, 6127700, 6916183)
world_pop_t

world_pop_t[0]
world_pop_t[4:]

# we cannot change the values of a tuple; this will raise an error:
# world_pop_t[2] = 100

#-- Set --#

# a set contains only unique values
print(mixed_list)

set(mixed_list)

# Section 1.3.5: Pandas Data Structures: Series and DataFrames

#-- Series --#

# a series is a vector with an index
world_pop_s = pd.Series(world_pop)
world_pop_s

# selection and slicing are similar to lists
world_pop_s[1]
world_pop_s[:4]

# select non-consecutive observations
world_pop_s[[0, 2]]

# select everything except the third observation
world_pop_s.drop(2)

# select everything except the last observation
world_pop_s[:-1]

# the index is flexible
world_pop_s2 = pd.Series(world_pop_dict)
world_pop_s2
world_pop_s2['1990']

# series are vectorized
pop_million = world_pop_s / 1000
pop_million

pop_rate = world_pop_s / world_pop_s[0]
pop_rate

'''
Vector arithmetic with series requires that the indices match. One way to 
ensure this is to reset the index after slicing the series. Specify
`drop=True` in `reset_index` to avoid adding the old index as a column.
'''

pop_increase = (
    world_pop_s.drop(0).reset_index(drop=True) -
    world_pop_s.drop(6).reset_index(drop=True)
)
pop_increase

percent_increase = pop_increase / world_pop_s.drop(6) * 100
percent_increase

# series have many useful methods that help perform calculations
world_pop_s.pct_change() * 100

# series are mutable
percent_increase[[0,1]] = [20, 22]
percent_increase

#-- DataFrame --#

world_pop_df = pd.DataFrame(
    # build the data frame from a dictionary of lists
    {'year': list(range(1950, 2011, 10)),
    'pop': world_pop}
)

world_pop_df

world_pop_df.columns

world_pop_df.shape

world_pop_df.describe()

# display the summary as integers
world_pop_df.describe().astype(int)

# extract the 'pop' column; returns a series
world_pop_df['pop']

# extract the first three rows (and all columns)
world_pop_df[:3]

'''
To select a mixture of rows and columns, use the `.loc` and `.iloc` methods. 
The former enables selection with labels, while the latter enables selection
with integers.
'''

# select the first three rows and the 'pop' column
world_pop_df.loc[:2, 'pop']

# select the first three rows and both columns (but switch the column order)
world_pop_df.loc[:2, ['pop', 'year']]

'''
Notice that with .loc, the last index is included. This differs from typical 
Python slicing semantics. The reason is that .loc is a label-based method of 
selection. 
'''

# select the first three rows and the 'pop' column using integers
world_pop_df.iloc[:3, 1] # now the last index is excluded

# select the first three rows and both columns (but switch the column order)
world_pop_df.iloc[:3, [1, 0]]

# take elements 1, 3, 5, ... of the 'pop' variable using step size 2
world_pop_df['pop'][::2]

# concatenate 'pop' with an NA; use numpy to generate the NA
import numpy as np

world_pop = pd.concat([world_pop_df['pop'], pd.Series([np.nan])], 
                      ignore_index=True)

world_pop

# pandas ignores NA values by default
world_pop.mean().round(2)

# we can override this behavior
world_pop.mean(skipna=False)

# Section 1.3.6: Functions and Methods

world_pop = world_pop_df['pop']
world_pop

len(world_pop)

# methods are functions that are attached to objects
world_pop.min() # access methods using the dot notation

world_pop.max()

world_pop.mean()

# round the result
world_pop.mean().round(2)

world_pop.sum() / len(world_pop)

# Use numpy to generate a sequence of decades
year = np.arange(1950, 2011, 10)
year

np.arange(start=1950, step=10, stop=2011)

# reverse sequence and convert to a series
pd.Series(np.arange(2010, 1949, -10))

world_pop.index

list(world_pop.index)

# set the index to the year
world_pop.index = year
world_pop.index
world_pop

'''
def myfunction(input1, input2, ..., inputN):
    DEFINE `output` USING INPUTS
    return output
'''

def my_summary(x): # function takes one input
    s_out = x.sum()
    l_out = len(x)
    m_out = x.mean()
    # define the output
    out = pd.Series([s_out, l_out, m_out], index=['sum', 'length', 'mean'])
    return out # end function by calling output

z = np.arange(1, 11)

my_summary(z)

my_summary(world_pop).astype(int) # return summary as integers

type(my_summary) # functions are objects

type(np.arange)

# Section 1.3.7: Loading and Saving Data

'''
Many modern IDEs enable you to set up a workspace or a project that 
automatically configures the working directory and enables you to work with 
relative paths, rather than setting the working directory manually. In cases 
where you need to manipulate the working directory, use the `os` module. 
'''

import os

# get the current working directory
# os.getcwd()

# change the working directory
# os.chdir('<path_name>')

# Import a CSV
un_pop = pd.read_csv('UNpop.csv')
un_pop

# Import a DTA
un_pop_dta = pd.read_stata('UNpop.dta')
un_pop_dta

# Pandas supports reading a wide variety of file types

# Write to a CSV
un_pop.to_csv('UNpop.csv', index=False)

# Write to a pickle; a pickle serializes the data
un_pop.to_pickle('UNpop.pkl')
