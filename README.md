# Python code for *Quantitative Social Science: An Introduction*

Welcome! This repository contains Python companion guides for Kosuke Imai's [Quantitative Social Science](https://press.princeton.edu/books/paperback/9780691175461/quantitative-social-science) (QSS). 

The repository is modeled after the [QSS GitHub repository](https://github.com/kosukeimai/qss), which contains base R and tidyverse code for the original and [tidyverse](https://press.princeton.edu/books/hardcover/9780691222271/quantitative-social-science) editions of QSS, along with the datasets that are analyzed in book. 

The python scripts are available in the following file formats: .py, .ipynb, and .pdf. 

## Development Status

The Python companion guides are a work-in-progress. The table below summarizes the current development status. 

| Chapter | Status |
| --- |:---:|
| 1. Introduction | **X** |
| 2. Causality | [Complete](https://github.com/jeffallen13/qsspy/tree/main/CAUSALITY) |
| 3. Measurement | [Complete](https://github.com/jeffallen13/qsspy/tree/main/MEASUREMENT) |
| 4. Prediction| In progress |
| 5. Discovery | **X** |
| 6. Probability | **X** |
| 7. Uncertainty| **X** |

## Setup

Chapter 1 will include guidance on setting up a Python analytical environment. In the meantime, below are some recommendations for getting started.

### Installation and Packages

I recommend following the [Installation and Setup](https://wesmckinney.com/book/preliminaries#installation_and_setup) instructions in Wes McKinney's [Python for Data Analysis, 3E](https://wesmckinney.com/book/). The instructions include guidance on installing [miniconda](https://docs.conda.io/projects/miniconda/en/latest/) and installing packages from [conda-forge](https://conda-forge.org/).

The companion guides use the following packages extensively:

- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/) 
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)
- [statsmodels](https://www.statsmodels.org/stable/index.html)
- [scikit-learn](https://scikit-learn.org/stable/)

### Integrated Development Environments

There are a variety of popular integrated development environments (IDEs) for conducting data analysis in Python, including:

- [VS Code](https://code.visualstudio.com/)
- [PyCharm](https://www.jetbrains.com/pycharm/)
- [Spyder](https://www.spyder-ide.org/) 
- [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/)

The companion guides were built using the VS Code IDE and VS Code's Jupyter notebooks integration. If you elect to use VS Code, helpful documentation pages include:

- [Getting Started with Python in VS Code](https://code.visualstudio.com/docs/python/python-tutorial)
- [Jupyter Notebooks in Visual Studio Code](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)
- [Data Science in VS Code Tutorial](https://code.visualstudio.com/docs/datascience/data-science-tutorial)
- [Python Interactive Window](https://code.visualstudio.com/docs/python/jupyter-support-py)

For those coming from an R background, Spyder is similar to RStudio.

## References

>Imai, Kosuke (2017). *Quantitative Social Science: An Introduction.* Princeton University Press. 
>
>McKinney, Wes (2022). *Python for Data Analysis: Data Wrangling with pandas, NumPy, and Jupyter, 3rd Edition.* O'Reilly Media.