# Python code for *Quantitative Social Science: An Introduction*

Welcome! This repository contains Python companion guides for Kosuke Imai's [Quantitative Social Science](https://press.princeton.edu/books/paperback/9780691175461/quantitative-social-science) (QSS). 

The repository is modeled after the [QSS GitHub repository](https://github.com/kosukeimai/qss), which contains base R and tidyverse code for the original and [tidyverse](https://press.princeton.edu/books/hardcover/9780691222271/quantitative-social-science) editions of QSS, along with the datasets that are analyzed in book. 

The python scripts are available in the following file formats: `.py`, `.ipynb`, and `.pdf`. The `.py` files are best for copying and running the code locally. The`.ipynb` files offer the best web-based viewing experience, including on GitHub. The `.pdf` files are best for printing. 

## Development Status

The Python companion guides are a work-in-progress. The table below summarizes the current development status. 

| Chapter | Status |
| --- |:---:|
| 1. Introduction | In Progress |
| 2. Causality | [Complete](https://github.com/jeffallen13/qsspy/tree/main/CAUSALITY) |
| 3. Measurement | [Complete](https://github.com/jeffallen13/qsspy/tree/main/MEASUREMENT) |
| 4. Prediction| [Complete](https://github.com/jeffallen13/qsspy/tree/main/PREDICTION) |
| 5. Discovery | [Complete](https://github.com/jeffallen13/qsspy/tree/main/DISCOVERY) |
| 6. Probability | [Complete](https://github.com/jeffallen13/qsspy/tree/main/PROBABILITY) |
| 7. Uncertainty| [Complete](https://github.com/jeffallen13/qsspy/tree/main/UNCERTAINTY) |

## Setup

The companion guides focus on replicating the QSS analysis in Python. Currently, they do not contain detailed instructions on setting up a Python environment. Fortunately, there are many excellent and free resources available online that provide guidance in this respect. Below are a few recommendations.

### Installation and Packages

I recommend following the [Installation and Setup](https://wesmckinney.com/book/preliminaries#installation_and_setup) instructions in Wes McKinney's [Python for Data Analysis, 3E](https://wesmckinney.com/book/). The instructions walk readers through a few important steps: 

- Installing Python via [miniconda](https://docs.conda.io/projects/miniconda/en/latest/) 
- Working with [conda environments](https://conda.io/projects/conda/en/latest/user-guide/index.html)
- Installing packages from the [conda-forge](https://conda-forge.org/) channel

The companion guides use the following packages extensively:

- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/) 
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)
- [statsmodels](https://www.statsmodels.org/stable/index.html)
- [scikit-learn](https://scikit-learn.org/stable/)
- [scipy](https://www.scipy.org/)

The installation and package management approach outlined above is one among many options. Other popular ways to install Python and manage packages include: 

- Installing Python from the [Python Software Foundation](https://www.python.org/downloads/), using the [pip](https://pip.pypa.io/en/stable/) package manager to install packages from the [Python Package Index](https://pypi.org/) (PyPI), and managing packages with Python [virtual environments](https://docs.python.org/3/tutorial/venv.html). This approach is similar to installing miniconda and using conda, conda-forge, and conda environments. Note that these approaches are not mutually exclusive. For example, you can use pip to install packages from PyPI into a conda environment.
- Installing an [Anaconda](https://www.anaconda.com/) distribution of Python. Anaconda comes pre-loaded with many data analysis packages, including those listed above. Anaconda offers a free [individual version](https://www.anaconda.com/pricing/individuals/). 
- Using container-based approaches, such as [Docker](https://www.docker.com/). This is a more advanced workflow that is widely used for deploying applications. 

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