# CycleFlow

This is the Python package CycleFlow. It implements a method of the same name, which infers proliferative heterogeneity and cell cycle parameters based of thymidine analogue pulse-chase labeling data. The method is described in Jolly et al (https://doi.org/10.1101/2020.09.10.291088).


### Example

We provide an example of usage of the package in the Jupyter Notebook `example/CycleFlowExample.ipynb`.

### Prerequisites

The package was tested with with python version 3.8.5 and the following packages: 

| name  | version |  
|-------|:-------:|
| pandas| v1.1.2  | 
| numpy | v1.19.2 |  
| emcee | v3.0.2  |  
| scipy | v1.5.2  |
| corner| v2.1.0  |
| numba | v0.51.2 |

### Installation

Download or clone the repository, install the prerequisites and then `python setup.py`.
