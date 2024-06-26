# CycleFlow

This is the Python package CycleFlow. It implements a method of the same name, which can be used to analyze flow cytometry time series data. 
CycleFlow infers proliferative heterogeneity and cell cycle parameters based on thymidine analogue pulse-chase labeling. 
The method is described in [Jolly et al.](https://doi.org/10.1016/j.crmeth.2022.100315)


### Prerequisites

The package was tested with python version 3.8.5 and the following packages: 

| name  | version |  
|-------|:-------:|
| pandas| v1.1.2  | 
| numpy | v1.19.2 |  
| emcee | v3.0.2  |  
| scipy | v1.5.2  |
| corner| v2.1.0  |
| numba | v0.51.2 |

### Installation

Download or clone the repository, install the prerequisites and then from the package root directory, execute `python setup.py install`.

Alternatively, CyleFlow can be installed using pip: `pip install CycleFlow`.

### Example

We provide an example of usage of the package in the Jupyter Notebook `cycleflow/example/CycleFlowExample.ipynb`.

## Input

As detailed in the example notebook, CycleFlow reads input data from 4 CSV files:
- The mean EdU positive G1, S and G2M fractions over time and the respective standard errors of the mean (see the [example](cycleflow/example/Tet21N.csv) for the format).

- The set of parameter values to initiate the mcmc sampling ([example](cycleflow/example/InitThetaTet.csv)).

- The mean steady state fraction in G1, S and G2M gate ([example](cycleflow/example/TetsteadyState.csv)).

- The lower and upper bounds of the flat prior distributions of the parameters ([example](cycleflow/example/prior.csv)).

## Output

CycleFlow provides functions that build a posterior distribution to be sampled from by the Emcee MCMC sampler.
The main output is a sample of the posterior distribution of parameters given the data, generated by the function `sampler.run_mcmc()`.
The sampled marginal posterior distributions for the length of each cell cycle phase in hours, the growth rate and the steady state fraction in each cell cycle phase can be then obtained using the `get_cycle()` function.
