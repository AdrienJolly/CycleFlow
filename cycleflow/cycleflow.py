#-*- coding: utf-8 -*-

import numpy as np
from scipy.integrate import odeint
from scipy.sparse.linalg import eigs
import pandas
from scipy.integrate import solve_ivp
from scipy.linalg.blas import dgemm, daxpy
from numba import jit


def convert_data(df, ssdf):
    '''Converts pandas dataframe into numpy array and rearranges data for the following steps.
      
    Arguments:
    df   -- dataframe containing means labeled cells fractions and errors over time
    ssdf -- dataframe containing steady state fractions in S and G2, mean values and errors
    '''
    array = df.to_numpy()
    vector_data = np.transpose(array).reshape(1, np.size(array))[0]
    data, error = np.split(vector_data, [len(vector_data) // 2])
    data = np.append(data, ssdf['mean'])
    error = np.append(error, ssdf['error'])
    return data, error


def _make_transitions(theta):
    '''Helper function to construct a transition matrix from parameters'''
    lambda_, mu, nu = theta[:3] # transition rates in G1, S, G2
    l = abs(int(theta[4])) # number of substeps in G1
    m = 15 # number of substeps in S; fixed
    n = 15 # number of substeps in G2; fixed
    a = theta[5] # probability to enter G0 upon mitosis
    g1, s, g2, g0, size = 0, l, l+m, l+m+n, l+m+n+1 # convenience: starting indices
    trans = np.zeros((size, size))
    for i in range(g1, s):
        trans[i+1, i] = lambda_
        trans[i, i] = -lambda_
    for i in range(s, g2):
        trans[i+1, i] = mu
        trans[i, i] = -mu
    for i in range(g2, g0):
        trans[i+1, i] = nu 
        trans[i, i] = -nu
    trans[g1, g0-1] = (1. - a) * nu * 2.
    trans[g0, g0-1] = a * nu * 2.
    return trans


class log_flat_prior:
    '''Makes a log prior that corresponding to a given allowed parameter range. 
    
    The returned log prior function is not normalized.
  
    Arguments:
    min -- lower bounds for parameters
    max -- upper bounds
    '''
    def __init__(self, min, max):
          self.min, self.max = min, max
      
    def __call__(self, theta):        
      if np.logical_and(self.min < theta, theta < self.max).all():
        return 0.0
      else:
        return -np.inf


# the jit decorator speeds up sampling by using numba. If this is not desired,
# simply comment out the decorator.
@jit('f8[:](f8,f8[::1],f8[:,::1],f8[::1],f8[::1],f8,f8[:,::1])', nopython=True)
def model_jit(t, y, transitions, theta, ss_fractions, kappa, labeling):
    '''ODE model for labeled cells
  
    Arguments:
    t            -- time point
    y            -- vector of initial conditions
    transitions  -- the transition matrix
    theta        -- vector of parameters
    ss_fractions -- vector of steady state fractions in each sub-phase
    kappa        -- growth rate kappa
    labeling     -- labeling matrix
    '''
    lbl_fractions = y 
    # l is the integer number of G1 substeps, but emcee only knows how to
    # handle real params. We work around this by letting emcee see l as a real
    # parameter but using it as an integer internally. As a result, emcee will
    # generate stepwise constant posteriors for the fake real l, which is
    # slightly inefficient but correct.
    l = abs(int(theta[4]))  
    m = 15 # S substeps; fixed
    n = 15 # G2 substeps; fixed
    eps_0 = theta[8] # intial EdU labeling rate
    tau = theta[3] # EdU labeling time constant
    mu = theta[1] #transition rate 
    eps = eps_0 * np.exp(-t / tau)
  
    # update of the labeling matrix
    # labeling is passed as a function argument and updated with low-level
    # numpy functions for speed
    labeling_sub_2 = labeling[l:l+n, l:l+n]
    np.fill_diagonal(labeling_sub_2, eps)
  
    dldt2 = (transitions.dot(lbl_fractions) - kappa * lbl_fractions 
             - labeling.dot(lbl_fractions - ss_fractions))

    return dldt2


class log_likelihood:
    '''Make a log likelihood function for a given set of data. 
     
    Initialization arguments:
    tdata   -- vector of time points
    data    -- vector, mean fractions to which the model is fitted, 
            generated with function convert_data()
    dataerr -- vector, error of the means, generated with function \
            TODO: THIS IS MISSING
  
    The returned callable evaluates the likelihood as a function of theta. 

    Argument:
    theta -- vector of parameters
    '''
    def __init__(self, tdata, data, dataerr):
        self.tdata, self.data, self.dataerr = tdata, data, dataerr
    def __call__(self, theta):
    #definitition of the parameters
        l = abs(int(theta[4])) #number of substeps in G1
        m = 15 # number of substeps in G2M
        n = 15 # number of substeps in S
        a = theta[5] # probability to enter G0 upon mitosis
        earlyS = int(theta[6] * n)
        lateS = int(theta[7] * n)
        y0 = np.zeros(l+n+m+1)
        # construct the transition matrix
        transitions = _make_transitions(theta)
        # calculate the steady-growth state
        eig = np.linalg.eig(transitions)
        index = np.argmax(eig[0])
        k = eig[0][index]
        if not np.isclose(k, k.real, rtol=1e-8):
            return -np.inf
        else:
            k = k.real
            ss_fractions = np.ascontiguousarray(eig[1][:, index].real)
            ss_fractions /= np.sum(ss_fractions)
            ss_G1, ss_S, ss_G2, ss_G0 = np.split(ss_fractions, [l, l+m, l+m+n])
            ss_earlyS, ss_midS, ss_lateS = np.split(ss_S, [earlyS, -lateS])
            ss_gate_S = np.sum(ss_midS)
            ss_gate_G2 = np.sum(ss_lateS) + np.sum(ss_G2)
            # now solve the ODE system 
            labeling = np.zeros((l+n+m+1, l+m+n+1)) # allocate labeling matrix for speed
            sol = solve_ivp(model_jit, [0, self.tdata[-1]], y0, t_eval=self.tdata, 
                    args=(transitions, theta, ss_fractions, k, labeling)).y  
            fit_G1l = np.sum(sol[0:l+earlyS, :], axis=0)
            fit_G0l = sol[l+m+n, :]
            fit_G0G1l = fit_G1l + fit_G0l
            fit_Sl = np.sum(sol[l+earlyS:l+n-lateS, :], axis=0)
            fit_G2l = np.sum(sol[l+n-lateS:l+n+m, :], axis=0)
            fit = np.concatenate([fit_G0G1l, fit_Sl, fit_G2l, [ss_gate_S, ss_gate_G2]])
            chi_squared = np.sum(((self.data - fit) ** 2 / (self.dataerr) ** 2))
            return -0.5 * chi_squared


class log_posterior:
    '''Make a log-posterior function from the given likelihood and prior.

    Initialization arguments:
    likelihood --- callable that yields the log likelihood
    prior      --- callable that yields the log prior
    
    The returned callable gives the posterior as a function of theta.

    Argument:
    theta --- parameter vector
    '''
    def __init__(self, log_likelihood, log_prior):
        self.likelihood, self.prior = log_likelihood, log_prior

    def __call__(self, theta):
        lp = self.prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.likelihood(theta)

    
def get_cycle(sampled):
    '''Return cell phase length, fractions, and growth rate from a full
    posterior sample.
    
    Argument:
    sampled -- one sample of parameters
    '''
    lambda_, mu, nu, tau = sampled[:4]
    l = abs(int(sampled[4])) # number of substeps in G1
    m = 15 # number of substeps in G2M; fixed
    n = 15 # number of substeps in S; fixed
    a = sampled[5]
    earlyS, lateS = [int(sampled[i] * n) for i in [6,7]]
    result = np.zeros(8)
    result[0] = l / lambda_ # length of G1
    result[1] = m / mu # length of S
    result[2] = n / nu # length of G2
    transitions = _make_transitions(sampled)
    eig = np.linalg.eig(transitions)
    index = np.argmax(eig[0])
    result[3] = eig[0][index].real  # growth rate
    ss_fractions = eig[1][:, index].real
    ss_fractions /= np.sum(ss_fractions)
    result[4] = np.sum(ss_fractions[0:l]) #G1 fraction
    result[5] = np.sum(ss_fractions[l:l+m]) #S fraction
    result[6] = np.sum(ss_fractions[l+m:l+m+n]) #G2 fraction
    result[7] = ss_fractions.real[l+m+n] # G0 fraction
    return result

 
