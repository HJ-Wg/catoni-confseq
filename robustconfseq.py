# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 12:51:31 2023

Implementations of
    (1) Catoni-Style Confidence Sequences
        authored by Hongjian Wang and Aaditya Ramdas
        https://arxiv.org/abs/2202.01250
    (2) Huber-Robust Confidence Sequences
        authored by Hongjian Wang and Aaditya Ramdas
        https://arxiv.org/abs/2301.09573
        AISTATS 2023 (oral)

@author: Hongjian Wang
"""

import numpy
from numpy import log, sqrt, pi, abs, exp
from toolkit import find_zero_decreasing


def phi_wide(x, p):
    if x > 0:
        return log(1 + x + abs(x)**p/p)
    else:
        return -log(1 - x + abs(x)**p/p) 

def phi_narrow(x, p):
    if x > 1:
        return log(p)
    elif x > 0:
        return - log(1 - x + abs(x)**p/p)
    elif x > -1:
        return log(1 + x + abs(x)**p/p)
    else:
        return -log(p)


class RCS_Generator:  
    '''
    Class for the Huber-robust (or non-robust Catoni-style) confidence sequence 
    that supports online inference/estimation.
    ''' 
    _data = numpy.empty(0)
    _t = 0
    _ucbs = numpy.empty(0)
    _lcbs = numpy.empty(0)
    _ts = numpy.empty(0, dtype=int)
    _lambs = numpy.empty(0)
    
    
    def __init__(self, eps, moment, alpha=.05, p=2, null=None, side='2', optimizer='BS'):
        '''
        Parameters
        ----------
        eps : float
            The distribution noise tolerance level.
        moment : float
            Upper bound on the p-th central moment (e.g. variance).
        alpha : float, optional
            Confidence parameter alpha (aka type 1 error rate). The default is .05.
        p : float, optional
            The order of moment. The default is 2 (which means "moment" is variance).
        null : float, optional
            The null hypothesis being tested. The default is None (meaning not interested in any particular null; just do estimation).
        side : char, optional
            '2' for two-sided, CS is [lcb, ucb]; null hypothesis is "true mean = mu_0";
            'l' for left one-sided, CS is (-inf, ucb]; null hypothesis is "true mean <= mu_0";
            'r' for right two-sided, CS is [lcb, inf); the null hypothesis is "true mean >= mu_0".
            The default is '2'.
        optimizer : string, optional
            root finding algorithm to use. Currently only support 'BS' which is the built-in binary search.

        Raises
        ------
        ValueError
            Apart from the basic violation of value bounds (like alpha < 0), an error will be raised if
            the noise tolerance level eps is too large: p < 1 + (p-1/p)*eps.
            When this happens, the CS will always be the entire R, aka we reach the "break-down point".
            

        '''
        if p <= 1:
            raise ValueError("Moment order less than or equal to 1.")
        if moment <= 0:
            raise ValueError("Negative moment.")
        if eps < 0:
            raise ValueError("Negative robustness parameter.")
        if p < 1 + (p-1/p)*eps:
            raise ValueError("Robustness parameter too large; CS always spans the entire real line.")
        if alpha <= 0 or alpha >= 0.5:
            raise ValueError("Confidence parameter alpha not in (0, 0.5).")
        if not side in { 2, '2', 'l', 'r' }:
            raise ValueError("'side' must be '2', 'l', or 'r'")
        if side == 2:
            side = '2'
            
        self._p = p
        self._mmt = moment
        self._eps = eps
        self._alpha = alpha
        
        if eps == 0: # decreasing lambda seq when noiseless & unbounded phi
            self._default_lamb = lambda t: min(1,0.5 * (2*p*log(2/alpha)/(t*moment))**(1/p))
            self._phi = lambda t: phi_wide(t,p)
        else: # constant lambda seq & narrowest phi
            self._default_lamb = lambda t: 0.5 * (eps/moment)**(1/p)
            self._phi = lambda t: phi_narrow(t,p)
                    
        self._mu0, self._side = null, side
        # side = 'l': CS is (-inf, ucb]; the null hypothesis is "true mean <= mu_0"
        # side = 'r': CS is [lcb, inf); the null hypothesis is "true mean >= mu_0"
        # side = '2': CS is [lcb, ucb]; the null hypothesis is "true mean = mu_0"
        
        self._effective_alpha = alpha/2 if side == 2 else alpha
            
        if not null is None:
            self._logM = self._logN = 0
                
                
    def observe(self, observation, lamb=None, calculate_CS=True, verbose=False):
        '''
        Take a data point online, to calculate the confidence sequence or conduct sequential inference.

        Parameters
        ----------
        observation : float
            The raw real-valued data point $Xi$.
        lamb : float, optional
            The weight $\lambda_i$ on $Xi$. Typically decreasing when eps = 0, and constant when eps > 0.
            The default is None (meaning the built-in minimax optimal weight sequence will be used).
        calculate_CS : bool, optional
            Calculate the confidence sequence at this time or not. The default is True.
        verbose : bool, optional
            Print out the returns or not. The default is False.

        Returns
        -------
        dict
            Possibly contains:
                'lcb', 'ucb' : float
                    The lower and upper confidence bound of the confidence sequence at this time (if requested). 
                'e-value' : float.
                    An indicator of gathered evidence against the null.
                    50-50 mixture of the left null and right null e-values if side is '2'
                'p-value' : float.
                    1/e-value.
                    
        '''
        self._t += 1
        p, k, eps, efalpha, t, mu0, side = self._p, self._mmt, self._eps, self._effective_alpha, self._t, self._mu0, self._side
        
        self._data = numpy.append(self._data, observation)
        
        if lamb is None:
            lamb = self._default_lamb(t)
        lamb = abs(lamb)
        self._lambs = numpy.append(self._lambs, lamb)
        
        to_return = dict()
        
        if calculate_CS:               
            if side != 'l':
                last_lcb = observation if self._lcbs.shape == (0,) else max(self._lcbs[-1], -1e6)
                lcb = find_zero_decreasing(
                    lambda arg: sum([ self._phi(self._lambs[i]*(self._data[i] - arg)) - log(1 + (self._lambs[i]**p)*k/p + (p-1/p)*eps) for i in range(t) ]) - log(1/efalpha),
                    guess=last_lcb, radius=k**(1/p) * t**((1-p)/p))
                to_return['lcb'] = lcb
                self._lcbs = numpy.append(self._lcbs, lcb)
            if side != 'r':
                last_ucb = observation if self._ucbs.shape == (0,) else min(self._ucbs[-1], 1e6)        
                ucb = find_zero_decreasing(
                    lambda arg: sum([ self._phi(self._lambs[i]*(self._data[i] - arg)) + log(1 + (self._lambs[i]**p)*k/p + (p-1/p)*eps) for i in range(t) ]) + log(1/efalpha),
                    guess=last_ucb, radius=k**(1/p) * t**((1-p)/p) )
                to_return['ucb'] = ucb
                self._ucbs = numpy.append(self._ucbs, ucb)
            self._ts = numpy.append(self._ts, t)
            
        
        if not mu0 is None:
            self._logM += (self._phi(lamb*(observation - mu0))) - log( 1 + lamb**p*k/p + (p-1/p)*eps )
            self._logN += (-self._phi(lamb*(observation - mu0))) - log( 1 + lamb**p*k/p + (p-1/p)*eps )
            
            self._logM, self._logN = max(min(self._logM, 80), -80), max(min(self._logN, 80), -80) # avoid overflow
            
            if side == 'l':
                to_return['e-value'] = exp(self._logM)                
            elif side == 'r':
                to_return['e-value'] = exp(self._logN)
            elif side == '2':
                to_return['e-value'] = 0.5*exp(self._logM) + 0.5*exp(self._logN)
            to_return['p-value'] = min(1, 1/to_return['e-value'])
        
        
        if verbose:
            print(to_return)
        return to_return
        

    def get_CS(self):
        '''
        Query the entire confidence sequences

        Returns
        -------
        dict
            times : numpy.ndarray
                A list of times (i.e. counts of data points) when CS is computed
            lcbs : numpy.ndarray
                A list of all lower confidence bounds
            ucbs : numpy.ndarray
                A list of all upper confidence bounds

        '''
        return {
            'times': self._ts, 'lcbs': self._lcbs, 'ucbs': self._ucbs
        }
    
    def get_e_value(self):
        if self._mu0 == None:
            raise RuntimeError("No null specified. So no e-value or p-value.")
        else:
            if self._side == 'l':
                return exp(self._logM)                
            elif self._side == 'r':
                return exp(self._logN)
            elif self._side == '2':
                return 0.5*exp(self._logM) + 0.5*exp(self._logN)
    
    def get_p_value(self):
        return min(1, 1/self.get_e_value(self))

