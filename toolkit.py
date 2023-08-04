# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 00:05:46 2023

@author: wangh
"""

# binary-find the zero of a decreasing function
def find_zero_decreasing(f, guess, radius=1, x_tolerance=None, y_tolerance=0.0001, bound=1e12):
    '''
    find the zero of a decreasing function, by binary search

    Parameters
    ----------
    f : function float -> float
        A non-increasing, real-valued function
    guess : float
        The initial center of the iterative search
    radius : float, optional
        The initial radius of search. The default is 1.
    x_tolerance : float, optional
        The maximum value of iterates difference before the search stops. Will adjust dynamically if the initial guess fails. The default is radius * 0.00001 (by leaving as None).
    y_tolerance : float, optional
        The maximum value of |f(x)| for the search to continue. The default is 0.0001.
    bound : float, optional
        The maximum of |iterate| during the search. The default is 1e12.

    Returns
    -------
    float
        The zero of f found by the algorithm. If f is positive within |x| < bound, return inf; negative, -inf.
    '''
    if x_tolerance is None:
        x_tolerance = radius * 0.00001
    
    if guess > bound:
        return float("inf")
    if guess < -bound:
        return float("-inf")
    
    upper = guess + radius
    lower = guess - radius

    # adjust to f(lower) > 0 > f(upper)
    while(f(upper) > 0):
        upper = guess + 5*radius
        lower = guess + radius
        guess = guess + 3*radius
        radius = 2*radius

    while(f(lower) < 0):
        upper = guess - radius
        lower = guess - 5*radius
        guess = guess - 3*radius
        radius = 2*radius
    
    f_upper = f(upper)
    f_lower = f(lower)

    '''
    if f_upper > 0:
        return find_zero_decreasing(f, guess + 3*radius, 2*radius)
    if f_lower < 0:
        return find_zero_decreasing(f, guess - 3*radius, 2*radius)
    '''
      
    if abs(f_upper + f_lower) < y_tolerance:
        return guess
      

    while upper - lower > x_tolerance:
        mid = (upper + lower)/2
        if f(mid) > 0:
            lower = mid
        else:
            upper = mid
      
    return (upper + lower)/2


