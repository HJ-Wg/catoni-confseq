# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 01:26:50 2023

Demonstration: 
    Robust vs. non-robust CS under heavy-tailed noises,
    reproducing the Figure 2 of Wang and Ramdas (2023) (https://arxiv.org/abs/2301.09573)

@author: wangh
"""

import scipy
from matplotlib import pyplot
import numpy
from robustconfseq import RCS_Generator


sample = [ ]
sum_sample = 0

stitched_ucbs = [ ]
stitched_lcbs = [ ]

time_start = 10
time_horizon = 10000
variance = 9
sd = numpy.sqrt(variance)

# robustness parameter
robust_eps = 1/variance
# so that 1 = sigma * sqrt(eps)


# geometrically spaced CS points
powers_of_1point1 = set()
pp = 2
while pp < time_horizon:
    powers_of_1point1.add(pp)
    pp = int(pp*1.1+1)
    
plot_this = lambda n : n in powers_of_1point1


rcs = RCS_Generator(eps=robust_eps, moment=variance)


for n in range(1, time_horizon + 1):
    # generate the distribution contaminated data point
    if scipy.stats.uniform.rvs() >= robust_eps:
        xn = scipy.stats.norm.rvs(loc=0, scale=sd)
    else:
        xn = scipy.stats.levy_stable.rvs(alpha=0.75, beta=0.5, loc=0, scale=sd)

    sum_sample += xn
    sample.append(xn)

    # our anytime-valid CS
    rcs.observe(xn, calculate_CS=plot_this(n))
    
    if plot_this(n):
        avg = sum_sample / n
    
        # stitched CS
        stitched_ucb = avg + 1.7 * sd * numpy.sqrt((numpy.log(numpy.log(2*n)) + 0.72 * numpy.log(10.4/.05))/n)
        stitched_lcb = avg - 1.7 * sd * numpy.sqrt((numpy.log(numpy.log(2*n)) + 0.72 * numpy.log(10.4/.05))/n)
        stitched_ucbs.append(stitched_ucb)
        stitched_lcbs.append(stitched_lcb)
        print(n, end=", ")



returns = rcs.get_CS()
n_list = returns["times"]
anytimevalid_ucbs = returns["ucbs"]
anytimevalid_lcbs = returns["lcbs"]

pyplot.figure(figsize = (3.1,3.1),tight_layout=True)
pyplot.xscale("log")  

pyplot.fill_between(n_list, anytimevalid_ucbs, anytimevalid_lcbs, color="red", alpha=0.3, label="Robust CS")

pyplot.fill_between(n_list, stitched_ucbs, stitched_lcbs, color="grey", alpha=0.6, label="Non-robust CS")

pyplot.xlabel("time")
pyplot.legend()
pyplot.grid()
pyplot.ylim([-20, 20])
pyplot.xlim([time_start, time_horizon])
pyplot.show()

