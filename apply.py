#!/usr/bin/env python
# encoding: utf-8
"""
nest.py
Based on
// apply.c     "LIGHTHOUSE" NESTED SAMPLING APPLICATION
// (GNU General Public License software, (C) Sivia and Skilling 2006)
//              u=0                                 u=1
//               -------------------------------------
//          y=2 |:::::::::::::::::::::::::::::::::::::| v=1
//              |::::::::::::::::::::::LIGHT::::::::::|
//         north|::::::::::::::::::::::HOUSE::::::::::|
//              |:::::::::::::::::::::::::::::::::::::|
//              |:::::::::::::::::::::::::::::::::::::|
//          y=0 |:::::::::::::::::::::::::::::::::::::| v=0
// --*--------------*----*--------*-**--**--*-*-------------*--------
//             x=-2          coastline -->east      x=2
// Problem:
//  Lighthouse at (x,y) emitted n flashes observed at D[.] on coast.
// Inputs:
//  Prior(u)    is uniform (=1) over (0,1), mapped to x = 4*u - 2; and
//  Prior(v)    is uniform (=1) over (0,1), mapped to y = 2*v; so that
//  Position    is 2-dimensional -2 < x < 2, 0 < y < 2 with flat prior
//  Likelihood  is L(x,y) = PRODUCT[k] (y/pi) / ((D[k] - x)^2 + y^2)
// Outputs:
//  Evidence    is Z = INTEGRAL L(x,y) Prior(x,y) dxdy
//  Posterior   is P(x,y) = L(x,y) / Z estimating lighthouse position
//  Information is H = INTEGRAL P(x,y) log(P(x,y)/Prior(x,y)) dxdy

Translated into Python by Daniel O'Donovan on 2009-10-07.

"""

import math

import nest

# define the number of objects n
n = 100

# define the max number of iterates
MAX = 1000

xlim = [-2, 2]
ylim = [ 0, 2]

Object = { 'u':0.0, 'v':0.0, 'x':0.0, 'y':0.0, 'logL':0.0, 'logWt':0.0 }

def LogLhood( x, y ):
    """ log Likelihood function """
    dd = [  4.73,  0.45, -1.73,  1.09,  2.19,  0.12,
            1.31,  1.00,  1.32,  1.07,  0.86, -0.49, -2.59,  1.73,  2.11,
            1.61,  4.98,  1.71,  2.23,-57.20,  0.96,  1.25, -1.56,  2.45,
            1.19,  2.17,-10.66,  1.91, -4.16,  1.92,  0.10,  1.98, -2.51,
            5.55, -0.47,  1.91,  0.95, -0.78, -0.84,  1.72, -0.01,  1.48,
            2.70,  1.21,  4.41, -4.79,  1.33,  0.81,  0.20,  1.58,  1.29,
           16.19,  2.75, -2.38, -1.79,  6.50,-18.53,  0.72,  0.94,  3.64,
            1.94, -0.11,  1.57,  0.57 ]
    logL = 0.0
    for d in dd:
        logL += math.log( (y / math.pi) / (math.pow( d - x, 2 ) + math.pow( y, 2 )))
    return logL

def prior( Object ):
    """ Set Object according to prior """
    Object['u'] = nest.uniform()
    Object['v'] = nest.uniform()
    Object['x'] = 4. * Object['u'] - 2.0
    Object['y'] = 2. * Object['v']
    Object['logL'] = LogLhood( Object['x'], Object['y'] )
    
def Explore( Object, logLstar ):
    """ Evolve object within likelihood constraint """
    step = 0.1  # initial guess suitable step-size in (0,1)
    m = 20      # MCMC counter (pre-judged # steps)
    accept = 0  # # MCMC acceptances
    reject = 0  # # MCMC rejections

    Try = Object.copy() # trial object

    while m > 0:
        m -= 1

        Try['u'] = Object['u'] + step * (2. * nest.uniform() - 1. ) # |move| < step
        Try['v'] = Object['v'] + step * (2. * nest.uniform() - 1. ) # |move| < step
        Try['u'] -= math.floor( Try['u'] )   # wraparound to stay within (0,1)
        Try['v'] -= math.floor( Try['v'] )   # wraparound to stay within (0,1)
        Try['x'] = 4.0 * Try['u'] - 2.0 # map to x
        Try['y'] = 2.0 * Try['v']       # map to y
        Try['logL'] = LogLhood( Try['x'], Try['y'] )    # trial likelihood value

        # Accept if and only if within hard likelihood constraint
        if Try['logL'] > logLstar:
            Object = Try.copy()
            accept += 1
        else:
            reject += 1
        # Refine step-size to let acceptance ratio converge around 50%
        if accept > reject: step *= math.exp(1.0 / accept)
        if accept < reject: step /= math.exp(1.0 / reject)

    return Object

def Results(Samples, nest, logZ):
    """ Posterior properties, here mean and stddev of x,y 
            -   Evidence (= total weight = SUM[Samples] Weight)
    """
    x = xx = 0.0    # 1st and 2nd moments of x
    y = yy = 0.0    # 1st and 2nd moments of y
    w = ww = 0.0    # Proportional weight

    for i in xrange( nest ):
        w   = math.exp( Samples[i]['logWt'] - logZ )
        ww += w
        # print '(%f - %f) = %f - %f' % (w, Samples[i]['logWt'], Samples[i]['x'], Samples[i]['y'])
        x   += w * Samples[i]['x']
        xx  += w * Samples[i]['x'] * Samples[i]['x']
        y   += w * Samples[i]['y']
        yy  += w * Samples[i]['y'] * Samples[i]['y']

    # print x, y, xx, yy, ww

    print "mean(x) = %g, stddev(x) = %g\n" % (x, math.sqrt(xx-x*x))
    print "mean(y) = %g, stddev(y) = %g\n" % (y, math.sqrt(yy-y*y))

        