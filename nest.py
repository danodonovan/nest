#!/usr/bin/env python
# encoding: utf-8
"""
nest.py

Baed on
//                   NESTED SAMPLING MAIN PROGRAM
// (GNU General Public License software, (C) Sivia and Skilling 2006)

Translated into Python by Daniel O'Donovan on 2009-10-07.

run with python nest.py

"""

import sys
import os

import math, random

import apply

def uniform():
    """ Return random in range (0,1) 
            - random.random() is (0,1]
    """
    a = 1.0
    while a == 1.0:
        a = random.random()
    return a

def plus( x, y ):
    """ Logarithmic addition log( exp( x ) + exp( y ) ) """
    if x > y:
        return x + math.log(1 + math.exp(y - x))
    else:
        return y + math.log(1 + math.exp(x - y))

def main( plot=False ):

    n = apply.n
    MAX = apply.MAX
    H = 0               # information, initially 0
    logZ = -1.0E8       # ln( Evidence Z, initially 0 )

    # collection of n objects
    Obj = []
    for i in xrange( n ):
        Obj.append( apply.Object.copy() )

    # objects stored for posterior results
    Samples = []
    for i in xrange( MAX ):
        Samples.append( apply.Object.copy() )

    # initialise priors
    for Object in Obj:
        apply.prior( Object )

    # outermost interval of prior mass
    logwidth = math.log( 1.0 - math.exp( -1.0 / n ) )

    # nested sampling loop
    for nest in xrange( MAX ):

        # worst object in the collection, with weight = width * likelihood
        worst = 0
        for i in xrange( 1, n ):
            if Obj[i]['logL'] < Obj[worst]['logL']: worst = i
        Obj[worst]['logWt'] = logwidth + Obj[worst]['logL']

        # update evidence Z and information H
        logZnew = plus( logZ, Obj[worst]['logWt'] )
        H = math.exp( Obj[worst]['logWt'] - logZnew ) * Obj[worst]['logL'] \
            + math.exp( logZ - logZnew ) * ( H + logZ ) - logZnew
        logZ = logZnew

        # posterior samples (optional)
        Samples[nest] = Obj[worst].copy()

        # Kill worst object in favour of copy of different survivor
        while True and (n > 1):             # don't kill if n is only 1
            copy = int( n * uniform() ) % n # force 0 <= copy < n
            if copy != worst:
                break
        logLstar = Obj[worst]['logL']       # new likelihood constraint
        Obj[worst] = Obj[copy].copy()       # overwrite the 

        # Evolve copied object within restraint
        Obj[worst] = apply.Explore( Obj[worst], logLstar ).copy()

        # shrink interval
        logwidth -= 1.0 / n

    # End nested sampling loop

    # Exit with evidence Z, information H, and optional posterior Samples
    print "# iterates = %d\n" %  ( nest )
    print "Evidence: ln(Z) = %g +- %g\n" % ( logZ, math.sqrt(H/n) ) 
    print "Information: H = %g nats = %g bits\n" % (H, H/math.log(2.))
    apply.Results(Samples, nest, logZ)        # optional

    return (Samples, nest, logZ)

if __name__ == '__main__':

    (Samples, nest, logZ) = main( plot=False )

