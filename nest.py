#!/usr/bin/env python
# encoding: utf-8
"""
nest.py

Created by Daniel O'Donovan on 2010-04-13.
Copyright (c) 2010 University of Cambridge. All rights reserved.
"""

import sys
import os

import copy

import numpy as np

class egg():
  """ most general sample class describes sample attributes:
        nmodels   - range for possible number of models
        (nparam   - number parameters)
        scope     - range for each parameter
        obsData   - the observed data (hopefully this is a link)
  """
  def __init__(self, nmodels, scope):

    # just some type checking
    if isinstance(nmodels, list) and isinstance(scope, list):
      self.nmodels  = np.array( nmodels )
      self.scope    = np.array( scope )
    else:
      print 'Egg class likes lists! Yum.'
      return

    self.nparam = len( scope )

    # data D - the observed data
    # self.D = np.array( obsData )

    # init and find these values upon creation of egg
    self.prior()

    self.lweight = None
    self.index = None

  # to be overidden
  def prior( self ):
    """ Initilise a sample """
    pass

  def update( self, data ):
    """ Calculate the log likelihood of sample in this state """
    # Must set self.llhood
    pass

class peakEgg( egg ):
  """ This class is tailored for the peak picker """

  def __init__(self, nmodels, scope):
    egg.__init__(self, nmodels, scope)

  def prior( self ):
    """ specific prior for peak picker """
    self.npeaks = np.random.random_integers( self.nmodels[0], self.nmodels[1] )

    self.A = np.zeros( self.npeaks )
    self.x = np.zeros( self.npeaks )
    self.s = np.zeros( self.npeaks )

    for i in range( self.npeaks ):
      self.A[i] = self.scope[0,0] + self.scope[0,1] * np.random.uniform()
      self.x[i] = self.scope[1,0] + self.scope[1,1] * np.random.uniform()
      self.s[i] = self.scope[2,0] + self.scope[2,1] * np.random.uniform()

  def update( self, D ):
    """ update log likelihood specific to peak picker """
    dataSize = self.scope[1,1]
    self.mock   = np.zeros( dataSize )
    for i in range( self.npeaks ):
      self.mock   += (self.A[i]/np.pi) * (0.5 * self.s[i]) / \
          (np.power(np.arange(dataSize)-self.x[i], 2.) + np.power( 0.5 * self.s[i], 2.0 ))
    self.llhood = -np.log( np.power(D - self.mock, 2.).sum() )


  def jiggle( self, step, data ):
    """ Jiggle self (when called to do so by explore) """

    # If npeaks = 0, we haven't initialised this egg yet - do so and exit
    if self.npeaks == 0:
      self.prior()
      return

    # add a jiggle proportional to step
    for i in range( self.npeaks ):
      # var = var +/- step : in prior range
      self.A[i] += self.scope[0,0] + self.scope[0,1] * (step * ( -1. + 2. * np.random.uniform() ))
      # wraparaund if exceeds region (ie keep in (0,1))
      self.A[i]  = self.A[i] % self.scope[0,1]

      self.x[i] += self.scope[1,0] + self.scope[1,1] * (step * ( -1. + 2. * np.random.uniform() ))
      self.x[i]  = self.x[i] % self.scope[1,1]

      self.s[i] += self.scope[2,0] + self.scope[2,1] * (step * ( -1. + 2. * np.random.uniform() ))
      self.s[i]  = self.s[i] % self.scope[2,1]

    # update the log likelihood for the new values
    self.update( data )

  def result( self, verbose=True ):
    """ return peak data and optionally print some feedback """
    if verbose:
      print '%6d peaks:' % (self.npeaks)
      for i in range( self.npeaks ):
        print '%6d Peak: A %f  - x %f  - s %f\n' % (i, self.A[i], self.x[i], self.s[i] )
    return

class basket():
  """ Class to hold all the eggs - apologies for stupid name 
        called with # of eggs to be held in basket
        also with example egg to be replicated neggs times
  """

  def __init__(self, neggs, baseEgg, data=None ):

    self.neggs = int(neggs)

    # fill the basket with eggs
    self.eggs = []
    for i in range( self.neggs ):
      self.eggs.append( copy.deepcopy(baseEgg) )

    # re-initialise all the eggs
    for i in range( self.neggs ):
      self.eggs[i].prior()

    if data:
      self.data = data
      self.update( self.data )
    else:
      print '* Without observed data, llhood is un-initialised'

  def update( self, data ):
    """ Update the loglikelihood for all eggs """
    for i in range( self.neggs ):
      self.eggs[i].update( data )

  def getBadEgg( self ):
    """ Return egg with worst log likelihood """
    eggs = self.eggs
    badEgg = eggs[0]
    for i, egg in enumerate( eggs ):
      if egg.llhood < badEgg.llhood:
        badEgg = egg
        badEgg.index = i
    return badEgg

  def getBestEgg( self ):
    """ Return egg with best log likelihood """
    eggs = self.eggs
    bestEgg = eggs[0]
    for i, egg in enumerate( eggs ):
      if egg.llhood > bestEgg.llhood:
        bestEgg = egg
        bestEgg.index = i
    return bestEgg

  def getEgg(self, index=None, notindex=None ):
    """ get egg at index i, or get random egg """
    eggs = self.eggs
    if index:
      egg = eggs[index]
    else:
      i = np.random.random_integers( len(eggs) - 1 )
      if notindex:
        while i == notindex:
          i = np.random.random_integers( len(eggs) - 1 )
      egg = eggs[i]
    return egg

  def remove( self, egg ):
    """ remove egg from basket """
    self.eggs.remove( egg )

  def append( self, egg ):
    """ add egg to basket """
    self.eggs.append( egg )

  def getDeviation( self ):
    """ get the variation of (llhood) for wholse basket """
    mean = 0.0
    for egg in self.eggs:
      mean += egg.llhood
    mean /= self.neggs
    dev = 0.0
    for egg in self.eggs:
      dev += np.power( egg.llhood - mean, 2. )
    dev = np.sqrt( dev / self.neggs )
    return dev


class Nest( object ):
  """ A nested sampling class """

  def __init__(self, basket, data, m=20, mstepsize=0.1, random=True, verbose=True):
    super(Nest, self).__init__()

    self.basket   = basket
    self.n        = basket.neggs         # the number of eggs in our nest (# samples)

    if isinstance(m, int) and isinstance(mstepsize, float):
      self.m = m
      self.mstepsize = mstepsize
    else:
      print 'Egg class likes ints and floats!'
      return

    self.data = np.array( data )

    self.H    = 0           # information, initially 0
    self.lZ   = -1.0E8      # ln( Evidence Z, initially 0 )

    self.pastEggs = []      # optional list to hold previous (bad) samples

    # outermost interval of prior mass
    self.lwidth = np.log( 1.0 - np.exp( -1.0 / self.n ) )

    # seed the random generator with a known value for debug
    if not random:
      np.random.seed(seed=1234)

    self.verbose = verbose

  def run(self, niter):
    """ Run the Nested Sampling Routine """

    self.basket.update( self.data )

    # do niter nested samples (or check for convergence - tbd)
    for sample in range( niter ):

      # worst object in the collection, with weight = width * likelihood
      badEgg = self.basket.getBadEgg()
      badEgg.lweight = self.lwidth + badEgg.llhood

      # update evidence Z and information H
      lZnew = self.plus( self.lZ, badEgg.lweight )
      self.H =  np.exp( badEgg.lweight - lZnew ) * badEgg.llhood \
              + np.exp( self.lZ - lZnew ) * ( self.H + self.lZ ) - lZnew
      self.lZ = lZnew

      # posterior samples (optional)
      self.pastEggs.append( badEgg )

      # Kill worst object in favour of copy of different survivor
      self.llstar = badEgg.llhood        # new likelihood constraint
      self.basket.remove( badEgg )

      goodEgg = self.basket.getEgg()

      # Evolve replaced egg within restraint
      goodEgg = self.explore( goodEgg )

      # put the evolved copy of the good egg back into the list
      self.basket.append( goodEgg )

      # shrink interval
      self.lwidth -= 1.0 / self.n

    if self.verbose:
      print '# iterates = %d' % niter
      print 'Evidence: ln(Z) = %f (+/- %f)' % (self.lZ, np.sqrt(self.H/float(self.n)))
      print 'Information: H = %f nats = %g bits' % ( self.H, self.H / np.log(2.) )

      self.results()

  def explore(self, egg):
    """ Given an egg, Evolve object within likelihood constraint  """
    # initial guess suitable step-size in (0,1)
    step = self.mstepsize 
    m = self.m  # # MCMC steps
    accept = 0  # # MCMC acceptances
    reject = 0  # # MCMC rejections

    trialEgg = copy.deepcopy( egg )

    while m > 0:
      m -= 1

      trialEgg.jiggle( step, self.data )

      if trialEgg.llhood > self.llstar:
        egg = copy.deepcopy( trialEgg )
        accept += 1
      else:
        reject += 1
      # Refine step-size to let acceptance ratio converge around 50%
      if accept > reject: step *= np.exp(1.0 / accept)
      if accept < reject: step /= np.exp(1.0 / reject)

    # probably don't need to clean up
    del( trialEgg )
    return egg

  def results( self ):
    """ TBD Not generic yet - need to think of clever way to do this """
    A = AA = 0.0
    x = xx = 0.0
    s = ss = 0.0
    for egg in self.pastEggs:
      w = np.exp( egg.lweight - self.lZ )
      A  += w * egg.A[0]
      AA += w * egg.A[0] * egg.A[0]
      x  += w * egg.x[0]
      xx += w * egg.x[0] * egg.x[0]
      s  += w * egg.s[0]
      ss += w * egg.s[0] * egg.s[0]

    print 'mean(A) = %12.6f, stdev(A) = %12.6f' % (A, np.sqrt( AA - (A * A) ))
    print 'mean(x) = %12.6f, stdev(x) = %12.6f' % (x, np.sqrt( xx - (x * x) ))
    print 'mean(s) = %12.6f, stdev(s) = %12.6f' % (s, np.sqrt( ss - (s * s) ))


  #---------------------------------------------------------------------------
  # Generic utility functions
  #
  def uniform(self):
      """ Return random in range (0,1) 
              - numpy.random.random() is (0,1]
      """
      a = np.random.uniform()
      while a == 1.0: a = np.random.uniform()
      return a

  def plus(self, x, y):
      """ Logarithmic addition log( exp( x ) + exp( y ) ) """
      if x > y: return x + np.log(1. + np.exp(y - x))
      else:     return y + np.log(1. + np.exp(x - y))
  #---------------------------------------------------------------------------



if __name__ == '__main__':

  npeaks = 1
  A = 05.123  # Volume of the Lorentzian
  x = 45.678  # Mean of the Lorentzian
  s = 0.9012  # Width of the Lorentzian
  data = (A/np.pi) * (0.5 * s) / (np.power(np.arange(128)-x, 2.) + np.power( 0.5 * s, 2.0 ))

  noise = 0.1
  observed_data = data + (- noise + 2.0 * noise * np.random.random( data.shape ))

  neggs =   500
  niter = 10000

  # Peak Egg (or any custom egg)
  P = peakEgg( [1,1], [ [0.1, 10.], [0., 128.], [0.1, 2.0] ] )

  B = basket( neggs, P )

  N = Nest( B, observed_data, m=300, random=False )

  N.run( niter )

  print '## deviation of samples %f' % ( B.getDeviation() )







