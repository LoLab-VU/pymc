# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 17:21:29 2014

@author: Erin
"""

from ..core import *
from .arraystep import *
import numpy as np
import random
import Dream_shared_vars
from datetime import datetime

__all__ = ['Dream']

class Dream(ArrayStep):
    def __init__(self, variables=None, nseedchains=100, convergenceCriteria=1.1, nCR=3, DEpairs=1, adaptationRate=.65, eps=5e-6, verbose=False, save_history = False, start_random=True, snooker=False, multitry=False, model=None, **kwargs):
        
        model = modelcontext(model)
                
        if variables is None:
            variables = model.cont_vars
        
        if not set(variables).issubset(set(model.cont_vars)):
            raise Exception('The implemented version of Dream should only be run on continuous variables.')
        
        self.model = model
        self.variables = variables
        self.nseedchains = nseedchains
        self.DEpairs = DEpairs
        self.eps = eps
        self.last_logp = None
        self.total_var_dimension = 0
        for var in variables:
            var_name = getattr(model, str(var))
            self.total_var_dimension += var_name.dsize
        print self.total_var_dimension
        
        self.iter = 0  
        self.len_history = 0
        self.save_history = save_history
        self.start_random = start_random
        
        super(Dream, self).__init__(variables, [model.fastlogp], **kwargs)
    
    def astep(self, q0, logp):
        # On first iteration, check that shared variables have been initialized (which only occurs if multiple chains have been started).
        if self.iter == 0:   
            print 'Dream has started'
            try:
                # Assuming the shared variables exist, seed the history with nseedchain draws from the prior
                with Dream_shared_vars.history_seeded.get_lock():
                    if Dream_shared_vars.history_seeded.value == 'F':
                        print 'Seeding history with draws from prior'
                        print self.nseedchains
                        for i in range(self.nseedchains):
                            start_loc = i*self.total_var_dimension
                            end_loc = start_loc+self.total_var_dimension
                            Dream_shared_vars.history[start_loc:end_loc] = self.draw_from_prior(self.model, self.variables)
                            #print 'Adding draw: '+str(i)+ ' : '+str(Dream_shared_vars.history[start_loc:end_loc])
                        #print 'Current history: '+str(Dream_shared_vars.history[0:1440])
                        #np.save('history_at_start.npy', np.frombuffer(Dream_shared_vars.history.get_obj()))
                    Dream_shared_vars.history_seeded.value = 'T'
                    if self.start_random:
                        print 'Setting start to random draw from prior.'
                        q0 = self.draw_from_prior(self.model, self.variables)
                        print 'Start: ',q0
                # Also get length of history array so we know when to save it at end of run.
                if self.save_history:
                    with Dream_shared_vars.history.get_lock():
                        self.len_history = len(np.frombuffer(Dream_shared_vars.history.get_obj()))
                        print 'setting len history = ', self.len_history
            
            except AttributeError:
                raise Exception('Dream should be run with multiple chains in parallel.  Set njobs > 1.')
                
        # Set gamma depending on iteration; every 5th iteration over all chains, large jumps are allowed
        gamma = self.set_gamma(self.iter, self.DEpairs, self.total_var_dimension)
        
        with Dream_shared_vars.history.get_lock() and Dream_shared_vars.count.get_lock():
            # Sample without replacement from the population history
            sampled_chains = self.sample_from_history(self.nseedchains, self.DEpairs, self.total_var_dimension)
    
        # Sum DEpairs number of chain pairs to give two chains, and take difference of chain sums
        #chain_differences = np.sum(sampled_chains[0:2*self.DEpairs], axis=0)-np.sum(sampled_chains[2*self.DEpairs:self.DEpairs*4], axis=0)
        chain_differences = np.array(sampled_chains[0])-np.array(sampled_chains[1])
        
        print 'Took difference of chains: '+str(sampled_chains[0:2])+' chain differences = ', chain_differences        
        
        # e is a random sample from a normal distribution with small sd
        e = np.random.normal(0, self.eps, self.total_var_dimension)
        
        # Generate proposal point
        q = q0 + gamma*chain_differences + e        
        
        #print 'proposed point = ', q        
        
        q_logp = logp(q)        
        
#        print 'logp of proposed point = ', q_logp        
        
        if self.last_logp == None:
            self.last_logp = logp(q0)
        
        
        q_new = metrop_select(q_logp - self.last_logp, q, q0)
        
        if np.array_equal(q0, q_new):
            print 'Did not accept point. Old logp: '+str(self.last_logp)+' Tested logp: '+str(q_logp)
        else:
            print 'Accepted point. Old logp: ',str(self.last_logp)+ ' New logp: ',str(q_logp)
            self.last_logp = q_logp
        
        #Place new point in history
        with Dream_shared_vars.history.get_lock() and Dream_shared_vars.count.get_lock():
            self.record_history(self.nseedchains, self.total_var_dimension, q_new, self.len_history)
        
        self.iter += 1
        return q_new
        
    def set_gamma(self, iteration, DEpairs, ndimensions):
        if iteration > 0 and iteration%5 == 0:
            gamma = np.array([1.0])
        
        else:
            gamma = np.array([2.38 / np.sqrt( 2 * DEpairs  * ndimensions)])
        
        return gamma

    def draw_from_prior(self, model, model_vars):
        draw = np.array([])
        for variable in model_vars:
            var_name = getattr(model, str(variable))
            try:
                var_draw = var_name.distribution.random()
            except AttributeError:
                raise Exception('Random draw from distribution for variable %s not implemented yet.' % var_name)
            draw = np.append(draw, var_draw)
        return draw.flatten()

    def sample_from_history(self, nseedchains, DEpairs, ndimensions):
        chain_num = random.sample(range(Dream_shared_vars.count.value+nseedchains), DEpairs*4)
        start_locs = [i*ndimensions for i in chain_num]
        end_locs = [i+ndimensions for i in start_locs]
        sampled_chains = [Dream_shared_vars.history[start_loc:end_loc] for start_loc, end_loc in zip(start_locs, end_locs)]
#        arr = np.frombuffer(Dream_shared_vars.history.get_obj())
#        b = arr.reshape(((50+self.nseedchains), self.total_var_dimension))
#        print 'sampling from history chain numbers: '+str(chain_num)+' sampled chains: '+str(sampled_chains[0:2])+' Current history: '+str(b)
        #print 'sampled chains: ', sampled_chains
        return sampled_chains
        
    def record_history(self, nseedchains, ndimensions, q_new, len_history):
        nhistoryrecs = Dream_shared_vars.count.value+nseedchains
        start_loc = nhistoryrecs*ndimensions
        end_loc = start_loc+ndimensions
#        print 'array location: ',start_loc
#        print 'history at position array_loc: ',Dream_shared_vars.history[start_loc:end_loc]
        Dream_shared_vars.history[start_loc:end_loc] = np.array(q_new).flatten()
#        arr = np.frombuffer(Dream_shared_vars.history.get_obj())
#        b = arr.reshape(((50+self.nseedchains), self.total_var_dimension))
#        print 'Added new point '+str(q_new)+' to history at position: '+str(nhistoryrecs)+' History now: '+str(b)
        Dream_shared_vars.count.value += 1
        if self.save_history and len_history == (nhistoryrecs+1)*ndimensions:
            date_time_str = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')+'_'
            self.save_history_to_disc(np.frombuffer(Dream_shared_vars.history.get_obj()), date_time_str)
            
    def save_history_to_disc(self, history, prefix):
        print 'saving history'
        filename = prefix+'DREAM_chain_history.npy'
        np.save(filename, history)
        
        