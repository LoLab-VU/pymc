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
import multiprocessing as mp
import multiprocessing.pool as mp_pool
import logging
import traceback

__all__ = ['Dream']

class Dream(ArrayStep):
    def __init__(self, variables=None, nseedchains=100, convergenceCriteria=1.1, nCR=3, DEpairs=1, adaptationRate=.65, eps=5e-6, verbose=False, save_history = False, start_random=True, snooker=10, multitry=False, model=None, **kwargs):
        
        model = modelcontext(model)
                
        if variables is None:
            variables = model.cont_vars
        
        if not set(variables).issubset(set(model.cont_vars)):
            raise Exception('The implemented version of Dream should only be run on continuous variables.')
        
        self.model = model
        self.variables = variables
        self.nseedchains = nseedchains
        self.DEpairs = DEpairs
        self.snooker = snooker
        if multitry == False:
            self.multitry = 1
        else:
            self.multitry = multitry
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
        #Gamma for snooker update is drawn from U[1.2, 2.2] as suggested in ter Braak 2008.     
        
        gamma = self.set_gamma(self.iter, self.DEpairs, self.total_var_dimension, self.snooker)
        
        
        with Dream_shared_vars.history.get_lock() and Dream_shared_vars.count.get_lock():
            if self.iter == 0 or self.snooker == 0 or self.iter % self.snooker != 0:
                proposed_pts = self.generate_proposal_points(self.multitry, gamma, q0, snooker=False)

            else:
                proposed_pts = self.generate_proposal_points(self.multitry, gamma, q0, snooker=True)
#            # Sample without replacement from the population history
#            sampled_chains = self.sample_from_history(self.nseedchains, self.DEpairs, self.total_var_dimension)
#    
#        # Sum DEpairs number of chain pairs to give two chains, and take difference of chain sums
#        #chain_differences = np.sum(sampled_chains[0:2*self.DEpairs], axis=0)-np.sum(sampled_chains[2*self.DEpairs:self.DEpairs*4], axis=0)
#        chain_differences = np.array(sampled_chains[0])-np.array(sampled_chains[1])
#        
#        #print 'Took difference of chains: '+str(sampled_chains[0:2])+' chain differences = ', chain_differences        
#        
#        # e is a random sample from a normal distribution with small sd
#        e = np.random.normal(0, self.eps, self.total_var_dimension)
#        
#        # Generate proposal point
#        q = q0 + gamma*chain_differences + e        
        
        print 'proposed point = ', proposed_pts 
        print 'proposed points squeezed = ',np.squeeze(proposed_pts)
        
        if self.multitry == 1:
            q_logp = logp(np.squeeze(proposed_pts))
            q = np.squeeze(proposed_pts)
        else:
            mp.log_to_stderr(logging.DEBUG)
            p = mp.Pool(self.multitry)
            args = zip([self]*self.multitry, np.squeeze(proposed_pts))
            log_ps = p.map(call_logp, args)
            p.close()
            q_logp_loc = np.argmax(log_ps)
            q_logp = log_ps[q_logp_loc]
            q = np.squeeze(proposed_pts[q_logp_loc])
            print 'logps = '+str(log_ps)+' Selected logp = '+str(q_logp)+' Point = '+str(q)
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
        
    def set_gamma(self, iteration, DEpairs, ndimensions, snooker_frequency):
        if iteration > 0 and iteration%5 == 0:
            gamma = np.array([1.0])
        
        elif iteration > 0 and snooker_frequency != 0 and iteration % snooker_frequency == 0:        
            gamma = np.random.uniform(1.2, 2.2)
            
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

    def sample_from_history(self, nseedchains, DEpairs, ndimensions, snooker=False):
        if snooker is False:
            chain_num = random.sample(range(Dream_shared_vars.count.value+nseedchains), DEpairs*4)
        else:
            chain_num = random.sample(range(Dream_shared_vars.count.value+nseedchains), 1)
        start_locs = [i*ndimensions for i in chain_num]
        end_locs = [i+ndimensions for i in start_locs]
        sampled_chains = [Dream_shared_vars.history[start_loc:end_loc] for start_loc, end_loc in zip(start_locs, end_locs)]
#        arr = np.frombuffer(Dream_shared_vars.history.get_obj())
#        b = arr.reshape(((50+self.nseedchains), self.total_var_dimension))
#        print 'sampling from history chain numbers: '+str(chain_num)+' sampled chains: '+str(sampled_chains[0:2])+' Current history: '+str(b)
        #print 'sampled chains: ', sampled_chains
        return sampled_chains
        
    def generate_proposal_points(self, n_proposed_pts, gamma, q0, snooker):
        if snooker is False:
            sampled_history_pts = np.array([self.sample_from_history(self.nseedchains, self.DEpairs, self.total_var_dimension) for i in range(n_proposed_pts)])
            if self.DEpairs != 0:
                chain_differences = [np.sum(sampled_history_pts[i][0:2*self.DEpairs], axis=0)-np.sum(sampled_history_pts[i][2*self.DEpairs:self.DEpairs*4], axis=0) for i in range(len(sampled_history_pts))]
            else:
                chain_differences = [sampled_history_pts[0]- sampled_history_pts[1] for i in range(len(sampled_history_pts))]
            e = np.array([np.random.normal(0, self.eps, self.total_var_dimension) for i in range(n_proposed_pts)])
            proposed_pts = q0 + gamma*chain_differences + e
        
        else:
            print 'n_proposed_pts: ',n_proposed_pts
            print 'gamma: ',gamma
            print 'q0: ', q0
            proposed_pts = self.snooker_update(n_proposed_pts, gamma, q0)
            
        return proposed_pts
        
    def snooker_update(self, n_proposed_pts, gamma, q0):
        sampled_history_pt = [self.sample_from_history(self.nseedchains, self.DEpairs, self.total_var_dimension, snooker=True) for i in range(n_proposed_pts)]
        print 'sampled history pt: ',sampled_history_pt
        print 'sampled history pts squeezed: ',np.squeeze(sampled_history_pt)
        print 'point 1: ',np.squeeze(sampled_history_pt)[0]
        print 'point 2: ',np.squeeze(sampled_history_pt)[1]
        #Find mutually orthogonal vectors spanning current location and a randomly chosen chain from history
        ortho_vecs = []
        for i in range(n_proposed_pts):        
            vecs, r = np.linalg.qr(np.column_stack((q0, np.squeeze(sampled_history_pt)[i])))
            print 'appended vector'
            ortho_vecs.append(vecs)
        print 'ortho vecs: ', ortho_vecs
        print 'ortho vec list: ', [ortho_vecs[0][:,0], ortho_vecs[0][:,1]]
        print 'len ortho_vecs: ',len(ortho_vecs)

        #Determine orthogonal projection of two other randomly chosen chains onto this span
        chains_to_be_projected = np.squeeze([np.array([self.sample_from_history(self.nseedchains, self.DEpairs, self.total_var_dimension, snooker=True) for i in range(2)]) for x in range(n_proposed_pts)])
        print 'chains_to_be_projected: ',chains_to_be_projected
        print 'len chains ',len(chains_to_be_projected)
        projected_pts = []
        for vec_set in range(len(ortho_vecs)):
            print 'vec set: ',vec_set
            pts_for_set = []
            for chain_n in range(2):
                print 'chain_n: ',chain_n
                print 'all ortho vecs: ',ortho_vecs
                ortho_vec_list = [ortho_vecs[vec_set][:,0], ortho_vecs[vec_set][:,1]]
                print 'Ortho vec list: ',np.array(ortho_vec_list)
                print 'all chains to be project: ',chains_to_be_projected
                print 'Chains to be projected: ',chains_to_be_projected[vec_set]
                print 'chain 1: ',chains_to_be_projected[vec_set][0]
                print 'chain 2: ',chains_to_be_projected[vec_set][1]
                
                pts_for_set.append([self.project_chains(ortho_vec_list, chains_to_be_projected[vec_set][chain_n])])
            projected_pts.append(pts_for_set)
            
        print 'projected pts: ',projected_pts
        print 'len projected pts: ', len(projected_pts)
        print 'point 1: ', projected_pts[0][0]
        print 'point 2: ', projected_pts[0][1]
        
        #Calculate difference between projected points
        chain_differences = np.array([np.array(projected_pts[i][0]) - np.array(projected_pts[i][1]) for i in range(n_proposed_pts)])
        print 'chain_differences: ',chain_differences
        
        #And use difference to propose a new point
        proposed_pts = q0 + gamma*chain_differences
        
        return proposed_pts
    
    def project_chains(self, ortho_vecs, chain_to_be_projected):
        sigmadict = {len(ortho_vecs):1}
        b0 = chain_to_be_projected
        for i, vec in enumerate(ortho_vecs):
            sigma = np.dot(chain_to_be_projected, vec)/np.dot(vec, vec) if np.dot(vec, vec) > 1e-20 else 0
            sigmadict[i] = sigma
            chain_to_be_projected = chain_to_be_projected - sigma*vec
        ortho_proj = chain_to_be_projected
        pt_on_line = b0 - ortho_proj
        return pt_on_line
    
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
    
def call_logp(args):
    #Defined at top level so it can be pickled.
    instance = args[0]
    point = args[1]
    
    logp_fxn = getattr(instance, 'fs')[0]
    ordering = getattr(instance, 'ordering')
    bij = DictToArrayBijection(ordering, {})
    logp = bij.mapf(logp_fxn)
    return logp(point)

        
class NoDaemonProcess(mp.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

#A subclass of multiprocessing.pool.Pool that allows processes to launch child processes (this is necessary for Dream to use multi-try)
#Taken from http://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic
class DreamPool(mp_pool.Pool):
    Process = NoDaemonProcess
        