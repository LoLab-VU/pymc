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
    def __init__(self, variables=None, nseedchains=100, nCR = 3, crossover_burnin=1000, DEpairs=1, adaptationRate=.65, eps=10e-6, verbose=False, save_history = False, start_random=True, snooker=.10, multitry=False, model=None, **kwargs):
        
        model = modelcontext(model)
                
        if variables is None:
            variables = model.cont_vars
        
        if not set(variables).issubset(set(model.cont_vars)):
            raise Exception('The implemented version of Dream should only be run on continuous variables.')
        
        self.model = model
        self.variables = variables
        self.nseedchains = nseedchains
        self.nCR = nCR
        self.crossover_burnin = crossover_burnin
        self.CR_probabilities = [1/float(self.nCR) for i in range(self.nCR)]
        self.CR_values = np.array([m/float(self.nCR) for m in range(1, self.nCR+1)])
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
        self.chain_n = None
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
                        print 'Setting crossover probability starting values.'
                        with Dream_shared_vars.cross_probs.get_lock():
                            starting_cross_probs = np.array([1/(float(self.nCR)) for i in range(self.nCR)])
                            Dream_shared_vars.cross_probs[0:self.nCR] = starting_cross_probs
                            print 'set prob of different crossover values to: ',Dream_shared_vars.cross_probs[0:self.nCR+1]
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
        
        if self.snooker != 0:
            snooker_choice = np.where(np.random.multinomial(1, [self.snooker, 1-self.snooker])==1)
            print 'Snooker choice: ',snooker_choice
            if snooker_choice[0] == 0:
                run_snooker = True
            else:
                run_snooker = False
            print 'Snooker decision: ',run_snooker
        else:
            run_snooker = False
        
        if self.multitry > 1:
            #Set CR value for generating proposal point
            CR_loc = np.where(np.random.multinomial(1, self.CR_probabilities)==1)
            #print 'CR_loc chosen: ',CR_loc
            CR = self.CR_values[CR_loc]
            #print 'Set CR to: ',CR
        else:
            CR = 1    
        
        gamma = self.set_gamma(self.iter, self.DEpairs, self.total_var_dimension, run_snooker, CR)
        
        with Dream_shared_vars.history.get_lock() and Dream_shared_vars.count.get_lock():
            if self.iter == 0 or self.snooker == 0 or run_snooker == False:
                #print 'Proposing pts with no snooker update. q0: ',q0,' CR: ',CR
                proposed_pts = self.generate_proposal_points(self.multitry, gamma, q0, CR, snooker=False)

            else:
                #print 'Proposing pts with snooker update. q0: ',q0,' CR: ',CR
                proposed_pts = self.generate_proposal_points(self.multitry, gamma, q0, CR, snooker=True)
        
        if self.multitry == 1:
            q_logp = logp(np.squeeze(proposed_pts))
            q = np.squeeze(proposed_pts)
        else:
            #mp.log_to_stderr(logging.DEBUG)
            p = mp.Pool(self.multitry)
            args = zip([self]*self.multitry, np.squeeze(proposed_pts))
            log_ps = p.map(call_logp, args)
            p.close()
            p.join()
            #Randomly select one of the tested points with probability proportional to the probability density at the point
            q_logp_min_loc = np.argmin(log_ps)
            q_logp_min = log_ps[q_logp_min_loc]
            positive_logps = log_ps + abs(q_logp_min)+1
            print 'logps: ',log_ps
            print 'positive logps: ',positive_logps
            sum_proposal_logps = np.sum(log_ps)
            sum_positive_proposal_logps = np.sum(positive_logps)
            logp_draw_prob = abs(positive_logps/sum_positive_proposal_logps)
            print 'logp draw prob: ',logp_draw_prob
            random_logp_loc = np.where(np.random.multinomial(1, logp_draw_prob)==1)[0]
            #print 'logp location: ',random_logp_loc
            q_proposal = np.squeeze(proposed_pts[random_logp_loc])
            #print 'logps proposed = '+str(log_ps)+' Selected logp = '+str(log_ps[random_logp_loc])+' Point = '+str(q_proposal)
            
            #Draw reference points around the randomly selected proposal point
            with Dream_shared_vars.history.get_lock() and Dream_shared_vars.count.get_lock():
                reference_pts = self.generate_proposal_points(self.multitry-1, gamma, q_proposal, CR, snooker=False)
                #print 'Generated reference points: ',reference_pts
            
            #Compute posterior density at reference points.
            if self.multitry > 2:
                p = mp.Pool(self.multitry-1)
                args = zip([self]*self.multitry, np.squeeze(reference_pts))
                ref_log_ps = p.map(call_logp, args)
                p.close()
                p.join()
            else:
                ref_log_ps = np.array([logp(np.squeeze(reference_pts))])
            
#        print 'logp of proposed point = ', q_logp        
        #print 'Reference logps = ',ref_log_ps
        
        if self.last_logp == None:
            self.last_logp = logp(q0)
        
        if self.multitry > 1:
            np.append(ref_log_ps, self.last_logp)
#            ref_logp_min_loc = np.argmin(ref_log_ps)
#            ref_logp_min = ref_log_ps[ref_logp_min_loc]
#            positive_ref_logps = ref_log_ps + abs(ref_logp_min)+1   
#            print 'Positive ref logps: ',positive_ref_logps
            #sum_reference_logps = np.sum(positive_ref_logps)+ self.last_logp + abs(ref_logp_min)+1
            all_log_ps = np.concatenate((log_ps, ref_log_ps))
            min_all_log_ps_loc = np.argmin(all_log_ps)
            all_log_p_min = all_log_ps[min_all_log_ps_loc]
            positive_proposal_logps = log_ps + abs(all_log_p_min)+1
            positive_reference_logps = ref_log_ps + abs(all_log_p_min)+1
            sum_pos_proposal_logps = np.sum(positive_proposal_logps)
            sum_reference_logps = np.sum(ref_log_ps)
            sum_pos_reference_logps = np.sum(positive_reference_logps)
            print 'Sum reference logps: ',sum_reference_logps
            print 'Sum proposal logps: ',sum_proposal_logps
            print 'log pos ref logps: ',np.log10(sum_pos_reference_logps)
            print 'log pos prop logps: ',np.log10(sum_pos_proposal_logps)
            print 'ratio prop/ref: ',sum_proposal_logps - sum_reference_logps
            print 'ratio pos prop/pos ref: ',sum_pos_proposal_logps - sum_pos_reference_logps
            print 'ratio logp prop/logp ref: ',np.log10(sum_pos_proposal_logps) - np.log10(sum_pos_reference_logps)
            q_new = metrop_select(sum_proposal_logps - sum_reference_logps, q_proposal, q0)
            q_logp = log_ps[random_logp_loc]
        else:    
            q_new = metrop_select(q_logp - self.last_logp, q, q0)
        
        if np.array_equal(q0, q_new):
            print 'Did not accept point. Old logp: '+str(self.last_logp)+' Old sum logps: '+str(sum_reference_logps)+' Tested sum logps: '+str(sum_proposal_logps)+' Tested logp: '+str(q_logp)+' Logp ratio: ',sum_proposal_logps-sum_reference_logps
        else:
            print 'Accepted point. Old logp: ',str(self.last_logp)+' Old sum logps: ',sum_reference_logps, 'Tested sum logps: ',sum_proposal_logps,' New logp: ',str(q_logp)+' Logp ratio: ',sum_proposal_logps-sum_reference_logps
            self.last_logp = q_logp
        
        #Place new point in history
        with Dream_shared_vars.history.get_lock() and Dream_shared_vars.count.get_lock() and Dream_shared_vars.current_positions.get_lock():
            self.record_history(self.nseedchains, self.total_var_dimension, q_new, self.len_history)
        
        #If using multi-try DREAM, estimate ideal crossover probabilities for each dimension during burn-in.
        #Don't do this for the first 10 iterations to give all chains a chance to fill in the shared current position array
        if self.multitry > 1 and self.iter > 10 and self.iter < self.crossover_burnin:
            with Dream_shared_vars.cross_probs.get_lock() and Dream_shared_vars.count.get_lock() and Dream_shared_vars.ncr_updates.get_lock() and Dream_shared_vars.current_positions.get_lock() and Dream_shared_vars.delta_m.get_lock():
               self.CR_probabilities = self.estimate_crossover_probabilities(self.iter, self.total_var_dimension, gamma, q0, q_new, CR)        
        
        self.iter += 1
        return q_new
    
    def estimate_crossover_probabilities(self, iteration, ndim, gamma, q0, q_new, CR):
        cross_probs = Dream_shared_vars.cross_probs[0:self.nCR]   
#        print 'Pulled out crossover probabilities.  Starting values = ',cross_probs
#        m_loc = np.where(np.random.multinomial(1, cross_probs)==1)[0]
#        #print 'Calculated m_loc = ',m_loc
#        CR = crossover_values[m_loc]
#        print 'Selected CR = ',CR

#        #print 'Updated shared L = ',Dream_shared_vars.ncr_updates
#        
#        proposed_pt = self.generate_proposal_points(1, gamma, q0, CR, snooker=False)
#        print 'Generated proposal point with CR.'
#        
#        if self.last_logp == None:
#            self.last_logp = logp(q0)
#            
#        q_new = metrop_select(logp(np.squeeze(proposed_pt)) - self.last_logp, np.squeeze(proposed_pt), q0)
#        #print 'Selected proposal point. q_new: ',q_new
        
        current_positions = np.frombuffer(Dream_shared_vars.current_positions.get_obj())
        nchains = len(current_positions)/ndim
        #print 'nchains: ',nchains
        current_positions = current_positions.reshape((nchains, ndim))
        #print 'Current positions: ',current_positions
        current_positions[self.chain_n] = q_new
        #print 'Replaced current position of current chain with new point. Current positions: ',current_positions
        sd_by_dim = np.std(current_positions, axis=0)
        #print 'SD by dimension: ',sd_by_dim
        
        #Compute squared normalized jumping distance
        print 'Shared delta m array before change: ',Dream_shared_vars.delta_m[0:self.nCR]
        #print 'constant: ',Dream_shared_vars.delta_m[m_loc]
        print 'diff q_new and q0: ',q_new-q0
        #print 'diff squared: ',(q_new-q0)**2
        #print 'sd squared: ',sd_by_dim**2
        #print 'num/den: ',(q_new-q0)**2/sd_by_dim**2
        #print 'Sum: ',np.sum((q_new-q0)**2/sd_by_dim**2)
        m_loc = np.where(self.CR_values == CR)[0]
        Dream_shared_vars.ncr_updates[m_loc] += 1
        Dream_shared_vars.delta_m[m_loc] = Dream_shared_vars.delta_m[m_loc] + np.sum((q_new - q0)**2/sd_by_dim**2)
        #print 'Squared normalized jumping distance for m = ',m_loc,' = ',Dream_shared_vars.delta_m[m_loc]
        
        #Update probabilities of tested crossover value
        tN = Dream_shared_vars.count.value - (nchains*10)
        #print 'tN : ',tN
        print 'delta_m[m_loc]: ',Dream_shared_vars.delta_m[m_loc]
        print 'ncr_updates[m_loc]: ',Dream_shared_vars.ncr_updates[m_loc]
        #print 'sum of delta_ms: ',np.sum(Dream_shared_vars.delta_m[0:self.nCR])
        #print 'num: ',tN*(Dream_shared_vars.delta_m[m_loc]/Dream_shared_vars.ncr_updates[m_loc])
        
        #Leave probabilities unchanged until all possible crossover values have had at least one successful move so that a given value's probability isn't prematurely set to 0, preventing further testing.
        delta_ms = np.array(Dream_shared_vars.delta_m[0:self.nCR])
        ncr_updates = np.array(Dream_shared_vars.ncr_updates[0:self.nCR])
        print 'truth status: ',np.all(delta_ms != 0)
        sum_delta_m_per_iter = np.sum(delta_ms/ncr_updates)
        if np.all(delta_ms != 0) == True:
            print 'All values have been successful at least once.  Changing crossover probabilities to reflect delta m values.'
            for m in range(self.nCR):
                cross_probs[m] = (Dream_shared_vars.delta_m[m]/Dream_shared_vars.ncr_updates[m])/sum_delta_m_per_iter
        
        Dream_shared_vars.cross_probs[0:self.nCR] = cross_probs
        
        print 'Current crossover value probabilities: ',cross_probs 
        
        self.CR_probabilities = cross_probs
        
        return cross_probs
         
    def set_gamma(self, iteration, DEpairs, ndimensions, snooker_choice, CR):
        if iteration > 0 and iteration%5 == 0:
            gamma = np.array([1.0])
        
        elif iteration > 0 and snooker_choice == True:        
            gamma = np.random.uniform(1.2, 2.2)
            
        else:
            d_prime = ndimensions*CR
            gamma = np.array([2.38 / np.sqrt( 2 * DEpairs  * d_prime)])
        
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
        
    def generate_proposal_points(self, n_proposed_pts, gamma, q0, CR, snooker):
        if snooker is False:
            #print 'Generating pts with no snooker update. n proposed pts= ',n_proposed_pts
            sampled_history_pts = np.array([self.sample_from_history(self.nseedchains, self.DEpairs, self.total_var_dimension) for i in range(n_proposed_pts)])
            #print 'Sampled history pts: ',sampled_history_pts            
            if self.DEpairs != 0:
                chain_differences = [np.sum(sampled_history_pts[i][0:2*self.DEpairs], axis=0)-np.sum(sampled_history_pts[i][2*self.DEpairs:self.DEpairs*4], axis=0) for i in range(len(sampled_history_pts))]
                #print 'Generated chain differences with DEpairs>0.  chain differences = ',chain_differences
            else:
                chain_differences = [sampled_history_pts[0]- sampled_history_pts[1] for i in range(len(sampled_history_pts))]
                #print 'Generated chain differences with DEpairs=0.  chain differences = ',chain_differences
            epsilon = np.array([np.random.normal(0, self.eps, self.total_var_dimension) for i in range(n_proposed_pts)])
            e = np.array([np.random.uniform(-.05, .05, self.total_var_dimension) for i in range(n_proposed_pts)])
            proposed_pts = q0 + e*gamma*chain_differences + epsilon
            #print n_proposed_pts,' proposed pts generated without snooker update. Proposed pts = ',proposed_pts  
        
        else:
            proposed_pts = self.snooker_update(n_proposed_pts, gamma, q0)
            #print n_proposed_pts,' proposed pts generated with snooker update. Proposed pts = ',proposed_pts
        
        if self.multitry > 1:
            #print 'points before crossover: ',proposed_pts
            if n_proposed_pts > 1:
                #Perform crossover
                for point in proposed_pts:
                    for d in range(len(point)):
                        U = np.random.uniform(0, 1)
                        #print 'U: ',U,' CR: ',CR
                        if U <= 1-CR:
                            #print 'Changed dimension: ',d,' to original'
                            point[d] = q0[d]
            else:
               #Perform crossover
                #print 'proposed pt[0]: ',proposed_pts[0]
                #print 'squeezed proposed_pt[0]: ',np.squeeze(proposed_pts[0])
                #print 'q0[0]: ',q0[0]
                #print 'q0: ',q0
                #print 'proposed pt[0][0]: ',proposed_pts[0][0]
                #print 'proposed_pt[0][1]: ',proposed_pts[0][1]
                for d in range(len(proposed_pts[0])):
                    U = np.random.uniform(0, 1)
                    #print 'U: ',U,' CR: ',CR
                    if U <= 1-CR:
                        #print 'Changed dimension: ',d,' to original'
                        proposed_pts[0][d] = q0[d] 
                        
        #print 'points after crossover: ',proposed_pts
        
        return proposed_pts
        
    def snooker_update(self, n_proposed_pts, gamma, q0):
        print 'iteration: ',self.iter,' running snooker update'
        sampled_history_pt = [self.sample_from_history(self.nseedchains, self.DEpairs, self.total_var_dimension, snooker=True) for i in range(n_proposed_pts)]

        #Find mutually orthogonal vectors spanning current location and a randomly chosen chain from history
        ortho_vecs = []
        if n_proposed_pts == 1:
           vecs, r = np.linalg.qr(np.column_stack((q0, np.squeeze(sampled_history_pt))))
           ortho_vecs.append(vecs) 
        else:
            for i in range(n_proposed_pts):        
                vecs, r = np.linalg.qr(np.column_stack((q0, np.squeeze(sampled_history_pt)[i])))
                ortho_vecs.append(vecs)

        #Determine orthogonal projection of two other randomly chosen chains onto this span
        chains_to_be_projected = np.squeeze([np.array([self.sample_from_history(self.nseedchains, self.DEpairs, self.total_var_dimension, snooker=True) for i in range(2)]) for x in range(n_proposed_pts)])
        projected_pts = []
        for vec_set in range(len(ortho_vecs)):
            pts_for_set = []
            for chain_n in range(2):
                ortho_vec_list = [ortho_vecs[vec_set][:,0], ortho_vecs[vec_set][:,1]]
                pts_for_set.append([self.project_chains(ortho_vec_list, chains_to_be_projected[vec_set][chain_n])])
            projected_pts.append(pts_for_set)
        
        #Calculate difference between projected points
        chain_differences = np.array([np.array(projected_pts[i][0]) - np.array(projected_pts[i][1]) for i in range(n_proposed_pts)])
        
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
        if self.chain_n is None:
            with Dream_shared_vars.nchains.get_lock():
                self.chain_n = Dream_shared_vars.nchains.value-1
                Dream_shared_vars.nchains.value -= 1
        
        #We only need to have the current position of all chains for estimating the crossover probabilities during burn-in so don't bother updating after that
        if self.iter < self.crossover_burnin:
            start_cp = self.chain_n*ndimensions
            #print 'Chain number: ',self.chain_n
            end_cp = start_cp+ndimensions
            Dream_shared_vars.current_positions[start_cp:end_cp] = np.array(q_new).flatten()
            #print 'Added chain position to current position array.'
            #cp_array = np.frombuffer(Dream_shared_vars.current_positions.get_obj())
            #b = cp_array.reshape((2, 12))
            #print 'Current position array: ',b, ' Chain #: ',self.chain_n,' Total iterations: ',Dream_shared_vars.count.value
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
        