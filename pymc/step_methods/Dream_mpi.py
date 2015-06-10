# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 17:21:29 2014

@author: Erin
"""

from ..core import *
from .arraystep import *
import numpy as np
from datetime import datetime
import itertools
from mpi4py import MPI

__all__ = ['Dream_mpi']

class Dream_mpi(ArrayStep):
    def __init__(self, variables=None, nseedchains=None, nCR = 3, adapt_crossover = True, crossover_burnin=None, DEpairs=1, adaptationRate=.65, lamb=.05, zeta=1e-12, verbose=False, save_history = False, history_file = False, crossover_file = False, history_thin = 10, start_random=True, start_from_history=False, snooker=.10, p_gamma_unity = .20, multitry=False, parallel=False, model=None, **kwargs):
        
        model = modelcontext(model)
                
        if variables is None:
            variables = model.cont_vars
        
        if not set(variables).issubset(set(model.cont_vars)):
            raise Exception('The implemented version of Dream should only be run on continuous variables.')
        
        self.model = model
        self.variables = variables
        self.nseedchains = nseedchains
        self.nCR = nCR
        self.adapt_crossover = adapt_crossover
        self.crossover_burnin = crossover_burnin
        self.crossover_file = crossover_file
        self.CR_probabilities = np.array([1/float(self.nCR) for i in range(self.nCR)])
        self.CR_values = np.array([m/float(self.nCR) for m in range(1, self.nCR+1)])        
        self.DEpairs = np.linspace(1, DEpairs, num=DEpairs) #This is delta in original Matlab code
        self.snooker = snooker
        self.gamma = None
        self.p_gamma_unity = p_gamma_unity #This is the probability of setting gamma=1
        if multitry == False:
            self.multitry = 1
        elif multitry == True:
            self.multitry = 5
        else:
            self.multitry = multitry
        self.parallel = parallel
        self.lamb = lamb #This is e sub d in DREAM papers
        self.zeta = zeta #This is epsilon in DREAM papers
        self.last_logp = None
        self.total_var_dimension = 0
        for var in variables:
            var_name = getattr(model, str(var))
            self.total_var_dimension += var_name.dsize
        if self.nseedchains == None:
            self.nseedchains = self.total_var_dimension*10
        gamma_array = np.zeros((self.total_var_dimension, self.DEpairs))
        for delta in range(1, self.DEpairs+1):
            gamma_array[:,delta-1] = np.array([2.38 / np.sqrt(2*delta*np.linspace(1, self.total_var_dimension, num=self.total_var_dimension))])
        self.gamma_arr = gamma_array
        self.iter = 0  
        self.chain_n = None
        self.len_history = 0
        self.save_history = save_history
        self.history_file = history_file
        self.history_thin = history_thin
        self.start_random = start_random
    
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank() 
        self.nchains = None
        self.draws = None
        
        super(Dream_mpi, self).__init__(variables, [model.fastlogp], allvars=True, **kwargs)
    
    def astep(self, q0, logp, all_vars_point):
        # On first iteration, check that shared variables have been initialized (which only occurs if multiple chains have been started).
        if self.iter == 0:   
            print 'Dream has started'
            
            self.rand_state = np.random.RandomState(self.rank)
            print 'Set random seed to: ',self.rank       
            
            if self.rank == 0:
                if self.history_file != False:
                    old_history = np.load(self.history_file)
                    len_old_history = len(old_history.flatten())
                    arr_dim = np.floor((((self.nchains*self.draws)*self.total_var_dimension)/self.history_thin))+len_old_history
                    history_arr = np.zeros(arr_dim)
                    history_arr[0:len_old_history] = old_history.flatten()
                    print 'Loaded history file.'
                else:
                    #Set this because the draw from prior is not set to use a method of the RandomState Class set above.
                    np.random.seed(self.rank)
                    arr_dim = np.floor(((self.nchains*self.draws*self.total_var_dimension)/self.history_thin))+(self.nseedchains*self.total_var_dimension)                  
                    history_arr = np.zeros(arr_dim)
                    print 'Seeding history with draws from prior'
                    for i in range(self.nseedchains):
                        start_loc = i*self.total_var_dimension
                        end_loc = start_loc+self.total_var_dimension
                        history_arr[start_loc:end_loc] = self.draw_from_prior(self.model, self.variables)
                self.history_arr = history_arr
                
                if self.crossover_file != False:
                    crossover_probabilities = np.load(self.crossover_file)
                    self.CR_probabilities = crossover_probabilities
                else:
                    crossover_probabilities = self.CR_probabilities

                for rank in range(1, self.nchains):
                    self.comm.Send(arr_dim, dest=rank, tag=3)
                    self.comm.Send(history_arr, dest=rank, tag=1)
                    print 'Sent history array to rank: ',rank
                    self.comm.Send(crossover_probabilities, dest=rank, tag=2)
                    print 'Sent crossover probabilities to rank: ',rank
                
                self.ncr_updates = np.zeros(self.nCR)
                self.delta_m = np.zeros(self.nCR)

            else:
                arr_dim = np.zeros(1)
                self.comm.Recv(arr_dim, source=0, tag=3)
                self.history_arr = np.zeros(arr_dim)
                crossover_probabilities = np.zeros(self.nCR)
                self.comm.Recv(self.history_arr, source=0, tag=1)
                print 'Received history array on rank: ',self.rank
                self.comm.Recv(crossover_probabilities, source=0, tag=2)
                self.CR_probabilities = crossover_probabilities
                print 'Received crossover probabilities on rank: ',self.rank
            
            if self.start_random:
                #Set this because the draw from prior function isn't set to use a method of the RandomState class set above
                if self.rank != 0:
                    np.random.seed(self.rank)
                print 'Setting start to random draw from prior.'
                q0 = self.draw_from_prior(self.model, self.variables)
                print 'Start: ',q0
            
            self.count = np.array(0)
            self.current_positions = np.zeros(self.nchains*self.total_var_dimension)
            self.rank_tags = {rank: rank for rank in self.assigned_ranks}
            
            print 'Assigned ranks: ',self.assigned_ranks,' to rank: ',self.rank
                
        if self.snooker != 0:
            snooker_choice = np.where(self.rand_state.multinomial(1, [self.snooker, 1-self.snooker])==1)
            print 'In rank: ',self.rank,' snooker choice: ',snooker_choice
            #print 'Snooker choice: ',snooker_choice
            if snooker_choice[0] == 0:
                run_snooker = True
            else:
                run_snooker = False
        else:
            run_snooker = False
        


        #Set CR value for generating proposal point
        CR_loc = np.where(self.rand_state.multinomial(1, self.CR_probabilities)==1)
        print 'In rank: ',self.rank,' CR_loc: ',CR_loc
        #print 'CR_loc chosen: ',CR_loc
        CR = self.CR_values[CR_loc]
            
        #print 'Selected CR: ',CR
            
        if len(self.DEpairs)>1:
            DEpair_choice = self.rand_state.randint(1, len(self.DEpairs)+1, size=1)
        else:
            DEpair_choice = 1
        
        print 'In rank: ',self.rank,' DE pairs: ',DEpair_choice
        
        if self.snooker == 0 or run_snooker == False:
            #print 'Proposing pts with no snooker update. q0: ',q0,' CR: ',CR
            proposed_pts = self.generate_proposal_points(self.multitry, q0, CR, DEpair_choice, snooker=False)

        else:
            #print 'Proposing pts with snooker update. q0: ',q0,' CR: ',CR
            proposed_pts, snooker_logp_prop, z = self.generate_proposal_points(self.multitry, q0, CR, DEpair_choice, snooker=True)
        
        if self.multitry == 1:
            q_logp = logp(np.squeeze(proposed_pts))
            q = np.squeeze(proposed_pts)

        else:
               
            if self.parallel:
                sent_ranks = []
                tags = []
                for rank_pt in zip(itertools.cycle(self.assigned_ranks), np.squeeze(proposed_pts)):
                    #print 'Sending rank pt combo: ',rank_pt,' from rank: ',self.rank
                    pts_to_send = [rank_pt[1], all_vars_point]
                    self.comm.send(obj=pts_to_send, dest=rank_pt[0], tag=self.rank_tags[rank_pt[0]])
#                    comm.send(obj=self, dest=rank, tag=4)
#                    print 'Sent instance: ',self,' to rank: ',rank
#                    comm.send(obj=np.squeeze(proposed_pts)[rank], dest=rank, tag=5)
                    sent_ranks.append(rank_pt[0])
                    tags.append(self.rank_tags[rank_pt[0]])
                    self.rank_tags[rank_pt[0]] += 1
                log_ps = []
                for rank, tag in zip(sent_ranks, tags):
                    eval_pt = self.comm.recv(source=rank, tag=tag)
                    log_ps.append(eval_pt)

            else:
                log_ps = []
                for pt in np.squeeze(proposed_pts):
                    log_ps.append(logp(pt))
            #Check if all logps are -inf, in which case they'll all be impossible and we need to generate more proposal points
            while np.all(np.isfinite(np.array(log_ps))==False):
                print 'All logps infinite. Generating new proposal. Old logps: ',log_ps
                if run_snooker is True:
                    proposed_pts, snooker_logp_prop, z = self.generate_proposal_points(self.multitry, q0, CR, DEpair_choice, snooker=run_snooker)
                else:
                    proposed_pts = self.generate_proposal_points(self.multitry, q0, CR, DEpair_choice, snooker=run_snooker)
                log_ps = []
                for pt in np.squeeze(proposed_pts):
                    log_ps.append(logp(pt))
                print 'Generated new logps. New logps: ',log_ps

            #Randomly select one of the tested points with probability proportional to the probability density at the point
            log_ps = np.array(log_ps)
                
            #Substract largest logp from all logps (this from original Matlab code)
            max_logp = np.amax(log_ps)
            log_ps_sub = np.exp(log_ps - max_logp)
                
            #Calculate probabilities
            sum_proposal_logps = np.sum(log_ps_sub)
            logp_prob = log_ps_sub/sum_proposal_logps
            best_logp_loc = np.where(self.rand_state.multinomial(1, logp_prob)==1)[0]
            #print 'logps: ',log_ps,'max_logp: ',max_logp,'log_ps_sub: ',log_ps_sub,'sum_proposal_logps: ',sum_proposal_logps,'logp_prob: ',logp_prob,'best_logp_loc: ',best_logp_loc
            q_proposal = np.squeeze(proposed_pts[best_logp_loc])
            #print 'logps proposed = '+str(log_ps)+' Selected logp = '+str(log_ps[best_logp_loc])+' Point = '+str(q_proposal)
            
            #Draw reference points around the randomly selected proposal point
            if run_snooker is True:
                reference_pts, snooker_logp_ref, z_ref = self.generate_proposal_points(self.multitry-1, q_proposal, CR, DEpair_choice, snooker=run_snooker)
            else:
                reference_pts = self.generate_proposal_points(self.multitry-1, q_proposal, CR, DEpair_choice, snooker=run_snooker)
            #print 'Generated reference points: ',reference_pts
            
            #Compute posterior density at reference points.
            if self.multitry > 2:
                if self.parallel:
                    sent_ranks = []
                    tags = []
                    for rank_pt in zip(itertools.cycle(self.assigned_ranks), np.squeeze(reference_pts)):
                        pts_to_send = [rank_pt[1], all_vars_point]
                        self.comm.send(obj=pts_to_send, dest=rank_pt[0], tag=self.rank_tags[rank_pt[0]])
                        sent_ranks.append(rank_pt[0])
                        tags.append(self.rank_tags[rank_pt[0]])
                        self.rank_tags[rank_pt[0]] += 1
                    ref_log_ps = []
                    for rank, tag in zip(sent_ranks, tags):
                        eval_pt = self.comm.recv(source=rank, tag=tag)
                        ref_log_ps.append(eval_pt)
                    
                else:
                    ref_log_ps = []
                    for pt in np.squeeze(reference_pts):
                        ref_log_ps.append(logp(pt))
            else:
                ref_log_ps = np.array([logp(np.squeeze(reference_pts))])
            
            #print 'logp of proposed point = ', q_logp        
            #print 'Reference logps = ',ref_log_ps
        
        if self.last_logp == None:
            self.last_logp = logp(q0)
        
        if self.multitry > 1:
            ref_log_ps = np.append(ref_log_ps, self.last_logp)

            if run_snooker is True:
                total_proposal_logp = log_ps + snooker_logp_prop
                #Goal is to determine the ratio =  p(y) * p(y --> X) / p(Xref) * p(Xref --> X) where y = proposal point, X = current point, and Xref = reference point
                # First determine p(y --> X) (i.e. moving from proposed point y to original point X)
                # p(y --> X) equals ||y - z||^(n-1), i.e. the snooker_logp for the proposed point
                # p(Xref --> X) is equal to p(Xref --> y) * p(y --> X) (i.e. moving from Xref to proposed point y to original point X)
                snooker_logp_ref = np.append(snooker_logp_ref, 0)
                total_reference_logp = ref_log_ps + snooker_logp_ref + snooker_logp_prop
                
            else:
                total_proposal_logp = log_ps
                total_reference_logp = ref_log_ps
                
            #Determine max logp for all proposed and reference points
            max_logp = np.amax(np.concatenate((total_proposal_logp, total_reference_logp)))
            
            #Calculate weights for proposed points
            weight_proposed = np.amax(np.exp(total_proposal_logp - max_logp))
            weight_reference = np.amax(np.exp(total_reference_logp - max_logp))
                
            q_new = metrop_select(weight_proposed - weight_reference, q_proposal, q0)
            q_logp = logp(q_proposal)   
            
        else:  
            if run_snooker is True:
                total_proposed_logp = q_logp + snooker_logp_prop
                snooker_current_logp = np.log(np.linalg.norm(q0-z))*(self.total_var_dimension-1)
                total_old_logp = self.last_logp + snooker_current_logp
                    
                q_new = metrop_select(total_proposed_logp - total_old_logp, q, q0)
            else:
                q_new = metrop_select(q_logp - self.last_logp, q, q0)  
#                
#            #if np.array_equal(q0, q_new) and self.multitry > 1:
#                #print 'Did not accept point. Old logp: '+str(self.last_logp)+' Old weighted logps: '+str(weight_reference)+' Tested weighted logps: '+str(weight_proposed)+' Tested logp: '+str(q_logp)+' Logp ratio: ',weight_proposed - weight_reference
#            #elif np.array_equal(q0, q_new) and run_snooker == True:
#                #print 'Did not accept point. Old logp: '+str(self.last_logp)+' Old weighted logp '+str(total_old_logp)+' Tested weighted logp: '+str(total_proposed_logp)+' Tested logp: '+str(q_logp)
#            #elif np.array_equal(q0, q_new) and run_snooker == False:
#                #print 'Did not accept point. Old logp: ',str(self.last_logp)+' New logp: ',str(q_logp)
#            #else:
#                #if self.multitry > 1:
#                    #print 'Accepted point.  Old weighted logps: '+str(weight_reference)+' Tested weighted logps: '+str(weight_proposed)+' Old logp: '+str(self.last_logp)+' Tested logp: '+str(q_logp)
#                #elif run_snooker == True:
#                    #print 'Accepted point.  Old weighted logp: '+str(total_old_logp)+' Tested weighted logp: '+str(total_proposed_logp)+' Old logp: ',self.last_logp+' Tested logp: '+str(q_logp)
#                #else:
#                    #print 'Accepted point. Old logp: '+str(self.last_logp)+' New logp: '+str(q_logp)
#                    
        self.last_logp = q_logp
        
        #Place new point in history given history thinning rate
        if self.iter % self.history_thin == 0:
            print 'iteration: '+str(self.iter)+' adding point to history.'
            self.record_history(self.nseedchains, self.total_var_dimension, q_new)
        
        #If adapting crossover values, estimate ideal crossover probabilities for each dimension during burn-in.
        #Don't count iterations where gamma was set to 1 in crossover adaptation calculations
        if self.adapt_crossover is True and self.iter < self.crossover_burnin:
            #If a snooker update was run, then regardless of the originally selected CR, a CR=1.0 was used.
            if run_snooker is False:
                self.estimate_crossover_probabilities(self.iter, self.total_var_dimension, q0, q_new, CR) 
            else:
                self.estimate_crossover_probabilities(self.iter, self.total_var_dimension, q0, q_new, CR=1) 
        
        self.iter += 1

        if self.iter==self.draws:
            if self.rank == 0 and self.save_history:
                date_time_str = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')+'_'
                self.save_history_to_disc(self.history_arr, date_time_str)
            
            #Send quit signals to other ranks.
            for slave in self.assigned_ranks:
                self.comm.send(obj='quit', dest=slave, tag=self.rank_tags[slave])

        self.chain_comm.Barrier()
        return q_new
    
    def estimate_crossover_probabilities(self, iteration, ndim, q0, q_new, CR):
        
        if self.rank in range(1, self.nchains):
            self.comm.Send(q_new, dest=0, tag=14)
        else:
            pt_to_add = np.zeros(q_new.shape)
            for rank in range(1, self.nchains):
                self.comm.Recv(pt_to_add, source=rank, tag=14)
                start_cp = rank*ndim
                end_cp = start_cp+ndim
                self.current_positions[start_cp:end_cp] = np.array(pt_to_add).flatten()
            start_cp = self.rank*ndim
            end_cp = start_cp+ndim
            self.current_positions[start_cp:end_cp] = np.array(q_new).flatten()
            
        if self.rank == 0:
            current_posit_reshape = self.current_positions.reshape((self.nchains, ndim))
        
            #print 'Replaced current position of current chain with new point. Current positions: ',current_positions
            
            sd_by_dim = np.std(current_posit_reshape, axis=0)
            for rank in range(1, self.nchains):
                self.comm.Send(sd_by_dim, dest=rank, tag=15)
        
        if self.rank in range(1, self.nchains):
            sd_by_dim = np.zeros(self.total_var_dimension)
            self.comm.Recv(sd_by_dim, source=0, tag=15)
            m_loc = np.where(self.CR_values == CR)[0]
            delta_m = np.nan_to_num(np.sum((q_new - q0)**2/sd_by_dim**2))
            use = True
            #Don't count iterations where gamma was set to 1 in crossover adaptation calculations
            if np.any(np.array(self.gamma)==1.0) != True:
                use = False
            delta_data = [m_loc, delta_m, use]
            self.comm.send(delta_data, dest=0, tag=16)
        
        self.chain_comm.Barrier()        
        
        if self.rank == 0:
            m_loc = np.where(self.CR_values == CR)[0]
            self.ncr_updates[m_loc] += 1
            self.delta_m[m_loc] = self.delta_m[m_loc] + np.nan_to_num(np.sum((q_new - q0)**2/sd_by_dim**2))
            for rank in range(1, self.nchains):
                delta_data = self.comm.recv(source=rank, tag=16)
                if delta_data[2] is True:
                    self.ncr_updates[delta_data[0]] += 1
                    self.delta_m[delta_data[0]] += delta_data[1]
    
            sum_delta_m_per_iter = np.sum(self.delta_m/self.ncr_updates)
            
            if np.all(self.delta_m != 0) == True:
                #print 'All values have been successful at least once.  Changing crossover probabilities to reflect delta m values.'
                for m in range(self.nCR):
                    self.CR_probabilities[m] = (self.delta_m[m]/self.ncr_updates[m])/sum_delta_m_per_iter
            
                for rank in range(1, self.nchains):
                    self.comm.Send(self.CR_probabilities, dest=rank, tag=17)
        
                if self.rank in range(1, self.nchains):
                    self.comm.Recv(self.CR_probabilities, source=0, tag=17)
            
         
    def set_gamma(self, iteration, DEpairs, snooker_choice, CR, d_prime):
        gamma_unity_choice = np.where(self.rand_state.multinomial(1, [self.p_gamma_unity, 1-self.p_gamma_unity])==1)
        
        if snooker_choice == True:
            gamma = self.rand_state.uniform(1.2, 2.2)
            
        elif gamma_unity_choice[0] == 0:
            gamma = 1.0
        
        else:
            #gamma = np.array([2.38 / np.sqrt( 2 * DEpairs  * d_prime)])
            gamma = self.gamma_arr[d_prime-1][DEpairs-1]
        
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
            limit = self.count + nseedchains - 1
            chain_num = self.rand_state.random_integers(limit, size=DEpairs*4)
        else:
            limit = self.count + nseedchains - 1
            chain_num = self.rand_state.random_integers(limit, size=1)
            
        print 'In rank: ',self.rank,' selected chain #s: ',chain_num
        start_locs = [i*ndimensions for i in chain_num]
        end_locs = [i+ndimensions for i in start_locs]
        sampled_chains = [self.history_arr[start_loc:end_loc] for start_loc, end_loc in zip(start_locs, end_locs)]

        return sampled_chains
        
    def generate_proposal_points(self, n_proposed_pts, q0, CR, DEpairs, snooker):
        if snooker is False:
            #print 'Generating pts with no snooker update. n proposed pts= ',n_proposed_pts
            sampled_history_pts = np.array([self.sample_from_history(self.nseedchains, DEpairs, self.total_var_dimension) for i in range(n_proposed_pts)])
    
            #print 'history shape: ',sampled_history_pts.shape           
            chain_differences = np.array([np.sum(sampled_history_pts[i][0:2*DEpairs], axis=0)-np.sum(sampled_history_pts[i][2*DEpairs:DEpairs*4], axis=0) for i in range(len(sampled_history_pts))])
            #print 'chain_differences_shape: ',chain_differences.shape            
            #print 'Generated chain differences with DEpairs>0.  chain differences = ',chain_differences
            zeta = np.array([self.rand_state.normal(0, self.zeta, self.total_var_dimension) for i in range(n_proposed_pts)])
            e = np.array([self.rand_state.uniform(-self.lamb, self.lamb, self.total_var_dimension) for i in range(n_proposed_pts)])
            d_prime = self.total_var_dimension
            U = self.rand_state.uniform(0, 1, size=chain_differences.shape)
            if n_proposed_pts > 1:
                d_prime = [len(U[point][np.where(U[point]<CR)]) for point in range(n_proposed_pts)]
                self.gamma = [self.set_gamma(self.iter, DEpairs, snooker, CR, d_p) for d_p in d_prime]
                #print 'd_primes: ',d_prime
                #print 'gammas: ',self.gamma
            else:
                d_prime = len(U[np.where(U<CR)])
                self.gamma = self.set_gamma(self.iter, DEpairs, snooker, CR, d_prime)
            #else:
            #    self.gamma = self.set_gamma(self.iter, DEpairs, snooker, CR, d_prime)
            
            if n_proposed_pts > 1:
                proposed_pts = [q0 + e[point]*gamma*chain_differences[point] + zeta[point] for point, gamma in zip(range(n_proposed_pts), self.gamma)]
            else:
                proposed_pts = q0+ e*self.gamma*chain_differences + zeta
            #print 'proposed points: ',proposed_pts
            if self.adapt_crossover is True and d_prime != self.total_var_dimension:
                if n_proposed_pts > 1:
                    for point, pt_num in zip(proposed_pts, range(n_proposed_pts)):
                        point[np.where(U[pt_num]>CR)] = q0[np.where(U[pt_num]>CR)]
                else:
                    proposed_pts[np.where(U>CR)] = q0[np.where(U>CR)[1]]
            #print n_proposed_pts,' proposed pts generated without snooker update. Proposed pts = ',proposed_pts  
        
        else:
            self.gamma = self.set_gamma(self.iter, DEpairs, snooker, CR, self.total_var_dimension)
            proposed_pts, snooker_logp, z = self.snooker_update(n_proposed_pts, q0)
            #print n_proposed_pts,' proposed pts generated with snooker update. Proposed pts = ',proposed_pts
        
        if snooker is False:
            return proposed_pts
        else:
            return proposed_pts, snooker_logp, z
        
    def snooker_update(self, n_proposed_pts, q0):
        #print 'iteration: ',self.iter,' running snooker update'
        sampled_history_pt = [self.sample_from_history(self.nseedchains, self.DEpairs, self.total_var_dimension, snooker=True) for i in range(n_proposed_pts)]
        #print 'sampled history point: ',sampled_history_pt
        chains_to_be_projected = np.squeeze([np.array([self.sample_from_history(self.nseedchains, self.DEpairs, self.total_var_dimension, snooker=True) for i in range(2)]) for x in range(n_proposed_pts)])
        #print 'chains to be projected shape: ',chains_to_be_projected.shape        
        #Define projection vector
        proj_vec_diff = np.squeeze(q0-sampled_history_pt)
        #print 'proj vec diff: ',proj_vec_diff
        if n_proposed_pts > 1:
            D = [np.dot(proj_vec_diff[point], proj_vec_diff[point]) for point in range(len(proj_vec_diff))]
            
            #Orthogonal projection of chains_to_projected onto projection vector
            diff_chains_to_be_projected = [(chains_to_be_projected[point][0]-chains_to_be_projected[point][1]) for point in range(n_proposed_pts)]
            zP = np.nan_to_num(np.array([np.dot(diff_chains_to_be_projected[point], proj_vec_diff[point])/D[point] for point in range(n_proposed_pts)]))
            dx = self.gamma*zP
            proposed_pts = [q0 + dx[point] for point in range(n_proposed_pts)]
            snooker_logp = [np.log(np.linalg.norm(proposed_pts[point]-sampled_history_pt[point]))*(self.total_var_dimension-1) for point in range(n_proposed_pts)]
        else:
            D = np.dot(proj_vec_diff, proj_vec_diff)

            #Orthogonal projection of chains_to_projected onto projection vector  
            diff_chains_to_be_projected = chains_to_be_projected[0]-chains_to_be_projected[1]
            zP = np.nan_to_num(np.array([np.dot(diff_chains_to_be_projected, proj_vec_diff)/D]))
            dx = self.gamma*zP
            proposed_pts = q0 + dx
            snooker_logp = np.log(np.linalg.norm(proposed_pts-sampled_history_pt))*(self.total_var_dimension-1)
        
        return proposed_pts, snooker_logp, sampled_history_pt
    
    def record_history(self, nseedchains, ndimensions, q_new):
        if self.rank != 0:
            self.comm.Send(q_new, dest=0, tag=11)
            
        else:
            pt_to_add = np.zeros(q_new.shape)
            for rank in range(1, self.nchains):
                self.comm.Recv(pt_to_add, source=rank, tag=11)
                print 'Adding point: ',pt_to_add,' from rank: ',rank
                nhistoryrecs = self.count+nseedchains+rank
                start_loc = nhistoryrecs*ndimensions
                end_loc = start_loc+ndimensions
                self.history_arr[start_loc:end_loc] = np.array(pt_to_add).flatten()
            nhistoryrecs = self.count+nseedchains+self.rank
            start_loc = nhistoryrecs*ndimensions
            end_loc = start_loc+ndimensions
            print 'Adding point: ',q_new,' from rank 0.'
            self.history_arr[start_loc:end_loc] = np.array(q_new).flatten()
        
        self.chain_comm.Barrier()        
        
        if self.rank == 0:
            self.count += self.nchains
            
            for rank in range(1, self.nchains):
                self.comm.Send(self.history_arr, dest=rank, tag=12)
                self.comm.Send(self.count, dest=rank, tag=13)
        else:
            self.comm.Recv(self.history_arr, source=0, tag=12)
            self.comm.Recv(self.count, source=0, tag=13)
        
        self.chain_comm.Barrier()
            
    def save_history_to_disc(self, history, prefix):
        filename = prefix+'DREAM_chain_history.npy'
        print 'Saving history to file: ',filename
        #Also save crossover probabilities if adapted
        np.save(filename, history)
        filename = prefix+'DREAM_chain_adapted_crossoverprob.npy'
        print 'Saving fitted crossover values: ',self.CR_probabilities,' to file.'
        np.save(filename, self.CR_probabilities)
    
        