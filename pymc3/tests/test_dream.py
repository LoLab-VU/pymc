# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 16:50:59 2014

@author: Erin
"""

import unittest
from pymc.tests.models import simple_model, mv_simple_discrete, multidimensional_model, mv_simple
import pymc as pm
from pymc.step_methods import Dream_shared_vars
import multiprocessing as mp
import numpy as np
from os import remove

class Test_Dream_Initialization(unittest.TestCase):
    
    def test_fail_with_one_chain(self):
        self.start, self.model, (self.mu, self.C) = simple_model() 
        with self.model:        
            self.step = pm.Dream()
        self.assertRaisesRegexp(Exception, 'Dream should be run with multiple chains in parallel.  Set njobs > 1.', pm.sample, draws=1, step=self.step, model=self.model, njobs=1)
    
    def test_fail_with_discrete_vars(self):
        self.start, self.model, (self.mu, self.C) = mv_simple_discrete()
        self.assertRaisesRegexp(Exception, 'The implemented version of Dream should only be run on continuous variables.', pm.Dream, variables=[self.model.x], model=self.model)
    
    def test_total_var_dimension_init(self):
        self.start, self.model, (self.mu, self.C) = simple_model()
        with self.model as model:
            step = pm.Dream()
        self.assertEqual(step.total_var_dimension, 2)
        self.start, self.model, (self.mu, self.C) = multidimensional_model()
        with self.model as model:
            step = pm.Dream()
        self.assertEqual(step.total_var_dimension, 6)
        self.start, self.model, (self.mu, self.C) = mv_simple()
        with self.model as model:
            step = pm.Dream()
        self.assertEqual(step.total_var_dimension, 3)

class Test_Dream_Algorithm_Components(unittest.TestCase):
    
    def test_gamma_unityfraction(self):
        self.start, self.model, (self.mu, self.C) = simple_model()
        n_unity_choices = 0
        with self.model as model:
            step = pm.Dream()
        fraction = step.p_gamma_unity
        for iteration in range(10000):
           choice = pm.Dream(model=self.model).set_gamma(iteration, DEpairs=1, snooker_choice=False, d_prime=step.total_var_dimension) 
           if choice == 1:
               n_unity_choices += 1
        emp_frac = n_unity_choices/10000.0
        self.assertAlmostEqual(emp_frac, fraction, places=1)
    
    def test_gamma_array(self):
        self.start, self.model, (self.mu, self.C) = simple_model()
        true_gamma_array = np.array([[1.683, 1.19, .972, .841, .753], [1.19, .841, .687, .595, .532]])
        with self.model as model:        
            step = pm.Dream(DEpairs=5, p_gamma_unity=0)
        for d_prime in range(1, step.total_var_dimension+1):
            for n_DEpair in range(1, 6):
                self.assertAlmostEqual(true_gamma_array[d_prime-1][n_DEpair-1], step.set_gamma(iteration=0, DEpairs=n_DEpair, snooker_choice=False, d_prime=d_prime), places=3)
    
    def test_gamma_snooker_choice(self):
        self.start, self.model, (self.mu, self.C) = simple_model()
        with self.model as model:        
            step = pm.Dream()
        self.assertGreaterEqual(step.set_gamma(iteration=0, DEpairs=1, snooker_choice=True, d_prime=3), 1.2)
        self.assertLess(step.set_gamma(iteration=0, DEpairs=1, snooker_choice=True, d_prime=3), 2.2)
    
    def test_snooker_fraction(self):
        self.start, self.model, (self.mu, self.C) = simple_model()
        n_snooker_choices = 0
        with self.model as model:        
            step = pm.Dream()
        fraction = step.snooker
        for iteration in range(10000):
           choice = pm.Dream(model=self.model).set_snooker()
           if choice == True:
               n_snooker_choices += 1
        emp_frac = n_snooker_choices/10000.0
        self.assertAlmostEqual(emp_frac, fraction, places=1)   
        
    def test_CR_fraction(self):
        self.start, self.model, (self.mu, self.C) = simple_model()
        nCR1 = 0
        nCR2 = 0
        nCR3 = 0
        crossoverprobs = np.array([.10, .65, .25])
        crossovervals = np.array([.33, .66, 1.0])
        with self.model as model:
            for iteration in range(10000):
                choice = pm.Dream().set_CR(crossoverprobs, crossovervals)
                if choice == crossovervals[0]:
                    nCR1 += 1
                elif choice == crossovervals[1]:
                    nCR2 += 1
                else:
                    nCR3 += 1
        emp_frac1 = nCR1/10000.0
        emp_frac2 = nCR2/10000.0
        emp_frac3 = nCR3/10000.0
        self.assertAlmostEqual(emp_frac1, crossoverprobs[0], places=1)
        self.assertAlmostEqual(emp_frac2, crossoverprobs[1], places=1)
        self.assertAlmostEqual(emp_frac3, crossoverprobs[2], places=1)
    
    def test_DEpair_selec(self):
        self.start, self.model, (self.mu, self.C) = simple_model()
        single_DEpair = np.array([1.0])
        multi_DEpair = np.array([1.0, 2.0, 3.0])
        nDE1 = 0
        nDE2 = 0
        nDE3 = 0
        with self.model as model:
            self.assertEqual(pm.Dream().set_DEpair(single_DEpair), 1)
            for iteration in range(10000):
                choice = pm.Dream().set_DEpair(multi_DEpair)
                if choice == multi_DEpair[0]:
                    nDE1 += 1
                elif choice == multi_DEpair[1]:
                    nDE2 += 1
                else:
                    nDE3 += 1
        emp_frac1 = nDE1/10000.0
        emp_frac2 = nDE2/10000.0
        emp_frac3 = nDE3/10000.0
        self.assertAlmostEqual(emp_frac1, .3, places=1)
        self.assertAlmostEqual(emp_frac2, .3, places=1)
        self.assertAlmostEqual(emp_frac3, .3, places=1)
    
    def test_prior_draw(self):
        self.start, self.model, (self.mu, self.C) = simple_model()
        self.assertEqual(len(pm.Dream(model=self.model).draw_from_prior(self.model, self.model.cont_vars)), 2)
        self.start, self.model, (self.mu, self.C) = multidimensional_model()
        self.assertEqual(len(pm.Dream(model=self.model).draw_from_prior(self.model, self.model.cont_vars)), 6)
        self.start, self.model, (self.mu, self.C) = mv_simple()
        self.assertRaises(Exception, pm.Dream(model=self.model).draw_from_prior, self.model, self.model.cont_vars)

    def test_chain_sampling_simple_model(self):
        self.start, self.model, (self.mu, self.C) = simple_model()
        with self.model as model:
            step = pm.Dream()
        history_arr = mp.Array('d', [0]*2*step.total_var_dimension)
        n = mp.Value('i', 0)
        Dream_shared_vars.history = history_arr
        Dream_shared_vars.count = n
        chains_added_to_history = []
        for i in range(2):
            start = i*step.total_var_dimension
            end = start+step.total_var_dimension
            chain = step.draw_from_prior(step.model, step.variables)
            Dream_shared_vars.history[start:end] = chain
            chains_added_to_history.append(chain)
        sampled_chains = step.sample_from_history(nseedchains=2, DEpairs=1, ndimensions=step.total_var_dimension)
        sampled_chains = np.array(sampled_chains)
        chains_added_to_history = np.array(chains_added_to_history)
        self.assertIs(np.array_equal(chains_added_to_history[chains_added_to_history[:,0].argsort()], sampled_chains[sampled_chains[:,0].argsort()]), True)
    
    def test_chain_sampling_multidim_model(self):
        self.start, self.model, (self.mu, self.C) = multidimensional_model()
        with self.model as model:
            step = pm.Dream()
        history_arr = mp.Array('d', [0]*2*step.total_var_dimension)
        n = mp.Value('i', 0)
        Dream_shared_vars.history = history_arr
        Dream_shared_vars.count = n
        chains_added_to_history = []
        for i in range(2):
            start = i*step.total_var_dimension
            end = start+step.total_var_dimension
            chain = step.draw_from_prior(step.model, step.variables)
            Dream_shared_vars.history[start:end] = chain
            chains_added_to_history.append(chain)       
        sampled_chains = step.sample_from_history(nseedchains=2, DEpairs=1, ndimensions=step.total_var_dimension)
        sampled_chains = np.array(sampled_chains)
        chains_added_to_history = np.array(chains_added_to_history)
        self.assertIs(np.array_equal(chains_added_to_history[chains_added_to_history[:,0].argsort()], sampled_chains[sampled_chains[:,0].argsort()]), True)
    
    def test_proposal_generation_nosnooker_CR1(self):
        self.start, self.model, (self.mu, self.C) = multidimensional_model()
        with self.model as model:
            step = pm.Dream()
        history_arr = mp.Array('d', range(120))
        n = mp.Value('i', 0)
        Dream_shared_vars.history = history_arr
        Dream_shared_vars.count = n
        step.nseedchains = 20
        q0 = np.array([2, 3, 4, 5, 6, 7])
        dims_kept = 0
        for iteration in range(10000):
            proposed_pt = step.generate_proposal_points(n_proposed_pts=1, q0=q0, CR=1, DEpairs=1, snooker=False)
            if iteration == 1:
                self.assertEqual(len(proposed_pt), 1)
            dims_change_vec = np.squeeze(q0 == proposed_pt)
            for dim in dims_change_vec:
                if dim:
                    dims_kept += 1
        frac_kept = dims_kept/(step.total_var_dimension*10000.0)
        self.assertAlmostEqual(frac_kept, 0, places=1)
        dims_kept = 0
        for iteration in range(1000):
            proposed_pts = step.generate_proposal_points(n_proposed_pts=5, q0=q0, CR=1, DEpairs=1, snooker=False)
            if iteration == 1:
                self.assertEqual(len(proposed_pts), 5)
            for pt in proposed_pts:
                dims_change_vec = (q0 == pt)
                for dim in dims_change_vec:
                    if dim:
                        dims_kept += 1
        frac_kept = dims_kept/(step.total_var_dimension*1000.0*5)
        self.assertAlmostEqual(frac_kept, 0, places=1)
                
    
    def test_proposal_generation_nosnooker_CR33(self):
        self.start, self.model, (self.mu, self.C) = multidimensional_model()
        with self.model as model:
            step = pm.Dream()
        history_arr = mp.Array('d', range(120))
        n = mp.Value('i', 0)
        Dream_shared_vars.history = history_arr
        Dream_shared_vars.count = n
        step.nseedchains = 20
        q0 = np.array([2, 3, 4, 5, 6, 7])
        dims_kept = 0
        for iteration in range(100000):
            proposed_pt = step.generate_proposal_points(n_proposed_pts=1, q0=q0, CR=.33, DEpairs=1, snooker=False)
            if iteration == 1:
                self.assertEqual(len(proposed_pt), 1)
            dims_change_vec = np.squeeze(q0 == proposed_pt)
            for dim in dims_change_vec:
                if dim:
                    dims_kept += 1
        frac_kept = dims_kept/(step.total_var_dimension*100000.0)
        self.assertAlmostEqual(frac_kept, 1-.33, places=1)
        dims_kept = 0
        for iteration in range(10000): 
            proposed_pts = step.generate_proposal_points(n_proposed_pts=5, q0=q0, CR=.33, DEpairs=1, snooker=False)
            if iteration == 1:
                self.assertEqual(len(proposed_pts), 5)
            for pt in proposed_pts:
                dims_change_vec = (q0 == pt)
                for dim in dims_change_vec:
                    if dim:
                        dims_kept += 1
        frac_kept = dims_kept/(step.total_var_dimension*10000.0*5)
        self.assertAlmostEqual(frac_kept, 1-.33, places=1)
    
    def test_proposal_generation_nosnooker_CR66(self):
        self.start, self.model, (self.mu, self.C) = multidimensional_model()
        with self.model as model:
            step = pm.Dream()
        history_arr = mp.Array('d', range(120))
        n = mp.Value('i', 0)
        Dream_shared_vars.history = history_arr
        Dream_shared_vars.count = n
        step.nseedchains = 20
        q0 = np.array([2, 3, 4, 5, 6, 7])
        dims_kept = 0
        for iteration in range(100000):
            proposed_pt = step.generate_proposal_points(n_proposed_pts=1, q0=q0, CR=.66, DEpairs=1, snooker=False)
            if iteration == 1:
                self.assertEqual(len(proposed_pt), 1)
            dims_change_vec = np.squeeze(q0 == proposed_pt)
            for dim in dims_change_vec:
                if dim:
                    dims_kept += 1
        frac_kept = dims_kept/(step.total_var_dimension*100000.0)
        self.assertAlmostEqual(frac_kept, 1-.66, places=1)
        dims_kept = 0
        for iteration in range(10000): 
            proposed_pts = step.generate_proposal_points(n_proposed_pts=5, q0=q0, CR=.66, DEpairs=1, snooker=False)
            if iteration == 1:
                self.assertEqual(len(proposed_pts), 5)
            for pt in proposed_pts:
                dims_change_vec = (q0 == pt)
                for dim in dims_change_vec:
                    if dim:
                        dims_kept += 1
        frac_kept = dims_kept/(step.total_var_dimension*10000.0*5)
        self.assertAlmostEqual(frac_kept, 1-.66, places=1)
    
    def test_proposal_generation_snooker(self):
        self.start, self.model, (self.mu, self.C) = multidimensional_model()
        with self.model as model:
            step = pm.Dream()
        history_arr = mp.Array('d', range(120))
        n = mp.Value('i', 0)
        Dream_shared_vars.history = history_arr
        Dream_shared_vars.count = n
        step.nseedchains = 20
        q0 = np.array([2, 3, 4, 5, 6, 7])
        proposed_pt, snooker_logp, z = step.generate_proposal_points(n_proposed_pts=1, q0=q0, CR=1, DEpairs=1, snooker=True)
        self.assertEqual(len(proposed_pt), step.total_var_dimension)
        proposed_pts, snooker_logp, z = step.generate_proposal_points(n_proposed_pts=5, q0=q0, CR=1, DEpairs=1, snooker=True)
        self.assertEqual(len(proposed_pts), 5)
    
    def test_multitry_logp_eval(self):
        self.start, self.model, (self.mu, self.C) = multidimensional_model()
        with self.model as model:
            step = pm.Dream()
            logp_fxn = model.fastlogp
            variables = step.variables
            ordering = pm.blocking.ArrayOrdering(variables)
            all_vars_point = {'x': np.array([[5, 7], [2, 3], [1, 9]])}
            bij = pm.blocking.DictToArrayBijection(ordering, all_vars_point)
            logp = bij.mapf(logp_fxn)
            proposed_pts = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18]]
            logps = step.mt_evaluate_logps(parallel=False, multitry=3, proposed_pts=proposed_pts, logp=logp, all_vars_pt=all_vars_point)
            correct_logps = []
            for pt in proposed_pts:
                correct_logps.append(logp(pt))
            correct_logps = np.array(correct_logps)
            self.assertEqual(np.array_equal(logps, correct_logps), True)
        
            logps = step.mt_evaluate_logps(parallel=True, multitry=3, proposed_pts=proposed_pts, logp=logp, all_vars_pt=all_vars_point)
            self.assertEqual(np.array_equal(logps, correct_logps), True)
    
    def test_multitry_proposal_selection(self):
        self.start, self.model, (self.mu, self.C) = multidimensional_model()
        with self.model as model:
            step = pm.Dream()
        log_ps = np.array([1000, 500])
        proposed_pts = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
        for iteration in range(100):
            q_proposal, q_logp = step.mt_choose_proposal_pt(log_ps, proposed_pts)
            self.assertEqual(np.array_equal(q_proposal, proposed_pts[0]), True)
    
    def test_crossover_prob_estimation(self):
        self.start, self.model, (self.mu, self.C) = simple_model()
        with self.model as model:
            step = pm.Dream(save_history=False)
        starting_crossover = step.CR_probabilities
        crossover_probabilities = mp.Array('d', starting_crossover)
        n = mp.Value('i', 0)
        nCR = step.nCR
        CR_vals = step.CR_values
        ncrossover_updates = mp.Array('d', [0]*nCR)
        current_position_arr = mp.Array('d', [1, 2, 3, 4, 5, 6])
        delta_m = mp.Array('d', [0]*nCR)
        step.chain_n = 0
        Dream_shared_vars.cross_probs = crossover_probabilities
        Dream_shared_vars.count = n
        Dream_shared_vars.ncr_updates = ncrossover_updates
        Dream_shared_vars.current_positions = current_position_arr
        Dream_shared_vars.delta_m = delta_m
        q0 = np.array([1, 2])
        q_new = np.array([1, 2])
        new_cr_probs = step.estimate_crossover_probabilities(step.total_var_dimension, q0, q_new, CR_vals[0])
        self.assertEqual(np.array_equal(new_cr_probs, starting_crossover), True)
        q_new = np.array([7, 8])
        new_cr_probs = step.estimate_crossover_probabilities(step.total_var_dimension, q0, q_new, CR_vals[0])
        self.assertEqual(np.array_equal(new_cr_probs, starting_crossover), True)
        q_new = np.array([9, 10])
        new_cr_probs = step.estimate_crossover_probabilities(step.total_var_dimension, q0, q_new, CR_vals[1])
        self.assertEqual(np.array_equal(new_cr_probs, starting_crossover), True)
        q_new = np.array([11, 12])
        new_cr_probs = step.estimate_crossover_probabilities(step.total_var_dimension, q0, q_new, CR_vals[2])
        self.assertEqual(np.array_equal(new_cr_probs, starting_crossover), False)
        self.assertGreater(new_cr_probs[2], starting_crossover[2])
        self.assertAlmostEqual(np.sum(new_cr_probs), 1.0, places=1)
        old_cr_probs = new_cr_probs
        for i, q_new in zip(range(5), [np.array([15, 16]), np.array([17, 18]), np.array([19, 20]), np.array([21, 22]), np.array([23, 24])]):
            new_cr_probs = step.estimate_crossover_probabilities(step.total_var_dimension, q0, q_new, CR_vals[1])  
        self.assertEqual(np.array_equal(new_cr_probs, old_cr_probs), False)
        
    def test_history_recording_simple_model(self):
        self.start, self.model, (self.mu, self.C) = simple_model()
        with self.model as model:
            step = pm.Dream(model_name='test_history_recording')
        history_arr = mp.Array('d', [0]*4*step.total_var_dimension)
        n = mp.Value('i', 0)
        nchains = mp.Value('i', 3)
        Dream_shared_vars.history = history_arr
        Dream_shared_vars.count = n
        Dream_shared_vars.nchains = nchains
        test_history = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

        for point in [1, 2], [3, 4], [5,6], [7, 8]:
            step.record_history(nseedchains=0, ndimensions=step.total_var_dimension, q_new=point, len_history=len(history_arr))
        history_arr_np = np.frombuffer(Dream_shared_vars.history.get_obj())
        history_arr_np_reshaped = history_arr_np.reshape(np.shape(test_history))
        self.assertIs(np.array_equal(history_arr_np_reshaped, test_history), True)
        remove('test_history_recording_DREAM_chain_history.npy')
        remove('test_history_recording_DREAM_chain_adapted_crossoverprob.npy')
        
    def test_history_recording_multidim_model(self):
        self.start, self.model, (self.mu, self.C) = multidimensional_model()
        with self.model as model:
            step = pm.Dream(model_name='test_history_recording')
        history_arr = mp.Array('d', [0]*4*step.total_var_dimension)
        n = mp.Value('i', 0)
        nchains = mp.Value('i', 3)
        Dream_shared_vars.history = history_arr
        Dream_shared_vars.count = n
        Dream_shared_vars.nchains = nchains
        test_history = np.array([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]], [[13, 14], [15, 16], [17, 18]], [[19, 20], [21, 22], [23, 24]]])
        for point in [[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]], [[13, 14], [15, 16], [17, 18]], [[19, 20], [21, 22], [23, 24]]:
            step.record_history(nseedchains=0, ndimensions=step.total_var_dimension, q_new=point, len_history=len(history_arr))
        history_arr_np = np.frombuffer(Dream_shared_vars.history.get_obj())
        history_arr_np_reshaped = history_arr_np.reshape(np.shape(test_history))
        self.assertIs(np.array_equal(history_arr_np_reshaped, test_history), True)
        remove('test_history_recording_DREAM_chain_history.npy')
        remove('test_history_recording_DREAM_chain_adapted_crossoverprob.npy')
     
    def test_history_saving_to_disc_sanitycheck(self):
        self.start, self.model, (self.mu, self.C) = simple_model()
        with self.model as model:
            step = pm.Dream()
        history = np.array([[5, 8, 10, 12], [13, 18, 20, 21], [1, .5, 9, 1e9]])
        step.save_history_to_disc(history, 'testing_history_save_')
        history_saved = np.load('testing_history_save_DREAM_chain_history.npy')
        self.assertIs(np.array_equal(history, history_saved), True)
        remove('testing_history_save_DREAM_chain_history.npy')
        remove('testing_history_save_DREAM_chain_adapted_crossoverprob.npy')
    
    def test_history_file_loading(self):
        self.start, self.model, (self.mu, self.C) = simple_model()
        with self.model as model:
            step = pm.Dream()
        old_history = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]).flatten()
        step.save_history_to_disc(old_history, 'testing_history_load_')
        with self.model as model:
            step = pm.Dream(history_file='testing_history_load_DREAM_chain_history.npy', save_history=True, model_name='test_history_loading')
            trace = pm.sample(draws=3, step=step, njobs=3)
        new_history = np.load('test_history_loading_DREAM_chain_history.npy')
        self.assertEqual(len(new_history), (len(old_history.flatten())+(3*step.total_var_dimension)))
        new_history_seed = new_history[:len(old_history.flatten())]
        new_history_seed_reshaped = new_history_seed.reshape(old_history.shape)
        self.assertIs(np.array_equal(old_history, new_history_seed_reshaped), True)
        
        added_history = new_history[len(old_history.flatten())::]
        for i, variable in enumerate(step.variables):
            concatenated_traces = trace.get_values(str(variable), combine=True).flatten()
            var_history = np.array([])
            start = i
            for saved_draw in range(3):
                var_history = np.append(var_history, added_history[start:start+variable.dsize])
                start += step.total_var_dimension
            self.assertIs(np.array_equal(var_history[0:variable.dsize], concatenated_traces[0:variable.dsize]), True)
            self.assertIs(np.array_equal(var_history[variable.dsize:variable.dsize*2], concatenated_traces[3*variable.dsize:3*variable.dsize+variable.dsize]), True)
            self.assertIs(np.array_equal(var_history[variable.dsize*2:variable.dsize*3], concatenated_traces[2*3*variable.dsize:2*3*variable.dsize+variable.dsize]), True)
        remove('testing_history_load_DREAM_chain_history.npy')
        remove('testing_history_load_DREAM_chain_adapted_crossoverprob.npy')
        remove('test_history_loading_DREAM_chain_adapted_crossoverprob.npy')
        remove('test_history_loading_DREAM_chain_history.npy')
        
    def test_crossover_file_loading(self):
        self.start, self.model, (self.mu, self.C) = simple_model()
        old_crossovervals = np.array([.45, .20, .35])
        np.save('testing_crossoverval_load_DREAM.npy', old_crossovervals)
        with self.model as model:
            step = pm.Dream(crossover_file='testing_crossoverval_load_DREAM.npy', save_history=True, model_name='testing_crossover_load')
            trace = pm.sample(draws=3, step=step, njobs=3)
        self.assertIs(np.array_equal(step.CR_probabilities, old_crossovervals), True)
        
        with self.model as model:
            trace = pm.sample(draws=100, step=step, njobs=3)
        
        crossover_vals_after_sampling = np.load('testing_crossover_load_DREAM_chain_adapted_crossoverprob.npy')
        self.assertIs(np.array_equal(crossover_vals_after_sampling, old_crossovervals), True)
        remove('testing_crossover_load_DREAM_chain_adapted_crossoverprob.npy')
        remove('testing_crossoverval_load_DREAM.npy')
        remove('testing_crossover_load_DREAM_chain_history.npy')
        
    
class Test_Dream_Full_Algorithm(unittest.TestCase):

    def test_history_correct_after_sampling_simple_model(self):
        self.start, self.model, (self.mu, self.C) = simple_model()
        with self.model as model:
            step = pm.Dream(save_history=True, history_thin=1, model_name='test_history_correct', adapt_crossover=False)
            trace = pm.sample(draws=10, step=step, njobs=5)
        history = np.load('test_history_correct_DREAM_chain_history.npy')
        self.assertEqual(len(history), step.total_var_dimension*((10*5/step.history_thin)+step.nseedchains))
        history_no_seedchains = history[(step.total_var_dimension*step.nseedchains)::] 
        for i, variable in enumerate(step.variables):
            concatenated_traces = trace.get_values(str(variable), combine=True).flatten()
            var_history = np.array([])
            start = i
            for draw in range(50):
                var_history = np.append(var_history, history_no_seedchains[start:start+variable.dsize])
                start += step.total_var_dimension
            self.assertIs(np.array_equal(var_history, concatenated_traces), True)
        remove('test_history_correct_DREAM_chain_history.npy')
        remove('test_history_correct_DREAM_chain_adapted_crossoverprob.npy')
            
        
    def test_history_correct_after_sampling_multidim_model(self):
        self.start, self.model, (self.mu, self.C) = multidimensional_model()
        with self.model as model:
            step = pm.Dream(save_history=True, history_thin=1, model_name='test_history_correct', adapt_crossover=False)
            trace = pm.sample(draws=10, step=step, njobs=5)
        history = np.load('test_history_correct_DREAM_chain_history.npy')
        self.assertEqual(len(history), step.total_var_dimension*((10*5/step.history_thin)+step.nseedchains))
        history_no_seedchains = history[(step.total_var_dimension*step.nseedchains)::] 
        for i, variable in enumerate(step.variables):
            concatenated_traces = trace.get_values(str(variable), combine=True).flatten()
            var_history = np.array([])
            start = i
            for draw in range(50):
                var_history = np.append(var_history, history_no_seedchains[start:start+variable.dsize])
                start += step.total_var_dimension
            self.assertIs(np.array_equal(var_history, concatenated_traces), True)
        remove('test_history_correct_DREAM_chain_history.npy')

if __name__ == '__main__':
    unittest.main()
    
