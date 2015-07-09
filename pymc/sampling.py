from . import backends
from .backends.base import merge_traces, BaseTrace, MultiTrace
from .backends.ndarray import NDArray
import multiprocessing as mp
from time import time
from .core import *
from . import step_methods
from .progressbar import progress_bar
from numpy.random import seed
import logging
from step_methods.Dream import DreamPool
import traceback
try:
    from mpi4py import MPI
    mpi_avail = True
except ImportError:
    mpi_avail = False

__all__ = ['sample', 'iter_sample']


def sample(draws, step, start=None, trace=None, chain=0, njobs=1, tune=None,
           progressbar=True, model=None, random_seed=None, use_mpi=False):
    """
    Draw a number of samples using the given step method.
    Multiple step methods supported via compound step method
    returns the amount of time taken.

    Parameters
    ----------

    draws : int
        The number of samples to draw
    step : function
        A step function
    start : dict
        Starting point in parameter space (or partial point)
        Defaults to trace.point(-1)) if there is a trace provided and
        model.test_point if not (defaults to empty dict)
    trace : backend, list, or MultiTrace
        This should be a backend instance, a list of variables to track,
        or a MultiTrace object with past values. If a MultiTrace object
        is given, it must contain samples for the chain number `chain`.
        If None or a list of variables, the NDArray backend is used.
        Passing either "text" or "sqlite" is taken as a shortcut to set
        up the corresponding backend (with "mcmc" used as the base
        name).
    chain : int
        Chain number used to store sample in backend. If `njobs` is
        greater than one, chain numbers will start here.
    njobs : int
        Number of parallel jobs to start. If None, set to number of cpus
        in the system - 2.
    tune : int
        Number of iterations to tune, if applicable (defaults to None)
    progressbar : bool
        Flag for progress bar
    model : Model (optional if in `with` context)
    random_seed : int or list of ints
        A list is accepted if more if `njobs` is greater than one.

    Returns
    -------
    MultiTrace object with access to sampling values
    """
    if njobs is None:
        njobs = max(mp.cpu_count() - 2, 1)
                  
    if njobs > 1:
        try:
            if not len(random_seed) == njobs:
                random_seeds = [random_seed] * njobs
            else:
                random_seeds = random_seed
        except TypeError:  # None, int
            random_seeds = [random_seed] * njobs

        chains = list(range(chain, chain + njobs))

        pbars = [progressbar] + [False] * (njobs - 1)
        
        if isinstance(start, list):
             argset = zip([draws] * njobs,
                     [step] * njobs,
                     start,
                     [trace] * njobs,
                     chains,
                     [tune] * njobs,
                     pbars,
                     [model] * njobs,
                     random_seeds)   
        else:
            argset = zip([draws] * njobs,
                         [step] * njobs,
                         [start] * njobs,
                         [trace] * njobs,
                         chains,
                         [tune] * njobs,
                         pbars,
                         [model] * njobs,
                         random_seeds)
        
        if not mpi_avail or use_mpi==False:
            sample_func = _mp_sample
            sample_args = [njobs, argset]
        else:
            print 'About to enter mpi sample function.'
            sample_func = _mpi_sample
            sample_args = [njobs, argset]
    else:
        sample_func = _sample
        sample_args = [draws, step, start, trace, chain,
                       tune, progressbar, model, random_seed]
    return sample_func(*sample_args)


def _sample(draws, step, start=None, trace=None, chain=0, tune=None,
            progressbar=True, model=None, random_seed=None):
    sampling = _iter_sample(draws, step, start, trace, chain,
                            tune, model, random_seed)
    progress = progress_bar(draws)
    try:
        for i, trace in enumerate(sampling):
            if progressbar:
                progress.update(i)
    except KeyboardInterrupt:
        trace.close()
    return MultiTrace([trace])


def iter_sample(draws, step, start=None, trace=None, chain=0, tune=None,
                model=None, random_seed=None):
    """
    Generator that returns a trace on each iteration using the given
    step method.  Multiple step methods supported via compound step
    method returns the amount of time taken.


    Parameters
    ----------

    draws : int
        The number of samples to draw
    step : function
        A step function
    start : dict
        Starting point in parameter space (or partial point)
        Defaults to trace.point(-1)) if there is a trace provided and
        model.test_point if not (defaults to empty dict)
    trace : backend, list, or MultiTrace
        This should be a backend instance, a list of variables to track,
        or a MultiTrace object with past values. If a MultiTrace object
        is given, it must contain samples for the chain number `chain`.
        If None or a list of variables, the NDArray backend is used.
    chain : int
        Chain number used to store sample in backend. If `njobs` is
        greater than one, chain numbers will start here.
    tune : int
        Number of iterations to tune, if applicable (defaults to None)
    model : Model (optional if in `with` context)
    random_seed : int or list of ints
        A list is accepted if more if `njobs` is greater than one.

    Example
    -------

    for trace in iter_sample(500, step):
        ...
    """
    sampling = _iter_sample(draws, step, start, trace, chain, tune,
                            model, random_seed)
    for i, trace in enumerate(sampling):
        yield MultiTrace([trace[:i + 1]])


def _iter_sample(draws, step, start=None, trace=None, chain=0, tune=None,
                 model=None, random_seed=None):
    model = modelcontext(model)
    draws = int(draws)
    seed(random_seed)
    if draws < 1:
        raise ValueError('Argument `draws` should be above 0.')

    if start is None:
        start = {}

    trace = _choose_backend(trace, chain, model=model)

    if len(trace) > 0:
        _soft_update(start, trace.point(-1))
    else:
        _soft_update(start, model.test_point)

    try:
        step = step_methods.CompoundStep(step)
    except TypeError:
        pass

    point = Point(start, model=model)

    trace.setup(draws, chain)
    for i in range(draws):
        if i == tune:
            step = stop_tuning(step)
        point = step.step(point)
        trace.record(point)
        yield trace
    else:
        trace.close()


def _choose_backend(trace, chain, shortcuts=None, **kwds):
    if isinstance(trace, BaseTrace):
        return trace
    if isinstance(trace, MultiTrace):
        return trace._traces[chain]
    if trace is None:
        return NDArray(**kwds)

    if shortcuts is None:
        shortcuts = backends._shortcuts

    try:
        backend = shortcuts[trace]['backend']
        name = shortcuts[trace]['name']
        return backend(name, **kwds)
    except TypeError:
        return NDArray(vars=trace, **kwds)
    except KeyError:
        raise ValueError('Argument `trace` is invalid.')

def _mp_sample(njobs, args):
    # If using DREAM stepping method, allocate a shared history array, a count variable, and a variable denoting whether or not the history has been seeded with draws from the prior.
    #mp.log_to_stderr(logging.DEBUG)    
    if 'Dream' in str(args[0]):
       step_method = args[0][1]
       min_njobs = (2*len(step_method.DEpairs))+1
       if njobs < min_njobs:
           raise Exception('Dream should be run with at least (2*DEpairs)+1 number of chains.  For current algorithmic settings, set njobs>=%s.' %str(min_njobs))
       if step_method.history_file != False:
           old_history = np.load(step_method.history_file)
           len_old_history = len(old_history.flatten())
           nold_history_records = len_old_history/step_method.total_var_dimension
           for job in range(njobs):
               sm = args[job][1]
               sm.nseedchains = nold_history_records
           if np.floor(args[0][0]/step_method.history_thin) > 0:
               arr_dim = (np.floor(args[0][0]/step_method.history_thin)*njobs*step_method.total_var_dimension)+len_old_history
           else:
               arr_dim = (njobs*step_method.total_var_dimension)+len_old_history
       else:
           if np.floor(args[0][0]/step_method.history_thin) > 0:
               arr_dim = (np.floor(args[0][0]/step_method.history_thin)*njobs*step_method.total_var_dimension)+(step_method.nseedchains*step_method.total_var_dimension)
           else:
               arr_dim = (njobs*step_method.total_var_dimension)+(step_method.nseedchains*step_method.total_var_dimension)
       min_nseedchains = 2*len(step_method.DEpairs)*njobs
       if step_method.nseedchains < min_nseedchains:
           raise Exception('The size of the seeded starting history is insufficient.  Increase nseedchains>=%s.' %str(min_nseedchains))
       current_position_dim = njobs*step_method.total_var_dimension
       history_arr = mp.Array('d', [0]*arr_dim)
       if step_method.history_file != False:
           history_arr[0:len_old_history] = old_history.flatten()
       nCR = step_method.nCR
       starting_crossover = step_method.CR_probabilities
       crossover_probabilities = mp.Array('d', starting_crossover)       
       ncrossover_updates = mp.Array('d', [0]*nCR)
       delta_m = mp.Array('d', [0]*nCR)
       current_position_arr = mp.Array('d', [0]*current_position_dim)
       nchains = mp.Value('i', njobs)
       n = mp.Value('i', 0)
       tf = mp.Value('c', 'F')
       if step_method.crossover_burnin == None:
           step_method.crossover_burnin = int(np.floor(args[0][0]/10))
       p = DreamPool(njobs, initializer=_mp_dream_init, initargs=(history_arr, current_position_arr, nchains, crossover_probabilities, ncrossover_updates, delta_m, n, tf, ))
    else:
       p = mp.Pool(njobs)
    try:
        traces = p.map(argsample, args)
    except Exception as e:
        traceback.print_exc()
        raise e
    p.close()
    p.join()
    return merge_traces(traces)

def _mp_dream_init(arr, cp_arr, nchains, crossover_probs, ncrossover_updates, delta_m, val, switch):
      step_methods.Dream_shared_vars.history = arr
      step_methods.Dream_shared_vars.current_positions = cp_arr
      step_methods.Dream_shared_vars.nchains = nchains
      step_methods.Dream_shared_vars.cross_probs = crossover_probs
      step_methods.Dream_shared_vars.ncr_updates = ncrossover_updates
      step_methods.Dream_shared_vars.delta_m = delta_m
      step_methods.Dream_shared_vars.count = val
      step_methods.Dream_shared_vars.history_seeded = switch
      
def _mpi_sample(njobs, args):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if 'Dream' in str(args[0]):
        #Assign extra available ranks to chain running ranks for use with multitry
        rank_assignments = {}
        size = comm.Get_size()
        unused_ranks = size - njobs
        ranks_per_chain = unused_ranks/njobs
        extra_ranks = unused_ranks%njobs
        for parent in range(njobs):
            start_idx = njobs + (parent*ranks_per_chain)
            end_idx = start_idx + ranks_per_chain
            assigned_ranks = range(size)[start_idx:end_idx]
            if parent in range(extra_ranks):
                start_idx = size - extra_ranks
                assigned_ranks.append(range(start_idx, size)[parent])
            rank_assignments[parent] = assigned_ranks
            for child in assigned_ranks:
                rank_assignments[child] = parent
            
        print 'Assigned ranks: ',rank_assignments
        
        #Create communicator for just the ranks running chains
        all_group = comm.Get_group()
        chains = range(njobs)
        chain_group = all_group.Incl(chains)
        newcomm = comm.Create(chain_group)
        chain_comm = newcomm
      
    if rank < njobs:
        if 'Dream' in str(args[0]):
            args = _mpi_dream_init(njobs, args, rank, rank_assignments, chain_comm)
        trace = _sample(*args[rank])
        print 'Finished trace in rank: ',rank
        if rank != 0:
            comm.send(trace, dest=0, tag=1)
            traces = []
    
    else:
        logp_fxn = args[0][1].fs[0]
        ordering = args[0][1].ordering
        running = True
        assigned_source = rank_assignments[rank]
        tag = rank
        while running == True:
            pts = comm.recv(source=assigned_source, tag=tag)
            if pts == 'quit':
                break
            pt_to_eval = pts[0]
            old_pt = pts[1]
            bij = DictToArrayBijection(ordering, old_pt)
            mapped_logp = bij.mapf(logp_fxn)
            logp = mapped_logp(pt_to_eval)    
            comm.send(logp, dest=assigned_source, tag=tag)
            tag += 1
    if rank == 0:
        traces = [trace]
        for rank in range(1, njobs):
            trace = comm.recv(source=rank, tag=1)
            traces.append(trace)
        return merge_traces(traces)

def _mpi_dream_init(njobs, args, rank, rank_assignments, chain_comm):
    step_method = args[0][1]
    min_njobs = (2*len(step_method.DEpairs))+1
    if njobs < min_njobs:
       raise Exception('Dream should be run with at least (2*DEpairs)+1 number of chains.  For current algorithmic settings, set njobs>=%s.' %str(min_njobs))
    if step_method.history_file != False:
       old_history = np.load(step_method.history_file)
       len_old_history = len(old_history.flatten())
       nold_history_records = len_old_history/step_method.total_var_dimension
       for job in range(njobs):
           sm = args[job][1]
           sm.nseedchains = nold_history_records
    
    min_nseedchains = 2*len(step_method.DEpairs)*njobs
    if step_method.nseedchains < min_nseedchains:
       raise Exception('The size of the seeded starting history is insufficient.  Increase nseedchains>=%s.' %str(min_nseedchains))
       
    if step_method.crossover_file != False:
        for job in range(njobs):
            sm = args[job][1]
            sm.adapt_crossover = False
       
    if step_method.crossover_burnin == None:
        for job in range(njobs):
            sm = args[job][1]
            sm.crossover_burnin = int(np.floor(args[0][0]/10))
    
    for job in range(njobs):
        sm = args[job][1]
        sm.draws = args[0][0]
        sm.nchains = njobs
        sm.assigned_ranks = rank_assignments[rank]
        sm.chain_comm = chain_comm
        
    return args
    
def stop_tuning(step):
    """ stop tuning the current step method """

    if hasattr(step, 'tune'):
        step.tune = False

    elif hasattr(step, 'methods'):
        step.methods = [stop_tuning(s) for s in step.methods]

    return step


def argsample(args):
    """ defined at top level so it can be pickled"""
    return _sample(*args)


def _soft_update(a, b):
    """As opposed to dict.update, don't overwrite keys if present.
    """
    a.update({k: v for k, v in b.items() if k not in a})
