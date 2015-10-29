from . import backends
from .backends.base import merge_traces, BaseTrace, MultiTrace
from .backends.ndarray import NDArray
import multiprocessing as mp
from time import time
from .core import *
from .step_methods import *
from .progressbar import progress_bar
from numpy.random import seed
from step_methods.Dream import DreamPool
from collections import defaultdict

__all__ = ['sample', 'iter_sample']

def assign_step_methods(model, step=None,
        methods=(NUTS, HamiltonianMC, Metropolis, BinaryMetropolis, 
        Slice, ElemwiseCategoricalStep)):
    '''
    Assign model variables to appropriate step methods. Passing a specified 
    model will auto-assign its constituent stochastic variables to step methods
    based on the characteristics of the variables. This function is intended to
    be called automatically from `sample()`, but may be called manually. Each
    step method passed should have a `competence()` method that returns an
    ordinal competence value corresponding to the variable passed to it. This
    value quantifies the appropriateness of the step method for sampling the 
    variable. 
    
    Parameters
    ----------
    
    model : Model object
        A fully-specified model object
    step : step function or vector of step functions
        One or more step functions that have been assigned to some subset of
        the model's parameters. Defaults to None (no assigned variables).
    methods : vector of step method classes
        The set of step methods from which the function may choose. Defaults
        to the main step methods provided by PyMC3.
        
    Returns
    -------
    List of step methods associated with the model's variables.
    '''
        
    steps = []
    assigned_vars = set()
    if step is not None:
        steps = np.append(steps, step).tolist()
        for s in steps:
            try:
                assigned_vars = assigned_vars | set(s.vars)
            except AttributeError:
                for m in s.methods:
                    assigned_vars = assigned_vars | set(m.vars)
    
    # Use competence classmethods to select step methods for remaining variables
    selected_steps = defaultdict(list)
    for var in model.free_RVs:
        if not var in assigned_vars:
               
            competences = {s:s._competence(var) for s in methods}

            selected = max(competences.keys(), key=(lambda k: competences[k]))
            
            print('Assigned {0} to {1}'.format(selected, var))
            selected_steps[selected].append(var)
    
    # Instantiate all selected step methods
    steps += [s(vars=selected_steps[s]) for s in selected_steps if selected_steps[s]]
    
    if len(steps)==1:
        steps = steps[0]    
                
    return steps

def sample(draws, step=None, start=None, trace=None, chain=0, njobs=1, tune=None,
           progressbar=True, model=None, random_seed=None):
    """
    Draw a number of samples using the given step method.
    Multiple step methods supported via compound step method
    returns the amount of time taken.

    Parameters
    ----------

    draws : int
        The number of samples to draw
    step : function or iterable of functions
        A step function or collection of functions. If no step methods are 
        specified, or are partially specified, they will be assigned 
        automatically (defaults to None).
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
    model = modelcontext(model)
    
    step = assign_step_methods(model, step)

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
        
        sample_func = _mp_sample
        sample_args = [njobs, argset]
    else:
        sample_func = _sample
        sample_args = [draws, step, start, trace, chain,
                       tune, progressbar, model, random_seed]
    return sample_func(*sample_args)


def _sample(draws, step=None, start=None, trace=None, chain=0, tune=None,
            progressbar=True, model=None, random_seed=None):
    sampling = _iter_sample(draws, step, start, trace, chain,
                            tune, model, random_seed)
    progress = progress_bar(draws)
    try:
        for i, strace in enumerate(sampling):
            if progressbar:
                progress.update(i)
    except KeyboardInterrupt:
        strace.close()
    return MultiTrace([strace])


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
        Step function
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
    for i, strace in enumerate(sampling):
        yield MultiTrace([strace[:i + 1]])


def _iter_sample(draws, step, start=None, trace=None, chain=0, tune=None,
                 model=None, random_seed=None):
    model = modelcontext(model)
    draws = int(draws)
    seed(random_seed)
    if draws < 1:
        raise ValueError('Argument `draws` should be above 0.')

    if start is None:
        start = {}

    strace = _choose_backend(trace, chain, model=model)

    if len(strace) > 0:
        _soft_update(start, strace.point(-1))
    else:
        _soft_update(start, model.test_point)

    try:
        step = CompoundStep(step)
    except TypeError:
        pass

    point = Point(start, model=model)

    strace.setup(draws, chain)
    for i in range(draws):
        if i == tune:
            step = stop_tuning(step)
        point = step.step(point)
        strace.record(point)
        yield strace
    else:
        strace.close()


def _choose_backend(trace, chain, shortcuts=None, **kwds):
    if isinstance(trace, BaseTrace):
        return trace
    if isinstance(trace, MultiTrace):
        return trace._straces[chain]
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
    if 'Dream' in str(args[0]):
       p = _mp_dream_pool(njobs, args)
    else:
       p = mp.Pool(njobs)

    traces = p.map(argsample, args)

    p.close()
    p.join()
    return merge_traces(traces)

def _mp_dream_pool(njobs, args):
    step_method = args[0][1]
    min_njobs = (2*len(step_method.DEpairs))+1
    if njobs < min_njobs:
        raise Exception('Dream should be run with at least (2*DEpairs)+1 number of chains.  For current algorithmic settings, set njobs>=%s.' %str(min_njobs))
    if step_method.history_file != False:
        old_history = np.load(step_method.history_file)
        len_old_history = len(old_history.flatten())
        nold_history_records = len_old_history/step_method.total_var_dimension
        for chain in range(len(args)):
            method = args[chain][1]
            method.nseedchains = nold_history_records
        if args[0][0] < step_method.history_thin:
            arr_dim = ((np.floor(njobs*args[0][0]/step_method.history_thin)+njobs)*step_method.total_var_dimension)+len_old_history
        else:
            arr_dim = np.floor((((njobs*args[0][0])*step_method.total_var_dimension)/step_method.history_thin))+len_old_history
    else:
        if args[0][0] < step_method.history_thin:
            arr_dim = ((np.floor(njobs*args[0][0]/step_method.history_thin)+njobs)*step_method.total_var_dimension)+(step_method.nseedchains*step_method.total_var_dimension)
        else:
            arr_dim = np.floor(((njobs*args[0][0]*step_method.total_var_dimension)/step_method.history_thin))+(step_method.nseedchains*step_method.total_var_dimension)
    min_nseedchains = 2*len(step_method.DEpairs)*njobs
    if step_method.nseedchains < min_nseedchains:
        raise Exception('The size of the seeded starting history is insufficient.  Increase nseedchains>=%s.' %str(min_nseedchains))
    current_position_dim = njobs*step_method.total_var_dimension
    history_arr = mp.Array('d', [0]*arr_dim)
    if step_method.history_file != False:
        history_arr[0:len_old_history] = old_history.flatten()
    nCR = step_method.nCR
    crossover_setting = step_method.CR_probabilities
    crossover_probabilities = mp.Array('d', crossover_setting)   
    ncrossover_updates = mp.Array('d', [0]*nCR)
    delta_m = mp.Array('d', [0]*nCR)
    current_position_arr = mp.Array('d', [0]*current_position_dim)
    nchains = mp.Value('i', njobs)
    n = mp.Value('i', 0)
    tf = mp.Value('c', 'F')
    if step_method.crossover_burnin == None:
        for chain in range(len(args)):
            method = args[chain][1]
            method.crossover_burnin = int(np.floor(args[0][0]/10))
    
    if args[0][2] != None:
        if step_method.start_random:
            print('Warning: start position provided but random_start set to True.  Overrode random_start value and starting walk at provided start position.')
            for chain in range(len(args)):
                method = args[chain][1]
                method.start_random = False

    p = DreamPool(njobs, initializer=_mp_dream_init, initargs=(history_arr, current_position_arr, nchains, crossover_probabilities, ncrossover_updates, delta_m, n, tf, ))
    
    return p

def _mp_dream_init(arr, cp_arr, nchains, crossover_probs, ncrossover_updates, delta_m, val, switch):
      step_methods.Dream_shared_vars.history = arr
      step_methods.Dream_shared_vars.current_positions = cp_arr
      step_methods.Dream_shared_vars.nchains = nchains
      step_methods.Dream_shared_vars.cross_probs = crossover_probs
      step_methods.Dream_shared_vars.ncr_updates = ncrossover_updates
      step_methods.Dream_shared_vars.delta_m = delta_m
      step_methods.Dream_shared_vars.count = val
      step_methods.Dream_shared_vars.history_seeded = switch

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
