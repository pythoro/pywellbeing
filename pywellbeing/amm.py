# -*- coding: utf-8 -*-
"""
Created on Sat May  7 13:17:45 2022

@author: Reuben
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genpareto, norm, logistic
from pathlib import Path

settings = {
    'n': 80,  # The number of cues along x.
    'x_max': 3,  # The maximum x-value magnitude cues.
    'use_error': True,
    'use_error_factor': 0.01,
}


def get_xs():
    return np.linspace(-settings['x_max'], settings['x_max'], settings['n'])


class Random():
    def __init__(self, random_seed=None):
        self.set_random_seed(random_seed)
    
    def set_random_seed(self, random_seed):
        self._rng = np.random.default_rng(random_seed)
        
    def get_rng(self):
        return self._rng


random = Random()


class Context():
    """ Generates events with some effect with some likelihood 
    
    Args:
        n (float): The number of cues along x.
        x_max (float): The maximum x-value of a cue.
        scale (float): The standard for the (normal) distribution of
            probabilities for each cue. Defaults to 1.0.
        prob (float): The probability factor on each cue. Defaults to 1.0.
        loc (float): The mean of the normal distribution. Defaults to 0.0.
        
    """
    def __init__(self, scale=1.0, prob=1.0, loc=0.0):
        self._scale = scale
        self._prob = prob
        self._loc = loc
    
    def setup(self):
        """ Setup the cues 
        
        """
        xs = get_xs()
        n = settings['n']
        nom = norm.pdf(xs, loc=self._loc)
        probs = [1 - self._prob, self._prob]
        ps = random.get_rng().choice([0, 1], size=n, p=probs) * nom
        self.xs = xs
        self.inds = np.arange(n)
        self.ps = ps / sum(ps)  # Normalise they so sum to 1

    def sample(self):
        return self.ps.copy()
    
    def expected_value(self):
        return np.mean(self.xs * self.ps)

    def plot(self):
        plt.figure()
        plt.scatter(self.xs, self.ps)
        plt.xlabel('Change in fitness')
        plt.ylabel('Occurence likelihood')
        plt.grid(visible=True, axis='y')
        plt.tight_layout()


class Motivator():
    """ Base motivator class 
    
    This class is used as a base for other classes. It provides common
    functionality such as the setup of random values, breeding with another
    of the same time, and history recording.
    
    Args:
        init (ndarray): An optional set of initial values. Used for breeding.
        decay (float): The rate at which learned values decay per period. 
            This represents 'forgetting'. Defaults to 0.98, which means 
            the next period has 0.98 times the previous value.
        z_range (float): To avoid selected values from increasing in 
            magnitude to unrealistic levels, they are capped to this 
            absolute magnitude. Defaults to 1.5.
        z_sd (float): The standard deviation of random error in values.
            Defaults to 0.5.
    
    """
    
    START_ZEROED = False  # True if no inheritence occurs
    
    def __init__(self,
                 init=None,
                 decay=0.98,
                 z_range=2.0,
                 z_sd=0.4):
        self._decay = decay
        self._z_range = z_range
        self._z_sd = z_sd
        self._history = {'vals': [],
                         'behaviour_tendency': [],
                         'cue_dist_modifier': []}
        self.setup(init=init)
        
    def setup(self, init=None):
        n = settings['n']
        if self.START_ZEROED:
            self._base = np.zeros(n)
            self._learned_vals = np.zeros(n)
        elif init is None:
            base = random.get_rng().normal(scale=self._z_sd, size=n)
            self._base = base
            self._learned_vals = np.zeros_like(base)
        else:
            self._base = init.copy()
            self._learned_vals = np.zeros_like(self._base)
        self.n = n
    
    def reset(self):
        self._learned_vals = np.zeros_like(self._base)
    
    @property
    def base(self):
        return self._base
    
    @property
    def learned_vals(self):
        return self._learned_vals
    
    def record_history(self):
        pass
    
    def get_occurances(self, cue_dist, behaviour_dist):
        """ Get the occurance frequency
        
        Args:
            cue_dist (ndarray): An array of cue frequencies presented by
                the environment.
            behaviour_dist (ndarray): An array of behavioural responses that
                determine whether the situation has effect (an occurance).
                The behaviour_dist values should be between 0 and 1.
        
        Returns:
            ndarray: The occurance frequency
        """
        return cue_dist * behaviour_dist
    
    def process(self, cue_dist, behaviour_dist, weighted_error):
        raise NotImplementedError
    
    def get_behaviour_tendency(self):
        raise NotImplementedError
    
    def get_weighted_error(self):
        raise NotImplementedError
    
    def get_cue_dist_modifier(self):
        return np.zeros(self.n)

    def breed(self, other):
        """ Breed this motivator with another of the same kind 
        
        Args:
            other: Another instance of the same class of motivator.
        
        Returns:
            A child motivator of the same class, with values randomly chosen
            from each parent.
        
        """
        n = self.n
        from_other = random.get_rng().choice([False, True], size=n)
        inherited = self._base.copy()
        inherited[from_other] = other._base[from_other]
        inherited = np.clip(inherited, -self._z_range, self._z_range)
        error = random.get_rng().normal(scale=self._z_sd, size=n)
        init = inherited + error
        new_motivator = self.__class__(init=init)
        return new_motivator
    
    def decay_vals(self, cue_dist):
        self._learned_vals *= self._decay
        
    
class Instincts(Motivator):
    """ Instinct based motivator
    
    This simply outputs a behaviour tendency based on the inherited values.

    """
    
    UNIQUE_SEED_MODIFIER = 25
    
    def process(self, cue_dist, behaviour_dist, weighted_error):
        # There's no function of cues or learning
        pass
    
    def get_behaviour_tendency(self):
        return self._base


class Prediction_Error(Motivator):
    """ Calculation of (reward) prediction error 
    
    This class uses inherited cue value to calculate prediction error.
    It uses a decay rate per period to ensure it never goes to 0 for 
    any cue.
    
    Args:
        [Defaults from Motivator]
        rate (float): A rate at which predicted values are learned from
            occurances. Defaults to 2.0.
    
    """
    
    
    def __init__(self, *args, rate=2, **kwargs):
        self._rate = rate
        super().__init__(*args, **kwargs)
        self._prediction_error = np.zeros_like(self._base)
        self._weighted_error = np.zeros_like(self._base)
    
    def learn(self, weighted_error):
        """ Learning rates based on occurance frequency 

        Note that learning only occurs when the situation has effect - this
        is termed an occurance. The rate of learning is proportional to
        the frequency of occurances.
        
        Warning:
            For stability, the maximum weighted error should be
            less than 1. This is normally the case. 

        Args:
            occurances (ndarray): An array of occurance frequencies.
            prediction_error (ndarray): The prediction error per cue.
                
        """
        self._learned_vals += weighted_error * self._rate
    
    def process(self, cue_dist, behaviour_dist, weighted_error):
        """ Main processing step - called each period 
        
        Note weighted error is calculated here.
        
        Args:
            cue_dist (ndarray): An array of cue frequencies presented by
                the environment.
            behaviour_dist (ndarray): An array of behavioural responses that
                determine whether the situation has effect (an occurance).
            weighted_error (ndarray): An array of prediction errors weighted
                by occurance frequency.
        
        """
        actual = self._base
        predicted = self._learned_vals
        if settings['use_error']:
            prediction_error = actual - predicted
        else:
            prediction_error = actual * settings['use_error_factor']
        occurances = self.get_occurances(cue_dist, behaviour_dist)
        weighted_error = prediction_error * occurances
        self._prediction_error = prediction_error  # Logged in history
        self._weighted_error = weighted_error  # Needed later
        self.learn(weighted_error)
        self.decay_vals(cue_dist)
        
    def get_prediction_error(self):
        return self._prediction_error
        
    def get_weighted_error(self):
        return self._weighted_error


class Routines(Motivator):
    """ Routines due to model-free reinforcement learning 
    
    This simple class reinforces behaviour on the basis of prediction errors
    that are weighted by frequency of occurance.
    
    Args:
        [Defaults from Motivator]
        rate (float): The learning rate. Defaults to 3.0.
    
    """
    
    START_ZEROED = True
    
    def __init__(self, *args, rate=3.0, **kwargs):
        self._rate = rate
        self._do_learn = True
        super().__init__(*args, **kwargs)
    
    def learn(self, weighted_error):
        self._learned_vals += weighted_error * self._rate

    def set_do_learn(self, flag):
        """ Turn learning off or on """
        self._do_learn = flag

    def get_behaviour_tendency(self):
        return self._learned_vals

    def process(self, cue_dist, behaviour_dist, weighted_error):
        if not self._do_learn:
            return
        self.learn(weighted_error)
        self.decay_vals(cue_dist)
    

class Planned_Control(Motivator):
    """ Decision making for niche construction 
    
    This implimentation assumes a finite amount of time/effort is available
    with which to change the environment. It is used to calculate a set of 
    effort values for each cue that are ultimately used as a multiplier on
    the normal environmental cues distribution.
    
    In each period, a number of 'decisions' are made. Once effort capacity 
    is reached, these decisions compare two cues and calculate whether it
    would increase net weighted prediction error if the effort was swapped
    or not.

    Args:
        [Defaults from Motivator]
        rate (float): The discrete effort increment size. Defaults to 0.01.
        num (int): The number of 'decisions' made per period.
    
    """

    START_ZEROED = True
    
    def __init__(self, *args,
                 rate=0.01,
                 num=20,
                 f_tot=0.1,
                 **kwargs):
        self._rate = rate
        self._num = num
        self._do_learn = True
        super().__init__(*args, **kwargs)
        self._n = settings['n']
        self._tot = self._n * f_tot

    def set_do_learn(self, flag):
        """ Turn learning off or on """
        self._do_learn = flag
    
    def cue_mod(self, effort):
        """ This calculates the multiplier used on the cue distribution 
        
        Args:
            modifier (ndarray): An array of learned y values that correspond
            with effort.
            
        Returns:
            ndarray: An array of modification factors.
        """
        return 5.0**effort
    
    def _learn_under_effort_capacity(self, weighted_error):
        """ Learning that happens initially when effort is under capacity 

        Args:
            weighted_error (ndarray): An array of prediction errors weighted
                by occurance frequency.
        
        Returns:
            ndarray: The array of incremental changes to make to effort.
        """
        err = weighted_error
        changes = np.ones(self._n) * self._rate
        changes = np.copysign(changes, err)
        return changes
    
    def _learn_at_effort_capacity(self, cue_dist, weighted_error):
        """ Learning that happens when effort is at capacity 
        
        Args:
            cue_dist (ndarray): An array of cue frequencies presented by
                the environment.
            weighted_error (ndarray): An array of prediction errors weighted
                by occurance frequency.
                
        Returns:
            ndarray: The array of incremental changes to make to effort.
        """
        err = weighted_error
        effort = self._learned_vals
        signs = np.copysign(1, effort)
        d1 = (self.cue_mod(effort + self._rate * signs) 
                  - self.cue_mod(effort))
        d2 = (self.cue_mod(effort - self._rate * signs) 
                  - self.cue_mod(effort))
        inds = np.arange(0, self._n, 1)
        A = random.get_rng().choice(inds, size=self._num)
        B = random.get_rng().choice(inds, size=self._num)
        delta = err[A] * d1[A] + err[B] *  d2[B]
        direction = np.ones_like(delta)
        direction[delta < 0] = -1
        changes = np.zeros(self._n)
        changes[A] = self._rate * signs[A] * direction
        changes[B] = -self._rate * signs[B] * direction
        return changes
    
    def learn(self, cue_dist, weighted_error):
        """ Main learning step - called each period 
        
        Args:
            cue_dist (ndarray): An array of cue frequencies presented by
                the environment.
            weighted_error (ndarray): An array of prediction errors weighted
                by occurance frequency.
        """
        if np.sum(np.abs(self._learned_vals)) >= self._tot:
            changes = self._learn_at_effort_capacity(cue_dist, weighted_error)
        else:
            changes = self._learn_under_effort_capacity(weighted_error)
        self._learned_vals += changes
    
    def process(self, cue_dist, behaviour_dist, weighted_error):
        """ Main processing step - called each period 
        
        Args:
            cue_dist (ndarray): An array of cue frequencies presented by
                the environment (and modified by effort).
            behaviour_dist (ndarray): An array of behavioural responses that
                determine whether the situation has effect (an occurance).
            weighted_error (ndarray): An array of prediction errors weighted
                by occurance frequency.
        """
        if not self._do_learn:
            return
        self.learn(cue_dist, weighted_error)
        self.decay_vals(cue_dist)

    def get_behaviour_tendency(self):
        """ Return zeros to not modify behavioural responses """
        return np.zeros(self._n)
    
    def get_cue_dist_mod(self):
        """ Return the modification factors based on effort"""
        return self.cue_mod(self._learned_vals)
    
    def get_modified_cue_dist(self, cue_dist):
        """ Return the modified cue distribution that's changed by effort 
        
        Args:
            cue_dist (ndarray): An array of cue frequencies presented by
                the environment (and modified by effort).
                
        Returns:
            ndarray: The new distribution of cues modified by effort to be 
            used in the next period.
        """
        mod = self.cue_mod(self._learned_vals)
        return mod * cue_dist
        

class Person():
    def __init__(self, motivators=None, a_decay=0.97,
                 history=False):
        self.setup(motivators)
        self._a_decay = a_decay
        self._record_history = history
            
    def setup(self, motivators=None):
        if motivators is None:
            self._predictor = Prediction_Error()
            self._instincts = Instincts()
            self._routines = Routines()
            self._planned_control = Planned_Control()
        else:
            self.set_motivators(motivators)
        self.history = {
            'instincts': [],
            'reinforcement': [],
            'niche_effort': [],
            'cue_dist': [],
            'behaviour_dist': [],
            'weighted_error': [],
            'prediction_error': []
            }
    
    def reset(self):
        self.history = {
            'instincts': [],
            'reinforcement': [],
            'niche_effort': [],
            'cue_dist': [],
            'behaviour_dist': [],
            'weighted_error': [],
            'prediction_error': []
            }
        for motivator in self.motivators:
            motivator.reset()
    
    def copy(self):
        return Person(motivators=self.motivators,
                      a_decay=self._a_decay)
    
    def set_record_history(self, history):
        self._record_history = history
    
    def set_do_learn(self, flag):
        self._planned_control.set_do_learn(flag)
        self._routines.set_do_learn(flag)
    
    @property
    def predictor(self):
        return self._predictor

    @property
    def planner(self):
        return self._planned_control

    @property
    def value(self):
        return self.predictor.base

    @property
    def instincts(self):
        return self._instincts.get_behaviour_tendency()

    @property
    def reinforcement(self):
        return self._routines.get_behaviour_tendency()
    
    @property
    def niche_effort(self):
        return self.planner.learned_vals
        
    @property
    def niche_mod(self):
        return self.planner.cue_mod(self.niche_effort)
    
    @property
    def cue_dist(self):
        return self.history['cue_dist'][-1]

    @property
    def behaviour_dist(self):
        return self.history['behaviour_dist'][-1]

    @property
    def weighted_error(self):
        return self.history['weighted_error'][-1]
    
    @property
    def prediction_error(self):
        return self.predictor.get_prediction_error()
    
    @property
    def motivators(self):
        return [self._predictor, self._instincts, self._routines,
                self._planned_control]
    
    def set_motivators(self, motivators):
        self._predictor, self._instincts, self._routines, \
                self._planned_control = motivators
    
    def store_history(self, cue_dist, behaviour_dist, weighted_error):
        self.history['cue_dist'].append(cue_dist)
        self.history['behaviour_dist'].append(behaviour_dist)
        self.history['weighted_error'].append(weighted_error)
        if not self._record_history:
            return
        self.history['instincts'].append(self.instincts)
        self.history['reinforcement'].append(self.reinforcement)
        self.history['niche_effort'].append(self.niche_effort)
        self.history['prediction_error'].append(self.prediction_error)
    
    def process(self, cue_dist):
        behaviour_dist_ys = np.ones_like(cue_dist)
        motivators = [self._instincts,
                      self._routines,
                      self._planned_control,
                      ]
        for motivator in motivators:
            behaviour_dist_ys += motivator.get_behaviour_tendency()
        weighted_error = self.predictor.get_weighted_error()
        behaviour_dist = logistic.cdf(behaviour_dist_ys)
        mod_cue_dist = self.planner.get_modified_cue_dist(cue_dist)
        for motivator in self.motivators:
            motivator.process(mod_cue_dist, behaviour_dist, weighted_error)
        self.store_history(mod_cue_dist, behaviour_dist, weighted_error)

    def subj_wb_history(self, k=3, n=1, decay=None):
        n_hist = len(self.history['cue_dist'])
        indices = np.arange(2, n_hist, 1)
        wb = [self.subjective_wellbeing(i=i, n=n, decay=decay)[k] for i in indices]
        return indices, np.array(wb)
    
    def subj_wb_history_decayed(self, k=3, n=100, decay=0.9):
        return self.subj_wb_history(k=k, n=n, decay=decay)
    
    def plot_wb_history(self, xlim=None, fignum=None, label=None, **kwargs):
        plt.figure(num=fignum)
        inds, vals = self.subj_wb_history()
        means = np.mean(vals, axis=0)
        std = np.std(vals, axis=0)
        plt.plot(inds, vals, label=label, **kwargs)
        plt.xlabel('Period')
        plt.ylabel('Subjective wellbeing')
        plt.xlim(xlim)
        plt.tight_layout()
        return plt.gcf()
    
    def fitness_history(self):
        n = len(self.history['cue_dist'])
        indices = np.arange(2, n, 1)
        wb = [self.fitness(i) for i in indices]
        return indices, np.array(wb)

    def subjective_wellbeing(self, i=None, n=1, decay=None):
        end = len(self.history['cue_dist']) - 1 if i is None else i
        start = max(0, end - n)
        errors = np.array(self.history['weighted_error'][start:end])
        inds = np.arange(end, start, -1)
        decay = self._a_decay if decay is None else decay
        weights = np.power(decay, inds - start)
        totals = weights.reshape(-1, 1) * errors
        neg_vals = totals < 0
        neg = np.sum(totals[neg_vals])
        pos = np.sum(totals[~neg_vals])
        net = np.sum(totals)
        abs_sum = np.sum(np.abs(totals))
        ratio = pos / abs_sum
        return neg, pos, net, ratio
    
    def fitness(self, i=None, n=1000, decay=1):
        end = len(self.history['cue_dist']) - 1 if i is None else i
        start = max(0, end - n)
        inds = np.arange(end, start, -1)
        weights = np.power(self._a_decay, inds) + 1e-2
        cue_dist = np.array(self.history['cue_dist'][start:end])
        behaviour_dist = np.array(self.history['behaviour_dist'][start:end])
        xs = get_xs().reshape(1, -1)
        costs_benefits = xs * cue_dist * behaviour_dist
        totals = weights.reshape(-1, 1) * costs_benefits * 1000
        return np.sum(totals) / np.sum(weights)
    
    def breed(self, other):
        """ Create a child """
        motivators = []
        for m1, m2 in zip(self.motivators, other.motivators):
            new = m1.breed(m2)
            motivators.append(new)
        child = Person(motivators=motivators)
        return child


class Life_History():
    """ Put someone through a set of Contexts 
    
    These may change differ at different stages, which is why attention 
    bias may help.
    """
    def __init__(self):
        self.person = None
        self.contexts = []

    def set_person(self, person):
        self.person = person

    def add_context(self, context, n_events):
        d = {'context': context,
             'n': n_events}
        self.contexts.append(d)
    
    def run_one(self, context, n):
        person = self.person
        for i in range(n):
            cue_dist = context.sample()
            person.process(cue_dist)
    
    def run(self):
        for d in self.contexts:
            self.run_one(**d)

                
class Population():
    """ Put lots of people through life histories and generations """
    def __init__(self):
        self.history = {'value': [],
                        'instincts': [],
                        'niche_effort': [],
                        'response_effort': [],
                        'reinforcement': [],
                        'cue_dist': [],
                        'behaviour_dist': [],
                        'occurances': [],
                        'weighted_error': [],
                        'prediction_error': [],
                        'fitness': [],
                        'subj_wb': [],
                        }
    
    def set_life_history(self, lh):
        self.lh = lh
    
    def set_pop(self, population, reset=True):
        self.pop = [p.copy() for p in population.pop]
        if reset:
            for p in self.pop:
                p.reset()
    
    def set_population(self, pop_size, n_history=10, *args, **kwargs):
        pop = []
        for i in range(pop_size):
            history = True if i < n_history else False
            person = Person(*args, history=history, **kwargs)
            pop.append(person)
        self.pop = pop
    
    def get_ave_fitness(self):
        wbs = [p.fitness() for p in self.pop]
        return np.mean(wbs), np.min(wbs), np.max(wbs)

    def get_ave_subj_wb(self):
        wbs = [p.subjective_wellbeing() for p in self.pop]
        return np.mean(wbs, axis=0), np.min(wbs, axis=0), np.max(wbs, axis=0)
    
    def get_ave_value(self):
        vals = np.array([p.value for p in self.pop])
        return np.mean(vals, axis=0), np.min(vals, axis=0), np.max(vals, axis=0)
    
    def get_ave_instincts(self):
        vals = np.array([p.instincts for p in self.pop])
        return np.mean(vals, axis=0), np.min(vals, axis=0), np.max(vals, axis=0)
    
    def get_ave_niche_effort(self):
        vals = np.array([p.niche_effort for p in self.pop])
        return np.mean(vals, axis=0), np.min(vals, axis=0), np.max(vals, axis=0)
    
    def get_ave_reinforcement(self):
        vals = np.array([p.reinforcement for p in self.pop])
        return np.mean(vals, axis=0), np.min(vals, axis=0), np.max(vals, axis=0)
    
    def get_ave_cue_dist(self):
        vals = np.array([p.cue_dist for p in self.pop])
        return np.mean(vals, axis=0), np.min(vals, axis=0), np.max(vals, axis=0)

    def get_ave_behaviour_dist(self):
        vals = np.array([p.behaviour_dist for p in self.pop])
        return np.mean(vals, axis=0), np.min(vals, axis=0), np.max(vals, axis=0)
    
    def get_ave_weighted_error(self):
        vals = np.array([p.weighted_error for p in self.pop])
        return np.mean(vals, axis=0), np.min(vals, axis=0), np.max(vals, axis=0)
    
    def get_ave_prediction_error(self):
        vals = np.array([p.prediction_error for p in self.pop])
        return np.mean(vals, axis=0), np.min(vals, axis=0), np.max(vals, axis=0)
    
    def get_ave_occurances(self):
        vals = np.array([p.cue_dist * p.behaviour_dist for p in self.pop])
        return np.mean(vals, axis=0), np.min(vals, axis=0), np.max(vals, axis=0)
    
    def breed(self, p_survive=0.6, p_mates=0.1):
        pop = self.pop
        fitness = [p.fitness() for p in pop]
        fitness_rank = np.argsort(-np.array(fitness))  # descending order
        S = len(pop)
        i_survive = int(S * p_survive)
        i_mates = int(S * p_mates)
        mate_pool = np.array(pop)[fitness_rank[:i_mates]]
        breeder_pool = np.array(pop)[fitness_rank[:i_survive]]
        mates = random.get_rng().choice(mate_pool, size=S)
        others = random.get_rng().choice(breeder_pool, size=S)
        new = [mate.breed(other) for mate, other in zip(mates, others)]
        self.pop = new
    
    def run_generation(self):
        lh = self.lh
        for i, p in enumerate(self.pop):
            # print('   ' + str(i))
            lh.set_person(p)
            lh.run()
    
    def record_hist(self):
        hist = self.history
        hist['value'].append(self.get_ave_value())
        hist['instincts'].append(self.get_ave_instincts())
        hist['reinforcement'].append(self.get_ave_reinforcement())
        hist['niche_effort'].append(self.get_ave_niche_effort())
        hist['cue_dist'].append(self.get_ave_cue_dist())
        hist['behaviour_dist'].append(self.get_ave_behaviour_dist())
        hist['occurances'].append(self.get_ave_occurances())
        hist['weighted_error'].append(self.get_ave_weighted_error())
        hist['prediction_error'].append(self.get_ave_prediction_error())
        hist['fitness'].append(self.get_ave_fitness())
        hist['subj_wb'].append(self.get_ave_subj_wb())
    
    def run(self, gen=50, p_survive=0.6):
        for i in range(gen):
            self.run_generation()
            self.record_hist()
            print(len(self.history['value']),
                  self.history['fitness'][-1][0],
                  self.history['subj_wb'][-1][0])
            self.breed(p_survive=p_survive)
        self.run_generation()
        
    def _save_fig(self, var, folder, fmt='png', i=-1):
        if folder is None:
            return
        path = Path(folder)
        fname = (path / var).as_posix() + '_i' + str(i) + '.' + fmt
        plt.savefig(fname=fname, dpi=300, format=fmt)
     
    def plot_gen_history(self, var='subj_wb', ylabel='Subjective wellbeing',
                         label=None,
                     folder=None, fmt='png', xlim=None, start=None, end=10,
                     fignum=None, **kwargs):
        plt.figure(num=fignum)
        vals = []
        for p in self.pop[start:end]:
            inds, v = p.subj_wb_history(**kwargs)
            vals.append(v)
        vals = np.array(vals)
        means = np.mean(vals, axis=0)
        std = np.std(vals, axis=0)
        plt.plot(inds, vals.T, label=label)
        plt.xlabel('Period')
        plt.ylabel(label)
        plt.xlim(xlim)
        plt.tight_layout()
        self._save_fig(var + '_gen_hist', folder, fmt)
        
     
    def plot_history(self, ax=None, var='fitness',
                     label='Fitness',
                     folder=None, fmt='png'):
        if ax is None:
            fig, ax = plt.subplots()
        inds = np.arange(0, len(self.history['value']), 1)
        vals = np.array(self.history[var])
        if var == 'subj_wb':
            vals = vals[:,:,3]
        err_minus = -(vals[:,1] - vals[:,0])
        err_plus = vals[:,2] - vals[:,0]
        ax.errorbar(inds,
                    vals[:,0],
                    yerr=[err_minus, err_plus]
                    )
        ax.set_xlabel('Generation')
        ax.grid(visible=True, axis='y')
        ax.set_ylabel(label)
        fig.tight_layout()
        self._save_fig(var, folder, fmt)
        
    def plot(self, var='value', label='value', i=-1, folder=None,
             fmt='png'):
        plt.figure()
        vals = self.history[var][i]
        xs = get_xs()
        means = vals[0]
        err_minus = -(vals[1] - vals[0])
        err_plus = vals[2] - vals[0]
        plt.errorbar(xs,
                     means,
                     yerr=[err_minus, err_plus],
                     fmt='o')
        plt.xlabel('Change in fitness')
        plt.ylabel(label)
        plt.grid(visible=True, axis='y')
        plt.tight_layout()
        self._save_fig(var, folder, fmt, i=i)

    def plot_all(self, folder=None, fmt='png', i=-1):
        self.plot_history(var='fitness', 
                          label='Fitness',
                          folder=folder,
                          fmt=fmt)
        self.plot_history(var='subj_wb',
                          label='Reflective wellbeing',
                          folder=folder,
                          fmt=fmt)
        self.plot(var='value', label='Cue value',
                  i=i, folder=folder, fmt=fmt)
        self.plot(var='instincts', label='Instincts',
                  i=i, folder=folder, fmt=fmt)
        self.plot(var='reinforcement', label='Routines',
                  i=i, folder=folder, fmt=fmt)
        self.plot(var='niche_effort', label='Niche effort',
                  i=i, folder=folder, fmt=fmt)
        self.plot(var='prediction_error', label='Prediction error',
                  i=i, folder=folder, fmt=fmt)
        self.plot(var='cue_dist', label='Cue distribution',
                  i=i, folder=folder, fmt=fmt)
        self.plot(var='behaviour_dist', label='Behaviour factor',
                  i=i, folder=folder, fmt=fmt)
        self.plot(var='occurances', label='Occurence likelihood',
                  i=i, folder=folder, fmt=fmt)
        self.plot(var='weighted_error', label='Weighted error',
                  i=i, folder=folder, fmt=fmt)
        
        
        
        

        
