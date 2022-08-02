# -*- coding: utf-8 -*-
"""
Created on Sat May  7 13:17:45 2022

@author: Reuben
"""

import numpy as np
from scipy.stats import genpareto, norm, logistic


def get_xs(x_max, n):
    return np.linspace(-x_max, x_max, n)


class Context():
    """ Generates events with some effect with some likelihood """
    def __init__(self, n, x_max, c=1.0, scale=1.0, prob=1.0):
        self._params = {'n': n,
                        'x_max': x_max,
                        'c': c,
                        'scale': scale,
                        'prob': prob}
    
    def setup(self, random_seed=None):
        p = self._params
        xs = get_xs(p['x_max'], p['n'])
        nom = norm.pdf(xs)
        rs = np.random.default_rng(random_seed)
        probs = [1 - p['prob'], p['prob']]
        ps = rs.choice([0, 1], size=p['n'], p=probs) * nom
        self.xs = xs
        self.inds = np.arange(p['n'])
        self.ps = ps / sum(ps)  # Normalise they so sum to 1

    def sample(self):
        return self.ps.copy()
    
    def expected_value(self):
        return np.mean(self.xs * self.ps)


class Motivator():
    UNIQUE_SEED_MODIFIER = 37
    START_ZEROED = False
    RANGE = 1.5
    SD = 0.5
    
    def __init__(self, n, x_max, random_seed=None, init=None, decay=0.96):
        self._decay = decay
        self._history = {'vals': [],
                         'behaviour_tendency': [],
                         'cue_dist_modifier': []}
        self.setup(n, x_max, random_seed=random_seed, init=init)
        
    def setup(self, n, x_max, random_seed=None, init=None):
        if self.START_ZEROED:
            self._base = np.zeros(n)
            self._learned_vals = np.zeros(n)
        elif init is None:
            random_seed = int(np.random.rand(1)*1e6) if random_seed is None else random_seed
            random_seed += self.UNIQUE_SEED_MODIFIER
            rs = np.random.default_rng(random_seed)
            base = rs.normal(scale=self.SD, size=n)
            self._base = base
            self._learned_vals = np.zeros_like(base)
        else:
            self._base = init.copy()
            self._learned_vals = np.zeros_like(self._base)
        self.xs = xs = get_xs(x_max, n)
    
    @property
    def base(self):
        return self._base
    
    @property
    def learned_vals(self):
        return self._learned_vals
    
    def record_history(self):
        pass
    
    def process(self, cue_dist, behaviour_dist, weighted_error):
        raise NotImplementedError
    
    def get_behaviour_tendency(self):
        raise NotImplementedError
    
    def get_weighted_error(self):
        raise NotImplementedError
    
    def get_cue_dist_modifier(self):
        return np.zeros_like(self.xs)

    def breed(self, other):
        n = len(self.xs)
        x_max = self.xs[-3]
        from_other = np.random.choice([False, True], size=n)
        inherited = self._base.copy()
        inherited[from_other] = other._base[from_other]
        inherited = np.clip(inherited, -self.RANGE, self.RANGE)
        rs = np.random.default_rng()
        error = rs.normal(scale=self.SD, size=n)
        init = inherited + error
        new_motivator = self.__class__(n=n, x_max=x_max, init=init)
        return new_motivator
    
    def decay_vals(self, cue_dist):
        # This seems to be not right
        cue_dist_norm = cue_dist / np.max(cue_dist)
        norm_period = 1 / cue_dist_norm
        decay = self._decay**norm_period
        self._learned_vals *= decay
        
    
class Instincts(Motivator):
    UNIQUE_SEED_MODIFIER = 25
    
    def process(self, cue_dist, behaviour_dist, weighted_error):
        # There's no function of cues or learning
        pass
    
    def get_behaviour_tendency(self):
        return self._base


class Prediction_Error(Motivator):
    UNIQUE_SEED_MODIFIER = 16
    RANGE = 3.0
    
    def __init__(self, *args, rate=0.1, **kwargs):
        self._rate = rate
        super().__init__(*args, **kwargs)
        self._prediction_error = np.zeros_like(self._base)
        self._weighted_error = np.zeros_like(self._base)
        self._weighted_error_rate = np.zeros_like(self._base)
        
    def process(self, cue_dist, behaviour_dist, weighted_error):
        actual = self._base
        predicted = self._learned_vals
        prediction_error = actual - predicted
        weighted_error = prediction_error * cue_dist * behaviour_dist * 100
        self._prediction_error = prediction_error
        self._weighted_error = weighted_error
        self._weighted_error_rate = weighted_error / cue_dist
        self._learned_vals += weighted_error * self._rate
        self.decay_vals(cue_dist)
        
    def get_prediction_error(self):
        # print('pred', self._prediction_error)
        return self._prediction_error
        
    def get_weighted_error(self):
        return self._weighted_error
    
    def get_weighted_error_rate(self):
        return self._weighted_error_rate
    
    def get_behaviour_tendency(self):
        return np.zeros_like(self._base)


class Routines(Motivator):
    UNIQUE_SEED_MODIFIER = 564
    START_ZEROED = True
    
    def __init__(self, *args, rate=0.1, **kwargs):
        self._rate = rate
        super().__init__(*args, **kwargs)
    
    def learn(self, weighted_error):
        self._learned_vals += weighted_error * self._rate

    def get_behaviour_tendency(self):
        return self._learned_vals

    def process(self, cue_dist, behaviour_dist, weighted_error):
        self.learn(weighted_error)
        self.decay_vals(cue_dist)
    

class Planned_Control(Motivator):
    UNIQUE_SEED_MODIFIER = 531
    START_ZEROED = True
    
    """ Niche construction """
    
    def __init__(self, *args, rate=0.1, num=20, f_tot=0.2,
                 **kwargs):
        self._rate = rate
        self._num = num
        super().__init__(*args, **kwargs)
        self._tot = len(self.xs) * f_tot

    def set_behaviour(self, behaviour):
        self._behaviour = behaviour
    
    def cue_mod(self, modifier):
        adj = modifier
        return 5.0**adj 
    
    def _learn_unlim(self, weighted_error):
        err = weighted_error
        changes = np.ones_like(self.xs) * self._rate
        changes = np.copysign(changes, err)
        self._learned_vals += changes
    
    def _learn_lim(self, cue_dist, weighted_error):
        """ TODO: Cap effort somehow """
        err = weighted_error
        effort = self._learned_vals
        d_effort = (self.cue_mod(effort + self._rate) - self.cue_mod(effort)) * cue_dist
        inds = np.arange(0, len(self.xs), 1)
        A = np.random.choice(inds, size=self._num)
        B = np.random.choice(inds, size=self._num)
        diff = err[A] * d_effort[A] - err[B] *  d_effort[B]
        direction = np.ones_like(A)
        direction[diff < 0] = -1
        changes = np.zeros_like(self.xs)
        changes[A] = direction * self._rate
        changes[B] = -direction * self._rate
        self._learned_vals += changes
    
    def learn(self, cue_dist, weighted_error):
        if np.sum(np.abs(self._learned_vals)) >= self._tot:
            return self._learn_lim(cue_dist, weighted_error)
        else:
            return self._learn_unlim(weighted_error)
    
    def process(self, cue_dist, behaviour_dist, weighted_error):
        self.learn(cue_dist, weighted_error)
        self.decay_vals(cue_dist)

    def get_behaviour_tendency(self):
        """ Don't modify behavioural responses """
        return np.zeros_like(self.xs)
    
    def get_cue_dist_mod(self):
        mod = self.cue_mod(self._learned_vals)
        return mod
    
    def get_modified_cue_dist(self, cue_dist):
        mod = self.cue_mod(self._learned_vals)
        return mod * cue_dist
        

class Person():
    def __init__(self, n, x_max, motivators=None, a_decay=0.98):
        self.history = {'cue_dist': [], 'behaviour_dist': [],
                        'weighted_error': []}
        self.xs = get_xs(x_max, n)
        self.setup(motivators)
        self._a_decay = a_decay
            
    def setup(self, motivators=None, random_seed=None):
        n = len(self.xs)
        x_max = self.xs[-1]
        if motivators is None:
            self._predictor = Prediction_Error(n, x_max, random_seed=random_seed)
            self._instincts = Instincts(n, x_max, random_seed=random_seed)
            self._routines = Routines(n, x_max, random_seed=random_seed)
            self._planned_control = Planned_Control(n, x_max,
                                                    random_seed=random_seed)
        else:
            self.set_motivators(motivators)
        self._random_seed = random_seed
    
    def reset(self):
        self.setup(random_seed=self._random_seed)
    
    @property
    def predictor(self):
        return self._predictor

    @property
    def planner(self):
        return self._planned_control

    @property
    def valence(self):
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
    def response_effort(self):
        return self.self.planner.get_behaviour_tendency()
    
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
        self.history['weighted_error'].append(weighted_error) # THIS WAS WRONG! behaviour_dist
    
    def process(self, cue_dist):
        behaviour_dist_xs = np.ones_like(cue_dist)
        motivators = [self._instincts,
                      self._routines,
                      self._planned_control,
                      ]
        for motivator in motivators:
            behaviour_dist_xs += motivator.get_behaviour_tendency()
        # TODO: Use dict for motivators
        weighted_error = self.predictor.get_weighted_error()
        behaviour_dist = logistic.cdf(behaviour_dist_xs)
        mod_cue_dist = self.planner.get_modified_cue_dist(cue_dist)
        for motivator in self.motivators:
            motivator.process(mod_cue_dist, behaviour_dist, weighted_error)
        self.store_history(mod_cue_dist, behaviour_dist, weighted_error)

    def subjective_wellbeing(self, i=None, n=1000):
        end = len(self.history['cue_dist']) - 1 if i is None else i
        start = max(0, end - n)
        neg_xs = self.xs < 0
        costs_benefits = np.array(self.history['weighted_error'][start:end])
        inds = np.arange(end, start, -1)
        weights = np.power(self._a_decay, inds - start)
        totals = weights.reshape(-1, 1) * costs_benefits
        sums = np.sum(totals, axis=0) / np.sum(weights)
        neg = np.sum(sums[neg_xs])
        pos = np.sum(sums[~neg_xs])
        net = np.sum(sums)
        ratio = pos / (pos - neg)
        return neg, pos, net, ratio
    
    def objective_wellbeing(self, i=None, n=1000, decay=0.999):
        end = len(self.history['cue_dist']) - 1 if i is None else i
        start = max(0, end - n)
        inds = np.arange(end, start, -1)
        weights = np.power(self._a_decay, inds) + 1e-2
        cue_dist = np.array(self.history['cue_dist'][start:end])
        behaviour_dist = np.array(self.history['behaviour_dist'][start:end])
        xs = self.xs.reshape(1, -1)
        costs_benefits = xs * cue_dist * behaviour_dist
        totals = weights.reshape(-1, 1) * costs_benefits * 1000
        return np.sum(totals) / np.sum(weights)
    
    def breed(self, other):
        """ Create a child """
        motivators = []
        for m1, m2 in zip(self.motivators, other.motivators):
            new = m1.breed(m2)
            motivators.append(new)
        n = len(self.xs)
        x_max = self.xs[-1]
        child = Person(n, x_max, motivators=motivators)
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
        self.history = {'valence': [],
                        'instincts': [],
                        'niche_effort': [],
                        'response_effort': [],
                        'reinforcement': [],
                        'obj_wb': [],
                        'subj_wb': [],
                        }
    
    def set_life_history(self, lh):
        self.lh = lh
    
    def set_population(self, pop_size, random_seed=None, *args, **kwargs):
        pop = []
        for i in range(pop_size):
            person = Person(*args, **kwargs)
            seed = random_seed + 33 * i
            person.setup(random_seed=seed)
            pop.append(person)
        self.pop = pop
    
    def _split(self, n_fail, n_top=5):
        wbs = [p.objective_wellbeing() for p in self.pop]
        ind_sorted = np.argsort(wbs)
        top = ind_sorted[-n_top:]
        bottom = ind_sorted[:n_fail]
        breeders = ind_sorted[n_fail + n_top:-n_top]
        pop = np.array(self.pop)
        breeders_lst = pop[breeders].tolist() + pop[top].tolist()  # Elitist selection
        return pop[top].tolist(), pop[bottom].tolist(), breeders_lst
    
    def get_ave_obj_wb(self):
        wbs = [p.objective_wellbeing() for p in self.pop]
        return np.mean(wbs)

    def get_ave_subj_wb(self):
        wbs = [p.subjective_wellbeing() for p in self.pop]
        return np.mean(wbs, axis=0)
    
    def get_ave_valence(self):
        return np.mean(np.array([p.valence for p in self.pop]), axis=0)
    
    def get_ave_instincts(self):
        return np.mean(np.array([p.instincts for p in self.pop]), axis=0)
    
    def get_ave_niche_effort(self):
        return np.mean(np.array([p.niche_effort for p in self.pop]), axis=0)
    
    def get_ave_reinforcement(self):
        return np.mean(np.array([p.reinforcement for p in self.pop]), axis=0)
    
    def breed(self, p_survive=0.5):
        n_fail = int(np.floor((1 - p_survive) * len(self.pop)))
        top, bottom, breeders = self._split(n_fail=n_fail)
        new = []
        for i in range(len(self.pop)):
            mate = other = np.random.choice(top, size=1)[0]
            other = np.random.choice(breeders, size=1)[0]
            new.append(mate.breed(other))
        self.pop = new
    
    def run_generation(self):
        lh = self.lh
        for i, p in enumerate(self.pop):
            # print('   ' + str(i))
            lh.set_person(p)
            lh.run()
    
    def record_hist(self):
        hist = self.history
        hist['valence'].append(self.get_ave_valence())
        hist['instincts'].append(self.get_ave_instincts())
        hist['niche_effort'].append(self.get_ave_niche_effort())
        hist['reinforcement'].append(self.get_ave_reinforcement())
        hist['obj_wb'].append(self.get_ave_obj_wb())
        hist['subj_wb'].append(self.get_ave_subj_wb())
    
    def run(self, gen=50, p_survive=0.5):
        for i in range(gen):
            self.run_generation()
            self.record_hist()
            print(len(self.history['valence']),
                  self.history['obj_wb'][-1],
                  self.history['subj_wb'][-1])
            self.breed(p_survive=p_survive)
        self.run_generation()
        
    
