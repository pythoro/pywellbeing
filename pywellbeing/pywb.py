# -*- coding: utf-8 -*-
"""
Created on Sat May  7 13:17:45 2022

@author: Reuben
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genpareto, norm, logistic
from pathlib import Path

def get_xs(x_max, n):
    return np.linspace(-x_max, x_max, n)


class Context():
    """ Generates events with some effect with some likelihood """
    def __init__(self, n, x_max, c=1.0, scale=1.0, prob=1.0, loc=0.0):
        self._params = {'n': n,
                        'x_max': x_max,
                        'c': c,
                        'scale': scale,
                        'prob': prob,
                        'loc': loc}
    
    def setup(self, random_seed=None):
        p = self._params
        xs = get_xs(p['x_max'], p['n'])
        nom = norm.pdf(xs, loc=p['loc'])
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

    def plot(self):
        plt.figure()
        plt.scatter(self.xs, self.ps)
        plt.xlabel('RL impact')
        plt.ylabel('Occurance likelihood')
        plt.tight_layout()


class Motivator():
    UNIQUE_SEED_MODIFIER = 37
    START_ZEROED = False
    RANGE = 1.5
    SD = 0.5
    
    def __init__(self, n, x_max, random_seed=None, init=None, decay=0.98):
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
        self._learned_vals *= self._decay
        
    
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
    
    def __init__(self, *args, rate=2, **kwargs):
        self._rate = rate
        super().__init__(*args, **kwargs)
        self._prediction_error = np.zeros_like(self._base)
        self._weighted_error = np.zeros_like(self._base)
    
    def get_rates(self, cue_dist, behaviour_dist):
        occurance_dist = cue_dist * behaviour_dist 
        occurances = occurance_dist
        rates = occurances * self._rate
        return rates
    
    def process(self, cue_dist, behaviour_dist, weighted_error):
        actual = self._base
        predicted = self._learned_vals
        prediction_error = actual - predicted
        weighted_error = prediction_error * cue_dist * behaviour_dist
        self._prediction_error = prediction_error
        self._weighted_error = weighted_error
        rates = self.get_rates(cue_dist, behaviour_dist)
        self._learned_vals += prediction_error * rates
        self.decay_vals(cue_dist)
        
    def get_prediction_error(self):
        # print('pred', self._prediction_error)
        return self._prediction_error
        
    def get_weighted_error(self):
        return self._weighted_error
        
    def get_behaviour_tendency(self):
        return np.zeros_like(self._base)


class Routines(Motivator):
    UNIQUE_SEED_MODIFIER = 564
    START_ZEROED = True
    
    def __init__(self, *args, rate=3.0, **kwargs):
        self._rate = rate
        self._do_learn = True
        super().__init__(*args, **kwargs)
    
    def learn(self, weighted_error):
        self._learned_vals += weighted_error * self._rate

    def set_do_learn(self, flag):
        self._do_learn = flag

    def get_behaviour_tendency(self):
        return self._learned_vals

    def process(self, cue_dist, behaviour_dist, weighted_error):
        if not self._do_learn:
            return
        self.learn(weighted_error)
        self.decay_vals(cue_dist)
    

class Planned_Control(Motivator):
    UNIQUE_SEED_MODIFIER = 531
    START_ZEROED = True
    
    """ Niche construction """
    
    def __init__(self, *args, rate=0.01, num=20, f_tot=0.1,
                 **kwargs):
        self._rate = rate
        self._num = num
        self._do_learn = True
        super().__init__(*args, **kwargs)
        self._tot = len(self.xs) * f_tot

    def set_do_learn(self, flag):
        self._do_learn = flag

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
        if not self._do_learn:
            return
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
    def __init__(self, n, x_max, motivators=None, a_decay=0.97,
                 history=False):
        self.xs = get_xs(x_max, n)
        self.setup(motivators)
        self._a_decay = a_decay
        self._record_history = history
            
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
        n = len(self.xs)
        x_max = self.xs[-1]
        return Person(n, x_max, motivators=self.motivators,
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

    def subj_wb_history(self, k=3, n=1, decay=None):
        n_hist = len(self.history['cue_dist'])
        indices = np.arange(2, n_hist, 1)
        wb = [self.subjective_wellbeing(i=i, n=n, decay=decay)[k] for i in indices]
        return indices, np.array(wb)
    
    def subj_wb_history_decayed(self, k=3, n=100, decay=0.9):
        return self.subj_wb_history(k=k, n=n, decay=decay)
    
    def plot_wb_history(self, xlim=None):
        plt.figure()
        vals = []
        for p in self.pop:
            inds, v = p.subj_wb_history()
            vals.append(v)
        vals = np.array(vals)
        means = np.mean(vals, axis=0)
        std = np.std(vals, axis=0)
        plt.plot(inds, vals[:10].T)
        plt.xlabel('Period')
        plt.ylabel('Subjective wellbeing')
        plt.xlim(xlim)
        plt.tight_layout()
    
    def obj_wb_history(self):
        n = len(self.history['cue_dist'])
        indices = np.arange(2, n, 1)
        wb = [self.objective_wellbeing(i) for i in indices]
        return indices, np.array(wb)

    def subjective_wellbeing(self, i=None, n=1, decay=None):
        end = len(self.history['cue_dist']) - 1 if i is None else i
        start = max(0, end - n)
        neg_xs = self.xs < 0
        costs_benefits = np.array(self.history['weighted_error'][start:end])
        inds = np.arange(end, start, -1)
        decay = self._a_decay if decay is None else decay
        weights = np.power(decay, inds - start)
        totals = weights.reshape(-1, 1) * costs_benefits
        sums = np.sum(totals, axis=0) / np.sum(weights)
        neg = np.sum(sums[neg_xs])
        pos = np.sum(sums[~neg_xs])
        net = np.sum(sums)
        ratio = pos / (abs(pos) + abs(neg))
        return neg, pos, net, ratio
    
    def objective_wellbeing(self, i=None, n=1000, decay=1):
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
                        'cue_dist': [],
                        'behaviour_dist': [],
                        'occurances': [],
                        'weighted_error': [],
                        'prediction_error': [],
                        'obj_wb': [],
                        'subj_wb': [],
                        }
    
    def set_life_history(self, lh):
        self.lh = lh
    
    def set_pop(self, population, reset=True):
        self.pop = [p.copy() for p in population.pop]
        if reset:
            for p in self.pop:
                p.reset()
    
    def set_population(self, pop_size, random_seed=None, 
                       n_history=10, *args, **kwargs):
        pop = []
        for i in range(pop_size):
            history = True if i < n_history else False
            person = Person(*args, history=history, **kwargs)
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
        return np.mean(wbs), np.std(wbs)

    def get_ave_subj_wb(self):
        wbs = [p.subjective_wellbeing() for p in self.pop]
        return np.mean(wbs, axis=0), np.std(wbs, axis=0)
    
    def get_ave_valence(self):
        vals = np.array([p.valence for p in self.pop])
        return np.mean(vals, axis=0), np.std(vals, axis=0)
    
    def get_ave_instincts(self):
        vals = np.array([p.instincts for p in self.pop])
        return np.mean(vals, axis=0), np.std(vals, axis=0)
    
    def get_ave_niche_effort(self):
        vals = np.array([p.niche_effort for p in self.pop])
        return np.mean(vals, axis=0), np.std(vals, axis=0)
    
    def get_ave_reinforcement(self):
        vals = np.array([p.reinforcement for p in self.pop])
        return np.mean(vals, axis=0), np.std(vals, axis=0)
    
    def get_ave_cue_dist(self):
        vals = np.array([p.cue_dist for p in self.pop])
        return np.mean(vals, axis=0), np.std(vals, axis=0)

    def get_ave_behaviour_dist(self):
        vals = np.array([p.behaviour_dist for p in self.pop])
        return np.mean(vals, axis=0), np.std(vals, axis=0)
    
    def get_ave_weighted_error(self):
        vals = np.array([p.weighted_error for p in self.pop])
        return np.mean(vals, axis=0), np.std(vals, axis=0)
    
    def get_ave_prediction_error(self):
        vals = np.array([p.prediction_error for p in self.pop])
        return np.mean(vals, axis=0), np.std(vals, axis=0)
    
    def get_ave_occurances(self):
        vals = np.array([p.cue_dist * p.behaviour_dist for p in self.pop])
        return np.mean(vals, axis=0), np.std(vals, axis=0)
    
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
        hist['reinforcement'].append(self.get_ave_reinforcement())
        hist['niche_effort'].append(self.get_ave_niche_effort())
        hist['cue_dist'].append(self.get_ave_cue_dist())
        hist['behaviour_dist'].append(self.get_ave_behaviour_dist())
        hist['occurances'].append(self.get_ave_occurances())
        hist['weighted_error'].append(self.get_ave_weighted_error())
        hist['prediction_error'].append(self.get_ave_prediction_error())
        hist['obj_wb'].append(self.get_ave_obj_wb())
        hist['subj_wb'].append(self.get_ave_subj_wb())
    
    def run(self, gen=50, p_survive=0.5):
        for i in range(gen):
            self.run_generation()
            self.record_hist()
            print(len(self.history['valence']),
                  self.history['obj_wb'][-1][0],
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
        
     
    def plot_history(self, ax=None, var='obj_wb',
                     label='Reproductive likelihood',
                     folder=None, fmt='png'):
        if ax is None:
            fig, ax = plt.subplots()
        inds = np.arange(0, len(self.history['valence']), 1)
        vals = np.array(self.history[var])
        if var == 'subj_wb':
            vals = vals[:,:,3]
        ax.errorbar(inds,
                    vals[:,0],
                    yerr=vals[:,1]*1.96
                    )
        ax.set_xlabel('Generation')
        ax.set_ylabel(label)
        fig.tight_layout()
        self._save_fig(var, folder, fmt)
        
    def plot(self, var='valence', label='Valence', i=-1, folder=None,
             fmt='png'):
        plt.figure()
        self.history[var]
        xs = self.pop[0].xs
        plt.errorbar(xs,
                     self.history[var][i][0],
                     yerr=self.history[var][i][1]*1.96,
                     fmt='o')
        plt.xlabel('RL impact')
        plt.ylabel(label)
        plt.tight_layout()
        self._save_fig(var, folder, fmt, i=i)

    def plot_all(self, folder=None, fmt='png', i=-1):
        self.plot_history(var='obj_wb', 
                          label='Reproductive likelihood',
                          folder=folder,
                          fmt=fmt)
        self.plot_history(var='subj_wb',
                          label='Subjective wellbeing',
                          folder=folder,
                          fmt=fmt)
        self.plot(var='valence', label='Cue valence',
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
        self.plot(var='occurances', label='Occurance likelihood',
                  i=i, folder=folder, fmt=fmt)
        
        
        

        
