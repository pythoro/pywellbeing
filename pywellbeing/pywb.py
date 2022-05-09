# -*- coding: utf-8 -*-
"""
Created on Sat May  7 13:17:45 2022

@author: Reuben
"""

import numpy as np
from scipy.stats import genpareto, norm, logistic


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
        x_half = np.linspace(p['x_max']/p['n'], p['x_max'], p['n'])
        nom = 1 / x_half  # Make expected value 1 for all points.
        xs = np.linspace(-p['x_max'], p['x_max'], p['n'] * 2)
        weights = np.concatenate((np.flip(nom), nom))
        rs = np.random.default_rng(random_seed)
        probs = [1 - p['prob'], p['prob']]
        ps = rs.choice([0, 1], size=p['n'] * 2, p=probs) * weights
        self.xs = xs
        self.inds = np.arange(p['n'] * 2)
        self.ps = ps / sum(ps)

    def sample(self, agency=None):
        if agency is None:
            return self.ps
        else:
            if len(agency) != self._params['n'] * 2:
                raise ValueError('mod is wrong size')
            else:
                return self.ps * agency
    
    def expected_value(self):
        return np.mean(self.xs * self.ps)



class Person():
    def __init__(self, n, x_max, alpha=0.0,
                 f_att=5, f_eff=10, f_att2=0.1, f_eff2=0.1, a_decay=0.999):
        self.history = {'ps': [], 'valence': [], 'attend': []}
        self._params = {'n': n,
                        'x_max': x_max,
                        'alpha': alpha,
                        'f_att': f_att,
                        'f_att2': f_att2,
                        'f_eff': f_eff,
                        'f_eff2': f_eff2,
                        'a_decay': a_decay}
        
    def set_attention(self, base=None, random_seed=None):
        p = self._params
        rs = np.random.default_rng(random_seed)
        base = np.zeros(p['n'] * 2) if base is None else base
        error = rs.normal(scale=1.0, size=p['n'] * 2)
        attention = base + error
        return attention
    
    def set_valence(self, xs, base=None, random_seed=None):
        p = self._params
        random_seed = int(np.random.rand(1)*1e6) if random_seed is None else random_seed
        random_seed += 10
        rs = np.random.default_rng(random_seed)
        base = np.zeros(p['n'] * 2) if base is None else base
        error = rs.normal(scale=1.0, size=p['n'] * 2)
        bias = xs * p['alpha']
        valence = base + error + bias
        # Ensure valence can't grow indefinately...
        valence = np.clip(valence, -9, 9)
        return valence
    
    def setup(self, attention=None, valence=None, random_seed=None):
        p = self._params
        xs = np.linspace(-p['x_max'], p['x_max'], p['n'] * 2)
        self.xs = xs
        self.base_attention = np.zeros(p['n'] * 2) if attention is None else attention
        self.attention = self.set_attention(base=attention,
                                            random_seed=random_seed)
        self.valence = self.set_valence(xs,
                                        base=valence,
                                        random_seed=random_seed)
        self.effort = np.zeros(p['n'] * 2)
    
    def determine_attend(self, ind):
        """ This needs checking """
        return True
        att = self.attention
        p = self._params
        a = att[ind] + self.base_attention[ind]
        p_will_attend = min(np.argsort(att)[ind] / len(att), 0.1)
        attend = p_will_attend >= np.random.rand()
        return attend

    def mod_attention(self, ps, valence):
        att = self.attention
        att = att + ps * abs(self.valence) * self._params['f_att2']
    
    def decay_attention(self):
        """ Decay toward uniform attention """
        self.attention = self.attention * self._params['a_decay'] 

    def mod_effort(self, ps, valence):
        eff = self.effort
        self.effort = eff + ps * valence * self._params['f_eff2']
    
    def get_agency(self):
        f = self._params['f_eff']
        s = np.max(np.abs(self.effort))
        if s == 0:
            adj = self.effort
        else:
            adj = self.effort / s
        return 5.0**adj
    
    def decay_effort(self):
        """ Decay toward uniform attention """
        self.effort = self.effort * self._params['a_decay']
    
    def store_history(self, ps, valence, attend):
        self.history['ps'].append(ps)
        self.history['valence'].append(valence)
        self.history['attend'].append(attend)
    
    def learn(self, ps, attend=None, learn=True):
        attend = self.determine_attend(ps) if attend is None else attend
        valence = self.valence
        if learn:
            if attend:
                # self.mod_attention(ind, valence)
                self.mod_effort(ps, valence)
            # self.decay_attention()
            self.decay_effort()
        self.store_history(ps, valence, attend)

    def subjective_wellbeing(self, i=None, n=1000):
        end = len(self.history['valence']) - 1 if i is None else i
        start = max(0, end - n)
        filt = np.array(self.history['attend'][start:end]).flatten()
        costs_benefits = self.xs * np.array(self.history['valence'][start:end])
        costs_benefits[~filt, :] = 0.0
        inds = np.arange(end, start, -1)
        weights = np.power(self._params['a_decay'], inds - start)
        totals = weights.reshape(-1, 1) * costs_benefits
        ratio = (np.sum(totals[totals > 0]) / 
                 -np.sum(totals[totals < 0]))
        weighted_ave = np.sum(totals) / sum(weights)
        return ratio, weighted_ave
    
    def objective_wellbeing(self, i=None, n=1000, decay=0.999):
        end = len(self.history['valence']) - 1 if i is None else i
        start = max(0, end - n)
        inds = np.arange(end, start, -1)
        weights = np.power(self._params['a_decay'], inds) + 1e-2
        costs_benefits = self.xs * self.history['ps'][start:end]
        totals = weights.reshape(-1, 1) * costs_benefits * 1000
        return np.sum(totals) / np.sum(weights)
    
    def breed(self, other):
        """ Create a child """
        n = self._params['n'] * 2
        from_other = np.random.choice([False, True], size=n)
        attention = self.base_attention
        attention[from_other] = other.base_attention[from_other]
        valence = self.valence
        valence[from_other] = other.valence[from_other]
        child = Person(**self._params)
        child.setup(attention=attention, valence=valence)
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
            agency = person.get_agency()
            ps = context.sample(agency=agency)
            person.learn(ps)
    
    def run(self):
        for d in self.contexts:
            self.run_one(**d)

                
class Population():
    """ Put lots of people through life histories and generations """
    def __init__(self):
        self.history = {'valence': [],
                        'effort': [],
                        'agency': [],
                        'obj_wb': [],
                        'subj_wb': [],
                        }
    
    def set_life_history(self, lh):
        self.lh = lh
    
    def set_population(self, pop_size, random_seed=None, *args, **kwargs):
        pop = []
        for i in range(pop_size):
            person = Person(*args, **kwargs)
            person.setup(random_seed=random_seed)
            pop.append(person)
        self.pop = pop
    
    def _split(self, n_fail, n_top=5):
        wbs = [p.objective_wellbeing() for p in self.pop]
        ind_sorted = np.argsort(wbs)
        top = ind_sorted[-n_top:]
        bottom = ind_sorted[:n_fail]
        breeders = ind_sorted[n_fail:-n_top]
        pop = np.array(self.pop)
        return pop[top], pop[bottom].tolist(), pop[breeders].tolist()
    
    def get_ave_obj_wb(self):
        wbs = [p.objective_wellbeing() for p in self.pop]
        return np.mean(wbs)

    def get_ave_subj_wb(self):
        wbs = [p.subjective_wellbeing()[1] for p in self.pop]
        return np.mean(wbs)
    
    def get_ave_valence(self):
        return np.mean(np.array([p.valence for p in self.pop]), axis=0)
    
    def get_ave_effort(self):
        return np.mean(np.array([p.effort for p in self.pop]), axis=0)
    
    def get_ave_agency(self):
        return np.mean(np.array([p.get_agency() for p in self.pop]), axis=0)
    
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
        hist['effort'].append(self.get_ave_effort())
        hist['agency'].append(self.get_ave_agency())
        hist['obj_wb'].append(self.get_ave_obj_wb())
        hist['subj_wb'].append(self.get_ave_subj_wb())
    
    def run(self, gen=50, p_survive=0.5):
        for i in range(gen):
            self.run_generation()
            self.record_hist()
            print(len(self.history['valence']),
                  self.get_ave_obj_wb(),
                  self.get_ave_subj_wb())
            self.breed(p_survive=p_survive)
        
    
