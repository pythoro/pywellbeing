# -*- coding: utf-8 -*-
"""
Created on Sat May  7 13:17:45 2022

@author: Reuben
"""

import numpy as np
from scipy.stats import genpareto, norm, logistic


class Context():
    """ Generates events with some effect with some likelihood """
    def __init__(self, n, x_max, c=1.0, scale=1.0, prob=0.3):
        self._params = {'n': n,
                        'x_max': x_max,
                        'c': c,
                        'scale': scale,
                        'prob': prob}
    
    def setup(self, random_seed=None):
        p = self._params
        x_half = np.linspace(p['x_max']/p['n'], p['x_max'], p['n'])
        nom = genpareto.pdf(x_half, c=p['c'], scale=p['scale'])
        xs = np.linspace(-p['x_max'], p['x_max'], p['n'] * 2)
        weights = np.concatenate((np.flip(nom), nom))
        rs = np.random.default_rng(random_seed)
        probs = [1 - p['prob'], p['prob']]
        ps = rs.choice([0, 1], size=p['n'] * 2, p=probs) * weights
        self.xs = xs
        self.inds = np.arange(p['n'] * 2)
        self.ps = ps / sum(ps)

    def sample(self, size=1, agency=None):
        if agency is None:
            ps = self.ps
        else:
            if len(agency) != self._params['n'] * 2:
                raise ValueError('mod is wrong size')
            else:
                ps = self.ps * agency
                ps = ps / sum(ps)
        ind = np.random.choice(a=self.inds, size=size, p=ps)
        x = self.xs[ind]
        return ind, x
    
    def expected_value(self):
        return np.mean(self.xs * self.ps)



class Person():
    def __init__(self, n, x_max, alpha=0.0,
                 f_att=5, f_eff=10, f_att2=0.01, f_eff2=0.01, a_decay=0.999):
        self.history = {'ind': [], 'x': [], 'valence': [], 'attend': []}
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
        rs = np.random.default_rng(random_seed)
        base = np.zeros(p['n'] * 2) if base is None else base
        error = rs.normal(scale=1.0, size=p['n'] * 2)
        bias = xs * p['alpha']
        valence = base + error + bias
        return valence
    
    def setup(self, attention=None, valence=None, random_seed=None):
        p = self._params
        xs = np.linspace(-p['x_max'], p['x_max'], p['n'] * 2)
        self.xs = xs
        self.base_attention = np.zeros(p['n'] * 2) if attention is None else attention
        self.attention = self.set_attention(base=attention,
                                            random_seed=random_seed)
        self._att_lasting = np.zeros_like(self.attention)
        self.valence = self.set_valence(xs,
                                        base=valence,
                                        random_seed=random_seed)
        self.effort = np.zeros(p['n'] * 2)
        self._eff_lasting = np.zeros_like(self.effort)
    
    def determine_attend(self, ind):
        att = self.attention
        p = self._params
        a = att[ind] + self.base_attention[ind]
        p_will_attend = logistic.cdf(att[ind] / p['f_att'])
        attend = p_will_attend >= np.random.rand()
        return attend

    def mod_attention(self, ind, valence):
        att = self.attention
        current = att[ind]
        att[ind] = current + abs(valence) * self._params['f_att2']
    
    def decay_attention(self):
        """ Decay toward uniform attention """
        diff = self.attention - self._att_lasting
        self._att_lasting += diff * 1e-2
        self.attention = (diff * self._params['a_decay'] \
                          + self._att_lasting)

    def mod_effort(self, ind, valence):
        current = self.effort[ind]
        self.effort[ind] = current + valence * self._params['f_eff2']
    
    def get_agency(self):
        f = self._params['f_eff']
        return logistic.cdf(self.effort / f)**2 / 0.5**2 + 1/f
    
    def decay_effort(self):
        """ Decay toward uniform attention """
        diff = self.effort - self._eff_lasting
        self._eff_lasting += diff * 1e-2
        self.effort = (diff * self._params['a_decay'] \
                          + self._eff_lasting)
    
    def store_history(self, ind, x, valence, attend):
        self.history['ind'].append(ind)
        self.history['x'].append(x)
        self.history['valence'].append(valence)
        self.history['attend'].append(attend)
    
    def learn(self, ind, x, attend=None, learn=True):
        attend = self.determine_attend(ind) if attend is None else attend
        valence = self.valence[ind]
        if learn:
            if attend:
                self.mod_attention(ind, valence)
                self.mod_effort(ind, valence)
            self.decay_attention()
            self.decay_effort()
        self.store_history(ind, x, valence, attend)

    def subjective_wellbeing(self, i=None, n=1000):
        end = len(self.history['ind']) - 1 if i is None else i
        start = max(0, end - n)
        filt = np.array(self.history['attend'][start:end]).flatten()
        vals = np.array(self.history['valence'][start:end]).flatten()
        vals[~filt] = 0.0
        inds = np.arange(end, start, -1)
        weights = np.power(self._params['a_decay'], inds)
        if sum(vals<0) == 0:
            ratio = 9999
        else:
            ratio = (np.sum(weights[vals>0] * vals[vals>0]) / 
                     -np.sum(weights[vals<0] * vals[vals<0]))
        weighted_ave = np.sum(vals * weights)  / sum(weights)
        return ratio, weighted_ave
    
    def objective_wellbeing(self, i=None, n=1000, decay=0.999):
        end = len(self.history['ind']) - 1 if i is None else i
        start = max(0, end - n)
        inds = np.arange(end, start, -1)
        weights = np.power(self._params['a_decay'], inds) + 1e-2
        xs = self.history['x'][start:end]
        return np.sum(xs * weights) / np.sum(weights)
    
    def breed(self, other):
        """ Create a child """
        n = self._params['p'] * 2
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
            ind, x = context.sample(size=1, agency=agency)
            person.learn(ind, x)
    
    def run(self):
        for d in self.contexts:
            self.run_one(**d)

                


class Population():
    """ Put lots of people through life histories and generations """
    pass