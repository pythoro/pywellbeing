# -*- coding: utf-8 -*-
"""
Created on Sat May  7 13:17:45 2022

@author: Reuben
"""

import numpy as np
from scipy.stats import genpareto, norm


class Context():
    """ Generates events with some effect with some likelihood """
    def __init__(self, n, x_max, c=1.0, scale=1.0, power=2):
        self._params = {'n': n,
                        'x_max': x_max,
                        'c': c,
                        'scale': scale,
                        'power': power}
    
    def setup(self, random_seed=None):
        p = self._params
        x_half = np.linspace(p['x_max']/p['n'], p['x_max'], p['n'])
        nom = genpareto.pdf(x_half, c=p['c'], scale=p['scale'])
        rs = np.random.default_rng(random_seed)
        xs = np.linspace(-p['x_max'], p['x_max'], p['n'] * 2)
        weights = np.concatenate((np.flip(nom), nom))
        ps = rs.uniform(0, 1, p['n'] * 2)**p['power'] * weights
        self.xs = xs
        self.inds = np.arange(p['n'] * 2)
        self.ps = ps / sum(ps)

    def sample(self, size=1, mod=None):
        if mod is None:
            ps = self.ps
        else:
            if len(mod) != self._params['n']:
                raise ValueError('mod is wrong size')
            else:
                ps = self.ps * mod
                ps = ps / sum(ps)
        ind = np.random.choice(a=self.inds, size=size, p=ps)
        x = self.xs[ind]
        return ind, x
    
    def expected_value(self):
        return np.mean(self.xs * self.ps)



class Person():
    def __init__(self, n, x_max, a_shape=1.0, a_scale=1.0, alpha=0.05,
                 f_att=10, f_eff=10, a_decay=0.999):
        self.history = {'ind': [], 'x': [], 'valence': [], 'attend': []}
        self._params = {'n': n,
                        'x_max': x_max,
                        'a_shape': a_shape,
                        'a_scale': a_scale,
                        'alpha': alpha,
                        'f_att': f_att,
                        'f_eff': f_eff,
                        'a_decay': a_decay}
        
    def _get_attention(self, random_seed=None):
        p = self._params
        rs = np.random.default_rng(random_seed)
        attention = rs.gamma(shape=p['a_shape'], size=p['n'] * 2)
        return attention
    
    def a_dist(self):
        loc = np.quantile(att, 0.5)
        return norm(loc=loc, scale=self._params['a_scale'])
    
    def _get_valence(self, xs, random_seed=None):
        p = self._params
        rs = np.random.default_rng(random_seed + 1000)
        valence = rs.standard_normal(size=p['n'] * 2)
        bias = xs * p['alpha']
        return valence + bias
            
    def setup(self, random_seed=None):
        p = self._params
        xs = np.linspace(-p['x_max'], p['x_max'], p['n'] * 2)
        self.xs = xs
        self.attention = self._get_attention(random_seed)
        self.valence = self._get_valence(xs, random_seed)
        self.effort = np.ones()
    
    def determine_attend(self, ind):
        att = self.attention
        y = att[ind]
        a_dist = self.a_dist()
        p_will_attend = a_dist.cdf(y)
        attend = np.random.rand() <= p_will_attend
        return attend

    def mod_attention(self, ind, valence):
        att = self.attention
        current = att[ind]
        att[ind] = current + abs(valance) / self._params['f_att']
    
    def decay_attention(self):
        """ Decay toward uniform attention """
        decay = self._params['a_decay']
        self.attention = (self.attention - 1) * decay + 1

    def mod_effort(self, ind, valence):
        current = self.effort[ind]
        self.current[ind] = current * ( 1 + valance / self._params['f_att'])
    
    def store_history(self, ind, x, valence, attend):
        self.history['ind'].append(ind)
        self.history['x'].append(x)
        self.history['valence'].append(valence)
        self.history['attend'].append(attend)
    
    def learn(self, ind, x, attend=None):
        attend = self.determine_attend(ind) if attend is None else attend
        if attend:
            valence = self._valence[ind]
            self.mod_attention(ind, valence)
            self.mod_effort(ind, valence)
        self.decay_attention()
        self.store_history(ind, x, valence, attend)

    def subjective_wellbeing(self, i=None, n=1000):
        end = len(self.history['ind']) - 1 if i is None else i
        start = max(0, end - n)
        filt = self.history['attend'][start:end]
        vals = self.history['valence'][start:end][filt]
        ratio = np.sum(vals[vals>0]) / -np.sum(vals[vals,0])
        mean = np.mean(vals) 
        return ratio, mean
    
    def objective_wellbeing(self, i=None, n=1000):
        end = len(self.history['ind']) - 1 if i is None else i
        start = max(0, end - n)
        xs = self.history['x'][start:end]
        return np.mean(xs)
    
    def breed(self, other):
        """ Create a child """
        raise NotImplementedError



class Life_History():
    """ Put someone through a set of Contexts """
    pass


class Population():
    """ Put lots of people through life histories and generations """
    pass