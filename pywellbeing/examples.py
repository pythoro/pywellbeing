# -*- coding: utf-8 -*-
"""
Created on Sun May  8 19:58:32 2022

@author: Reuben
"""

import matplotlib.pyplot as plt

from . import pywb


class Simplest():
    def run(self, pop_size=300, random_seed=0, j=200,
            gen=80, p_survive=0.6):
        pywb.random.set_random_seed(random_seed)
        context = pywb.Context()
        context.setup()
        lh = pywb.Life_History()
        lh.add_context(context, j)
        population = pywb.Population()
        population.set_population(pop_size,
                                  n_history=10)
        population.set_life_history(lh)
        population.run(gen=gen, p_survive=p_survive)
        return population
    
    
    def hedonic_adaptation(self, pop, random_seed=0,
                           j=199, k=200, l=0, loc=-0.5, do_learn=False):
        pywb.random.set_random_seed(random_seed)
        normal = pywb.Context()
        normal.setup()
        novel = pywb.Context(loc=loc)
        novel.setup()
        lh = pywb.Life_History()
        lh.add_context(normal, j)
        lh.add_context(novel, k)
        if l > 0:
            lh.add_context(normal, l)
        population = pywb.Population()
        population.set_pop(pop)
        population.set_life_history(lh)
        [p.set_do_learn(do_learn) for p in population.pop]
        population.run_generation()
        return population
        
    def hedonic_adaptation_2(self, pop, random_seed=0,
                           j=199, k=200, l=0, f=20.0, ind=20, do_learn=False):
        pywb.random.set_random_seed(random_seed)
        normal = pywb.Context()
        normal.setup()
        novel = pywb.Context()
        novel.setup()
        novel.ps[ind] *= f
        lh = pywb.Life_History()
        lh.add_context(normal, j)
        lh.add_context(novel, k)
        if l > 0:
            lh.add_context(normal, l)
        population = pywb.Population()
        population.set_pop(pop, reset=False)
        population.set_life_history(lh)
        [p.set_do_learn(do_learn) for p in population.pop]
        population.run_generation()
        return population
    
    def hedonic_adaptation_3(self, pop, **kwargs):
        return self.hedonic_adaptation_2(pop, ind=60, **kwargs)
    
    

