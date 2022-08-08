# -*- coding: utf-8 -*-
"""
Created on Sun May  8 19:58:32 2022

@author: Reuben
"""

from . import pywb

class Simplest():
    def run(self, pop_size=200, n=80, x_max=3, random_seed=0, j=200,
            gen=80, p_survive=0.6):
        context = pywb.Context(n=n, x_max=x_max)
        context.setup(random_seed + 1000)
        lh = pywb.Life_History()
        lh.add_context(context, j)
        population = pywb.Population()
        population.set_population(pop_size, random_seed=random_seed,
                                  n=n, x_max=x_max)
        population.set_life_history(lh)
        population.run(gen=gen, p_survive=p_survive)
        return population
    
    
    def hedonic_adaptation(self, person, n=80, x_max=3, random_seed=0,
                           j=1000, k=1000, loc=0.5):
        normal = pywb.Context(n=n, x_max=x_max)
        normal.setup(random_seed + 1000)
        novel = pywb.Context(n=n, x_max=x_max, loc=loc)
        novel.setup(random_seed + 1000)
        lh = pywb.Life_History()
        person.reset()
        lh.set_person(person)
        lh.add_context(normal, j)
        lh.add_context(novel, k)
        lh.run()
        
    def hedonic_adaptation_2(self, person, n=80, x_max=3, random_seed=0,
                           j=999, k=1000, f=50.0, ind=20):
        normal = pywb.Context(n=n, x_max=x_max)
        normal.setup(random_seed + 1000)
        novel = pywb.Context(n=n, x_max=x_max)
        novel.setup(random_seed + 1000)
        novel.ps[ind] *= f
        lh = pywb.Life_History()
        person.reset()
        lh.set_person(person)
        lh.add_context(normal, j)
        lh.add_context(novel, k)
        lh.run()
        
        