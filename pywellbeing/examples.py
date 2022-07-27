# -*- coding: utf-8 -*-
"""
Created on Sun May  8 19:58:32 2022

@author: Reuben
"""

from . import pywb

class Simplest():
    def run(self, pop_size=200, n=40, x_max=3, random_seed=0, j=200,
            gen=40, p_survive=0.6):
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
    
    