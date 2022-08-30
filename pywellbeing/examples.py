# -*- coding: utf-8 -*-
"""
Created on Sun May  8 19:58:32 2022

@author: Reuben
"""

import matplotlib.pyplot as plt
from pathlib import Path

from . import amm


class AMM_Demo():
    """ Standard demonstration of the Adaptive Motivation Model of wellbeing 
    
    """
    
    def run(self, pop_size=300, random_seed=0, j=200,
            gen=80, p_survive=0.6):
        amm.random.set_random_seed(random_seed)
        context = amm.Context()
        context.setup()
        lh = amm.Life_History()
        lh.add_context(context, j)
        population = amm.Population()
        population.set_population(pop_size,
                                  n_history=10)
        population.set_life_history(lh)
        population.run(gen=gen, p_survive=p_survive)
        return population
    
    
    def hedonic_adaptation_large_change(self, pop, random_seed=0,
                           j=199, k=200, l=0, loc=-0.5, do_learn=False):
        amm.random.set_random_seed(random_seed)
        normal = amm.Context()
        normal.setup()
        novel = amm.Context(loc=loc)
        novel.setup()
        lh = amm.Life_History()
        lh.add_context(normal, j)
        lh.add_context(novel, k)
        if l > 0:
            lh.add_context(normal, l)
        population = amm.Population()
        population.set_pop(pop)
        population.set_life_history(lh)
        [p.set_do_learn(do_learn) for p in population.pop]
        population.run_generation()
        return population
        
    def hedonic_adaptation_down(self, pop, random_seed=0,
                           j=199, k=40, l=60, f=20.0, ind=20, do_learn=False):
        amm.random.set_random_seed(random_seed)
        normal = amm.Context()
        normal.setup()
        novel = amm.Context()
        novel.setup()
        novel.ps[ind] *= f
        lh = amm.Life_History()
        lh.add_context(normal, j)
        lh.add_context(novel, k)
        if l > 0:
            lh.add_context(normal, l)
        population = amm.Population()
        population.set_pop(pop, reset=False)
        population.set_life_history(lh)
        [p.set_do_learn(do_learn) for p in population.pop]
        population.run_generation()
        return population
    
    def hedonic_adaptation_up(self, pop, **kwargs):
        return self.hedonic_adaptation_down(pop, ind=60, **kwargs)
    
    def run_all(self, folder=None, pop=None, fmt='svg'):
        pop = self.run() if pop is None else pop
        # pop.plot_all(folder=folder, i=0, fmt=fmt)
        pop.plot_all(folder=folder, i=-1, fmt=fmt)
        pop_ha_down = self.hedonic_adaptation_down(pop)
        pop_ha_up = self.hedonic_adaptation_up(pop)
        self.plot_adaptation(pop_ha_down, pop_ha_up, y=8, folder=folder,
                             fmt=fmt)
        pop.lh.contexts[0]['context'].plot()
            plt.savefig(Path(folder) / ('initial_cue_dist.' + fmt), dpi=300,
                        format=fmt)
        return pop
        
    def plot_adaptation(self, pop_ha_down, pop_ha_up, y, folder=None, fmt='svg'):
        fig = pop_ha_down.pop[y].plot_wb_history(xlim=(195, 270),
                                         label='Avoidance situation',
                                         linestyle='--')
        pop_ha_up.pop[y].plot_wb_history(xlim=(195, 270), fignum=fig.number,
                                         label='Approach situation')
        plt.figure(fig.number)
        plt.axhline(pop_ha_up.pop[y].subj_wb_history()[1][-1], linestyle=':',
                    label='Baseline')
        plt.legend()
        if folder is not None:
            plt.savefig(Path(folder) / ('hedonic_adaptation.' + fmt), dpi=300,
                        format=fmt)


def run_amm(folder=None, fmt='svg'):
    s = AMM_Demo()
    return s.run_all(folder=folder, fmt=fmt)
    

