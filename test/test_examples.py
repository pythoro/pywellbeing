# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 18:20:01 2022

@author: Reuben
"""
import pytest
import numpy as np
from pytest import approx

import pywellbeing as pywb


class Test_Simplest():
    def test_run(self):
         s = pywb.examples.Simplest()
         pop = s.run(pop_size=5, j=50, gen=3, random_seed=0)
         assert pop.history['obj_wb'][-1][0] == approx(4.010854417782909)
         assert pop.history['subj_wb'][-1][0][3] == approx(0.3745950860447285)