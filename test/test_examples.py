# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 18:20:01 2022

@author: Reuben
"""
import pytest
import numpy as np
from pytest import approx

import pywellbeing as pywb


class Test_AMM_Demo():
    def test_run(self):
         s = pywb.examples.AMM_Demo()
         pop = s.run(pop_size=20, j=50, gen=3, random_seed=0)
         assert pop.history['fitness'][-1][0] == approx(34.21923157299892)
         assert pop.history['subj_wb'][-1][0][3] == approx(0.6150275581721691)