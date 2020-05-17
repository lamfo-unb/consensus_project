# -*- coding: utf-8 -*-

import eikon as ek
import pandas as pd

ek.set_app_key('603a61c8427642b3816e14e3dea83f1ca4dc7f7c')

symbols = pd.read_excel('symbols.xlsx')
var = pd.read_excel('vars.xlsx', sheet_name='COD')

inc = list(set(var['INC'].dropna()))
bal = list(set(var['BAL']))

inc_cod = ['TR.'+variables for variables in inc]
bal_cod = ['TR.'+variables for variables in bal]

#for asset in symbols['RIC']:
df, err = ek.get_data('PETR4.SA', bal_cod)
