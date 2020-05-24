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


for asset in symbols['RIC'][240:]:
    print(f'Getting data : {asset}')
    df_bal, err = ek.get_data(asset, bal_cod,parameters = {'Period':'FQ0','Frq':'FQ','SDate':'2020-05-15','EDate':'2000-01-01'})
    df_inc, err = ek.get_data(asset, inc_cod,parameters = {'Period':'FQ0','Frq':'FQ','SDate':'2020-05-15','EDate':'2000-01-01'})
    df_merge = pd.merge(df_bal,df_inc,left_on='Balance Sheet Period End Date',right_on='Income Statement Period End Date')
    df_merge.to_csv(f'../data/{asset}_merge.csv')
