import numpy as np


sigma=np.sqrt(0.15)



#Location parameter
mu=0
#Scale parameter

#Intercpet
b0=.5
phi=[]

#In-sample size and out-of-sample size
T_train=200


T_test=50


#Percentage of relevant variables
Prct_relevant=0.3

#Number of variables in the simulation
Nbvar = 20

#Basic specifications for explanatory variables:
Spec={}

simul_dict = {}
#Daily information
temp_dict = {}
temp_dict['time_length'] = Nbvar//2
temp_dict['vector_size'] = 30
temp_dict['kappa'] = 60

simul_dict['daily'] = temp_dict
Spec['daily']=Nbvar//2
Spec['Kd']=30 # size of the weights vector
#kappad=60

#monthly
temp_dict = {}
temp_dict['time_length'] = Nbvar//2
temp_dict['vector_size'] = 12
temp_dict['kappa'] = 3
simul_dict['monthly'] = temp_dict

Spec['monthly']=Nbvar//2
Spec['Km']=12 # size of the weights vector
#kappam=3

#quarterly (not needed hereafter)

Spec['quarterly']=0
Spec['Kq']=0 #size of the weights vector

#Specifications of the MIDAS model
Spec['Name']='Simulations'
Spec['intercept']=1
Spec['AR']=0 #Here the code doesnt allow the use of an autoregressive term

#Weighting scheme
Spec['TwoParam']=1
Spec['almon']=1
Spec['Averaging']=0
Spec['betaFc']=0
Spec['nbvar']=Spec['daily']+Spec['monthly']


######################### temporary variables ###########################
# This should be automatic in the future

Spec['daily'] = 0 # number of daily variables
Spec['Kd']=21 # number of days in the period
Spec['monthly'] = 40 # number of monthly variables
Spec['Km']= 3 # number of months in the period
Spec['quarterly'] = 65 # number of quarterly variables
Spec['Kq']= 1 # number of quarters in the period
Spec['nbvar'] = Spec['daily']+ Spec['monthly'] + Spec['quarterly']
"""
"""
##################################################################################################
