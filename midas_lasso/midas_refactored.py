import math
import numpy as np
import statsmodels.api as sm
import scipy.optimize as op
import matplotlib.pyplot as plt
from utils import Constuct_Mat_DataFreqLag_WG, simulate_X
from midas_model import weights_midas_beta,SSE_midas_beta



np.random.seed(42)



#Location parameter
mu=0
#Scale parameter
sigma=math.sqrt(0.15)

#In-sample size and out-of-sample size
T_train=200
y=np.ones(T_train)

T_test=50
yoos=np.ones(T_test)

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
kappad=60

#monthly
temp_dict = {}
temp_dict['time_length'] = Nbvar//2
temp_dict['vector_size'] = 12
temp_dict['kappa'] = 3
simul_dict['monthly'] = temp_dict

Spec['monthly']=Nbvar//2
Spec['Km']=12 # size of the weights vector
kappam=3

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

#Relevant features
Nbrelevant=math.floor(Prct_relevant*Spec['nbvar'])
#Non relevant features
Nbirrelevant=Spec['nbvar']-Nbrelevant


#Constructing X matrices in the MIDAS form (each row corresponds to a time t
# indices ; and columns are the high-frequency data includes in [t-K;t])

X_train, X_test = simulate_X(T_train,T_test,simul_dict)
#Construction Y
theta1=0.1*np.ones(Spec['nbvar'])
theta2=-0.05*np.ones(Spec['nbvar'])

#Intercpet
b0=.5
phi=[]

#Betas
bt= np.random.binomial(1, Prct_relevant, Spec['nbvar'])*np.random.normal(0,1,Spec['nbvar'])
WM = weights_midas_beta(np.r_[theta1,theta2],bt, Spec)

##Add beta0 and phi
W = np.r_[WM,b0,phi]

#Adding a column of one in the covariates matrix
if b0 is not None:
    X_train = np.c_[X_train, np.ones(T_train)]
    X_test = np.c_[X_test,np.ones(T_test)]

# #Computing Y
Yreg=X_train.dot(W)
Yfor=X_test.dot(W)

#### LASSO Specifications
# Parameter (lambda)
lambda_par=[0,1] # <-- Here is a range for a loop
#Norm (e.g. if norme=1 it's the LASSO, norme=2 it's the ridge,... )
norme=np.r_[0,np.ones(len(lambda_par)-1)] # <-- same size than the lambda_par vector

# Number of models to simulate
Mdl=100
M={}
m=1

# DGP from gaussian noise
Ysim=Yreg+np.random.normal(mu,sigma,T_train) # In-sample Y
Yfsim=Yfor+np.random.normal(mu,sigma,T_test) # Out-of-sample Y
Relevant=np.where(np.abs(bt)>1e-8)[0] #Relevant betas

# loop on parameters lambda_par
results = []
for i in range(len(lambda_par)):
    #display
    print('# variables '+str(Nbvar))
    print('model no. ' +str(m))
    print('lambda= ' +str(lambda_par[i]))

    R={}
    R['Ysim'] = Ysim
    R['Yfsim'] = Yfsim

    # LASSO technical specifications
    LO={}
    LO['lambda']=lambda_par[i]
    LO['wCV']   =1
    LO['mu']    =.01 # <-- mu for Nesterov regularization
    LO['norm']  =norme[i]
    R['lambda']=LO['lambda']
    Spec['n_iter']        =5
    Spec['EPS']           =0
    Spec['SSE']           =1
    Spec['constraint']    =0
    # Initializing parameters
    param_init=np.zeros(Spec['nbvar']*3+1)

    # Case of LASSO-MIDAS model
    if i>0:
        xopt =  op.fmin(SSE_midas_beta, param_init,args=(X_train, R['Ysim'], Spec, LO,), xtol=1e-4, ftol=1e-4, maxiter=100000, maxfun=100000)

    #Classical MIDAS regression model
    else:
        xopt = op.fmin(SSE_midas_beta, param_init,args=(X_train, R['Ysim'], Spec,), xtol=1e-4, ftol=1e-4, maxiter=100000, maxfun=100000)

    #Saving parameters
    R['th_simul']=xopt[0:2*Spec['nbvar']]
    R['bt_simul']=xopt[2*Spec['nbvar']:(len(xopt)-1)]
    R['RelevantVar_simul']=0
    R['IrrelevantVar_simul']=0

    #Number of non-zeros beta coefficients (Sparsity)
    for ll in range(len(R['bt_simul'])):
        if -sigma/math.sqrt(len(R['bt_simul']))<R['bt_simul'][ll] and R['bt_simul'][ll]<sigma/math.sqrt(len(R['bt_simul'])):
            R['IrrelevantVar_simul']=R['IrrelevantVar_simul']+1
        else:
            R['RelevantVar_simul']=R['RelevantVar_simul']+1

    #Computing fitted values

    WM = weights_midas_beta(R['th_simul'],R['bt_simul'],Spec)
    W=np.r_[WM,xopt[-1]]
    R['xopt']=xopt

    #In-sample results
    R['Yhat']=X_train.dot(W)
    R['resid']=R['Ysim']-R['Yhat']
    R['MSE']=np.sum(np.square(R['resid']))
    R['RMSE']=np.sqrt(R['MSE'])

    #Out-of-sample results
    R['Yfhat']=X_test.dot(W)
    R['fresid']=R['Yfsim']-R['Yfhat']
    R['MSFE']=np.sum(np.square(R['fresid']))
    R['RMSFE']=np.sqrt(R['MSFE'])
    results.append(R)


#Plot the difference
R0 = results[0]
range = np.arange(1,len(R0['bt_simul'])+1)
print(R0['th_simul'])
print(R0['bt_simul'])
print(len(bt))
print(len(R0['bt_simul']))

fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
ax.scatter(range, bt, color='r')
ax.scatter(range, R0['bt_simul'], color='b')
ax.set_xlabel('Range')
ax.set_ylabel('Betas')
ax.set_title('MIDAS')
plt.show()


















