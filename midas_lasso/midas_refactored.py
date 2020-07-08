import math
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from utils import Constuct_Mat_DataFreqLag_WG, simulate_X, weights_midas_beta, store_results
from midas_model import MidasLasso
from variables import *


np.random.seed(42)



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
y_train=Yreg+np.random.normal(mu,sigma,T_train) # In-sample Y
y_test=Yfor+np.random.normal(mu,sigma,T_test) # Out-of-sample Y
Relevant=np.where(np.abs(bt)>1e-8)[0] #Relevant betas

# loop on parameters lambda_par
results = []
for i,lambda_value in enumerate(lambda_par):
    #display
    print('# variables '+str(Nbvar))
    print('model no. ' +str(m))
    print('lambda= ' +str(lambda_par[i]))

    # LASSO technical specifications
    LO={}
    LO['lambda']=lambda_value
    LO['wCV']   =1
    LO['mu']    =.01 # <-- mu for Nesterov regularization
    LO['norm']  =norme[i]
    Spec['n_iter']        =5
    Spec['EPS']           =0
    Spec['SSE']           =1
    Spec['constraint']    =0
    # Initializing parameters
    param_init=np.zeros(Spec['nbvar']*3+1)

    model = MidasLasso(Spec)

    # Case of LASSO-MIDAS model
    if lambda_value!=0:
        xopt = model.fit(X_train,y_train,LO)
    #Classical MIDAS regression model
    else:
        xopt = model.fit(X_train,y_train)

    #Computing fitted values

    R = store_results(xopt,X_train,X_test,y_train,y_test,LO['lambda'],model)

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


















