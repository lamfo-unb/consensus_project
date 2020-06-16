import math
import numpy as np
import statsmodels.api as sm
import scipy.optimize as op

#Number of variables in the simulation
numbervar = [20,100,200,300]

#for j in range(len(numbervar)):
j = 0

#Location parameter
mu=0
#Scale parameter
sigma=math.sqrt(0.15)

#In-sample size and out-of-sample size
T=200
y=np.ones(T)

Toos=50
yoos=np.ones(Toos)

#Percentage of relevant variables
Prct_relevant=0.3
Nbvar=numbervar[j]

#Basic specifications for explanatory variables:
Spec={}
#Daily information
Spec['dseries']=[]
Spec['daily']=Nbvar//2
Spec['Kd']=30 # size of the weights vector
kappad=60

#monthly
Spec['mseries']=[]
Spec['monthly']=Nbvar//2
Spec['Km']=12 # size of the weights vector
kappam=3

#quarterly (not needed hereafter)
Spec['qseries']=[]
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

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def Constuct_Mat_DataFreqLag_WG(g,d,K,m) :
    #param: g: vector of low frequency data (Y)
    #param: d: vector of high frequency data (X)
    #(d must ends at exactly the same data than the low frequency time series
    #e.g. if g is quarterly and ends at 2012q2, and d is monthly sampled, d must ends on June 2012.)
    #param K: number of high frequency data to involve in the MIDAS weighting scheme (K in MIDAS equations)
    #param m: difference in frequency between g and d (e.g. if g is quarterly and d is monthly, m=3)
    #return: D is matrix with the same size than g and K columns.
    lo = len(g)
    i = len(d)
    D = np.zeros([lo,K])
    for row in range(lo):
        for col in range(K):
            D[lo-row-1,col]=d[i-col-1]
        i=i-m
    return D


def weights_midas_beta(th, bt, Spc):
#construction covariates matrix as defined by MIDAS weighting scheme
#th: parameters theta for the weighting kernel
#bt: parameters beta for regression coefficients
#Spc: MIDAS specifications
    l=[]
    for i in range(Spc['daily']):
        dict_daily = {}
        dict_daily['one'] = np.ones(Spc['Kd'])
        dict_daily['w'] = np.arange(1,Spc['Kd']+1)/Spc['Kd']
        dict_daily['k'] =  np.arange(1,Spc['Kd']+1)
        dict_daily['kk'] =  Spc['Kd']
        l.append(dict_daily)

    for i in range(Spc['monthly']):
        dict_monthly = {}
        dict_monthly['one'] = np.ones(Spc['Km'])
        dict_monthly['w'] = np.arange(1,Spc['Km']+1)/Spc['Km']
        dict_monthly['k'] =  np.arange(1,Spc['Km']+1)
        dict_monthly['kk'] =  Spc['Km']
        l.append(dict_monthly)


    for i in range(Spc['quarterly']):
        dict_quarterly = {}
        dict_quarterly['one'] = np.ones(Spc['Kq'])
        dict_quarterly['w'] = np.arange(1, Spc['Kq'] + 1) / Spc['Kq']
        dict_quarterly['k'] = np.arange(1, Spc['Kq'] + 1)
        dict_quarterly['kk'] = Spc['Kq']
        l.append(dict_quarterly)

    if Spc['TwoParam']:
        th1=th[0:Spc['nbvar']]
        th2=th[Spc['nbvar']:2*Spc['nbvar']]
    else:
        th2=th

    for i in range(len(th2)):
        if Spc['TwoParam']:
            if Spc['almon']:
                W0=np.exp(th1[i]*l[i]['k'] + th2[i]*np.square(l[i]['k'])) / np.sum(np.exp(th1[i]*l[i]['k'] + th2[i]*np.square(l[i]['k'])))
            elif Spc['betaFc']:
                W0=np.exp(th1[i]*l[i]['k'] + th2[i]*np.square(l[i]['k'])) / np.sum(np.exp(th1[i]*l[i]['k'] + th2[i]*np.square(l[i]['k'])));
        elif Spc['Averaging']:
            W0=l[i]['one']/l[i]['kk']
        elif Spc['betaFc']:
            W0=np.power(th2[i]*(1-l[i]['w']),(th2[i]-1)) / sum(np.power(th2[i]*(1-l[i]['w']),(th2[i]-1)))
        elif Spc['betaFc_special']:
            W0=th2[i]*l[i]['w']*np.power((1-l[i]['w']),(th2[i]-1))/sum(th2[i]*l[i]['w']*np.power((1-l[i]['w']),(th2[i]-1)))
        if i==0:
            W = W0*bt[i]
            ww = W0
        else:
            W = np.r_[W,W0*bt[i]]
            ww = np.r_[ww,W0]

    return W.T, ww

def SSE_midas_beta_lasso(param, x, y, Spec, LO):
    b0 = None
    phi = None
    th1 = None
    # Sum of Squared Errors
    th2=param[0:Spec['nbvar']]
    bt=param[Spec['nbvar']:2*Spec.nbvar]
    if Spec['TwoParam']:
        th1=param[0:Spec.nbvar]
        th2=param[Spec.nbvar:2*Spec.nbvar]
        bt=param[2*Spec.nbvar:3*Spec.nbvar]
    if Spec['intercept']:
        b0=param(2*Spec.nbvar+Spec.TwoParam*Spec.nbvar+1)
    if Spec['AR']:
        phi=param[-1]

    WM, _ = weights_midas_beta(np.r_[th1,th2],bt,Spec)

    W = np.r_[WM,b0,phi]

    EPS=y-x.dot(W) # epsilon without LASSO penalty

    # LASSO
    LassoPen=np.zeros(Spec['nbvar'])

    mu=LO['mu']
    lambda_par=LO['lambda']

    for i in Spec['nbvar']:
        if np.abs(bt[i])<=mu:
            LassoPen[i]=bt[i]/mu
        else:
            LassoPen[i]=np.sign(bt[i])

    #NormNesterov=np.sum(LassoPen*bt)-(mu/2)*np.square(np.linalg.norm(LassoPen,2))

    pp=LO['norm']
    SSE=np.sum(np.square(EPS))+lambda_par*np.power(np.sum(np.power(np.abs(bt),pp)),(1/pp))#*np.linalg.norm(bt,1);
    return SSE

def SSE_midas_beta(param, x, y, Spec):
    # Sum of Squared Errors
    b0 = None
    phi = None
    th1 = None
    # Sum of Squared Errors
    th2=param[0:Spec['nbvar']]
    bt=param[Spec['nbvar']:2*Spec.nbvar]
    if Spec['TwoParam']:
        th1=param[0:Spec.nbvar]
        th2=param[Spec.nbvar:2*Spec.nbvar]
        bt=param[2*Spec.nbvar:3*Spec.nbvar]
    if Spec['intercept']:
        b0=param(2*Spec.nbvar+Spec.TwoParam*Spec.nbvar+1)
    if Spec['AR']:
        phi=param[-1]

    WM, _ = weights_midas_beta(np.r_[th1,th2],bt,Spec)

    W = np.r_[WM,b0,phi]

    EPS=y-x.dot(W)

    SSE=np.sum(np.square(EPS))
    return SSE
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#Constructing X matrices in the MIDAS form (each row corresponds to a time t
# indices ; and columns are the high-frequency data includes in [t-K;t])

# XX_Reg=[]
# XX_For=[]
Spec['sK']=[]
Spec['Type']=[]

#daily matrix
Xd = np.zeros([2*(T+Toos)*kappad,Spec['daily']])
for i in range(Spec['daily']):
    modeld = {'Constant': 0, 'AR': np.r_[1, np.random.uniform(-1, 1)],'MA':np.r_[1, 0], 'ARLags': [1], 'Variance': 0.15}
    Xd[:,i] = sm.tsa.arma_generate_sample(ar = modeld['AR'], ma=modeld['MA'], nsample= 2*(T+Toos)*kappad, scale = math.sqrt(modeld['Variance']))
    Spec['sK'].append(Spec['Kd'])
    Spec['Type'].append('D')
    if i==0 :
        x = Constuct_Mat_DataFreqLag_WG(np.r_[y,yoos],Xd[:,i],Spec['Kd'],kappad) #<-- Here is the function for constructing the high to low frequency matrix
    else:
        x0 = Constuct_Mat_DataFreqLag_WG(np.r_[y, yoos], Xd[:, i], Spec['Kd'],
                                        kappad)  # <-- Here is the function for constructing the high to low frequency matrix
        x = np.c_[x,x0]

XX_Reg = x[0:T,:]
XX_For = x[T:T+Toos,:]


#monthly matrix
Xm = np.zeros([2*(T+Toos)*kappam,Spec['monthly']])
for i in range(Spec['monthly']):
    modelm = {'Constant': 0, 'AR': np.r_[1, np.random.uniform(-1, 1)],'MA':np.r_[1, 0], 'ARLags': [1], 'Variance': 0.15}
    Xm[:,i] = sm.tsa.arma_generate_sample(ar = modelm['AR'], ma=modelm['MA'], nsample=2*(T+Toos)*kappam, scale = math.sqrt(modeld['Variance']))
    Spec['sK'].append(Spec['Km'])
    Spec['Type'].append('M')
    if i == 0:
        x = Constuct_Mat_DataFreqLag_WG(np.r_[y, yoos], Xm[:, i], Spec['Km'],
                                        kappam)
    else:
        x0 = Constuct_Mat_DataFreqLag_WG(np.r_[y, yoos], Xm[:, i], Spec['Km'],
                                        kappam)
        x = np.c_[x, x0]
XX_Reg = np.c_[XX_Reg,x[0:T,:]]
XX_For = np.c_[XX_For,x[T:T+Toos,:]]


#Construction Y
theta1=0.1*np.ones(Spec['nbvar'])
theta2=-0.05*np.ones(Spec['nbvar'])

#Intercpet
b0=.5
phi=[]

#Betas
bt= np.random.binomial(1, Prct_relevant, Spec['nbvar'])*np.random.normal(0,1,Spec['nbvar'])
WM,weiii = weights_midas_beta(np.r_[theta1,theta2],bt, Spec)

##Add beta0 and phi
W = np.r_[WM,b0,phi]

#Adding a column of one in the covariates matrix
if b0 is not None:
    XX_Reg = np.c_[XX_Reg, np.ones(T)]
    XX_For = np.c_[XX_For,np.ones(Toos)]

# #Computing Y
Yreg=XX_Reg.dot(W)
Yfor=XX_For.dot(W)

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
Ysim=Yreg+np.random.normal(mu,sigma,T) # In-sample Y
Yfsim=Yfor+np.random.normal(mu,sigma,Toos) # Out-of-sample Y
Relevant=np.where(np.abs(bt)>1e-8)[0] #Relevant betas

# loop on parameters lambda_par
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

    # Optimazation option
    #     Spec.options=optimset('GradObj','off','Display','iter','LargeScale','off', ...
    #                         'MaxFunEvals',100000,'MaxIter', 100000,'TolFun',1e-6, 'TolX',1e-6);
    # Case of LASSO-MIDAS model
    if i>0:
        # fun_est_midas=@(param)SSE_midas_beta_lasso(param, XX.Reg, R.Ysim, Spec, LO);
        xopt = op.fmin(SSE_midas_beta_lasso, param_init, xtol=1e-8, args=(XX_Reg, R['Ysim'], Spec, LO,))
        # [prm, Fval, EF]=fminunc(fun_est_midas,param_init,Spec.options);

    #Classical MIDAS regression model
    else:
        # fun_est_midas=@(param)SSE_midas_beta(param, XX.Reg, R.Ysim, Spec);
        xopt = op.fmin(SSE_midas_beta, param_init, xtol=1e-8, args=(XX_Reg, R['Ysim'], Spec,))
        # [prm, Fval, EF]=fminunc(fun_est_midas,param_init,Spec.options);
