import math
import numpy as np
import statsmodels.api as sm
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

#
#
#     WW=[]; ww=[];
    for i in range(len(th2)):
        W=np.zeros(Spc['sK'][i])
        if Spc['TwoParam']:
            if Spc['almon']:
                W=np.exp(th1[i]*l[i]['k'] + th2[i]*np.square(l[i]['k'])) / np.sum(np.exp(th1[i]*l[i]['k'] + th2[i]*np.square(l[i]['k'])))
                print(W)

        #     elseif Spc.betaFc
        #         W=exp(th1(i).*l(i).k + th2(i).*(l(i).k.^2)) / sum(exp(th1(i).*l(i).k + th2(i).*(l(i).k.^2)));
        #     end
        # elseif Spc.Averaging
        #     W=l(i).one./l(i).kk;
        # elseif Spc.betaFc
        #     W=(th2(i)*(1-l(i).w).^(th2(i)-1)) / sum(th2(i)*(1-l(i).w).^(th2(i)-1));
        # elseif Spc.betaFc_special
        #     W=th2(i)*l(i).w.*(1-l(i).w).^(th2(i)-1)/sum(th2(i)*l(i).w.*(1-l(i).w).^(th2(i)-1));
        # end
#
#         WW=[WW (W.*bt(i))];
#
#         ww=[ww W];
#
#     end
#
#     WM=WW';
#
# end
    return l

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

test = weights_midas_beta(np.r_[theta1,theta2],bt, Spec)

# WM,weiii = weights_midas_beta(np.r_[theta1,theta2],bt, Spec)
# W = WM.copy()
# W.append(b0)
# W.extend(phi)
#
#
# #Adding a column of one in the covariates matrix
# XX_Reg = np.c_[XX_Reg, np.ones(T)]
# XX_For = np.c_[XX_For,np.ones(Toos)]

# #Computing Y
# Yreg=XX_Reg*W
# Yfor=XX_For*W
 ##X_Reg 200, 2100
 ##X_For = 50, 2100