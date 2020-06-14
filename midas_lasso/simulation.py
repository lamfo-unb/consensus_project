import math
import numpy as np
import statsmodels.api as sm
#Number of variables in the simulation
numbervar = [20,100,200,300]
print(numbervar)

#for j in range(len(numbervar)):
j = 1

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


def weights_midas_beta( th, bt, Spc):
#construction covariates matrix as defined by MIDAS weighting scheme
#th: parameters theta for the weighting kernel
#bt: parameters beta for regression coefficients
#Spc: MIDAS specifications

    l=[]
    for i=1:Spc['daily']:
        l(i).one=ones(1,Spc['Kd'])
        l(i).w=(1:Spc['Kd']/Spc['Kd']
        l(i).k=1:Spc['Kd']
        l(i).kk=Spc['Kd']

    for i=1:Spc['monthly']:
        l(Spc['daily']+i).one=ones(1,Spc['Km']
        l(Spc['daily']+i).w=(1:Spc['Km']/Spc['Km'])
        l(Spc['daily']+i).k=1:Spc['Km']
        l(Spc['daily']+i).kk=Spc['Km']

    for i=1:Spc['quarterly']:
        l(Spc['daily']+Spc['monthly']+i).one=ones(1,Spc['Kq'])
        l(Spc['daily']+Spc['monthly']+i).w=(1:Spc['Kq)/Spc['Kq']
        l(Spc['daily']+Spc['monthly']+i).k=1:Spc['Kq']
        l(Spc['daily']+Spc['monthly']+i).kk=Spc['Kq']


    if Spc['TwoParam']:
        th1=th(1:Spc['nbvar']);
        th2=th(Spc['nbvar']+1:2*Spc['nbvar']);
    else:
        th2=th;

    WW=[]; ww=[];
    for i=1:length(th2):
        W=zeros(1,Spc['sK'](i));
        if Spc['TwoParam']:
            if Spc['almon']:
                W=exp(th1(i).*l(i).k + th2(i).*(l(i).k.^2)) / sum(exp(th1(i).*l(i).k + th2(i).*(l(i).k.^2)));
            elif Spc['betaFc']:
                W=exp(th1(i).*l(i).k + th2(i).*(l(i).k.^2)) / sum(exp(th1(i).*l(i).k + th2(i).*(l(i).k.^2)));
        elif Spc['Averaging']
            W=l(i).one./l(i).kk;
        elif Spc['betaFc']
            W=(th2(i)*(1-l(i).w).^(th2(i)-1)) / sum(th2(i)*(1-l(i).w).^(th2(i)-1));
        elif Spc['betaFc_special']
            W=th2(i)*l(i).w.*(1-l(i).w).^(th2(i)-1)/sum(th2(i)*l(i).w.*(1-l(i).w).^(th2(i)-1));

        WW=[WW (W.*bt(i))];
        ww=[ww W];

    WM=WW';

return([WM, ww])


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#Constructing X matrices in the MIDAS form (each row corresponds to a time t
# indices ; and columns are the high-frequency data includes in [t-K;t])

XX_Reg=[]
XX_For=[]
Spec['sK']=[]
Spec['Type']=[]

#daily matrix
Xd = np.zeros([2*(T+Toos)*kappad,Spec['daily']])
for i in range(Spec['daily']):
    modeld = {'Constant': 0, 'AR': np.r_[1, np.random.uniform(-1, 1)],'MA':np.r_[1, 0], 'ARLags': [1], 'Variance': 0.15}
    Xd[:,i] = sm.tsa.arma_generate_sample(ar = modeld['AR'], ma=modeld['MA'], nsample= 2*(T+Toos)*kappad, scale = math.sqrt(modeld['Variance']))
    x = Constuct_Mat_DataFreqLag_WG(np.r_[y,yoos],Xd[:,i],Spec['Kd'],kappad) #<-- Here is the function for constructing the high to low frequency matrix
    Spec['sK'].append(Spec['Kd'])
    Spec['Type'].append('D')
    XX_Reg.append(x[0:T,:]) # <-- Then the high-frequency matrices are splitted to go either In-Sample matrix (XX.Reg) or Out-of Sample matrix (XX.For)
    XX_For.append(x[T:T+Toos,:])


#monthly matrix
Xm = np.zeros([2*(T+Toos)*kappam,Spec['monthly']])
for i in range(Spec['monthly']):
    modelm = {'Constant': 0, 'AR': np.r_[0.2, np.random.uniform(-1, 1)],'MA':np.r_[1, 0], 'ARLags': [1], 'Variance': 0.15}
    Xm[:,i] = sm.tsa.arma_generate_sample(ar = modelm['AR'], ma=modelm['MA'], nsample=2*(T+Toos)*kappam, scale = math.sqrt(modeld['Variance']))
    x = Constuct_Mat_DataFreqLag_WG(np.r_[y,yoos],Xm[:,i],Spec['Km'],kappam) #<-- Here is the function for constructing the high to low frequency matrix
    Spec['sK'].append(Spec['Km'])
    Spec['Type'].append('M')
    XX_Reg.append(x[0:T,:]) # <-- Then the high-frequency matrices are splitted to go either In-Sample matrix (XX.Reg) or Out-of Sample matrix (XX.For)
    XX_For.append(x[T:T+Toos,:])


#Construction Y
theta1=0.1*np.ones(Spec['nbvar'])
theta2=-0.05*np.ones(Spec['nbvar'])
#Intercpet
b0=.5
phi=[]
#Betas
bt= np.random.binomial(1, Prct_relevant, Spec['nbvar'])*np.random.normal(0,1,Spec['nbvar'])
[WM,weiii]=weights_midas_beta(np.r_[theta1,theta2],bt, Spec) # <-- This function computes the MIDAS exp Almon Weights w.r.t. parameters theta

W=[WM,b0,phi]


print(bt)

# ar =  np.r_[0, 0.5]# add zero-lag and negate
# ma = np.r_[1, 0] # add zero-lag
# y = sm.tsa.arma_generate_sample(ar, ma, 250)
# print(y)




# for i=1:Spec.daily
#     Xd(:,i)=simulate(modeld,2*(T+Toos)*kappad);
#     x = Constuct_Mat_DataFreqLag_WG([y;yoos],Xd(:,i),Spec.Kd,kappad); % <-- Here is the function for constructing the high to low frequency matrix
#     Spec.sK=[Spec.sK Spec.Kd];    Spec.Type=[Spec.Type 'D'];
#     XX.Reg=[XX.Reg x(1:T,:)]; % <-- Then the high-frequency matrices are splitted to go either In-Sample matrix (XX.Reg) or Out-of Sample matrix (XX.For)
#     XX.For=[XX.For x(T+1:T+Toos,:)];
# end


#y = sm.tsa.arma_generate_sample(ar, ma, 250)
