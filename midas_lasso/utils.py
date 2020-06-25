import numpy as np
import statsmodels.api as sm


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


def generate_arma_sample(time_length,vector_size,kappa,T_train,T_test):
    Xd = np.zeros([2*(T_train+T_test)*kappa,time_length])
    y = np.ones(T_train+T_test)
    for i in range(time_length):
        modeld = {'Constant': 0, 'AR': np.r_[1, np.random.uniform(-1, 1)],'MA':np.r_[1, 0], 'ARLags': [1], 'Variance': 0.15}
        Xd[:,i] = sm.tsa.arma_generate_sample(ar = modeld['AR'], ma=modeld['MA'], nsample= 2*(T_train+T_test)*kappa, scale = np.sqrt(modeld['Variance']))
        if i==0 :
            x = Constuct_Mat_DataFreqLag_WG(y,Xd[:,i],vector_size,kappa) #<-- Here is the function for constructing the high to low frequency matrix
        else:
            x0 = Constuct_Mat_DataFreqLag_WG(y, Xd[:, i], vector_size,
                                            kappa)  # <-- Here is the function for constructing the high to low frequency matrix
            x = np.c_[x,x0]

    X_train = x[0:T_train,:]
    X_test = x[T_train:T_train+T_test,:]

    return X_train, X_test

def simulate_X(T_train,T_test,simul_dict):

    X_train = []
    X_test = []
    for period in simul_dict:
        temp_X_train, temp_X_test = generate_arma_sample(simul_dict[period]['time_length'],
                                    simul_dict[period]['vector_size'],simul_dict[period]['kappa'],
                                    T_train,T_test)
        X_train.append(temp_X_train)
        X_test.append(temp_X_test)

    X_train = np.column_stack(X_train)
    X_test = np.column_stack(X_test)

    return X_train,X_test