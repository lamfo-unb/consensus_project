import math
import numpy as np
import statsmodels.api as sm
import scipy.optimize as op
import matplotlib.pyplot as plt


def create_time_dicts(Spc):
    """
    Creates a dict with daily, monthly and quarterly information
    """

    abreviations = ['Kd','Km','Kq']
    daily_range = Spc['daily']
    monthly_range = Spc['monthly']
    quarterly_range = Spc['quarterly']
    all_ranges = np.cumsum([0,daily_range,monthly_range,quarterly_range])

    out_list = []
    for i,abrev in enumerate(abreviations):
        temp_dict = {}
        temp_dict['range'] = range(all_ranges[i],all_ranges[i+1])
        temp_dict['one'] = np.ones(Spc[abrev])
        temp_dict['w'] = np.arange(1,Spc[abrev]+1)/Spc[abrev]
        temp_dict['k'] =  np.arange(1,Spc[abrev]+1)
        temp_dict['kk'] =  Spc[abrev]
        out_list.append(temp_dict)

    return out_list


def weights_midas_beta(th, bt, Spc):
    """
    Constructs covariates matrix as defined by MIDAS weighting scheme

    Inputs
    ------
    th: parameters theta for the weighting kernel
    bt: parameters beta for regression coefficients
    Spc: MIDAS specifications

    Returns
    -------
    W : weights

    """

    dict_list = create_time_dicts(Spc)

    if Spc['TwoParam']:
        th1=th[0:Spc['nbvar']]
        th2=th[Spc['nbvar']:2*Spc['nbvar']]
    else:
        th2=th

    for time_period in dict_list:
        for i in time_period['range']:
            if Spc['TwoParam']:
                if Spc['almon']:
                    W0=np.exp(th1[i]*time_period['k'] + th2[i]*np.square(time_period['k'])) / np.sum(np.exp(th1[i]*time_period['k'] + th2[i]*np.square(time_period['k'])))
                elif Spc['betaFc']:
                    W0=np.exp(th1[i]*time_period['k'] + th2[i]*np.square(time_period['k'])) / np.sum(np.exp(th1[i]*time_period['k'] + th2[i]*np.square(time_period['k'])))
            elif Spc['Averaging']:
                W0=time_period['one']/time_period['kk']
            elif Spc['betaFc']:
                W0=np.power(th2[i]*(1-time_period['w']),(th2[i]-1)) / sum(np.power(th2[i]*(1-time_period['w']),(th2[i]-1)))
            elif Spc['betaFc_special']:
                W0=th2[i]*time_period['w']*np.power((1-time_period['w']),(th2[i]-1))/sum(th2[i]*time_period['w']*np.power((1-time_period['w']),(th2[i]-1)))
            if i==0:
                W = W0*bt[i]
                ww = W0
            else:
                W = np.r_[W,W0*bt[i]]
                ww = np.r_[ww,W0]

    return W.T


def SSE_midas_beta(param, x, y, Spec, LO=False):
    # Sum of Squared Errors
    b0 = None
    phi = None
    th1 = None
    # Sum of Squared Errors
    th2=param[0:Spec['nbvar']]
    bt=param[Spec['nbvar']:2*Spec['nbvar']]
    if Spec['TwoParam']:
        th1=param[0:Spec['nbvar']]
        th2=param[Spec['nbvar']:2*Spec['nbvar']]
        bt=param[2*Spec['nbvar']:3*Spec['nbvar']]
    if Spec['intercept']:
        b0=param[2*Spec['nbvar']+Spec['TwoParam']*Spec['nbvar']]
    if Spec['AR']:
        phi=param[-1]

    WM = weights_midas_beta(np.r_[th1,th2],bt,Spec)

    if b0 is not None and phi is not None:
        W = np.r_[WM,b0,phi]
    elif b0 is not None:
        W = np.r_[WM,b0]
    elif phi is not None:
        W = np.r_[WM,phi]
    else:
        W = WM

    EPS=y-x.dot(W)

    if not LO:
        SSE=np.sum(np.square(EPS))
        return SSE
    elif LO is not None and isinstance(LO,dict):
        LassoPen=np.zeros(Spec['nbvar'])

        mu=LO['mu']
        lambda_par=LO['lambda']

        for i in range(Spec['nbvar']):
            if np.abs(bt[i])<=mu:
                LassoPen[i]=bt[i]/mu
            else:
                LassoPen[i]=np.sign(bt[i])

        #NormNesterov=np.sum(LassoPen*bt)-(mu/2)*np.square(np.linalg.norm(LassoPen,2))

        pp=LO['norm']
        SSE=np.sum(np.square(EPS))+lambda_par*np.power(np.sum(np.power(np.abs(bt),pp)),(1/pp))#*np.linalg.norm(bt,1);
        return SSE
    else:
        raise ValueError('Lambda should be either a dict with parameters or False')