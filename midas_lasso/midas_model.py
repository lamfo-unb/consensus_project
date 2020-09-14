import math
import numpy as np
from utils import weights_midas_beta
from typing import Dict, List
import scipy.optimize as op


class MidasLasso():

    def __init__(self,settings:Dict[int,bool]):
        self.settings = settings
        self.param_init=np.zeros(settings['nbvar']*3+1)
        self.W = None
        

    def _apply_settings(
        self,param:np.array
        ) -> np.array:
        """ 
        Create the MIDAS model's weights by
        applying the defined settings

        Inputs
        ------
        param : array that containts beta, phi and theta values

        Returns
        ------
        W : mean error
        """

        b0 = None
        phi = None
        th1 = None

        th2=param[0:self.settings['nbvar']]
        bt=param[self.settings['nbvar']:2*self.settings['nbvar']]
        if self.settings['TwoParam']:
            th1=param[0:self.settings['nbvar']]
            th2=param[self.settings['nbvar']:2*self.settings['nbvar']]
            bt=param[2*self.settings['nbvar']:3*self.settings['nbvar']]
        if self.settings['intercept']:
            b0=param[2*self.settings['nbvar']+self.settings['TwoParam']*self.settings['nbvar']]
        if self.settings['AR']:
            phi=param[-1]

        WM = weights_midas_beta(np.r_[th1,th2],bt,self.settings)

        if b0 is not None and phi is not None:
            return np.r_[WM,b0,phi],bt
        elif b0 is not None:
            return np.r_[WM,b0],bt
        elif phi is not None:
            return np.r_[WM,phi],bt
        else:
            return WM,bt



    def SSE_midas_beta(
        self,param:np.array, x:np.array,
        y:np.array, L0=False
        ) -> float:

        """
        Calculate the mean square error of the MIDAS model

        Inputs
        ------
        param : array that containts beta, phi and theta values
        x : input shape (number of examples x number of features)
        y : target shape (number of examples)
        L0 : lambda parameters, either False (no lasso) or a dict with
        mu and lambda values

        Returns
        ------
        SSE : mean error

        """

        W,bt = self._apply_settings(param)

        EPS=y-x.dot(W)

        if not L0:
            SSE=np.sum(np.square(EPS))
            return SSE
        elif L0 is not None and isinstance(L0,dict):
            LassoPen=np.zeros(self.settings['nbvar'])

            mu=L0['mu']
            lambda_par=L0['lambda']

            for i in range(self.settings['nbvar']):
                if np.abs(bt[i])<=mu:
                    LassoPen[i]=bt[i]/mu
                else:
                    LassoPen[i]=np.sign(bt[i])

            #NormNesterov=np.sum(LassoPen*bt)-(mu/2)*np.square(np.linalg.norm(LassoPen,2))

            pp=L0['norm']
            SSE=np.sum(np.square(EPS))+lambda_par*np.power(np.sum(np.power(np.abs(bt),pp)),(1/pp))#*np.linalg.norm(bt,1);
            return SSE
        else:
            raise ValueError('Lambda should be either a dict with parameters or False')



    def fit(
        self,X_train:np.array,
        y_train:np.array,L0=False
        ) -> np.array:
        """
        Learn the weights for the MIDAS model

        Inputs
        ------
        X_train : input shape (number of examples x number of features)
        y_train : target shape (number of examples)
        L0 : lambda parameters, either False (no lasso) or a dict with
        mu and lambda values

        Returns
        ------
        xopt : learned weights (not the same as the MIDAS,
        which are given by self.W)

        """

        xopt =  op.fmin(self.SSE_midas_beta, self.param_init,args=(X_train, y_train, L0,), xtol=1e-4, ftol=1e-4, maxiter=100000, maxfun=100000)

        WM = weights_midas_beta(xopt[0:2*self.settings['nbvar']],xopt[2*self.settings['nbvar']:(len(xopt)-1)],self.settings)
        self.W=np.r_[WM,xopt[-1]]
        return xopt

    def predict(
        self,X_test:np.array
        ) -> np.array:
        """ Predict output given input """
        return np.dot(X_test,self.W)
