import math
import numpy as np
from typing import Dict, List
import scipy.optimize as op


class MidasLasso():

    def __init__(
        self,settings:Dict[int,bool]
        )-> None:
        """ Initialize Midas model

        Inputs
        ------
        settings : dict with settings
        """
        self.settings = settings
        self.param_init=np.zeros(settings['nbvar']*3+1)
        self.W = None

    def load_weights(
        self,file_name:str
        )-> None:
        """ Load trained weights from Midas model

        Inputs
        ------
        file_name : name of the file
        """

        if '.npy' not in file_name:
            file_name = file_name + '.npy'
        self.W = np.load(file_name)

    def _create_time_dicts(
        self) -> List[Dict]:
        """
        Create a dict with daily, monthly and quarterly information

        Inputs
        ------
        self.settings : dict with number of variables for each time frequency

        Returns
        ------
        out_list : list of dicts with each frequency configuration

        """
        abreviations = ['Kd','Km','Kq']
        daily_range = self.settings['daily']
        monthly_range = self.settings['monthly']
        quarterly_range = self.settings['quarterly']
        all_ranges = np.cumsum([0,daily_range,monthly_range,quarterly_range])

        out_list = []
        for i,abrev in enumerate(abreviations):
            temp_dict = {}
            temp_dict['range'] = range(all_ranges[i],all_ranges[i+1])
            temp_dict['one'] = np.ones(self.settings[abrev])
            temp_dict['w'] = np.arange(1,self.settings[abrev]+1)/self.settings[abrev]
            temp_dict['k'] =  np.arange(1,self.settings[abrev]+1)
            temp_dict['kk'] =  self.settings[abrev]
            out_list.append(temp_dict)

        return out_list
        
    def _construct_covariates(
        self,th:np.array,
        bt:np.array) -> np.array:

        """
        Construct covariates matrix as defined by MIDAS weighting scheme

        Inputs
        ------
        th: parameters theta for the weighting kernel
        bt: parameters beta for regression coefficients

        Returns
        -------
        W : MIDAS weights

        """

        dict_list = self._create_time_dicts()

        if self.settings['TwoParam']:
            th1=th[0:self.settings['nbvar']]
            th2=th[self.settings['nbvar']:2*self.settings['nbvar']]
        else:
            th2=th

        for time_period in dict_list:
            for i in time_period['range']:
                if self.settings['TwoParam']:
                    if self.settings['almon']:
                        W0=np.exp(th1[i]*time_period['k'] + th2[i]*np.square(time_period['k'])) \
                            / np.sum(np.exp(th1[i]*time_period['k'] + th2[i]*np.square(time_period['k'])))
                    elif self.settings['betaFc']:
                        W0=np.exp(th1[i]*time_period['k'] + th2[i]*np.square(time_period['k'])) \
                            / np.sum(np.exp(th1[i]*time_period['k'] + th2[i]*np.square(time_period['k'])))
                elif self.settings['Averaging']:
                    W0=time_period['one']/time_period['kk']
                elif self.settings['betaFc']:
                    W0=np.power(th2[i]*(1-time_period['w']),(th2[i]-1)) \
                        / sum(np.power(th2[i]*(1-time_period['w']),(th2[i]-1)))
                elif self.settings['betaFc_special']:
                    W0=th2[i]*time_period['w']*np.power((1-time_period['w']),(th2[i]-1))\
                        / sum(th2[i]*time_period['w']*np.power((1-time_period['w']),(th2[i]-1)))
                if i==0:
                    W = W0*bt[i]
                    ww = W0
                else:
                    W = np.r_[W,W0*bt[i]]
                    ww = np.r_[ww,W0]

        return W.T

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

        WM = self._construct_covariates(np.r_[th1,th2],bt)

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

        WM = self._construct_covariates(xopt[0:2*self.settings['nbvar']],xopt[2*self.settings['nbvar']:(len(xopt)-1)])
        self.W=np.r_[WM,xopt[-1]]
        return xopt

    def predict(
        self,X_test:np.array
        ) -> np.array:
        """ Predict output given input """
        return np.dot(X_test,self.W)
