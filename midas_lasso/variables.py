import numpy as np
from utils import simulate_X, generate_y, load_data
import math
from typing import Dict, Tuple

def load_variables(
    real_data_dict:Dict = None, simulation_mode:bool=False,
    )-> Tuple[np.array,np.array,np.array,np.array,Dict]:
    """ 
    Create the MIDAS's simulation data or load market data

    Inputs
    ------
    real_data_dict : dict with ticker and monthly data file name
    simulation_mode : Whether or not to simulate data

    Returns
    ------
    X_train, X_test : Training and testing inputs
    y_train, y_test : Training and testing outputs
    Spec : dict with model specifications
    """

    Spec = {}
    #Specifications of the MIDAS model
    Spec['intercept']=1
    Spec['AR']=0 #Here the code doesnt allow the use of an autoregressive term

    #Weighting scheme
    Spec['TwoParam']=1
    Spec['almon']=1
    Spec['Averaging']=0
    Spec['betaFc']=0

    if simulation_mode:
        y_train, y_test, X_train, X_test, _, Spec = load_simulation(Spec)
        return y_train, y_test, X_train, X_test, Spec

    else:
        X_train, X_test, y_train, y_test, number_variables = load_market_data(real_data_dict)
        Spec['daily'] = number_variables['daily']  # number of daily variables
        Spec['Kd']= 21 # number of days in the period
        Spec['monthly'] = number_variables['monthly'] # number of monthly variables
        Spec['Km']= 3 # number of months in the period
        Spec['quarterly'] = number_variables['quarterly']  # number of quarterly variables, 65 for petro
        Spec['Kq']= 1 # number of quarters in the period
        Spec['nbvar'] = sum(number_variables.values())

        if Spec['intercept']==1:
            bias_column = np.ones((X_train.shape[0],1))
            X_train = np.concatenate((X_train,bias_column),axis=1)
            bias_column = np.ones((X_test.shape[0],1))
            X_test = np.concatenate((X_test,bias_column),axis=1)

        return X_train, X_test, y_train, y_test, Spec


def load_market_data(
    real_data_dict:Dict
    )-> Tuple[np.array,np.array,np.array,np.array,int]:
    """ 
    Load market data

    Inputs
    ------
    real_data_dict : dict with ticker,monthly data file name
    and testing length

    Returns
    ------
    X_train, X_test : Training and testing inputs
    y_train, y_test : Training and testing outputs
    number_variables : number of variables 
    """    

    T_test = real_data_dict['T_test']
    ticker = real_data_dict['ticker']

    if len(real_data_dict['monthly']) == 0:
        file_name_monthly = None
    else:
        file_name_monthly = real_data_dict['monthly']

    if len(real_data_dict['quarterly']) == 0:
        file_name_quarterly = None
    else:
        file_name_quarterly = real_data_dict['quarterly']

    X_train, X_test, y_train, y_test, number_variables = load_data(
    ticker,T_test = T_test,file_name_quarterly=file_name_quarterly,
    file_name_monthly=file_name_monthly)

    return X_train, X_test, y_train, y_test, number_variables


def load_simulation(Spec):

    #In-sample size and out-of-sample size
    T_train=200


    T_test=50


    #Percentage of relevant variables
    Prct_relevant=0.3

    #Number of variables in the simulation
    Nbvar = 20

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
    
    #Relevant features
    #Nbrelevant=math.floor(Prct_relevant*Spec['nbvar'])
    #Non relevant features
    #Nbirrelevant=Spec['nbvar']-Nbrelevant


    #Constructing X matrices in the MIDAS form (each row corresponds to a time t
    # indices ; and columns are the high-frequency data includes in [t-K;t])

    X_train, X_test = simulate_X(T_train,T_test,simul_dict)
    #Construction Y

    y_train,y_test,X_train,X_test,bt = generate_y(
        T_train,T_test,X_train,X_test,Prct_relevant=Prct_relevant,Spec=Spec)
    bt= np.random.binomial(1, Prct_relevant, Spec['nbvar'])*np.random.normal(0,1,Spec['nbvar'])
    
    return y_train, y_test, X_train, X_test, bt, Spec

