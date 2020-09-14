import numpy as np
import pandas as pd
import statsmodels.api as sm
from variables import sigma, Spec, mu, b0, phi, Prct_relevant
from typing import Tuple, Union, Any, List, Dict

def naive_prediction(
    y_test:np.array) -> float:
    """
    Return the MSE naive prediction for y
    y_{t-1} = y_t
    """
    yhat = np.concatenate((np.zeros(1),y_test[:-1]))
    
    return np.sqrt(np.sum(np.square(yhat-y_test)))


def read_ticker(
    ticker:str) -> pd.DataFrame:
    """ 
    Reads ticker and returns X's and y's according to target column and 
    testing length

    Inputs
    ------
    ticker : company ticker

    Returns
    ------
    df : pandas dataframe with the percentual change of
    each indicator in the result

    """
    df = pd.read_csv(f'data/consolidate/{ticker}.csv',
                    index_col='Unnamed: 0',parse_dates=True).fillna(0)

    df = df.loc[:, (df != 0).any(axis=0)] # removing 0 columns
    df = df.pct_change().replace([np.inf, -np.inf, np.nan], 0)

    return df



def load_data(
    ticker:str,T_test:int,file_name_monthly:str,
    y_col='Lucro/Prejuízo do Período'
    )-> Tuple[np.array,np.array,np.array,np.array]:
    """ 
    Load data from target company and merge with data from IPEA

    Inputs
    ------
    ticker : company ticker
    T_test : number of testing periods
    file_name_monthly : name of the file with monthly data
    y_col : target column name
    Returns
    ------
    X_train, X_test, y_train, y_test : input/output data

    """
    df_monthly = read_ipea(file_name_monthly)
    df_quarterly = read_ticker(ticker)

    df_all = pd.concat([df_monthly,df_quarterly],join='inner',axis=1)

    y_all = df_all.loc[:,y_col].copy()
    X_all = df_all.drop(columns=y_col)

    X_train = X_all.iloc[:-T_test,:]
    X_test = X_all.iloc[-T_test:,:]

    y_train = y_all[:-T_test]
    y_test = y_all[-T_test:]

    return X_train.values, X_test.values, y_train.values, y_test.values


def lagger(
    dataset:pd.DataFrame, n_lags:int,
    price_columns : Union[str,List[str]]) -> pd.DataFrame:
    """
    Create columns of time lags

    Inputs
    ------
    dataset : dataframe to lag 
    n_lags : number of time lags
    price_columns :
    y_col : target column name(s)
    Returns
    ------
    result : lagged dataframe
    """
    from toolz.curried import reduce
    df = reduce(
        lambda df, lag: df.assign(**{col + '_' +str(lag): dataset[[col]].shift(lag).values for col in price_columns}),
        range(1, n_lags + 1),
        dataset[price_columns])

    result = df.assign(**{col: dataset[col] for col in dataset.drop(price_columns, axis=1).columns}).fillna(0)
    return result[sorted(result.columns)]


def read_ipea(
    file_name:str) -> pd.DataFrame:
    """
    Read data downloaded from IPEA and return it on quarterly frequency

    Inputs
    ------
    file_name : file name

    Returns
    ------
    result : dataframe
    
    """
    df = pd.read_csv(f'data/monthly/{file_name}.csv',
                    index_col='Date',parse_dates=True).fillna(0)
    df = df.pct_change().replace([np.inf, -np.inf, np.nan], 0)

    df = lagger(df,2,list(df))
    df.index = df.index - pd.Timedelta('1 days')
    df = df.resample('Q', convention='start').asfreq()

    return df

def calculate_residuals(
    model:Any,X_train:np.array,
    X_test:np.array,y_train:np.array,
    y_test:np.array) -> Tuple[float,float]:

    """
    Calculate residuals

    Inputs
    ------
    model : Midas model
    X_train : training input
    X_test : testing input
    y_train : training output
    y_test : testing output

    Returns
    ------
    mse_train : mean square error from training
    mse_test : mean square error from testing

    """

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    training_residuals = y_train - y_train_pred
    testing_residuals = y_test - y_test_pred
    mse_train = np.sqrt(np.sum(np.square(training_residuals)))
    mse_test = np.sqrt(np.sum(np.square(testing_residuals)))

    return mse_train, mse_test

def store_results(
    xopt:np.array,X_train:np.array,
    X_test:np.array,y_train:np.array,
    y_test:np.array,L0:Dict,
    model:Any) -> Dict:

    """
    Organize results in a dictionary 

    Inputs
    ------
    xopt : optimized weights
    X_train : training input
    X_test : testing input
    y_train : training output
    y_test : testing output
    model : Midas model

    Returns
    ------
    R : dictionary with results

    """
    R={}
    R['lambda']=L0
    R['th_simul']=xopt[0:2*model.settings['nbvar']]
    R['bt_simul']=xopt[2*model.settings['nbvar']:(len(xopt)-1)]
    R['RelevantVar_simul']=0
    R['IrrelevantVar_simul']=0
    R['xopt']=xopt

    for ll in range(len(R['bt_simul'])):
        if -sigma/np.sqrt(len(R['bt_simul']))<R['bt_simul'][ll] and R['bt_simul'][ll]<sigma/np.sqrt(len(R['bt_simul'])):
            R['IrrelevantVar_simul']=R['IrrelevantVar_simul']+1
        else:
            R['RelevantVar_simul']=R['RelevantVar_simul']+1

    # MSE for test and train

    R['MSE_train'], R['MSE_test'] = calculate_residuals(model,X_train,X_test,y_train,y_test)


    return R

def create_time_dicts(
    Spc:Dict) -> List[Dict]:
    """
    Create a dict with daily, monthly and quarterly information

    Inputs
    ------
    Spc : dict with number of variables for each time frequency

    Returns
    ------
    out_list : list of dicts with each frequency configuration

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

def weights_midas_beta(
    th:np.array, bt:np.array,
    Spc:Dict) -> np.array:
    """
    Construct covariates matrix as defined by MIDAS weighting scheme

    Inputs
    ------
    th: parameters theta for the weighting kernel
    bt: parameters beta for regression coefficients
    Spc: MIDAS specifications

    Returns
    -------
    W : MIDAS weights

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
                    W0=np.exp(th1[i]*time_period['k'] + th2[i]*np.square(time_period['k'])) \
                        / np.sum(np.exp(th1[i]*time_period['k'] + th2[i]*np.square(time_period['k'])))
                elif Spc['betaFc']:
                    W0=np.exp(th1[i]*time_period['k'] + th2[i]*np.square(time_period['k'])) \
                        / np.sum(np.exp(th1[i]*time_period['k'] + th2[i]*np.square(time_period['k'])))
            elif Spc['Averaging']:
                W0=time_period['one']/time_period['kk']
            elif Spc['betaFc']:
                W0=np.power(th2[i]*(1-time_period['w']),(th2[i]-1)) \
                    / sum(np.power(th2[i]*(1-time_period['w']),(th2[i]-1)))
            elif Spc['betaFc_special']:
                W0=th2[i]*time_period['w']*np.power((1-time_period['w']),(th2[i]-1))\
                    / sum(th2[i]*time_period['w']*np.power((1-time_period['w']),(th2[i]-1)))
            if i==0:
                W = W0*bt[i]
                ww = W0
            else:
                W = np.r_[W,W0*bt[i]]
                ww = np.r_[ww,W0]

    return W.T



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


def generate_y(T_train,T_test,X_train,X_test,Prct_relevant=Prct_relevant,Spec=Spec):
    theta1=0.1*np.ones(Spec['nbvar'])
    theta2=-0.05*np.ones(Spec['nbvar'])



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

    # DGP from gaussian noise
    y_train=Yreg+np.random.normal(mu,sigma,T_train) # In-sample Y
    y_test=Yfor+np.random.normal(mu,sigma,T_test) # Out-of-sample Y

    return y_train,y_test,X_train,X_test,bt