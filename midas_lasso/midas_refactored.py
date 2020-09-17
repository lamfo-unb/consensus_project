import math
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
from utils import store_results
from midas_model import MidasLasso
from variables import load_variables
from utils import load_data, naive_prediction, calculate_residuals
import argparse



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--ticker',
                        help='Stock ticker.')
    parser.add_argument('-m','--monthly', default = '',
                        help='CSV file with monthly data.')
    parser.add_argument('-o', '--out_of_sample', type=int, default=15,
                        help='Testing length.')
    parser.add_argument('-v', '--verbose', type=bool, default = True,
                        help='Whether or not to display results.')
    parser.add_argument('-s','--save', type=bool, default = True,
                        help='Whether or not to save weights to file.')    
    args = parser.parse_args()

    train_midas(args)



def train_midas(args):

    VERBOSE = args.verbose
    SAVE_WEIGHTS = args.save



    SIMULATION = False
    np.random.seed(42)
    # Parameter (lambda)
    lambda_par=[0,1] # <-- Here is a range for a loop
    #Norm (e.g. if norme=1 it's the LASSO, norme=2 it's the ridge,... )
    norme=np.r_[0,np.ones(len(lambda_par)-1)] # <-- same size than the lambda_par vector

    real_data_dict = {'ticker':args.ticker,'monthly':args.monthly,'T_test':args.out_of_sample}
    X_train, X_test, y_train, y_test, Spec = load_variables(real_data_dict=real_data_dict, simulation_mode=SIMULATION)


    print('Running MIDAS model on stock {} with {} monthly parameters,\
         testing period of {}'.format(args.ticker,len(args.monthly),args.out_of_sample))

    # loop on parameters lambda_par
    results = []
    for i,lambda_value in enumerate(lambda_par):
        #display
        print('lambda= ' +str(lambda_par[i]))
        SAVE_NAME = real_data_dict['ticker']

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

        model = MidasLasso(Spec)

        # Case of LASSO-MIDAS model
        if lambda_value!=0:
            xopt = model.fit(X_train,y_train,LO)
            SAVE_NAME = SAVE_NAME + '_lasso'
        #Classical MIDAS regression model
        else:
            xopt = model.fit(X_train,y_train)

        #Computing fitted values


        if SIMULATION:
            R = store_results(xopt,X_train,X_test,y_train,y_test,LO['lambda'],model)
        else:
            R = {}
            R['MSE_train'], R['MSE_test'] = calculate_residuals(model,X_train,X_test,y_train,y_test)

        if SAVE_WEIGHTS:
            np.save(SAVE_NAME, model.W)  

        results.append(R)


    if VERBOSE:
        
        print("Results for normal midas \n MSE train = {:.2f} \
        , MSE test = {:.2f}".format(results[0]['MSE_train'], results[0]['MSE_test']))
        print("Results for midas lasso \n MSE train = {:.2f} \
        , MSE test = {:.2f}".format(results[1]['MSE_train'], results[1]['MSE_test']))

        print("Results for naive prediction MSE train = {:.2f}\
        , MSE test {:.2f}".format(naive_prediction(y_train),naive_prediction(y_test)))


if __name__ == "__main__":
    main()













