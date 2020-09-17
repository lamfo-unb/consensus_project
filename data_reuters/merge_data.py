"""
merges fundamentus and reuters data
the final dataframe is saved in data/consolidate/
"""

import pandas as pd
import pickle as pk
from glob import glob



def clean_zero_columns(df):
    # removes zero columns of a dataframe
    df = df.loc[:, (df != 0).any(axis=0)]
    return df


def fix_duplicates(df,ticker,folder_name='consolidate'):
    """
    reads and saves dataframe 
    this is done to rename all duplicated columns

    Parameters
    ----------
    df : pd.DataFrame
        dataframe with duplicated columns
    ticker : string
        ticker of given equity

    """
    df.to_csv(f'../data/{folder_name}/{ticker}.csv')
    fixed_df = pd.read_csv(f'../data/{folder_name}/{ticker}.csv',
                            index_col = '0',parse_dates=True).sort_index().astype(float)
    return fixed_df


def check_existance(ticker):
    # checks existance of ticker in downloaded data
    name_reuters = glob(f'../data/{ticker[:-1]}*.SA_merge.csv')
    if len(name_reuters) > 0:
        print(f'Symbol {ticker} found!\n')
        found = True
        name_reuters = name_reuters[0]
        df_reuters = pd.read_csv(name_reuters,index_col='Balance Sheet Period End Date',parse_dates=True).fillna(0).sort_index()
    
        
    else:
        print(f'Symbol {ticker} not found! \n')
        found = False
        df_reuters = None

    return found,df_reuters

def get_translation(df_fund, df_reuters,ticker,
                    start_date = '2018-12-31',
                    end_date = '2019-12-31'):
    """
    gets translation of columns based on correlation

    Parameters
    ----------
    df_fund : pd.DataFrame
        dataframe from fundamentus
    df_reuters : pd.DataFrame
        dataframe from reuters
    ticker : string
        ticker of given equity
    start_date and end_data are used just to get a few samples
    to calculate the correlation
    """

    df_reuters = df_reuters.loc[start_date:end_date,:].drop(['Unnamed: 0','Instrument_x'],axis=1)
    df_fund = df_fund.loc[start_date:end_date,:].fillna(0)

    df_reuters = clean_zero_columns(df_reuters)
    df_fund = clean_zero_columns(df_fund)
    dict_names = {}
    for column_fund in df_fund:
        if df_reuters.corrwith(df_fund.loc[:,column_fund]).shape[0] > 0:
            columns_reuters = df_reuters.corrwith(df_fund.loc[:,column_fund]).sort_values(ascending=False).index[0]
            dict_names[columns_reuters] = column_fund
            df_reuters.drop(columns=columns_reuters,inplace=True) # removes selected column
        else:
            print('No correlation!')

    return dict_names


def join_data(df_fund_clean,df_reuters,new_names):
    """
    Merges two dataframes according to common columns

    Parameters
    ----------
    df_fund : pd.DataFrame
        dataframe from fundamentus
    df_reuters : pd.DataFrame
        dataframe from reuters
    new_names : dictionary
        dict with keys = reuters columns
        and values = fundamentus columns

    """
    df_reuters.rename(columns=new_names,inplace=True)
    columns_mask = [new_names[x] for x in new_names]
    row_mask = df_reuters.index < df_fund_clean.index[0]
    df_final = pd.concat((df_fund_clean,df_reuters.loc[row_mask,columns_mask]/1e3),join='outer',axis=0).sort_index()

    return df_final

def main():

    df_fund_raw = pd.read_pickle('../data/all_data.pkl')
    all_tickers = [name for name in df_fund_raw]

    for ticker in all_tickers:

        found, df_reuters = check_existance(ticker)
        if found:
            df_fund_clean = fix_duplicates(df_fund_raw[ticker],ticker)
            new_names = get_translation(df_fund_clean,df_reuters.copy(),ticker)
            df_final = join_data(df_fund_clean,df_reuters,new_names)
            df_final.to_csv(f'../data/consolidate/{ticker}.csv')

if __name__ == '__main__':
    main()