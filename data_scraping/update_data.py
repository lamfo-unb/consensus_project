from scrapping import data_scrapping
import pandas as pd
import argparse
from glob import glob
import sys
sys.path.append("../data_reuters/")
from merge_data import fix_duplicates

def join_old_new(df_latest,df_old):
    """
    Merges two dataframes according to common columns

    Parameters
    ----------
    df_latest : pd.DataFrame
        latest dataframe from fundamentus
    df_old : pd.DataFrame
        old dataframe from fundamentus
    """

    row_mask = df_old.index[-1] < df_latest.index
    print(row_mask)
    df_final = pd.concat((df_old,df_latest.loc[row_mask,:]))


    return df_final


def read_consolidate(ticker,old_folder_name='../data/consolidate/'):
    
    df = pd.read_csv(f'{old_folder_name}{ticker}.csv',index_col = 0,parse_dates=True)

    return df



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--time', type=float, default=3.0,
                        help='Time in seconds to sleep before operations.')
    parser.add_argument('-s','--symbol',
                        help='CSV file with the symbols.')
    parser.add_argument('-u', '--username',
                        help='Username deathbycpatcha.')
    parser.add_argument('-p', '--password',
                        help='Password deathbycpatcha.')
    parser.add_argument('-a','--attempts', type=int, default=3,
                        help='Number of atempts to solve the captcha.')
       #Get arguments
    args = parser.parse_args()
    #symbols = pd.read_csv(args.symbol)
    #Run the data_scraping scrapping
    fundamentus_dict = data_scrapping(args,save_data=False)
    folder_name = 'latest'

    for ticker in fundamentus_dict:
        print(f'Processing: {ticker}')
        df_latest = fix_duplicates(fundamentus_dict[ticker],ticker,folder_name=folder_name)
        df_latest.to_csv(f'{ticker}.csv')
        df_old = read_consolidate(ticker)
        df_final = join_old_new(df_latest,df_old)
        df_final.to_csv(f'../data/{folder_name}/{ticker}.csv')


if __name__ == "__main__":
    main()