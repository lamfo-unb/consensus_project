import pandas as pd
import pickle

#Read the Sectors
sectors = pd.read_csv(r"data/tickers_with_sectors.csv")

#Read the pickle file
with open(r'data/all_data.pkl', 'rb') as handle:
    symbols = pickle.load(handle)

#Concat by sectors
grouped = sectors.groupby('Subsetor')
#Create the dictionary
df_results = {}
for name, group in grouped:
    #Get the symbols
    sym = group["Symbol"].to_list()
    #Get the dataframe
    res = []
    for symbol in sym:
        df = group.loc[group["Symbol"]==sym]
        df["Symbol"] = symbol
        df["Sector"] = name
        df["Date"] = str(df.index)
        res.append(df)
    # Save results
    df_results[name] = pd.concat(res)


temp = df_results["Alimentos Processados"]
temp.rename(columns={ df.columns[0]: "Date2" }, inplace = True)
temp["Date"] = pd.datetime(temp["Date"].astype(int))