import pandas as pd
import ipeadatapy as ipea
import datetime as dt
import pytz
import pickle


# para frequencias trimestrais = "Trimestral", para freqs mensais = "Mensal"

def ipea_feeder(freq, tema='Macroeconômico', pais='BRA', ano_update=2020, mes_update=1, dia_update=1):
    utc = pytz.UTC

    q_vars = ipea.metadata(big_theme=tema, country=pais, frequency=freq)
    q_vars['LAST UPDATE'] = pd.to_datetime(q_vars['LAST UPDATE'])

    last_update = dt.datetime(ano_update, dia_update, mes_update)
    q_vars = q_vars[q_vars['LAST UPDATE'] > last_update.replace(tzinfo=utc)]

    q_df = pd.DataFrame()
    for code in q_vars['CODE']:
        tm_temp = ipea.timeseries(code, yearGreaterThan=2000).iloc[:, -1]
        tm_temp.rename(code, inplace=True)
        q_df = q_df.append(tm_temp)

    return q_df

dict_freq = {'Trimestral':'quarterly','Mensal':'monthly'}
for freq in dict_freq:
    print('Loading {}'.format(dict_freq[freq]))
    data = ipea_feeder(freq).T
    data.index.name = 'Date'
    data.to_csv(f'../{dict_freq[freq]}/test_{freq}.csv')
