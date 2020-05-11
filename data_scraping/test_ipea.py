import pandas as pd
import pyIpeaData as ipea

meta_dados = ipea.get_metadados()

macro_tri = meta_dados[(meta_dados.BASNOME == "Macroeconômico") & (meta_dados.PERNOME == "Trimestral") &
                       (meta_dados.SERSTATUS == "A")]

series = dict.fromkeys(macro_tri['SERCODIGO'])

for cod in macro_tri['SERCODIGO']:
    series[cod] = ipea.get_serie(cod, None)

print(macro_tri['SERNOME'])
print(len(series))
print(series)















