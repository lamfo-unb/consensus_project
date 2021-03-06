import argparse
import sys
import tempfile
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

import zipfile
import glob
from PIL import Image
from io import BytesIO
import deathbycaptcha
import time
import pickle
from pathlib import Path
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd

def transpose_df(raw_dict):
    out = pd.DataFrame()
    for sheet in raw_dict:
        temp = raw_dict[sheet]
        colnames = temp.iloc[0, :]
        colnames.iloc[0] = "Data"
        # get rownames
        rownames = temp.iloc[:, 0]
        # Delete the first row
        temp = temp.drop(temp.index[0])
        # Delete the first column
        temp = temp.drop(temp.columns[0], axis=1)
        # Transpose the data_scraping
        temp_transposed = temp.T
        # Define the column names
        temp_transposed.columns = rownames[1:]
        # Define the rownames names
        temp_transposed.index = pd.to_datetime(colnames[1:], )

        out = pd.concat((out,temp_transposed),axis=1)
        
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--time', type=float, default=3.0,
                        help='Time in seconds to sleep before operations.')
    parser.add_argument('-s','--symbol',
                        help='CSV file with the symbols.')
    parser.add_argument('-u', '--username',
                        help='Username deathbycpatcha.')
    parser.add_argument('-p', '--password',
                        help='Pssword deathbycpatcha.')
    parser.add_argument('-a','--attempts', type=int, default=3,
                        help='Number of atempts to solve the captcha.')
       #Get arguments
    args = parser.parse_args()

    #Run the data_scraping scrapping
    data_scrapping(args)

def data_scrapping(args,save_data=True):
    try:
        #Step 1: Read the CSV and run all over the companies
        symbols = pd.read_csv(args.symbol)
        df_results = {}
        for sym in symbols["Symbols"]:
            local_attempt = 0
            empty = True
            while local_attempt < args.attempts and empty:
                # Create the tempfile
                tmp_path = tempfile.mkdtemp()
                options = webdriver.ChromeOptions()
                options.add_argument("download.default_directory=" + tmp_path)
                prefs = {'download.default_directory': tmp_path};
                options.add_experimental_option('prefs', prefs)
                # initializite the driver
                driver = webdriver.Chrome(options=options)
                delay = args.time
                driver.get("http://fundamentus.com.br/balancos.php?papel="+str(sym))
                WebDriverWait(driver, delay).until(EC.presence_of_element_located((By.ID, 'containerMenu')))
                # Get captcha
                with open('captcha.png', 'wb') as file:
                    file.write(driver.find_element_by_xpath('/html/body/div[1]/div[2]/form/img').screenshot_as_png)

                # Send the captcha to the DeathByCaptcha
                username = args.username
                password = args.password
                # Send the captcha
                client = deathbycaptcha.SocketClient(username, password)
                captcha = client.decode("captcha.png", 15)
                if captcha is not None:
                    # Get the solution
                    solution = captcha["text"]
                    # Find the text box
                    web_elem = driver.find_element("name", "codigo_captcha")
                    # Send the captcha results
                    web_elem.send_keys(solution)
                    # Request the data_scraping
                    button = driver.find_element("name", "submit")
                    # Click the button
                    button.click()
                    # Sleep and wait for the download
                    time.sleep(args.time)
                    driver.close()
                    driver.quit()

                    #Extract the zip file
                    zip_file = glob.glob(tmp_path + "/" + "*.zip")

                    #Make sure that something was downloaded
                    local_attempt = local_attempt + 1
                    if len(zip_file) > 0: empty = False
                else:
                    empty = True
                    local_attempt = local_attempt + 1


            #Get the name (bal_Empresa.zip)
            if Path(zip_file[0]).name == "bal_Empresa.zip":
                empty == True
            else:
                # Sleep and wait for zip extract
                time.sleep(args.time)
                with zipfile.ZipFile(zip_file[0], 'r') as zip_ref:
                    zip_ref.extractall(tmp_path + "/")

                # Sleep and wait for zip extract
                time.sleep(args.time)

                # List all XLS files
                xls_file = glob.glob(tmp_path + "/" + "*.xls")

                # Read all sheets in a PandasDataFrame
                fund_dict = pd.read_excel(xls_file[0], sheet_name=None, header=None, skiprows=1)

                transposed_df = transpose_df(fund_dict)
                #Save results
                df_results[sym] = transposed_df
            if empty == True:
                print("ERROR: Symbol: "+sym+" was not downloaded.")
            else:
                print("OK: Symbol: "+sym+" was downloaded.")
    except Exception as e:
        print(str(e))

    #Save the results
    if save_data:
        pickle.dump(df_results, open("../data/all_data.pkl", "wb"))
    else:
        return df_results
if __name__ == '__main__':
    main()