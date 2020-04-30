import argparse
import sys
import tempfile
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import pandas as pd
import zipfile
import glob
from PIL import Image
from io import BytesIO
import deathbycaptcha
import time
import pickle


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
    parser.add_argument('-l', '--left', type=float, default=230,
                        help='Left position crop captcha.')
    parser.add_argument('-w', '--upper', type=float, default=630,
                        help='Top position crop captcha.')
    parser.add_argument('-r', '--right', type=float, default=630,
                        help='Right position crop captcha.')
    parser.add_argument('-b', '--bottom', type=float, default=780,
                        help='Bottom position crop captcha.')

    #Get arguments
    args = parser.parse_args()

    #Run the data scrapping
    data_scrapping(args)

def data_scrapping(args):
    #Step 1: Read the CSV and run all over the companies
    symbols = pd.read_csv(args.symbol)
    df_results = []
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
            # Screen shot
            element = driver.find_element_by_class_name("captcha")
            # Do the screen shot
            png = driver.get_screenshot_as_png()
            # Cut the image
            im = Image.open(BytesIO(png))
            # TODO Get the location automatically
            im = im.crop((args.left, args.upper, args.right, args.bottom))  # defines crop points
            im.save('captcha.png')
            # Send the captcha to the DeathByCaptcha
            username = args.username
            password = args.password
            # Send the captcha
            client = deathbycaptcha.SocketClient(username, password)
            captcha = client.decode("captcha.png", 15)
            # Get the solution
            solution = captcha["text"]
            # Find the text box
            web_elem = driver.find_element("name", "codigo_captcha")
            # Send the captcha results
            web_elem.send_keys(solution)
            # Request the data
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

        # Sleep and wait for zip extract
        time.sleep(args.time)

        with zipfile.ZipFile(zip_file[0], 'r') as zip_ref:
            zip_ref.extractall(tmp_path + "/")

        # Sleep and wait for zip extract
        time.sleep(args.time)

        # List all XLS files
        xls_file = glob.glob(tmp_path + "/" + "*.xls")

        # Read all sheets in a PandasDataFrame
        sheet1 = pd.read_excel(xls_file[0], sheet_name=0, header=None, skiprows=1)

        # get column names
        colnames = sheet1.iloc[0, :]
        colnames.iloc[0] = "Data"
        # get rownames
        rownames = sheet1.iloc[:, 0]
        # Delete the first row
        sheet1 = sheet1.drop(sheet1.index[0])
        # Delete the first column
        sheet1 = sheet1.drop(sheet1.columns[0], axis=1)
        # Transpose the data
        sheet1_transposed = sheet1.T
        # Define the column names
        sheet1_transposed.columns = rownames[1:]
        # Define the rownames names
        sheet1_transposed.index = pd.to_datetime(colnames[1:], )
        #Save results
        df_results.append(sheet1_transposed)
        if empty == True:
            print("ERROR: Symbol: "+sym+" was not downloaded.")
        else:
            print("OK: Symbol: "+sym+" was downloaded.")

    #Save the results
    pickle.dump(df_results, open("resukts.pkl", "wb"))
if __name__ == '__main__':
    main()


#https://pythonprogramming.net/argparse-cli-intermediate-python-tutorial/
#python example.py --username=pedrobsb --password=****** --symbol=symbols.csv