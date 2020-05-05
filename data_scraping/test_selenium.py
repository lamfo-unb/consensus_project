##BGigin here
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
from data_scraping import deathbycaptcha
import time
import os

#Create the tempfile
tmp_path = tempfile.mkdtemp()
options = webdriver.ChromeOptions()
options.add_argument("download.default_directory="+tmp_path)
prefs = {'download.default_directory': tmp_path};
options.add_experimental_option('prefs', prefs)

#initializite the driver
driver = webdriver.Chrome('/usr/local/bin/chromedriver', options=options)
#TODO Loop in case of empty zip_file path (wrong captcha)
# f_download = False
# while not f_download:
delay = 3  # seconds
driver.get("http://fundamentus.com.br/balancos.php?papel=PETR4")
WebDriverWait(driver, delay).until(EC.presence_of_element_located((By.ID, 'containerMenu')))
# Screen shot
element = driver.find_element_by_class_name("captcha")
# Get location
location = element.location
# Get size
size = element.size
# Do the screen shot
png = driver.get_screenshot_as_png()
# Cut the image
im = Image.open(BytesIO(png))
# Get the location automatically
left = element.location_once_scrolled_into_view['x']
top = element.location_once_scrolled_into_view['y']  # +300
right = element.location_once_scrolled_into_view['x'] + size['width']
bottom = element.location_once_scrolled_into_view['y'] + size['height']  # +300
im = im.crop((left, top, right, bottom))  # defines crop points
# im = im.crop((230, 690, 630, 800))  defines crop points
im.save('captcha.png')

# Send the captcha to the DeathByCaptcha
username = "pedrobsb"
password = "*********"

# Send the captcha
client = deathbycaptcha.SocketClient(username, password)
captcha = client.decode("captcha.png", 15)

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
time.sleep(2)
driver.close()
driver.quit()

# Extract the zip file
zip_file = glob.glob(tmp_path + "/" + "*.zip")
#TODO Loop in case of empty zip_file path (wrong captcha)
#    if zip_file:
#        print("File exists!")
#        f_download = True
#    else:
#        print("No file in path, trying again...")

#Sleep and wait for zip extract
time.sleep(2)

with zipfile.ZipFile(zip_file[0], 'r') as zip_ref:
    zip_ref.extractall(tmp_path+"/")

#Sleep and wait for zip extract
time.sleep(2)

#List all XLS files
xls_file = glob.glob(tmp_path+"/"+"*.xls")

#Read all sheets in a PandasDataFrame
sheet1 = pd.read_excel(xls_file[0], sheet_name=0, header=None, skiprows = 1)

#get column names
colnames  = sheet1.iloc[0,:]
colnames.iloc[0] = "Data"
#get rownames
rownames = sheet1.iloc[:,0]
#Delete the first row
sheet1 = sheet1.drop(sheet1.index[0])
#Delete the first column
sheet1 = sheet1.drop(sheet1.columns[0], axis=1)
#Transpose the data_scraping
sheet1_transposed = sheet1.T
#Define the column names
print(sheet1_transposed)
sheet1_transposed.columns = rownames[1:]
#Define the rownames names
sheet1_transposed.index = pd.to_datetime(colnames[1:],)


