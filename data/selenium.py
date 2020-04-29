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
from data import deathbycaptcha
import time

#Create the tempfile
tmp_path = tempfile.mkdtemp()
options = webdriver.ChromeOptions()
options.add_argument("download.default_directory="+tmp_path)
prefs = {'download.default_directory' : tmp_path};
options.add_experimental_option('prefs', prefs)

#initializite the driver
driver = webdriver.Chrome(options=options)
delay = 3 # seconds
driver.get("http://fundamentus.com.br/balancos.php?papel=PETR4")
WebDriverWait(driver, delay).until(EC.presence_of_element_located((By.ID, 'containerMenu')))
#Screen shot
element = driver.find_element_by_class_name("captcha")
#Get location
location = element.location
#Get size
size = element.size
#Do the screen shot
png = driver.get_screenshot_as_png()
#Cut the image
im = Image.open(BytesIO(png))
#TODO Get the location automatically
#left = location['x']
#top = location['y']+300
#right = location['x'] + size['width']
#bottom = location['y'] + size['height']+300
#im = im.crop((left, top, right, bottom)) # defines crop points
im = im.crop((230, 690, 630, 800)) # defines crop points
im.save('captcha.png')

#Send the captcha to the DeathByCaptcha
username = "pedrobsb"
password= "*********"

#Send the captcha
client = deathbycaptcha.SocketClient(username, password)
captcha = client.decode("captcha.png", 15)

#Get the solution
solution = captcha["text"]

#Find the text box
web_elem = driver.find_element("name", "codigo_captcha")

#Send the captcha results
web_elem.send_keys(solution)

#Request the data
button = driver.find_element("name", "submit")

#Click the button
button.click()

#Sleep and wait for the download
time.sleep(2)
driver.close()
driver.quit()

#Extract the zip file
zip_file = glob.glob(tmp_path+"/"+"*.zip")

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
print(sheet1.head())

