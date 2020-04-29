#BGigin here
from selenium import webdriver
from PIL import Image
from io import BytesIO
#from data import deathbycaptcha
import time

#initializite the driver
driver = webdriver.Chrome()

driver.get("http://fundamentus.com.br/balancos.php?papel=PETR4")