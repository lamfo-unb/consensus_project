##BGigin here
from selenium import webdriver
from PIL import Image
from io import BytesIO
from data import deathbycaptcha
import time

#initializite the driver
driver = webdriver.Chrome(executable_path='/usr/local/bin/chromedriver')

driver.get("http://fundamentus.com.br/balancos.php?papel=PETR4")
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



#Sleep
time.sleep(10)

#Send the captcha to the DeathByCaptcha
username = "pedrobsb"
password= "******"

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








try:
    balance = client.get_balance()

    # Put your CAPTCHA file name or file-like object, and optional
    # solving timeout (in seconds) here:
    captcha = client.decode("captcha.png", 15)
    if captcha:
        # The CAPTCHA was solved; captcha["captcha"] item holds its
        # numeric ID, and captcha["text"] item its text.
        print("CAPTCHA %s solved: %s" % (captcha["captcha"], captcha["text"]))

        if ...:  # check if the CAPTCHA was incorrectly solved
            client.report(captcha["captcha"])
except deathbycaptcha.AccessDeniedException:
    # Access to DBC API denied, check your credentials and/or balance


#driver.quit()





/html/body/div[1]/div[2]/form/img





driver.find_element_by_id('search_form_input_homepage').send_keys("realpython")
driver.find_element_by_id("search_button_homepage").click()
print(driver.current_url)
driver.quit()

# Inicia o servidor
startServer(args=c(
    paste("-Dwebdriver.chrome.driver=", getwd(), "/chromedriver.exe -Dwebdriver.chrome.args='--disable-logging'",
          sep="")), log=FALSE, invisible=FALSE)
remDr < - remoteDriver(browserName="chrome")

# Abre o navegador
remDr$open()

# Maximiza a janela
remDr$maxWindowSize()

# Vai para a pagina de interesse
site < -"http://fundamentus.com.br/balancos.php?papel=PETR4"
remDr$navigate(site)

# Faz um printscreen do site
library(base64enc)
img < -remDr$screenshot(display=FALSE, useViewer=TRUE, file=NULL)
writeBin(base64Decode(img, "raw"), 'teste.png')

# Recorta o captcha
library(installr)
# install.ImageMagick()
local < -system("where convert", intern=TRUE)
system('"C:/Program Files/ImageMagick-6.9.0-Q16/convert.exe" -crop 202x62+475+340 teste.png teste2.png', intern=TRUE)

# Usa o DeathByCaptcha http://static.deathbycaptcha.com/files/dbc_api_v4_2_wincli.zip
system(paste("deathbycaptcha.exe -l pedrobsb -p pedroh -c ", getwd(), "/teste2.png", " -t 60", sep=""))
txt < -scan("answer.txt", what="character")

# Encontra o objeto da caixa de texto
webElem < - remDr$findElement(using="name", "codigo_captcha")

# Manda o resultado dp captcha
webElem$sendKeysToElement(list(txt))

# Executa o botao
webElem$sendKeysToElement(list(initializing_parcel_number, key="enter"))

# Encontra o objeto da caixa de texto
webElem < - remDr$findElement(using="name", "submit")

# Clica no botao
webElem$clickElement()

# Fecha as conexoes
remDr$close()
remDr$closeServer()

