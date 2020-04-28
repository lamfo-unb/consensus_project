from PIL import Image
from selenium import webdriver

def get_captcha(driver, element, path):
    # now that we have the preliminary stuff out of the way time to get that image :D
    location = element.location
    size = element.size
    # saves screenshot of entire page
    driver.save_screenshot(path)

    # uses PIL library to open image in memory
    image = Image.open(path)

    left = location['x']
    top = location['y'] + 140
    right = location['x'] + size['width']
    bottom = location['y'] + size['height'] + 140

    image = image.crop((left, top, right, bottom))  # defines crop points
    image.save(path, 'jpeg')  # saves new cropped image


driver = webdriver.Firefox()
driver.get("http://sistemas.cvm.gov.br/?fundosreg")

# change frame
driver.switch_to.frame("Main")

# download image/captcha
img = driver.find_element_by_xpath(".//*[@id='trRandom3']/td[2]/img")
get_captcha(driver, img, "captcha.jpeg")



driver = webdriver.Firefox()
driver.get("http://sistemas.cvm.gov.br/?fundosreg")

# change frame
driver.switch_to.frame("Main")

# download image/captcha
img = driver.find_element_by_xpath(".//*[@id='trRandom3']/td[2]/img")
get_captcha(driver, img, "captcha.jpeg")