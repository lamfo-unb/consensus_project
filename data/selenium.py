
#Step 1: Install
#$ pip install selenium

#brew cask upgrade && brew upgrade && brew cleanup


#$  brew tap homebrew/cask-cask
#$ brew cask install chromedriver


You might need to install it with : brew cask install chromedriver or brew install chromedriver

and then do which chromedriver

You will get the relevant path.



chromedriver = "/path/to/chromedriver/folder"
driver = webdriver.Chrome(chromedriver)
or chromedriver has to be in you PATH. You can add chromedriver to PATH with

export PATH=$PATH:/path/to/chromedriver/folder


#Step 2:
#https://www.kenst.com/2015/03/including-the-chromedriver-location-in-macos-system-path/

from selenium import webdriver
driver = webdriver.PhantomJS()
driver.set_window_size(1120, 550)
driver.get("https://duckduckgo.com/")
driver.find_element_by_id('search_form_input_homepage').send_keys("realpython")
driver.find_element_by_id("search_button_homepage").click()
print(driver.current_url)
driver.quit()


from selenium import webdriver


driver = webdriver.Firefox()
driver.get("http://www.fundamentus.com.br/balancos.php?papel=PETR4")

# change frame
driver.switch_to.frame("Main")

# download image/captcha
img = driver.find_element_by_xpath("/html/body/div[1]/div[2]/form/img")


tt = img

