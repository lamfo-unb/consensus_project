#Define the symbol
sym = "PETR4"
url = "http://www.fundamentus.com.br/balancos.php?papel="

#Define the temp folder
import tempfile

#Create the temporary folder to store the backtester results
tmp_path = tempfile.mkdtemp()

#Get the PNG

import urllib

f = open('local_file_name','wb')
f.write(urllib.urlopen(src).read())
f.close()