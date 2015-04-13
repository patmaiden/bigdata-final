from bs4 import BeautifulSoup
import urllib2
import codecs

response = urllib2.urlopen('http://www.beeradvocate.com/beer/style/116/')
html = response.read()
soup = BeautifulSoup(html)

table = soup.find("table")

print table
