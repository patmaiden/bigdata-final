from bs4 import BeautifulSoup
import urllib2 
import codecs

request = urllib2.Request('http://www.beeradvocate.com/beer/style/116/', headers={'User-Agent': 'Mozilla/5.0'})
response = urllib2.urlopen(request)
html = response.read()
soup = BeautifulSoup(html)

table = soup.find("table")

print table
