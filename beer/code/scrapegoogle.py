from bs4 import BeautifulSoup
import bs4
import urllib2 
import codecs
import re
import unicodedata

def scrape():
    request = urllib2.Request('https://www.google.com/search?q=what+is+data+science&oq=what+is+data+science&aqs=chrome..69i57j0l5.375j0j4&sourceid=chrome&es_sm=93&ie=UTF-8', headers={'User-Agent': 'Mozilla/5.0'})
    response = urllib2.urlopen(request)
    html = response.read()
    #print html
    soup = BeautifulSoup(html)
    #print soup.prettify()
    headers = soup.findAll("h3", attrs={'class':'r'})
    links = []
    for h3 in headers:
        link = str(h3.find("a"))
        link = re.search("<a href=\"\/url\?q=(.*)&amp;sa=U&amp;", link)
        link = link.group(1)
        link.join('')
        links.append(link)
    print links

    for link in links:
        pass

    request = urllib2.Request(links[0], headers={'User-Agent': 'Mozilla/5.0'})

    html = urllib2.urlopen(request).read()

    soup = BeautifulSoup(html)
    texts = soup.findAll(text=True)
    print type(texts[1])
    for property, value in vars(texts[1]).iteritems():
        print property    

    visible_texts = filter(lambda x: x != None and x != unicode('\n'), map(visible, texts))
    print visible_texts[:10]
    visible_texts = map(lambda x: unicodedata.normalize('NFKD', x).encode('ascii','ignore')
, visible_texts)
    text = reduce(lambda x,y: x + ' ' +y.strip(), visible_texts, '')
    text = ''.join(t for t in text.lower() if t.isalnum() or t == ' ')
    print text
    #response = urllib2.urlopen(request)
    #html = response.read()
    #print html
    #test = BeautifulSoup(html)
    #text = test.prettify()

def visible(element):

    if element.parent.name in ['style', 'script', '[document]', 'head', 'title']:
        return None
    elif isinstance(element,bs4.element.Comment):
        return None
    return unicode(element)
scrape()