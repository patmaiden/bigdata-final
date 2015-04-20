from bs4 import BeautifulSoup
import bs4
import urllib2 
import codecs
import re
import unicodedata
from gensim import corpora,models,similarities

def scrape(phrase, num_results):
    phrase = phrase.replace(' ', '+')
    urls = []
    for i in range(0,num_results,10):
        urls.append('https://www.google.com/search?q='+ phrase +'&oq='+ phrase +'&aqs=chrome..69i57j0l5.375j0j4&sourceid=chrome&es_sm=93&ie=UTF-8&start='+ str(i))
    #url = 'https://google.com/webhp?hl=en&gws_rd=ssl#hl=en&q=' + phrase
    html = ""
    for url in urls:
        request = urllib2.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

        response = urllib2.urlopen(request)
        html += response.read()
        #print  html
    soup = BeautifulSoup(html)
    #print soup.prettify()
    headers = soup.findAll("h3", attrs={'class':'r'})
    links = []

    for h3 in headers:
        link = str(h3.find("a"))
        link2 = re.search("<a href=\"\/url\?q=(.*)&amp;sa=U&amp;", link)
        if link2 != None:
            link = link2.group(1)
            link.join('')
        print link
        links.append(link)
    #print links

    documents = []
    for link in links:

        request = urllib2.Request(link, headers={'User-Agent': 'Mozilla/5.0'})
        try:
            html = urllib2.urlopen(request).read()
        except:
            print "error opening", link
            continue
        soup = BeautifulSoup(html)
        texts = soup.findAll(text=True)
        #print type(texts[1])
        #for property, value in vars(texts[1]).iteritems():
            #print property    

        visible_texts = filter(lambda x: x != None and x != unicode('\n'), map(visible, texts))
       
        visible_texts = map(lambda x: unicodedata.normalize('NFKD', x).encode('ascii','ignore')
, visible_texts)
        #print visible_texts[:10]
        text = reduce(lambda x,y: x + ' ' +y.strip(), visible_texts, '')
        text = ''.join(t for t in text.lower() if t.isalnum() or t == ' ')
        #print text
        documents.append(text)
        
        #response = urllib2.urlopen(request)
        #html = response.read()
        #print html
        #test = BeautifulSoup(html)
        #text = test.prettify()
    return documents

def topic_model(docs):
    stoplist = set('for a of the and to in'.split())
    texts = [[word for word in document.lower().split() if word not in stoplist]  for document in docs]
    
    from collections import defaultdict
    freq = defaultdict(int)
    for text in texts:
        for token in text:
            freq[token] += 1


    texts = [[token for token in text if freq[token] > 1] for text in texts]

    from pprint import pprint
    #print(texts[1:10])
    dictionary = corpora.Dictionary(texts)
    #pprint(dictionary)
    corpus = [dictionary.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize('/tmp/corpus.mm', corpus)
    corpus = corpora.MmCorpus('/tmp/corpus.mm')
    lsi = models.lsimodel.LsiModel(corpus=corpus, num_topics = 10) 
    print lsi
    lsi.print_topics()
    print("done")
from collections import defaultdict

def sort_dict(x):
    import operator
    return sorted(x.items(), key=operator.itemgetter(1))

def count_words(docs):
    freq = defaultdict(int)
    for doc in docs:
        doc = doc.split()
        for word in doc:
            freq[word] += 1
    sorted_list = sort_dict(freq)
    sorted_list.reverse()
    stopwords = set(['and', 'or', 'a', 'an', 'the', 'but', 'of'])
    for element in sorted_list:
        if element[0] not in stopwords and len(element[0]) > 2:
            print element[0], element[1]




def visible(element):

    if element.parent.name in ['style', 'script', '[document]', 'head', 'title']:
        return None
    elif isinstance(element,bs4.element.Comment):
        return None
    return unicode(element)
docs = scrape('what is computer science',10)
count_words(docs)
#model = topic_model(docs)
