from bs4 import BeautifulSoup
import bs4
import urllib2 
import codecs
import re
import unicodedata
from gensim import corpora,models,similarities
from collections import defaultdict
import operator
import sys

def scrape(phrase, num_results):
    #replace spaces with plusses to convert them for the url
    phrase = phrase.replace(' ', '+')
    
    #generate the URLs of the google searches
    urls = []
    for i in range(0,num_results,10):
        urls.append('https://www.google.com/search?q='+ phrase +'&oq='+ phrase +'&aqs=chrome..69i57j0l5.375j0j4&sourceid=chrome&es_sm=93&ie=UTF-8&start='+ str(i))
    
    #collect the html - would be nice to do this in parallel
    html = ""
    for url in urls:
        request = urllib2.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

        response = urllib2.urlopen(request)
        html += response.read()
    

    #select out the links associated with each search result
    soup = BeautifulSoup(html)    
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
  
    #make the HTML requests and store the results
    documents = []
    for link in links:
        request = urllib2.Request(link, headers={'User-Agent': 'Mozilla/5.0'})
        try:
            html = urllib2.urlopen(request).read()
        except: #should figure out root cause here
            print "error opening", link
            continue
        soup = BeautifulSoup(html)
        texts = soup.findAll(text=True)
        
        #get rid of empty lines
        visible_texts = filter(lambda x: x != None and x != unicode('\n'), map(visible, texts))
       
        #convert unicode to ascii
        visible_texts = map(lambda x: unicodedata.normalize('NFKD', x).encode('ascii','ignore')
, visible_texts)
        
        # concatenate lines
        text = reduce(lambda x,y: x + ' ' +y.strip(), visible_texts, '')
        
        #remove everything but words and spaces
        text = ''.join(t for t in text.lower() if t.isalnum() or t == ' ')

        documents.append(text)
        
    return documents


"""This isn't quite working yet - still trying to figure out gensim, since I think it is 
doing some sort of supervised topic modeling and I am only familar with unsupervised
"""
def topic_model(docs):
    stoplist = set('for a of the and to in'.split())
    
    #get rid of common words (articles, prepositions, conjunctions
    texts = [[word for word in document.lower().split() if word not in stoplist]  for document in docs]
    
    freq = defaultdict(int)
    for text in texts:
        for token in text:
            freq[token] += 1
    
    #remove common  
    texts = [[token for token in text if freq[token] > 1] for text in texts]

    dictionary = corpora.Dictionary(texts)

    corpus = [dictionary.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize('/tmp/corpus.mm', corpus)
    corpus = corpora.MmCorpus('/tmp/corpus.mm')
    lsi = models.lsimodel.LsiModel(corpus=corpus, num_topics = 10) 
    print lsi
    lsi.print_topics()
    print("done")
def sort_dict(x):
    import operator
    return sorted(x.items(), key=operator.itemgetter(1))

def count_words(docs):
    
    #count up the words
    freq = defaultdict(int)
    for doc in docs:
        doc = doc.split()
        for word in doc:
            freq[word] += 1
    
    #sort the list in descending order
    sorted_list = sort_dict(freq)
    sorted_list.reverse()
    stopwords = set(['and', 'or', 'a', 'an', 'the', 'but', 'of'])
    
    #print out words
    for element in sorted_list:
        if element[0] not in stopwords and len(element[0]) > 2:
            print element[0], element[1]
    



def visible(element):

    if element.parent.name in ['style', 'script', '[document]', 'head', 'title']:
        return None
    elif isinstance(element,bs4.element.Comment):
        return None
    return unicode(element)

if __name__ == "__main__":
    argc = len(sys.argv)
    if argc < 3:
        print "usage: python scrapegoogle.py <num_results> <search_query>"
        sys.exit(1)
    num_res = int(sys.argv[1])
    query = ' '.join(sys.argv[i] for i in xrange(2, argc))
    print "Query:", query
    print "Number of results:", num_res
    docs = scrape(query,num_res)
    count_words(docs)
    #model = topic_model(docs)
