from bs4 import BeautifulSoup
import bs4
import urllib2 
import codecs
import re
import unicodedata
from collections import defaultdict
import operator
import sys
from nltk.corpus import stopwords
from nltk import word_tokenize
import pickle
import enchant
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import lda
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
import matplotlib.pyplot as plt
import math
import networkx
import colour as cl

"""Scrapes google for the search query
and returns clean HTML of the pages"""
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
    to_remove = []
    
    for link in links:
        request = urllib2.Request(link, headers={'User-Agent': 'Mozilla/5.0'})
        try:
            html = urllib2.urlopen(request).read()
        except: #should figure out root cause here
            print "error opening", link
            to_remove.append(link)
            continue
        soup = BeautifulSoup(html)
        text = soup.findAll(text=True)
         
        #get rid of empty lines
        visible_texts = filter(lambda x: x != None and x != unicode('\n'), map(visible, text))
       
        #convert unicode to ascii
        visible_texts = map(lambda x: unicodedata.normalize('NFKD', x).encode('ascii', 'ignore'), visible_texts)

        # concatenate lines
        text = reduce(lambda x,y: x + ' ' +y.strip(), visible_texts, '')
        
        #remove everything but words and spaces
        text = ''.join(t for t in text.lower() if t.isalnum() or t == ' ')

        
        documents.append(text)
    
    #pull out bad links
    for x in to_remove:
        links.remove(x)

    return documents, links
"""Helper for scrape that returns text only if it is visible"""
def visible(element):
    if element.parent.name in ['style', 'script', '[document]', 'head', 'title']:
        return None
    elif isinstance(element,bs4.element.Comment):
        return None
    return unicode(element)


"""Saves the input dictionary for later use"""
def save_dict(docs, links,query_nr):
    pickle.dump(docs, open(query_nr + '.p',"wb"))
    pickle.dump(links, open(query_nr + '_urls.p','wb'))

"""loads a dictionary at the given filename"""
def load_dict(fn):
    return pickle.load(open(fn, "rb"))

"""Generates a list of (word,frequency pairs)"""
def count_words(docs):
    stop = set(stopwords.words('english'))
    stop = stop.union({'like','may','many','one','blog'})
    literal = set(['data', 'big', 'big data']) #set(['data','science','scientist','scientists', 'data science', 'data scientist', 'data scientists'])
    
    #count up the words
    freq = defaultdict(int)
    check_stop = lambda word: word not in stop and len(word) > 2
    for doc in docs:
        doc = doc.split()
        for word in doc:
            if check_stop(word) and  word not in literal:
                freq[word] += 1.0
            else:
                doc.remove(word)
        for word_ind in xrange(1,len(doc)):
            phrase = doc[word_ind - 1] + ' ' + doc[word_ind]
            if check_stop(doc[word_ind-1]) and check_stop(doc[word_ind]) and  phrase not in literal:
                freq[phrase] += 1.5 #bigrams get small bump
    
    #sort the list in descending order
    sorted_list = sort_dict(freq)
    sorted_list.reverse()
    
    return sorted_list 

"""Helper function for count_words that sorts the input dictionary"""
def sort_dict(dictionary):
    item_list = dictionary.items()
    import operator
    return sorted(item_list, key=operator.itemgetter(1))

"""Generates wordcloud from documents"""
def make_cloud(docs):
    flat_doc = count_words(docs)
    from wordcloud import WordCloud
    import wordcloud
   
    wc = WordCloud(ranks_only = True, font_path='/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf')
    wc.fit_words(flat_doc)
    
    
    plt.imshow(wc)
    plt.axis("off")
    plt.show()

    
"""cleans the string of literal and nonenglish words and tokenizes it"""   
def clean_vec(docs, literal):
    d = enchant.Dict("en_US")
    stop = set(stopwords.words('english'))
    stop = stop.union({'like','may'})

    #function to return whether to include word or not
    check_stop = lambda doc: word not in literal and word not in stop and len(word) > 2 and d.check(word)
    
    to_ret = []
    for doc in docs:
        tmp = []
        doc = doc.split()
        for word in doc:
            if check_stop(word):
                tmp.append(word)
        to_ret.append(tmp)
    return to_ret

"""joins a list of list of strings into a list of strings"""
def flatten_list(docs):
    out = []
    for doc in docs:
        out.append(' '.join(word for word in doc))
    return out

"""generates the wordcount vectorizer for topic modeling"""
def make_vectorizer(docs):
    vec = CountVectorizer(min_df = 5, ngram_range=(1,2))
    counts = vec.fit_transform(docs)
    names = vec.get_feature_names()
    return counts, names

"""generates the tfidf vectorizer for k-means"""
def make_tfidf_vec(docs):
    vec = TfidfVectorizer(min_df=5, ngram_range=(1,2))
    tfidf = vec.fit_transform(docs)
    names = vec.get_feature_names()
    return tfidf, names

"""generates a k-means model for the input data"""
def run_kmeans(vector=None, links=[], iters=500, clusters=8):
    km = KMeans(n_clusters=clusters, max_iters=iters)
    km.fit_transform(vec)
    clusters = defaultdict(list)
    for i in xrange(len(links)):
        clusters[km.labels[i]].append(links[i])
    for x in clusters:
        print x, clusters[x]
    return km.labels_

"""generates a topic model for the input data"""
def topic_model(counts, names,iters=500, topics = 6, rand_state=1 ):
    model = lda.LDA(n_iter=iters, n_topics=topics, random_state=rand_state)
    dist = model.fit_transform(counts)
    print_topics(model, names)
    return model, dist

"""prints out the top 20 words of each topic"""
def print_topics(model, names):
    topic_word = model.topic_word_
    n_top_words = 20 
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(names)[np.argsort(topic_dist)][:-n_top_words:-1]
        print("Topic {}: {}".format(i," | ".join(topic_words)))

"""Assigns a document its highest cluster given its distribution"""
def assign_cluster(dist, links, n_clusters=6):
    clusters = [[] for i  in xrange(n_clusters)]
    rev_lookup = dict()
    for i in xrange(len(links)):
        cluster = max( (v,j) for j,v in enumerate(dist[i]) ) [1]
        clusters[cluster].append(links[i])
        rev_lookup[links[i]] = cluster
    percents = []
    for i in xrange(n_clusters):
        percents.append(float(len(clusters[i]))/float(len(links)))
    return clusters, percents, rev_lookup

"""plots a bargraph of the % given towards each topic"""
def plot_bargraph(percents,width=.8):
    topics = xrange(len(percents))
    barlist = plt.bar(topics, percents, width=width)
    plt.xticks(np.arange(len(topics)) + width/2.,topics)
    plt.xlabel('Topic number')
    plt.show()

"""Generates a list of edges between documents along with their edge weights"""
def find_matches(dist, links, thresh=3):
    edges = []
    for i in xrange(len(links)):
        for j in xrange(i+1, len(links)):
            link1 = links[i]
            link2 = links[j]
            weight = determine_edge(dist[i],dist[j])
            if weight < thresh:
                edges.append((i,j,{'weight':weight}))
    return edges

"""helper to compute distance of log of distributions"""
def determine_edge(dist1,dist2):
    import math 
    diff = 0 
    for x,y in zip(dist1,dist2):
        diff += abs(math.log(x)-math.log(y))
    return diff 

"""builds the graph with documents as nodes and distances as edges"""
def build_graph(dist, links, lookup, thresh=3):
    import networkx as nx
    import colour as cl
    color_map = {0:'red',1:'blue', 2:'green', 3:'yellow',4:'orange', 5:'purple', 6: 'pink', 7:'magenta', 8:'brown', 9:'black' }
    G = nx.Graph()
    for i, l in enumerate(links):
        color = cl.Color(color_map[lookup[l]])
        G.add_node(i,label=l, cluster=lookup[l], r =int(color.red*255.0), b=int(color.blue*255.0),g=int(color.green*255.0))
    edges = find_matches(dist, links, thresh)
    G.add_edges_from(edges)
    return G, color_map

"""displays the given graph"""
def show_graph(G):
    import networkx as nx
    color_map = {0:'red',1:'blue', 2:'green', 3:'yellow',4:'orange', 5:'purple'}
    nx.draw(G,node_color=[color_map[G.node[node]['cluster']] for node in G])
    plt.show()



"""This was used for development, but now I made the interface at analyze.py, which 
is a much nicer way to interact with these tools"""
if  __name__ == "__main__":
    argc = len(sys.argv)
    if argc < 3:
        print "usage: python scrapegoogle.py <num_results> <search_query>"
        sys.exit(1)

    num_res = int(sys.argv[1])
    query = ' '.join(sys.argv[i] for i in xrange(2, argc))
    print "Query:", query
    print "Number of results:", num_res
    print "starting to scrape"
    literal = set(query.split())
    docs, links = scrape(query,num_res)
    print "saving dictionary"
    query = query.replace(' ','_')
    fn = query + '_' + str(num_res) + '.p'    
    print fn
    save_dict(docs,links,query + '_' + str(num_res))
    print "reloading dict"
    links = load_dict(query + '_' + str(num_res) + '_urls.p')
    docs = clean_vec(load_dict(fn), literal)
    print "cleaning data"
    flat_doc = flatten_list(docs)
    print "vectorizing data"
    counts, names = make_vectorizer(flat_doc)
    print "topic modeling"
    model,dist = topic_model(counts, names)
    print len(links), links
    clusters, percents, lookup = assign_cluster(dist, links)
    plot_bargraph(percents)
    

 
