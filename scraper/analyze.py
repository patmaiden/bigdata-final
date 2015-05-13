

import functions as func
import sys
import os

def run_loop():
    print_help()
    while True:
        print "Please enter your search query:",
        query = raw_input().strip()

        print "How many results would you like to scrape (will be rounded to closest multiple of 10:",
        num_res = int(raw_input())/10*10
        base_fn = 'scrape_data/'+ query.replace(' ','_') + '_' + str(num_res)
        fn = base_fn + '.p'
        link_fn = base_fn + '_urls.p'
        
        if os.path.exists(fn):
            print "It looks like this search has been one before, so you can use the saved data. Would you like to re-scrape the data? (y/n):",
            response = raw_input().lower()
            while not (response == 'y' or response =='n'):
                print "Please enter 'y' or 'n'"
                response = raw_input.lower()
            if response == 'y':
                docs, links = func.scrape(query, num_res)
                func.save_dict(docs, links, base_fn)
            else:
                docs = func.load_dict(fn)
                links = func.load_dict(link_fn)
        else:
            docs, links = func.scrape(query, num_res)
            func.save_dict(docs, links, base_fn)
        docs = func.flatten_list(func.clean_vec(docs, set(query.split())))
        while True: 
            print "Please enter your command. Press h to repeat the options:",
            entry = raw_input().strip()

            if entry == 'f':
                res = func.count_words(docs)     
                print "\nprinting the top 15 most frquent words\n"
                for x in res[:15]:
                    print x[0],x[1]    
            elif entry == 'w':
                print "\n  generating word cloud\n"
                func.make_cloud(docs)

            elif entry == 't':
                print "\n Topic modeling and generating bargraph of document distributions"
                print "\nHow many topics would you like?:",
                num_top = int(raw_input().strip())
                counts, names = func.make_vectorizer(docs)
                model, dist = func.topic_model(counts, names, topics=num_top)
                clusters, percents, lookup = func.assign_cluster(dist, links,n_clusters=num_top)
                func.plot_bargraph(percents)
            
            elif entry == 'g':
                print "\n topic modeling ang building graph with 6 topics\n"
                print "\nHow many topics would you like? (Limit 10 for visualiztion):",
                num_top = int(raw_input().strip())
               
                
                print "\nPlease set the threshold for a graph connection"
                print "The default is 3. Higher threshold means more connections:",
                thresh = int(raw_input().strip())
                counts, names = func.make_vectorizer(docs)
                model, dist = func.topic_model(counts, names, topics=num_top)
                
               
                clusters, percents, lookup = func.assign_cluster(dist,links)
                G, cmap = func.build_graph(dist, links, lookup, thresh)
                print "The colors are as follows:"
                for i in xrange(num_top):
                    print "Topic", i, "    Color:", cmap[i]

                func.show_graph(G)

            elif entry == 'h':
                print_help()
            
            elif entry == 'r':
                break
            
            elif entry == 'e':
                sys.exit(0)
            
            else:
                sys.stderr.write("\nInvalid command\n")
                print_help()
            

def print_help():
    string = "\nHere are the options\n"
    string += "f - word frequency\n"
    string += "t - topic model with bar graph\n"
    string += "g - topic model with network\n"
    string += "r - reset search query\n"
    string += "e - exit\n"
    print string








if __name__ == "__main__":
    run_loop()
