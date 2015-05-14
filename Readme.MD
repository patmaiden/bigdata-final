This repository contains the code for Patrick Maiden's final project for PHP2650: Statistical Mehods for Big Data. This project focuses on analyzing googl search results using topic modelling.

The functions used for analysis live in functions.py. I should probably break up this file at some point in order to make the individual pieces more resuable.

analysis.py is an interface that allow you to choose a search query and number of results. Then, it will either load chached data or scrape Google for new results. Using these data, you can run different analyses, such as word frequency analysis and topic modelling. 

Direct any questions to patmaiden@gmail.com
