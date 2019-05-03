# SNAPReddit
Link Prediction on directed, signed graph of the hyperlink in Reddit, where the network represents the directed connections between two subreddits. 

Used data from SNAP (Stanford Network Analysis Project) : http://snap.stanford.edu/data/soc-RedditHyperlinks.html

# Data
The data is downloaded from SNAP Social Network: Reddit Hyperlink Network.

Number of files required = 3

1) Network of subreddit-to-subreddit hyperlinks extracted from hyperlinks in the body of the post. [**Download here**](http://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv)

2) Network of subreddit-to-subreddit hyperlinks extracted from hyperlinks in the title of the post. [**Download here**](http://snap.stanford.edu/data/soc-redditHyperlinks-title.tsv).

3) Embedding vectors of subreddits (communities on Reddit). [**Download here**](http://snap.stanford.edu/data/web-redditEmbeddings-subreddits.csv). This file generates one numerical vector in low dimensional space (a.k.a. embeddings) for each subreddit. The embeddings are 300 dimensions each. Two subreddit embeddings are similar if the users who post in them are similar.

The *.csv files downloaded above need to be stored in ./input/ of your project directory.

# Workthrough
Implementation of the project can be found here: https://medium.com/@amanjain1397/link-prediction-in-directed-subreddit-hyperlink-network-part-1-fe867800b9e3
