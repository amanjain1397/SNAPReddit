# SNAPReddit
Link Prediction on hyperlinks graph of Reddit, where the network represents the connections between two subreddits. 

Used data from SNAP (Stanford Network Analysis Project) : http://snap.stanford.edu/data/soc-RedditHyperlinks.html

# Data
The data is downloaded from SNAP Social Network: Reddit Hyperlink Network.

Number of files required = 2

1) Network of subreddit-to-subreddit hyperlinks extracted from hyperlinks in the body of the post. [**Download here**](http://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv)

2) Network of subreddit-to-subreddit hyperlinks extracted from hyperlinks in the title of the post. [**Download here**](http://snap.stanford.edu/data/soc-redditHyperlinks-title.tsv).

The *.csv files downloaded above need to be stored in ./input/ of your project directory.

# Workthrough
Implementation of the project can be found here: https://medium.com/@amanjain1397/link-prediction-in-directed-subreddit-hyperlink-network-part-1-fe867800b9e3
