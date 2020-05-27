# SNAPReddit
Link Prediction on hyperlinks graph of Reddit, where the network represents the connections between two subreddits. We use [node2vec](https://snap.stanford.edu/node2vec/) for generating node embeddings. The binary operations on the two node features gives us the edge features, which can be used for link prediction.  

Used data from SNAP (Stanford Network Analysis Project) : http://snap.stanford.edu/data/soc-RedditHyperlinks.html

### Data
The data is downloaded from SNAP Social Network: Reddit Hyperlink Network.
Number of files required = 2

1) Network of subreddit-to-subreddit hyperlinks extracted from hyperlinks in the body of the post. [**Download here**](http://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv)

2) Network of subreddit-to-subreddit hyperlinks extracted from hyperlinks in the title of the post. [**Download here**](http://snap.stanford.edu/data/soc-redditHyperlinks-title.tsv).

The *.csv files downloaded above need to be stored in .src//input/ of your project directory.

### Getting Started
- Clone this repo:
```bash
git clone https://github.com/amanjain1397/SNAPReddit.git
cd SNAPReddit
```
- Usage
```bash
cd src
python main.py --help
usage: main.py [-h] [--task TASK] [--input [INPUT]] [--regen] [--iter ITER]
               [--workers WORKERS] [--num_experiments NUM_EXPERIMENTS]
               [--weighted] [--unweighted] [--directed] [--undirected]

Run node2vec.

optional arguments:
  -h, --help            show this help message and exit
  --task TASK           Task to run, one of 'gridsearch', 'edgeencoding', and
                        'sensitivity' (default: None)
  --input [INPUT]       Input graph path (default: ./input/graph.txt)
  --regen               Regenerate random positive/negative links (default:
                        False)
  --iter ITER           Number of epochs in SGD (default: 1)
  --workers WORKERS     Number of parallel workers. Default is 1. (default: 1)
  --num_experiments NUM_EXPERIMENTS
                        Number of experiments to average. Default is 5.
                        (default: 5)
  --weighted            Boolean specifying (un)weighted. Default is
                        unweighted. (default: False)
  --unweighted
  --directed            Graph is (un)directed. Default is undirected.
                        (default: False)
  --undirected
```

### Working Example
First we must prepare the text file for the graph edgelist.
```bash
cd src
python prepare.py
```
This creates the edgelist file at **./src/input/graph.txt**.
```bash
python main.py --tasks gridsearch --workers 2
```

### More Info
Learn about Stanford Network Analysis Project (SNAP) from [here](http://snap.stanford.edu/). The reference python implementation of node2vec can be found [here](https://github.com/aditya-grover/node2vec).
