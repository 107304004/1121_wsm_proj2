import argparse
from search import *
from util import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="indexes/collection", type=str)
    parser.add_argument("--query", default="../data/topics.401-450.txt", type=str)
    parser.add_argument("--method", default="bm25", type=str)
    parser.add_argument("--k", default=2000, type=int)
    parser.add_argument("--output", default='runs/bm25.run', type=str)
    
    args = parser.parse_args()

    if args.method == "bm25":
        searcher = LuceneSearcher(args.index)
        searcher.set_bm25(k1=2, b=0.75)
    elif args.method == "dir":
        searcher = LuceneSearcher(args.index)
        searcher.set_qld(mu=1000)
    elif args.method == "jm":
        searcher = JMSearcher(args.index)
        searcher.set_qljm(jm_lambda=0.8)

    query = read_title(args.query)
    search(searcher, query, args)
