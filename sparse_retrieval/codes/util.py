import re
import pandas as pd
from pyserini.pyclass import autoclass
from pyserini.search.lucene import LuceneSearcher


class JMSearcher(LuceneSearcher):

    def __init__(self, index):
        super().__init__(index)

    def set_qljm(self, jm_lambda=float(0.5)):
        """Configure query likelihood with Jelinek-Mercer smoothing as the scoring function.

        Parameters
        ----------
        jm_lambda: float
            Jeliek-Mercer smoothing parameter lambda.
        """
        LMJMSimilarity = autoclass('org.apache.lucene.search.similarities.LMJelinekMercerSimilarity')(float(jm_lambda))
        self.object.searcher = autoclass('org.apache.lucene.search.IndexSearcher')(self.object.reader)
        self.object.searcher.setSimilarity(LMJMSimilarity)


def read_title(query_path):
    with open(query_path, 'r', encoding='utf-8') as f:
        texts = f.read()
        qids = re.findall('<num> Number: (.*?)\n', texts)
        titles = re.findall('<title> (.*?)\n\n<desc>', texts)

    query = dict()
    for qid, title in zip(qids, titles):
        query[qid.strip()] = title

    return query


def read_topic(query_path):
    with open(query_path, 'r', encoding='utf-8') as f:
        texts = f.read()
        qids = re.findall('<num> Number: (.*?)\n', texts)
        descs = re.findall('<top>\n\n(.*?)\n</top>', texts, re.S)

    query = dict()
    for qid, desc in zip(qids, descs):
        query[qid.strip()] = desc

    return query


def read_run(run_path):
    colnames = ['qid', 'constant', 'docid', 'rank', 'score', 'method']
    df = pd.read_csv(run_path, sep=" ", header=None, names=colnames)

    method = df['method'][0]
    df.rename(columns={'score': method+'_score'}, inplace=True)

    df['ids'] = df['qid'].apply(lambda x: str(x)) + "_" + df['docid']
    df = df[['ids', 'rank', method+'_score']]
    df.rename(columns={'rank': method+'_rank'}, inplace=True)

    return df


def read_qrels(qrels_path):
    colnames = ['qid', 'constant', 'docid', 'relevance']
    df = pd.read_csv(qrels_path, sep=" ", header=None, names=colnames)

    df['ids'] = df['qid'].apply(lambda x: str(x)) + "_" + df['docid']
    df = df[['ids', 'relevance']]

    return df
