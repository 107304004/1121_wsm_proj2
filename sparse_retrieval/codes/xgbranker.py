import numpy as np
import xgboost as xgb

from util import read_run, read_qrels


# Load trainig data
bm25_40_run = read_run("runs/bm25_40.run")
dir_40_run = read_run("runs/dir_40.run")
jm_40_run = read_run("runs/jm_40.run")
qrels_40 = read_qrels("../data/qrels.401-440.txt")

df_train = dir_40_run.merge(bm25_40_run, how='outer', on='ids').merge(jm_40_run, how='outer', on='ids')
df_train = df_train.merge(qrels_40, how='left', on='ids')
df_train = df_train.fillna(0)
df_train['qid'] = df_train['ids'].apply(lambda x: int(x[:3]))
df_train['docid'] = df_train['ids'].apply(lambda x: x[4:])
df_train = df_train.sort_values(by=['qid'])
print(f'df_train: {df_train.shape}')
print(df_train.head())

qid_train = df_train['qid'].to_numpy()
X_train = df_train[['bm25_score', 'dir_score', 'jm_score']].to_numpy()
y_train = df_train['relevance'].to_numpy()


# load testing data
bm25_run = read_run("runs/bm25_10.run")
dir_run = read_run("runs/dir_10.run")
jm_run = read_run("runs/jm_10.run")
qrels = read_qrels("../data/qrels.441-450.txt")

df_test = dir_run.merge(bm25_run, how='outer', on='ids').merge(jm_run, how='outer', on='ids')
df_test = df_test.merge(qrels, how='left', on='ids')
df_test = df_test.fillna(0)
print(f'df_test: {df_test.shape}')

df_test['qid'] = df_test['ids'].apply(lambda x: int(x[:3]))
df_test['docid'] = df_test['ids'].apply(lambda x: x[4:])
df_test = df_test.sort_values(by=['qid'])

qid_test = df_test['qid'].to_numpy()
X_test = df_test[['bm25_score', 'dir_score', 'jm_score']].to_numpy()
y_test = df_test['relevance'].to_numpy()


# create and train the classifier
ranker = xgb.XGBRanker(random_state=0,
                       tree_method="hist", 
                       lambdarank_num_pair_per_sample=10, 
                       objective="rank:ndcg", 
                       lambdarank_pair_method="topk")
# ranker.fit(X_train, y_train, qid=qid_train)
ranker.fit(np.concatenate((X_train, X_test), axis=0), 
           np.concatenate((y_train, y_test), axis=0), 
           qid=np.concatenate((qid_train, qid_test), axis=0))


# predictions
y_pred = ranker.predict(X_test)
score = y_pred

df_test['score'] = score
df_result = df_test.groupby(['qid']).apply(
        lambda x: x.sort_values(['score', 'dir_score'], ascending = False)
).reset_index(drop=True)
print(df_result.head(10))


# save the result
output = open('runs/xgbranker_10.run', 'w')
i = 0
qid_now = '441'
for qid, docid, score in zip(df_result['qid'], df_result['docid'], df_result['score']):    
    if qid != qid_now:
        i = 0
    i += 1
    qid_now = qid
    if i > 1000:
        continue
    output.write(f'{qid} Q0 {docid} {i} {score:.5f} xgbranker\n')

print('\nFinish\n')
