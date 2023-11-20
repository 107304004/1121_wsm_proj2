from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from util import read_run, read_qrels


# Load trainig data
bm25_40_run = read_run("runs/bm25_40.run")
dir_40_run = read_run("runs/dir_40.run")
jm_40_run = read_run("runs/jm_40.run")
qrels_40 = read_qrels("../data/qrels.401-440.txt")


df_train = bm25_40_run.merge(dir_40_run, how='outer', on='ids').merge(jm_40_run, how='outer', on='ids')
df_train = df_train.merge(qrels_40, how='left', on='ids')
df_train = df_train.fillna(0)
df_train = df_train.sample(frac=1, random_state=0)
# print(df_train.describe())
print(f'df_train: {df_train.shape}')
print(df_train.head())

X_train = df_train[['bm25_score', 'dir_score', 'jm_score']]
y_train = df_train['relevance']


# load testing data
bm25_run = read_run("runs/bm25_10.run")
dir_run = read_run("runs/dir_10.run")
jm_run = read_run("runs/jm_10.run")
qrels = read_qrels("../data/qrels.441-450.txt")

df_test = bm25_run.merge(dir_run, how='outer', on='ids').merge(jm_run, how='outer', on='ids')
df_test = df_test.fillna(0)
print(f'df_test: {df_test.shape}')

X_test = df_test[['bm25_score', 'dir_score', 'jm_score']]


# feature scaling (important for convergence & accuracy)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# create and train the classifier
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)


# predictions
y_pred = classifier.predict_proba(X_test)
score = y_pred[:, 1]
X_test = scaler.inverse_transform(X_test)

df_test['score'] = score
df_test['qid'] = df_test['ids'].apply(lambda x: x[:3])
df_test['docid'] = df_test['ids'].apply(lambda x: x[4:])

df_result = df_test.groupby(['qid']).apply(
        lambda x: x.sort_values(['score', 'dir_score'], ascending = False)
).reset_index(drop=True)
print(df_result.head(10))


# save result
output = open('runs/logistic_regression_10.run', 'w')
i = 0
qid_now = '441'
for qid, docid, score in zip(df_result['qid'], df_result['docid'], df_result['score']):    
    if qid != qid_now:
        i = 0
    i += 1
    qid_now = qid
    if i > 1000:
        continue
    output.write(f'{qid} Q0 {docid} {i} {score:.5f} logistic_regression\n')

print('\nFinish\n')
