import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

def to_TF_IDF(texts):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix, vectorizer

df = pd.read_csv('data/pn.tsv', sep="\t", header=None, names=['num', 'label', 'text', 'judges', 'usage'])

train_df = df[df['usage'] == 'train']
dev_df   = df[df['usage'] == 'dev']
test_df  = df[df['usage'] == 'test']

print(train_df.info())
print(dev_df.info())
print(test_df.info())

train_tfidf, _ = to_TF_IDF(train_df['text'].values)
dev_tfidf, _ = to_TF_IDF(dev_df['text'].values)
test_tfidf, _ = to_TF_IDF(test_df['text'].values)

np.savez('data/tfidf', X_train = train_tfidf, y_train= train_df['label'].values,
                       X_dev = dev_tfidf, y_dev= dev_df['label'].values,
                       X_test = test_tfidf, y_test= test_df['label'].values)

