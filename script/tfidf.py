import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('data/pn.tsv', sep="\t", header=None, names=['num', 'label', 'text', 'judges', 'usage'])

"""
df_bin = df[df["label"] != 0]
df_bin["label"] = (df_bin["label"] == 1).astype(int)

train_df = df_bin[df_bin['usage'] == 'train']
dev_df   = df_bin[df_bin['usage'] == 'dev']
test_df  = df_bin[df_bin['usage'] == 'test']
"""

train_df = df[df['usage'] == 'train']
dev_df   = df[df['usage'] == 'dev']
test_df  = df[df['usage'] == 'test']

print(train_df.info())
print(dev_df.info())
print(test_df.info())

vectorizer = TfidfVectorizer()
train_tfidf = vectorizer.fit_transform(train_df["text"].values)
dev_tfidf   = vectorizer.transform(dev_df["text"].values)
test_tfidf  = vectorizer.transform(test_df["text"].values)

save_npz("data/X_train.npz", train_tfidf)
save_npz("data/X_dev.npz", dev_tfidf)
save_npz("data/X_test.npz", test_tfidf)

np.savez('data/ylabel', y_train= train_df['label'].values,
                       y_dev= dev_df['label'].values,
                       y_test= test_df['label'].values)

